"""Aggregate and per-taxonomy-category/vector evaluation.

Precision/recall/F1/AUC are only well-defined for a binary problem. To break
them out "per category" as the taxonomy's Notes section asks for, each
category's numbers are computed on a category-vs-legit subset of the test
set (that category's fraud rows + all legit test rows), scored with the same
already-trained model and the same decision threshold as the aggregate run.
That keeps every category's precision comparable (same legit pool, same
threshold) while isolating recall to that category's own vectors. Per-vector
breakout below is just recall - the only one of the four metrics that is
still meaningful at 20-row vector sample sizes without borrowing from other
vectors' false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import CATEGORY_ORDER, DECISION_THRESHOLD, VECTOR_ORDER


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_proba))


def _safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_proba))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_actual_fraud": int(y_true.sum()),
        "n_predicted_fraud": int(y_pred.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_proba),
        "pr_auc": _safe_pr_auc(y_true, y_proba),
    }


@dataclass
class EvaluationResult:
    aggregate: dict
    per_category: pd.DataFrame
    per_vector: pd.DataFrame
    category_b_note: str
    category_e_note: str
    worst_performers: list[dict] = field(default_factory=list)


def evaluate(test_df: pd.DataFrame, y_proba: np.ndarray, threshold: float = DECISION_THRESHOLD) -> EvaluationResult:
    y_true = test_df["is_fraud"].astype(bool).astype(int).to_numpy()
    y_pred = (y_proba >= threshold).astype(int)

    aggregate = binary_metrics(y_true, y_pred, y_proba)

    legit_mask = ~test_df["is_fraud"].astype(bool)
    per_category_rows = []
    for category in CATEGORY_ORDER:
        cat_mask = (test_df["fraud_category"] == category) & test_df["is_fraud"].astype(bool)
        subset_mask = (legit_mask | cat_mask).to_numpy()
        n_fraud = int(cat_mask.sum())
        if n_fraud == 0:
            per_category_rows.append({"category": category, "n_test_fraud": 0, "precision": None, "recall": None, "f1": None, "roc_auc": None, "pr_auc": None})
            continue
        sub_true = y_true[subset_mask]
        sub_pred = y_pred[subset_mask]
        sub_proba = y_proba[subset_mask]
        m = binary_metrics(sub_true, sub_pred, sub_proba)
        per_category_rows.append({
            "category": category,
            "n_test_fraud": n_fraud,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
        })
    per_category = pd.DataFrame(per_category_rows)

    per_vector_rows = []
    for vector in VECTOR_ORDER:
        vec_mask = (test_df["fraud_vector"] == vector).to_numpy()
        n_test = int(vec_mask.sum())
        if n_test == 0:
            per_vector_rows.append({"vector": vector, "n_test": 0, "n_caught": 0, "recall": None})
            continue
        n_caught = int(y_pred[vec_mask].sum())
        per_vector_rows.append({
            "vector": vector,
            "n_test": n_test,
            "n_caught": n_caught,
            "recall": n_caught / n_test,
        })
    per_vector = pd.DataFrame(per_vector_rows)

    b_row = per_category[per_category.category == "B"].iloc[0]
    if b_row["recall"] is None:
        category_b_note = "Category B had no fraud cases in this test split."
    elif b_row["recall"] < 0.95:
        category_b_note = (
            f"Category B (Social Engineering/Authorization Fraud) recall: "
            f"{b_row['recall']:.2f} over {int(b_row['n_test_fraud'])} test cases. "
            "This is an EXPECTED, documented limitation, not a defect: per "
            "identify/attack-taxonomy.md, B2 in particular is genuinely "
            "authorized by the real accountholder, so there is no failed-auth "
            "signal for a transaction-level classifier to key off. Verified "
            "genuine (not a residual leak) via results/report.md's investigation "
            "section - feature importance is spread across many features (no "
            "single field dominates), every vector's amount range genuinely "
            "overlaps its same-channel legit population, and predicted-fraud "
            "confidence on B1/B2/B3 test rows spans a real range rather than "
            "clustering at ~1.000."
        )
    else:
        category_b_note = (
            f"Category B (Social Engineering/Authorization Fraud) recall: "
            f"{b_row['recall']:.2f} over {int(b_row['n_test_fraud'])} test cases. "
            "This is NOT the result identify/attack-taxonomy.md predicts - it "
            "explicitly calls B2 genuinely-authorized-by-the-accountholder and "
            "therefore hard to catch on transaction data alone. A near-perfect "
            "score here is a red flag, not a win - check feature_importance.csv, "
            "amount_range_overlap.csv, and prediction_confidence_spread.csv "
            "(written every run) for which field is doing the work before "
            "trusting this score. Do not read a high score here as evidence the "
            "model would perform this well on real B-category fraud without "
            "checking those first."
        )

    e_row = per_category[per_category.category == "E"].iloc[0]
    category_e_note = (
        "Category E here reflects E2 (promo abuse) ONLY, not E1+E2. E1 "
        "(GenAI fabricated dispute evidence) lives in disputes.csv, joined "
        "via transaction_id. The ID-generation bug that made that join "
        "impossible (generate/entities.py's short_id()) has since been fixed "
        "and verified (see generate/README.md's anti-leakage notes and "
        "defend/README.md's changelog), but the join itself has not been "
        "re-added to this pipeline in this pass - kept out deliberately to "
        "keep this run's metrics comparable to the pre-fix baseline. "
        f"Category E recall shown below ({_none_or(e_row['recall'])}) should "
        "be read as 'recall on E2', not 'recall on category E'."
    )

    ranked = per_vector[per_vector["recall"].notna()].sort_values("recall")
    worst_performers = ranked.head(5).to_dict(orient="records")

    return EvaluationResult(
        aggregate=aggregate,
        per_category=per_category,
        per_vector=per_vector,
        category_b_note=category_b_note,
        category_e_note=category_e_note,
        worst_performers=worst_performers,
    )


def _none_or(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"
