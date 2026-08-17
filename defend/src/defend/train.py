"""Entrypoint: python -m defend.train

Loads generate/data/transactions.csv (read-only), trains the primary XGBoost
classifier, evaluates it aggregate + per-category + per-vector, and saves the
model + evaluation report to defend/model/ and defend/results/. E1/disputes.csv
is excluded this pass - see data.py's module docstring for why.
"""

from __future__ import annotations

from .config import RANDOM_SEED, TEST_SIZE, VECTOR_ORDER
from .data import build_dataset
from .diagnostics import amount_range_overlap, missingness_profile, prediction_confidence_spread
from .evaluate import evaluate
from .features import build_features
from .model import feature_importance, train_model
from .persist import save_model_artifacts
from .report import write_results
from .split import split_dataset


def main() -> None:
    df = build_dataset()
    train_df, test_df = split_dataset(df, test_size=TEST_SIZE, seed=RANDOM_SEED)

    X_train, y_train, spec = build_features(train_df)
    X_test, y_test, _ = build_features(test_df, category_levels=spec.category_levels)

    model = train_model(X_train[spec.feature_columns], y_train, seed=RANDOM_SEED)
    y_proba = model.predict_proba(X_test[spec.feature_columns])[:, 1]

    eval_result = evaluate(test_df, y_proba)
    fi = feature_importance(model, spec.feature_columns)
    top_features = fi.head(6)["feature"].tolist()
    missingness_df = missingness_profile(df, top_features, VECTOR_ORDER)
    amount_overlap_df = amount_range_overlap(df, VECTOR_ORDER)
    confidence_df = prediction_confidence_spread(test_df, y_proba, VECTOR_ORDER)

    save_model_artifacts(model, spec)

    dataset_stats = {
        "Total rows": len(df),
        "Total fraud rows": int(df["is_fraud"].astype(bool).sum()),
        "Train rows": len(train_df),
        "Test rows": len(test_df),
        "Test fraud rows": int(test_df["is_fraud"].astype(bool).sum()),
        "Feature count": len(spec.feature_columns),
        "Random seed": RANDOM_SEED,
        "Test size": TEST_SIZE,
    }
    write_results(eval_result, fi, dataset_stats, missingness_df, amount_overlap_df, confidence_df)

    print(f"Trained on {len(train_df)} rows, evaluated on {len(test_df)} rows.")
    print(f"Aggregate: precision={eval_result.aggregate['precision']:.3f} "
          f"recall={eval_result.aggregate['recall']:.3f} "
          f"f1={eval_result.aggregate['f1']:.3f} "
          f"roc_auc={eval_result.aggregate['roc_auc']}")
    print(eval_result.category_b_note)
    print(eval_result.category_e_note)
    print("Model saved to defend/model/, report saved to defend/results/report.md")


if __name__ == "__main__":
    main()
