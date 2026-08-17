"""Wraps Defend's real trained XGBoost classifier for scoring a single
proposed transaction.

This module only *reads* `defend/model/xgboost_model.json` and
`defend/model/feature_metadata.json` -- files defend/persist.py already
describes as being saved specifically "so the web prototype can reuse
it." No file under defend/ is modified, and none of defend/'s training
code is imported; the feature-encoding transform below is a small,
independent reimplementation of the same encoding
defend/src/defend/features.py uses (category dtype + fixed
category->level mapping from feature_metadata.json), needed because
mandate-demo's own project/venv is separate from defend's.

Architectural note on `transaction_within_mandate_envelope`: that column
IS one of Defend's trained features, and in the training data it's a
near-perfect predictor of category C (it's populated from the
generator's own ground truth). It is deliberately NEVER set here. The
three checks in this demo run in parallel and independently (see
runner.py) -- the mandate-envelope check's own verdict cannot be an
input to a check running alongside it, and in a live system nothing
would know that verdict before the mandate check itself computes it.
Leaving it unset means Defend's classifier has to make its call from the
transaction's other ~35 features alone, exactly as it would have to in
production before any independent mandate check has run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xgboost as xgb

DEFEND_MODEL_DIR = Path(__file__).resolve().parents[3] / "defend" / "model"


@dataclass
class ClassifierResult:
    fraud_probability: float
    decision_threshold: float
    risk_label: str  # "LOW_RISK" | "HIGH_RISK"
    features_sent: dict


class DefendClassifier:
    """Loads defend/model/*.json once and scores rows against it."""

    def __init__(self, model_dir: Path = DEFEND_MODEL_DIR) -> None:
        model_path = model_dir / "xgboost_model.json"
        metadata_path = model_dir / "feature_metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Defend's trained model artifacts not found under {model_dir}. "
                "Run `python -m defend.train` in defend/ first."
            )

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        with open(metadata_path, encoding="utf-8") as f:
            self.metadata: dict = json.load(f)

    def score(self, feature_values: dict) -> ClassifierResult:
        row = self._encode_row(feature_values)
        proba = float(self.model.predict_proba(row)[0, 1])
        threshold = float(self.metadata["decision_threshold"])
        risk_label = "HIGH_RISK" if proba >= threshold else "LOW_RISK"
        return ClassifierResult(
            fraud_probability=round(proba, 6),
            decision_threshold=threshold,
            risk_label=risk_label,
            features_sent=feature_values,
        )

    def _encode_row(self, feature_values: dict) -> pd.DataFrame:
        """Mirrors defend/src/defend/features.py's build_features() column
        encoding, applied to a single new row instead of a training batch."""

        numeric_columns = self.metadata["numeric_columns"]
        categorical_columns = self.metadata["categorical_columns"]
        boolean_columns = self.metadata["boolean_columns"]
        category_levels = self.metadata["category_levels"]
        feature_columns = self.metadata["feature_columns"]

        data: dict[str, object] = {}

        for col in numeric_columns:
            value = feature_values.get(col)
            data[col] = float(value) if value is not None else float("nan")

        for col in boolean_columns:
            value = feature_values.get(col)
            if value is None:
                data[col] = float("nan")
            else:
                data[col] = 1.0 if bool(value) else 0.0

        row = pd.DataFrame([data])

        for col in categorical_columns:
            value = feature_values.get(col)
            cats = category_levels[col]
            row[col] = pd.Categorical([value], categories=cats)

        return row[feature_columns]
