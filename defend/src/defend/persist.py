"""Saves the trained model and everything the web prototype needs to reuse it
on new rows: the feature column list/order, the fixed category->level mapping
categorical columns were encoded with, and the decision threshold.

XGBoost's own JSON format (not pickle) is used for the model so it isn't tied
to this exact Python/xgboost version when the web prototype loads it later.
"""

from __future__ import annotations

import json
from pathlib import Path

import xgboost as xgb

from .config import DECISION_THRESHOLD, MODEL_DIR
from .features import FeatureSpec


def save_model_artifacts(model: xgb.XGBClassifier, spec: FeatureSpec, out_dir: Path = MODEL_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "xgboost_model.json")

    metadata = {
        "numeric_columns": spec.numeric_columns,
        "categorical_columns": spec.categorical_columns,
        "boolean_columns": spec.boolean_columns,
        "feature_columns": spec.feature_columns,
        "category_levels": spec.category_levels,
        "decision_threshold": DECISION_THRESHOLD,
        "target_column": "is_fraud",
    }
    with open(out_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model_artifacts(model_dir: Path = MODEL_DIR) -> tuple[xgb.XGBClassifier, dict]:
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "xgboost_model.json")
    with open(model_dir / "feature_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata
