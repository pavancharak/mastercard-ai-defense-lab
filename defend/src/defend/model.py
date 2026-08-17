"""Trains the primary gradient-boosted tree classifier (XGBoost)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import RANDOM_SEED


def train_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int = RANDOM_SEED) -> xgb.XGBClassifier:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        enable_categorical=True,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def feature_importance(model: xgb.XGBClassifier, feature_columns: list[str]) -> pd.DataFrame:
    importances = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
