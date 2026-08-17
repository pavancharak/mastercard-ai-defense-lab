"""Feature engineering: turns the transaction dataset (data.py) into a
numeric/categorical feature matrix XGBoost can train on directly.

Categorical columns are cast to pandas 'category' dtype (not one-hot encoded)
and passed to XGBoost with enable_categorical=True. That keeps the feature
count small and lets the trees split on category natively, but it means the
category->code mapping has to be fixed once on the full dataset before the
train/test split (or before scoring new data later) - build_features() below
does that and returns it as part of FeatureSpec, which is saved as model
metadata so the web prototype can reproduce it exactly on new rows.

No dispute_* columns here: this pass trains on transactions.csv only (see
data.py for why E1/disputes.csv is excluded this pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import EXCLUDED_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMNS

CATEGORICAL_COLUMNS = [
    "channel",
    "currency",
    "merchant_category",
    "geo_country",
    "mandate_category_scope",
    "authorization_result",
]

# Present as 0/1/NaN or True/False in the CSV; NaN means "not applicable to
# this row's channel/vector" (e.g. session_frame_rate_anomaly is only ever
# set for A2). XGBoost's native missing-value handling treats that NaN as
# informative-by-absence, which is exactly what we want here.
BOOLEAN_COLUMNS = [
    "device_is_new", "geo_is_new", "session_frame_rate_anomaly",
    "login_event_flag", "account_detail_changed_before_txn",
    "is_first_time_payee", "request_urgency_flag", "approval_outside_hierarchy",
    "preceded_by_support_call", "credential_exposure_window_flag",
    "agent_attestation_present", "agent_attestation_verified",
    "mandate_recurring_flag", "transaction_within_mandate_envelope",
]


@dataclass
class FeatureSpec:
    numeric_columns: list[str]
    categorical_columns: list[str]
    boolean_columns: list[str]
    category_levels: dict[str, list[str]]  # fixed category->code mapping to reuse at inference time

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns + self.boolean_columns


def _derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["txn_hour"] = ts.dt.hour.astype("float64")
    return df


def build_features(df: pd.DataFrame, category_levels: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, pd.Series, FeatureSpec]:
    """Returns (X, y, spec). Pass a fitted spec's category_levels back in when
    transforming new data (e.g. the held-out test split, or future inference
    data) so category codes stay consistent with training."""
    df = _derive_time_features(df)

    boolean_columns = [c for c in BOOLEAN_COLUMNS if c in df.columns]
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    derived_numeric = ["txn_hour"]
    all_declared = set(EXCLUDED_COLUMNS) | set(TIMESTAMP_COLUMNS) | set(boolean_columns) | set(categorical_columns) | set(derived_numeric)
    numeric_columns = [c for c in df.columns if c not in all_declared and pd.api.types.is_numeric_dtype(df[c])] + derived_numeric

    X = pd.DataFrame(index=df.index)
    for col in numeric_columns:
        X[col] = pd.to_numeric(df[col], errors="coerce")

    for col in boolean_columns:
        # keep NaN as missing rather than coercing to False, per the docstring above
        X[col] = df[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype("float64")

    levels = dict(category_levels) if category_levels else {}
    for col in categorical_columns:
        values = df[col].astype("string")
        if col in levels:
            cats = levels[col]
        else:
            cats = sorted(values.dropna().unique().tolist())
            levels[col] = cats
        X[col] = pd.Categorical(values, categories=cats)

    y = df[TARGET_COLUMN].astype(bool).astype(int)

    spec = FeatureSpec(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        boolean_columns=boolean_columns,
        category_levels=levels,
    )
    return X, y, spec
