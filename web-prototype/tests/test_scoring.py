"""Sanity tests for scoring.py against Defend's real trained model. No
mocking of the classifier -- these load the actual
defend/model/xgboost_model.json, same as the running app does."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_prototype import data_sources, scoring  # noqa: E402


def test_scoring_a_real_high_confidence_fraud_row_from_the_dataset():
    sample = data_sources.sample_transactions(limit_per_vector=5)
    a1_row = next(row for row in sample if row.get("fraud_vector") == "A1")

    features = scoring.feature_values_from_csv_row(a1_row)
    result = scoring.score(features)

    assert 0.0 <= result.fraud_probability <= 1.0
    assert result.risk_label in ("LOW_RISK", "HIGH_RISK")
    assert result.decision_threshold == 0.5


def test_free_form_scoring_with_all_fields_blank_still_returns_a_score():
    features = scoring.feature_values_from_form({})
    result = scoring.score(features)
    assert 0.0 <= result.fraud_probability <= 1.0
    assert all(v is None for v in features.values())


def test_free_form_scoring_coerces_types_correctly():
    features = scoring.feature_values_from_form(
        {
            "amount": "42.50",
            "channel": "subscription",
            "device_is_new": "true",
            "geo_is_new": "false",
            "": "ignored",
        }
    )
    assert features["amount"] == 42.50
    assert features["device_is_new"] is True
    assert features["geo_is_new"] is False


def test_classifier_is_a_singleton_across_calls():
    c1 = scoring.get_classifier()
    c2 = scoring.get_classifier()
    assert c1 is c2
