"""Case browser scoring: wraps mandate_demo.classifier.DefendClassifier
(itself a read-only wrapper around defend/model/*.json -- see that
module's docstring) to score either a real row from
generate/data/transactions.csv or a judge's free-form input.

No scoring logic is reimplemented here. This module's only job is
building the feature dict DefendClassifier.score() expects, from two
different raw shapes (a CSV row's strings, or a submitted HTML form).
"""

from __future__ import annotations

from functools import lru_cache

from mandate_demo.classifier import ClassifierResult, DefendClassifier

# The most decision-relevant fields, shown as the free-form scoring
# form. The remaining ~20 of Defend's 38 features are left unset (missing
# -> NaN) for a free-form submission, exactly like a real row where a
# field genuinely doesn't apply to that channel/vector (see
# defend/src/defend/features.py's own docstring on this). This keeps the
# form usable by a judge in a demo setting rather than an intimidating
# 38-field wall, and is explicitly called out as a scope decision, not a
# hidden simplification of the model itself -- the classifier still sees
# a real 38-column row; the other columns are just genuinely absent.
FREE_FORM_FIELDS = [
    {"name": "amount", "label": "Amount (USD)", "type": "number", "step": "0.01"},
    {"name": "channel", "label": "Channel", "type": "select", "options": [
        "card_present", "card_not_present", "ach", "wire", "rtp", "p2p",
        "agentic_checkout", "promo_redemption",
    ]},
    {"name": "merchant_category", "label": "Merchant category", "type": "select", "options": [
        "apparel", "digital_goods", "electronics", "gas_station", "grocery",
        "home_goods", "marketplace", "restaurant", "subscription", "travel",
    ]},
    {"name": "geo_country", "label": "Geo country", "type": "text"},
    {"name": "account_age_days", "label": "Account age (days)", "type": "number"},
    {"name": "prior_transaction_count", "label": "Prior transaction count", "type": "number"},
    {"name": "historical_avg_amount", "label": "Historical avg amount (USD)", "type": "number", "step": "0.01"},
    {"name": "device_is_new", "label": "Device is new", "type": "bool"},
    {"name": "geo_is_new", "label": "Geo is new", "type": "bool"},
    {"name": "is_first_time_payee", "label": "First-time payee", "type": "bool"},
    {"name": "request_urgency_flag", "label": "Request urgency flag", "type": "bool"},
    {"name": "credential_exposure_window_flag", "label": "Credential exposure window flag", "type": "bool"},
    {"name": "agent_attestation_present", "label": "Agent attestation present", "type": "bool"},
    {"name": "agent_attestation_verified", "label": "Agent attestation verified", "type": "bool"},
    {"name": "agent_behavior_pattern_match_score", "label": "Agent behavior pattern match score (0-1)", "type": "number", "step": "0.001"},
    {"name": "mandate_amount_cap", "label": "Mandate amount cap (USD)", "type": "number", "step": "0.01"},
    {"name": "mandate_category_scope", "label": "Mandate category scope", "type": "select", "options": [
        "", "apparel", "digital_goods", "electronics", "gas_station", "grocery",
        "home_goods", "marketplace", "restaurant", "subscription", "travel",
    ]},
    {"name": "mandate_recurring_flag", "label": "Mandate recurring flag", "type": "bool"},
    {"name": "transaction_within_mandate_envelope", "label": "Transaction within mandate envelope", "type": "bool"},
    {"name": "authorization_result", "label": "Authorization result", "type": "select", "options": ["approved", "declined"]},
]


@lru_cache(maxsize=1)
def get_classifier() -> DefendClassifier:
    """Loaded once per process -- defend/model/*.json doesn't change
    while the server runs."""
    return DefendClassifier()


def _coerce_bool(raw: str | None) -> bool | None:
    if raw in (None, ""):
        return None
    return raw in ("true", "True", "1", "on", "yes")


def _coerce_number(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def feature_values_from_csv_row(row: dict) -> dict:
    """A transactions.csv row already has (almost) exactly Defend's
    feature column names -- see defend/src/defend/config.py's
    EXCLUDED_COLUMNS for what's deliberately left out (IDs, timestamps,
    label-leakage columns). This just type-coerces the CSV's strings and
    derives txn_hour from the timestamp, mirroring
    defend/src/defend/features.py's own derivation (re-derived here, not
    imported, since features.py works on a full DataFrame batch, not one
    row -- see that module for the batch version this matches)."""

    classifier = get_classifier()
    metadata = classifier.metadata
    features: dict = {}

    for col in metadata["numeric_columns"]:
        if col == "txn_hour":
            continue
        features[col] = _coerce_number(row.get(col))

    timestamp = row.get("timestamp")
    if timestamp:
        # "2026-08-15T03:12:46" -> hour component, matching features.py's
        # pd.to_datetime(...).dt.hour derivation.
        try:
            features["txn_hour"] = float(timestamp.split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            features["txn_hour"] = None
    else:
        features["txn_hour"] = None

    for col in metadata["categorical_columns"]:
        value = row.get(col)
        features[col] = value if value not in (None, "") else None

    for col in metadata["boolean_columns"]:
        raw = row.get(col)
        features[col] = _coerce_bool(raw) if raw not in (None, "", "nan") else None

    return features


def feature_values_from_form(form: dict) -> dict:
    """Builds a feature dict from the free-form scoring form. Every field
    not present in FREE_FORM_FIELDS (or left blank by the judge) is
    genuinely omitted -> DefendClassifier encodes it as missing, the same
    native-missing-value handling XGBoost already uses for every
    not-applicable-to-this-channel field in the real training data."""

    features: dict = {}
    for field in FREE_FORM_FIELDS:
        raw = form.get(field["name"])
        if field["type"] == "bool":
            features[field["name"]] = _coerce_bool(raw)
        elif field["type"] == "number":
            features[field["name"]] = _coerce_number(raw)
        else:
            features[field["name"]] = raw if raw not in (None, "") else None
    return features


def score(feature_values: dict) -> ClassifierResult:
    return get_classifier().score(feature_values)
