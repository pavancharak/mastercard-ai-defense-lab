"""Volumes, seeds, and shared constants for dataset generation.

Per-vector volume weights follow the Plausibility rating each vector was given in
identify/attack-taxonomy.md (Very high=5, High=3, Emerging=1), scaled by --scale.
This keeps the dataset's class balance a direct, traceable function of the
taxonomy document rather than an arbitrary choice.
"""

from __future__ import annotations

DEFAULT_SEED = 42
BASE_DATE = "2026-08-16"  # anchor "now" for relative timestamps; matches lab date
HISTORY_WINDOW_DAYS = 180

DEFAULT_N_LEGIT = 20_000
DEFAULT_N_DISPUTE_LEGIT = 1_200

# Plausibility -> relative weight, as stated per-vector in the taxonomy.
VERY_HIGH, HIGH, EMERGING = 5, 3, 1

VECTOR_WEIGHTS: dict[str, int] = {
    "A1": VERY_HIGH,  # Synthetic Identity Construction
    "A2": HIGH,       # Deepfake Document and Video KYC Bypass
    "B1": HIGH,       # Deepfake Voice Executive Impersonation
    "B2": HIGH,       # Deepfake Video Authorized Push Payment Fraud
    "B3": VERY_HIGH,  # Multivector AI Personalized Phishing and Vishing
    "C1": EMERGING,   # Agent Impersonation
    "C2": EMERGING,   # Malicious Storefront Targeting Legitimate Agents
    "C3": EMERGING,   # Delegated Mandate Scope Abuse
    "D1": VERY_HIGH,  # AI Accelerated Card Testing
    "D2": VERY_HIGH,  # AI Scaled Credential Stuffing and Account Takeover
    "E1": HIGH,       # GenAI Fabricated Refund and Chargeback Evidence
    "E2": HIGH,       # Promotion and Coupon Abuse at GenAI Scale
}

VECTOR_CATEGORY: dict[str, str] = {
    "A1": "A", "A2": "A",
    "B1": "B", "B2": "B", "B3": "B",
    "C1": "C", "C2": "C", "C3": "C",
    "D1": "D", "D2": "D",
    "E1": "E", "E2": "E",
}

VECTOR_NAME: dict[str, str] = {
    "A1": "Synthetic Identity Construction",
    "A2": "Deepfake Document and Video KYC Bypass",
    "B1": "Deepfake Voice Executive Impersonation",
    "B2": "Deepfake Video Authorized Push Payment Fraud",
    "B3": "Multivector AI Personalized Phishing and Vishing",
    "C1": "Agent Impersonation",
    "C2": "Malicious Storefront Targeting Legitimate Agents",
    "C3": "Delegated Mandate Scope Abuse",
    "D1": "AI Accelerated Card Testing",
    "D2": "AI Scaled Credential Stuffing and Account Takeover",
    "E1": "GenAI Fabricated Refund and Chargeback Evidence",
    "E2": "Promotion and Coupon Abuse at GenAI Scale",
}

# Vectors that produce dispute-shaped records (linked to a transaction) rather
# than a standalone point-of-sale transaction. See taxonomy "Notes" section.
DISPUTE_VECTORS = {"E1"}

UNITS_PER_WEIGHT = 20  # base volume multiplier before --scale is applied

MERCHANT_CATEGORIES = [
    "electronics", "grocery", "travel", "apparel", "digital_goods",
    "home_goods", "subscription", "restaurant", "gas_station", "marketplace",
]

GEOS = ["US", "GB", "CA", "AU", "DE", "FR", "IN", "BR", "MX", "NG", "SG", "NL"]
