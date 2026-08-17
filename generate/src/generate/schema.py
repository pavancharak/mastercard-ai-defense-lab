"""Canonical record shapes for the Generate pillar's output.

Two tables, matching the taxonomy's own guidance (see "Notes for the Generate
and Defend pillars" in identify/attack-taxonomy.md):

- TransactionRecord: one row per payment-adjacent event. Shared by categories
  A-D. Most fields are optional because different channels populate different
  subsets (a wire transfer doesn't have a device fingerprint; an agentic
  checkout doesn't have a login event). Every field maps to a "Transaction
  signal" sentence in the taxonomy for at least one vector.
- DisputeRecord: category E1/E2 needs transaction + dispute metadata, a
  different shape than point-of-sale fields, so it lives in its own table
  linked back to TransactionRecord via original_transaction_id.

Every field a generator doesn't set keeps its default (None/False/0), which
guarantees every row produced anywhere in vectors.py has the same columns when
the rows are assembled into a DataFrame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TransactionRecord:
    # --- identity / time ---
    transaction_id: str = ""
    timestamp: str = ""
    account_id: str = ""
    customer_id: str = ""
    channel: str = ""
    amount: float = 0.0
    currency: str = "USD"

    # --- merchant (C, D) ---
    merchant_id: str | None = None
    merchant_category: str | None = None
    merchant_registration_age_days: int | None = None
    merchant_reputation_score: float | None = None

    # --- account / identity history (A) ---
    account_age_days: int = 0
    prior_transaction_count: int = 0
    historical_avg_amount: float = 0.0
    credit_history_months: int | None = None
    claimed_employment_tenure_months: int | None = None
    identity_cross_source_correlation: float | None = None

    # --- device / session (A2, D2) ---
    device_id: str | None = None
    device_is_new: bool | None = None
    geo_country: str = ""
    geo_is_new: bool | None = None
    session_frame_rate_anomaly: bool | None = None
    session_video_artifact_score: float | None = None
    onboarding_velocity_seconds: float | None = None
    login_event_flag: bool | None = None
    account_detail_changed_before_txn: bool | None = None

    # --- payee / authorization (B) ---
    payee_id: str | None = None
    is_first_time_payee: bool | None = None
    payee_prior_interaction_count: int | None = None
    request_urgency_flag: bool | None = None
    approval_outside_hierarchy: bool | None = None
    preceded_by_support_call: bool | None = None
    time_since_support_call_minutes: float | None = None
    credential_exposure_window_flag: bool | None = None

    # --- agentic commerce (C) ---
    agent_id: str | None = None
    agent_attestation_present: bool | None = None
    agent_attestation_verified: bool | None = None
    agent_behavior_pattern_match_score: float | None = None
    mandate_id: str | None = None
    mandate_amount_cap: float | None = None
    mandate_category_scope: str | None = None
    mandate_recurring_flag: bool | None = None
    transaction_within_mandate_envelope: bool | None = None

    # --- velocity / automation (D) ---
    attempt_sequence_id: str | None = None
    attempts_in_window: int | None = None
    window_seconds: float | None = None
    authorization_result: str | None = None  # approved | declined

    # --- promo (E2) ---
    promo_code: str | None = None
    account_cluster_id: str | None = None

    # --- labels / meta ---
    is_fraud: bool = False
    fraud_category: str | None = None
    fraud_vector: str | None = None
    narrative_tag: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class DisputeRecord:
    dispute_id: str = ""
    original_transaction_id: str = ""
    account_id: str = ""
    dispute_filed_at: str = ""
    dispute_reason: str = ""
    evidence_type: str | None = None  # photo | receipt | none
    evidence_generated_flag: bool = False
    evidence_generative_artifact_score: float | None = None
    claimant_dispute_rate_30d: float = 0.0
    correlated_claim_count: int = 0

    is_fraud: bool = False
    fraud_category: str | None = None
    fraud_vector: str | None = None
    narrative_tag: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


TRANSACTION_COLUMNS = [f.name for f in fields(TransactionRecord)]
DISPUTE_COLUMNS = [f.name for f in fields(DisputeRecord)]
