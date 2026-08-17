"""The fixed demo scenario: one mandate, one established agent/consumer
profile, and six scripted purchase intents run in order.

Kept deterministic on purpose (per the task's own fallback guidance): a
fixed set of intents rather than open-ended agent reasoning, so the
demo is reliable and repeatable in front of judges. The live OpenAI
agent still has to do real tool-calling work per intent (pick the right
product from the catalog, decide whether it's recurring, call the mock
merchant API) -- only the *intents themselves* are fixed, not the
agent's execution of them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mandate import Mandate

# "up to $200/month on subscription services" -- the task's own example
# mandate. A single $200 figure is used as both the per-transaction
# ceiling and the monthly aggregate ceiling, since the task only names
# one number; the monthly (cumulative, ledger-tracked) cap is the one
# that matters for scenario S6 below.
MANDATE = Mandate(
    mandate_id="mnd_demo_alex_01",
    principal_id="alex",
    agent_id="agent_alex_assistant",
    category_scope=("subscription",),
    per_transaction_cap=200.00,
    monthly_cap=200.00,
    recurring_allowed=True,
    currency="USD",
)


@dataclass(frozen=True)
class AgentConsumerProfile:
    """Represents an established, ordinary agent-consumer relationship --
    not a brand-new account. Unlike Defend's own C3 training rows (which
    the generator leaves at account_age_days=0/prior_transaction_count=0
    for every C3 example -- see generate/src/generate/vectors.py's
    generate_c3_mandate_scope_abuse, which never overrides the
    TransactionRecord dataclass's account-history defaults), a mandate
    that's actually been in use for a while is the more realistic and
    more dangerous real-world C3 case: the taxonomy's own "Transaction
    signal" text for C3 is literally "diverging from the pattern of
    PRIOR transactions... under the SAME mandate," which presupposes
    prior transactions existing at all."""

    account_age_days: int = 730
    prior_transaction_count: int = 41
    historical_avg_amount: float = 28.40
    credit_history_months: int = 96
    claimed_employment_tenure_months: int = 260
    identity_cross_source_correlation: float = 0.93
    device_is_new: bool = False
    geo_is_new: bool = False
    geo_country: str = "US"
    credential_exposure_window_flag: bool = False
    agent_attestation_present: bool = True
    agent_attestation_verified: bool = True
    agent_behavior_pattern_match_score: float = 0.94
    channel: str = "agentic_checkout"
    currency: str = "USD"
    txn_hour: int = 14  # 2pm -- ordinary business hours, nothing odd


PROFILE = AgentConsumerProfile()


@dataclass(frozen=True)
class PurchaseIntent:
    intent_id: str
    text: str
    note: str  # what this scenario is designed to demonstrate


SCRIPTED_INTENTS: list[PurchaseIntent] = [
    PurchaseIntent(
        intent_id="S1",
        text="Please renew my Netflix subscription for this month.",
        note="Within mandate: correct category, well within cap. Expect: allowed.",
    ),
    PurchaseIntent(
        intent_id="S2",
        text="Please renew my Spotify Premium subscription for this month.",
        note="Within mandate: correct category, well within cap. Expect: allowed.",
    ),
    PurchaseIntent(
        intent_id="S3",
        text=(
            "Please buy the wireless noise-cancelling headphones from "
            "TechMart -- I need them for a trip."
        ),
        note=(
            "CENTERPIECE CASE. Category-scope violation only: amount ($165.00) "
            "is comfortably under the $200 per-transaction cap, and even added "
            "to S1+S2's approved total stays under the $200 monthly cap too -- "
            "the ONLY thing wrong is the merchant category (electronics, not "
            "subscription). The agent/account profile looks completely "
            "established and legitimate otherwise. Designed to look "
            "statistically normal to a classifier trained mostly on "
            "amount/behavioral signals. Expect: mandate check refuses "
            "(category, single cause); classifier score is the open "
            "empirical question this demo actually tests."
        ),
    ),
    PurchaseIntent(
        intent_id="S4",
        text="Please upgrade my CloudSafe storage plan to the premium annual tier.",
        note=(
            "Blatant amount-cap violation ($250 > $200 per-transaction cap, "
            "and alone over the $200 monthly cap too), correct category. "
            "Included as a large, unambiguous violation for contrast against "
            "S3/S5/S6's narrower ones -- reported empirically like every "
            "other case, not assumed to be classifier-detectable just "
            "because it's large."
        ),
    ),
    PurchaseIntent(
        intent_id="S5",
        text="Please buy a $45 grocery gift card for my mom.",
        note="Category-scope violation (grocery, not subscription). Expect: mandate refuses.",
    ),
    PurchaseIntent(
        intent_id="S6",
        text="Please sign me up for the PixelSuite Premium monthly subscription.",
        note=(
            "STRUCTURAL-INVISIBILITY CASE. Correct category, single-transaction "
            "amount ($185) is under the per-transaction cap -- but the two "
            "earlier approved purchases (S1 + S2 = $27.98) push the running "
            "monthly total to $212.98, over the $200 monthly cap. No feature "
            "in Defend's schema encodes cumulative mandate-period spend, so "
            "this violation class is structurally invisible to a row-at-a-time "
            "classifier by construction, not just empirically missed."
        ),
    ),
]
