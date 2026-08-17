"""One generator per taxonomy vector.

Each function returns a list of row-dicts shaped by schema.TransactionRecord
(or, for E1, schema.DisputeRecord plus its linked transaction). The
docstring on each function quotes the "Transaction signal" sentence from
identify/attack-taxonomy.md it is implementing, so the mapping from taxonomy
to code stays auditable.

Fraud rows are built by sampling the same fields a legitimate row would have,
but from a distribution shifted toward the anomaly the taxonomy describes.
That is deliberate: a detector trained on this data has to learn the shift in
distribution, not memorize a leakage flag.
"""

from __future__ import annotations

import numpy as np

from .config import GEOS, MERCHANT_CATEGORIES
from .entities import EntityPool, short_id
from .schema import DisputeRecord, TransactionRecord
from .utils import lognormal_amount, random_timestamp, timestamp_at_hour


def _txn(**kwargs) -> dict:
    return TransactionRecord(**kwargs).as_dict()


def _dispute(**kwargs) -> dict:
    return DisputeRecord(**kwargs).as_dict()


# Channels that have a payee/counterparty concept at all. Whether
# payee_prior_interaction_count/is_first_time_payee get populated depends on
# this (a real, already-modeled property of the transaction), not on which
# generator produced the row - see _sample_payee()'s docstring.
_PAYEE_CHANNELS = {"wire", "ach", "rtp", "p2p"}
_PROMO_CODES = ["WELCOME10", "NEWUSER15", "REFER25", "FIRSTBUY"]

# Per-channel legit amount distributions (lognormal mean, sigma in log-space).
# wire/ach/rtp/p2p used to share the same small-retail-purchase distribution
# as card_not_present, capping legit wire at ~$300 - nothing like the real
# population these rails actually carry, and specifically what let B1's
# amount alone perfectly separate it from legit (its range was completely
# disjoint from that unrealistic $3-$300 ceiling). Real-world grounding:
#   - wire: consumers are ~6% of wire volume by count but include large
#     one-off payments (real estate closings commonly land in the low-to-mid
#     six figures; median US home price is roughly $400-430k as of 2025),
#     alongside smaller business/vendor wires. mean=8.2, sigma=1.5 gives a
#     ~$3.6k median with a realistic tail into the hundreds of thousands
#     (~$119k at p99, ~$375k at p99.9).
#   - ach: Nacha reports ~$93T across ~35.2B payments in 2025, ~$2,642
#     average - dominated by payroll and B2B, with plenty of small consumer
#     bill-pay too. mean=7.26, sigma=1.1 targets that ~$2.6k mean with a
#     ~$1.4k median and a payroll/B2B tail into the five figures.
#   - rtp/p2p: RTP network's per-transaction cap is now $10M (raised from
#     $1M in Feb 2025), but individual banks cap Zelle-style P2P far lower -
#     commonly $2.5k-$15k/transaction for business use, less for personal.
#     mean=5.3, sigma=1.3 gives a ~$200 typical-P2P median with a realistic
#     tail into the low thousands (~$4.1k at p99) matching real transfer
#     limits, well above the old $330 hard ceiling.
# See defend/README.md's changelog for the investigation that found this.
_WIRE_LIKE_AMOUNT_PARAMS: dict[str, tuple[float, float]] = {
    "wire": (8.2, 1.5),
    "ach": (7.26, 1.1),
    "rtp": (5.3, 1.3),
    "p2p": (5.3, 1.3),
}

# Bias tiers for _novelty_flags(), grounded in taxonomy text:
_NOVELTY_LEGIT = 0.03  # legit's own base rate - familiar device/geo the vast majority of the time
_NOVELTY_ELEVATED = 0.85  # taxonomy explicitly names new device/geo as this vector's signal (A1, A2, B3, D2)
_NOVELTY_MODERATE = 0.5  # plausible given the mechanism, but not an explicit taxonomy signal (C1, D1, E2)

# Bias tiers for _sample_approval_context() (B1's named signal: urgency
# framing, hierarchy bypass) and _sample_support_call_context() (B2's named
# signal: call proximity). Legit tiers are deliberately low but nonzero -
# both behaviors happen for genuine reasons too.
_URGENCY_LEGIT, _URGENCY_B1, _URGENCY_OTHER = 0.05, 0.85, 0.12
_HIERARCHY_LEGIT, _HIERARCHY_B1, _HIERARCHY_OTHER = 0.05, 0.7, 0.1
_SUPPORT_CALL_LEGIT, _SUPPORT_CALL_B2, _SUPPORT_CALL_OTHER = 0.03, 0.85, 0.1

# credential_exposure_window_flag (B3's named signal) used to be True only
# for B3 and null for every other row including legit - the same shape as
# every other fix above. A legit account's credentials showing up in a
# breach dataset without being exploited is a real, if uncommon, false-
# positive-prone signal, so legit (and every other vector, not just B3 -
# see _sample_credential_exposure()) gets a low nonzero rate rather than a
# hard null.
_CRED_EXPOSURE_LEGIT = 0.02
_CRED_EXPOSURE_B3 = 0.85


def _sample_credential_exposure(rng: np.random.Generator, bias: float) -> dict:
    """credential_exposure_window_flag for every vector, same Bernoulli(bias)
    shape as _novelty_flags(). First fix pass only applied this to legit and
    B3 directly in their own kwargs, leaving it null for every other fraud
    vector - the exact same null-vs-populated shape as the original bugs,
    just on this field. Now applied everywhere via one helper so a future
    field addition can't repeat that mistake by omission."""
    return {"credential_exposure_window_flag": bool(rng.random() < bias)}


# claimed_employment_tenure_months/identity_cross_source_correlation are A1's
# own named signal (claimed tenure inconsistent with a thin file; low
# cross-source correlation). A near-miss repeat of the same missingness bug:
# adding these to legit-only (to give A1 a real contrast) without also
# generalizing to every OTHER vector reproduced the exact null-vs-populated
# shape this whole investigation has been fixing, just on a new field pair -
# caught immediately by the same diagnostics, fixed the same way via one
# shared helper instead of hand-adding to each generator individually.
_IDENTITY_VERIFICATION_NORMAL = (0.75, 0.99)  # well-established accounts: legit, and every fraud vector but A1
_IDENTITY_VERIFICATION_A1 = (0.05, 0.35)  # A1's own signal: thin file, low cross-source correlation


def _sample_identity_verification(rng: np.random.Generator, correlation_range: tuple[float, float]) -> dict:
    lo, hi = correlation_range
    return {
        "claimed_employment_tenure_months": int(rng.integers(0, 480)),
        "identity_cross_source_correlation": float(np.round(rng.uniform(lo, hi), 3)),
    }


def _novelty_flags(rng: np.random.Generator, bias: float) -> dict:
    """device_is_new/geo_is_new for every vector, drawn from the same
    Bernoulli(bias) shape everywhere - only bias itself differs by vector,
    grounded in whether that vector's taxonomy text actually describes new-
    device/geo as part of its signal (elevated bias) or the transaction is
    executed by/on behalf of the real accountholder on their normal setup
    (low bias, same as legit). Never a hard True-vs-unset split - that was
    the bug (see generate/README.md's anti-leakage notes)."""
    return {
        "device_is_new": bool(rng.random() < bias),
        "geo_is_new": bool(rng.random() < bias),
    }


def _sample_payee(rng: np.random.Generator, channel: str, first_time_bias: float) -> dict:
    """Populates payee_id/is_first_time_payee/payee_prior_interaction_count
    for payee-relevant channels only, with legit and fraud both drawing from
    the same {0} u uniform[1,50) support - only the MIX changes
    (first_time_bias), never a hard 0-vs-populated split. That mixture
    difference is deliberately weaker than perfect separation: a first-time
    payee happens for legitimate transfers too (new vendor, new
    family member), and social-engineered payees aren't always literally
    brand new (a mule account may already have some history) - the taxonomy
    describes this as "first-time OR RARE payee", a tendency, not a
    certainty. See generate/README.md's anti-leakage notes for why disjoint,
    deterministic distributions are treated as a bug and not a shortcut.
    """
    if channel not in _PAYEE_CHANNELS:
        return {}
    is_first_time = bool(rng.random() < first_time_bias)
    count = 0 if is_first_time else int(rng.integers(1, 50))
    return {
        "payee_id": short_id("payee", rng),
        "is_first_time_payee": is_first_time,
        "payee_prior_interaction_count": count,
    }


def _sample_approval_context(rng: np.random.Generator, channel: str, urgency_bias: float, hierarchy_bias: float) -> dict:
    """request_urgency_flag/approval_outside_hierarchy for payee-relevant
    channels on every vector, not just B1. These used to be populated only
    for B1 (request_urgency_flag deterministically True) and left null
    everywhere else - the same missingness-fingerprint shape as the
    credit_history_months/payee_prior_interaction_count and geo_is_new/
    device_is_new bugs, just not yet the empirically dominant one. Fixed the
    same way: every payee-channel row gets both fields from a real
    Bernoulli draw, with B1's bias elevated (taxonomy names this as B1's
    signal) and every other vector/legit given a low-but-real base rate
    (legitimate urgent wires and occasional non-standard approvals do
    happen - an overdue vendor invoice, an owner-approved emergency
    purchase)."""
    if channel not in _PAYEE_CHANNELS:
        return {}
    return {
        "request_urgency_flag": bool(rng.random() < urgency_bias),
        "approval_outside_hierarchy": bool(rng.random() < hierarchy_bias),
    }


def _sample_support_call_context(
    rng: np.random.Generator, channel: str, bias: float, max_minutes: float = 240
) -> dict:
    """preceded_by_support_call/time_since_support_call_minutes for
    payee-relevant channels on every vector, not just B2. Same fix pattern
    as _sample_approval_context: a real Bernoulli draw everywhere (elevated
    for B2, low base rate elsewhere - people do call support before a large
    legitimate transfer sometimes, e.g. to verify account details), with
    time_since_support_call_minutes only defined when a call actually
    happened (a genuine conditional null, not a vector fingerprint - it
    applies the same way regardless of which vector produced the row).
    max_minutes narrows the recency window for B2 specifically, since the
    taxonomy's signal there is a call "shortly" before the transfer, not
    merely that a call happened at some point - a wider window is used for
    every other vector/legit where recency isn't the named signal."""
    if channel not in _PAYEE_CHANNELS:
        return {}
    preceded = bool(rng.random() < bias)
    return {
        "preceded_by_support_call": preceded,
        "time_since_support_call_minutes": float(np.round(rng.uniform(2, max_minutes), 1)) if preceded else None,
    }


# ---------------------------------------------------------------------------
# Legitimate baseline
# ---------------------------------------------------------------------------

_LEGIT_CHANNEL_MIX = [
    "card_not_present", "card_present", "wire", "ach", "rtp", "p2p",
    "agentic_checkout", "promo_redemption",
]


def generate_legit(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    rows = []
    for _ in range(n):
        acct = entities.sample_account(rng)
        merchant = entities.sample_merchant(rng)
        channel = str(rng.choice(_LEGIT_CHANNEL_MIX, p=[0.30, 0.20, 0.03, 0.07, 0.10, 0.10, 0.15, 0.05]))
        if channel in _WIRE_LIKE_AMOUNT_PARAMS:
            mean, sigma = _WIRE_LIKE_AMOUNT_PARAMS[channel]
            amount = lognormal_amount(rng, mean, sigma)
        else:
            amount = min(lognormal_amount(rng, 3.6, 0.9), acct.historical_avg_amount * float(rng.uniform(0.4, 2.0)) + 5)

        # A wire/ACH/RTP/P2P transfer has a payee, not a merchant - forcing
        # merchant fields onto every row regardless of channel (the bug) let
        # a classifier fingerprint "not B1/B2/D2 on a payee channel" purely
        # from merchant field presence, the same missingness shape as the
        # credit_history_months/payee_prior_interaction_count and
        # geo_is_new/device_is_new bugs. Scoped the same way payee fields
        # already are, just to the complementary channel set.
        merchant_kwargs: dict = {}
        if channel not in _PAYEE_CHANNELS:
            merchant_kwargs = dict(
                merchant_id=merchant.merchant_id,
                merchant_category=merchant.category,
                merchant_registration_age_days=merchant.registration_age_days,
                merchant_reputation_score=merchant.reputation_score,
            )

        # E2 always sets promo_code on its promo_redemption rows; legit's own
        # promo_redemption rows never did, the same channel-scoped
        # missingness gap as merchant/payee fields - most promo code use is
        # genuinely legitimate, abuse is the rare case, not the norm.
        promo_kwargs: dict = {}
        if channel == "promo_redemption":
            promo_kwargs = dict(promo_code=str(rng.choice(_PROMO_CODES)))

        agent_kwargs: dict = {}
        if channel == "agentic_checkout":
            # A genuine agentic transaction always carries agent/mandate metadata;
            # leaving these null here would let the detector cheat off their mere
            # presence rather than off C1-C3's actual signal (verification,
            # behavior match, envelope fit).
            agent = entities.sample_agent(rng)
            amount = min(amount, agent.mandate_amount_cap * float(rng.uniform(0.1, 0.95)))
            agent_kwargs = dict(
                agent_id=agent.agent_id,
                agent_attestation_present=True,
                agent_attestation_verified=True,
                agent_behavior_pattern_match_score=float(np.round(rng.uniform(0.8, 0.99), 3)),
                mandate_id=agent.mandate_id,
                mandate_amount_cap=agent.mandate_amount_cap,
                mandate_category_scope=agent.mandate_category_scope,
                mandate_recurring_flag=agent.mandate_recurring_flag,
                transaction_within_mandate_envelope=True,
            )

        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=acct.account_id,
            customer_id=acct.customer_id,
            channel=channel,
            amount=amount,
            account_age_days=acct.account_age_days,
            prior_transaction_count=acct.prior_transaction_count,
            historical_avg_amount=acct.historical_avg_amount,
            credit_history_months=acct.credit_history_months,
            # Static identity-file attributes from this account's own,
            # already-completed onboarding - every real account has these on
            # file, not just A1's freshly-fabricated ones. A1's own version
            # is deliberately low/unverified (claimed tenure inconsistent
            # with a thin file, low cross-source correlation); a genuine,
            # already-established account's identity checks out well.
            claimed_employment_tenure_months=int(rng.integers(0, 480)),
            identity_cross_source_correlation=float(np.round(rng.uniform(0.75, 0.99), 3)),
            device_id=acct.device_id,
            geo_country=acct.geo_country,
            authorization_result="approved",
            is_fraud=False,
            credential_exposure_window_flag=bool(rng.random() < _CRED_EXPOSURE_LEGIT),
            **_novelty_flags(rng, _NOVELTY_LEGIT),
            **_sample_payee(rng, channel, first_time_bias=0.08),
            **_sample_approval_context(rng, channel, _URGENCY_LEGIT, _HIERARCHY_LEGIT),
            **_sample_support_call_context(rng, channel, _SUPPORT_CALL_LEGIT),
            **merchant_kwargs,
            **agent_kwargs,
            **promo_kwargs,
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Category A: Identity and Onboarding Fraud
# ---------------------------------------------------------------------------

def generate_a1_synthetic_identity(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"No prior credit history despite claimed age or employment tenure;
    thin-file behavior with unusually mature spending patterns from the first
    transaction; identity attributes that individually pass verification but
    show low cross-source correlation."
    """
    rows = []
    for _ in range(n):
        merchant = entities.sample_merchant(rng)
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=short_id("acct", rng),
            customer_id=short_id("cust", rng),
            channel="card_not_present",
            amount=lognormal_amount(rng, 5.6, 0.6),  # unusually mature first-txn spend
            merchant_id=merchant.merchant_id,
            merchant_category=merchant.category,
            merchant_registration_age_days=merchant.registration_age_days,
            merchant_reputation_score=merchant.reputation_score,
            account_age_days=int(rng.integers(0, 14)),
            prior_transaction_count=0,
            historical_avg_amount=0.0,
            credit_history_months=int(rng.integers(0, 6)),
            claimed_employment_tenure_months=int(rng.integers(24, 120)),
            identity_cross_source_correlation=float(np.round(rng.uniform(0.05, 0.35), 3)),
            device_id=short_id("dev", rng),
            geo_country=rng.choice(GEOS),
            authorization_result="approved",
            is_fraud=True,
            fraud_category="A",
            fraud_vector="A1",
            narrative_tag="synthetic identity: thin file, mature first-transaction spend, low cross-source identity correlation",
            **_novelty_flags(rng, _NOVELTY_ELEVATED),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
        )
        rows.append(row)
    return rows


def generate_a2_deepfake_kyc(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Verification completed but with device or session characteristics
    inconsistent with a live capture (frame rate anomalies, injected video
    artifacts, mismatched metadata); onboarding velocity inconsistent with a
    genuine document retrieval and capture flow."
    """
    rows = []
    for _ in range(n):
        merchant = entities.sample_merchant(rng)
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=short_id("acct", rng),
            customer_id=short_id("cust", rng),
            channel="card_not_present",
            amount=lognormal_amount(rng, 5.0, 0.7),
            merchant_id=merchant.merchant_id,
            merchant_category=merchant.category,
            merchant_registration_age_days=merchant.registration_age_days,
            merchant_reputation_score=merchant.reputation_score,
            account_age_days=int(rng.integers(0, 3)),
            prior_transaction_count=0,
            historical_avg_amount=0.0,
            credit_history_months=int(rng.integers(0, 4)),
            device_id=short_id("dev", rng),
            geo_country=rng.choice(GEOS),
            session_frame_rate_anomaly=bool(rng.random() < 0.75),
            session_video_artifact_score=float(np.round(rng.uniform(0.55, 0.97), 3)),
            onboarding_velocity_seconds=float(np.round(rng.uniform(4, 45), 1)),
            authorization_result="approved",
            is_fraud=True,
            fraud_category="A",
            fraud_vector="A2",
            narrative_tag="deepfake KYC bypass: liveness/session artifacts, implausibly fast onboarding",
            **_novelty_flags(rng, _NOVELTY_ELEVATED),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Category B: Social Engineering and Authorization Fraud
# ---------------------------------------------------------------------------

def generate_b1_bec_voice_deepfake(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"First-time or rare payee, urgency framing inconsistent with the
    payer's historical approval pattern, request originating or approved
    outside normal business hours or normal approval hierarchy."
    """
    rows = []
    for _ in range(n):
        acct = entities.sample_account(rng)
        # "outside normal business hours OR normal approval hierarchy" per the
        # taxonomy is a disjunction, not a certainty on either arm - most B1
        # attempts land off-hours (the taxonomy's headline example) but not
        # literally all of them, matching approval_outside_hierarchy's own
        # existing 70% (not 100%) rate below.
        if rng.random() < 0.8:
            off_hours = int(rng.choice([22, 23, 0, 1, 5, 6]))
            timestamp = timestamp_at_hour(rng, off_hours)
        else:
            timestamp = random_timestamp(rng)
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=timestamp,
            account_id=acct.account_id,
            customer_id=acct.customer_id,
            channel="wire",
            amount=lognormal_amount(rng, 9.5, 0.7),
            account_age_days=acct.account_age_days,
            prior_transaction_count=acct.prior_transaction_count,
            historical_avg_amount=acct.historical_avg_amount,
            credit_history_months=acct.credit_history_months,
            geo_country=acct.geo_country,
            authorization_result="approved",
            is_fraud=True,
            fraud_category="B",
            fraud_vector="B1",
            narrative_tag="deepfake voice BEC: urgent wire to first-time payee, off-hours/out-of-hierarchy approval",
            **_novelty_flags(rng, _NOVELTY_LEGIT),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
            **_sample_payee(rng, "wire", first_time_bias=0.82),
            **_sample_approval_context(rng, "wire", _URGENCY_B1, _HIERARCHY_B1),
            **_sample_support_call_context(rng, "wire", _SUPPORT_CALL_OTHER),
        )
        rows.append(row)
    return rows


def generate_b2_deepfake_video_app(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"A first time large transfer to a new payee shortly after a support or
    verification style call, unusual transfer timing relative to account
    history." Genuinely authorized by the accountholder, so authentication
    succeeds -- the signal has to live in payee novelty and call proximity.
    """
    rows = []
    for _ in range(n):
        acct = entities.sample_account(rng)
        channel = str(rng.choice(["rtp", "p2p"]))
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=acct.account_id,
            customer_id=acct.customer_id,
            channel=channel,
            amount=max(lognormal_amount(rng, 7.0, 0.6), acct.historical_avg_amount * float(rng.uniform(3, 12))),
            account_age_days=acct.account_age_days,
            prior_transaction_count=acct.prior_transaction_count,
            historical_avg_amount=acct.historical_avg_amount,
            credit_history_months=acct.credit_history_months,
            geo_country=acct.geo_country,
            authorization_result="approved",
            is_fraud=True,
            fraud_category="B",
            fraud_vector="B2",
            narrative_tag="deepfake video APP fraud: genuinely authorized, large first-time transfer shortly after a support-style call",
            **_novelty_flags(rng, _NOVELTY_LEGIT),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
            **_sample_payee(rng, channel, first_time_bias=0.82),
            **_sample_approval_context(rng, channel, _URGENCY_OTHER, _HIERARCHY_OTHER),
            **_sample_support_call_context(rng, channel, _SUPPORT_CALL_B2, max_minutes=60),
        )
        rows.append(row)
    return rows


def generate_b3_multivector_phishing(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Credential use with a session or device profile inconsistent with the
    account's history immediately after a plausible phishing exposure
    window."
    """
    rows = []
    for _ in range(n):
        acct = entities.sample_account(rng)
        merchant = entities.sample_merchant(rng)
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=acct.account_id,
            customer_id=acct.customer_id,
            channel="card_not_present",
            amount=lognormal_amount(rng, 5.2, 0.9),
            merchant_id=merchant.merchant_id,
            merchant_category=merchant.category,
            merchant_registration_age_days=merchant.registration_age_days,
            merchant_reputation_score=merchant.reputation_score,
            account_age_days=acct.account_age_days,
            prior_transaction_count=acct.prior_transaction_count,
            historical_avg_amount=acct.historical_avg_amount,
            credit_history_months=acct.credit_history_months,
            geo_country=str(rng.choice(GEOS)),
            device_id=short_id("dev", rng),
            credential_exposure_window_flag=bool(rng.random() < _CRED_EXPOSURE_B3),
            authorization_result="approved",
            is_fraud=True,
            fraud_category="B",
            fraud_vector="B3",
            **_novelty_flags(rng, _NOVELTY_ELEVATED),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
            narrative_tag="multivector AI phishing: credential use from new device/geo shortly after exposure window",
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Category C: Agentic Commerce Fraud
# ---------------------------------------------------------------------------

def generate_c1_agent_impersonation(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Agent-attributed transaction with a credential or attestation that is
    present but does not independently verify; mismatch between claimed
    agent identity and observed transaction behavior pattern for that agent."
    """
    rows = []
    for _ in range(n):
        spoofed_agent = entities.sample_agent(rng)
        merchant = entities.sample_merchant(rng)
        amount = lognormal_amount(rng, 5.5, 0.8)
        # C1's fraud is about attestation/identity, not mandate scope - the
        # attacker isn't targeting the envelope, so whether this transaction
        # happens to fall within the spoofed agent's real mandate is a
        # genuine consequence of the independently-sampled amount/merchant,
        # not an arbitrary flag. Mirrors C3's own derivation below.
        within_envelope = bool(
            amount <= spoofed_agent.mandate_amount_cap
            and merchant.category == spoofed_agent.mandate_category_scope
        )
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=spoofed_agent.linked_account_id,
            customer_id=short_id("cust", rng),
            channel="agentic_checkout",
            amount=amount,
            merchant_id=merchant.merchant_id,
            merchant_category=merchant.category,
            merchant_registration_age_days=merchant.registration_age_days,
            merchant_reputation_score=merchant.reputation_score,
            agent_id=spoofed_agent.agent_id,
            agent_attestation_present=True,
            agent_attestation_verified=False,
            agent_behavior_pattern_match_score=float(np.round(rng.uniform(0.05, 0.35), 3)),
            mandate_id=spoofed_agent.mandate_id,
            mandate_amount_cap=spoofed_agent.mandate_amount_cap,
            mandate_category_scope=spoofed_agent.mandate_category_scope,
            mandate_recurring_flag=spoofed_agent.mandate_recurring_flag,
            transaction_within_mandate_envelope=within_envelope,
            credit_history_months=spoofed_agent.linked_account.credit_history_months,
            geo_country=str(rng.choice(GEOS)),
            authorization_result="approved",
            is_fraud=True,
            fraud_category="C",
            fraud_vector="C1",
            narrative_tag="agent impersonation: attestation present but unverifiable, behavior pattern mismatch vs claimed agent",
            **_novelty_flags(rng, _NOVELTY_MODERATE),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
        )
        rows.append(row)
    return rows


def generate_c2_malicious_storefront(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Agent initiated transaction to a newly registered or low reputation
    merchant, transaction characteristics inconsistent with the consumer's
    own historical merchant preferences."
    """
    rows = []
    for _ in range(n):
        agent = entities.sample_agent(rng)
        acct = entities.sample_account(rng)
        merchant = entities.sample_malicious_merchant(rng)
        amount = lognormal_amount(rng, 4.8, 0.7)
        # C2's agent is genuinely legitimate - only the merchant is malicious
        # - so envelope compliance is a real, derived consequence of the
        # independently-sampled amount/merchant category, same as C1 above.
        within_envelope = bool(
            amount <= agent.mandate_amount_cap
            and merchant.category == agent.mandate_category_scope
        )
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=acct.account_id,
            customer_id=acct.customer_id,
            channel="agentic_checkout",
            amount=amount,
            merchant_id=merchant.merchant_id,
            merchant_category=merchant.category,
            merchant_registration_age_days=merchant.registration_age_days,
            merchant_reputation_score=merchant.reputation_score,
            agent_id=agent.agent_id,
            agent_attestation_present=True,
            agent_attestation_verified=True,
            agent_behavior_pattern_match_score=float(np.round(rng.uniform(0.7, 0.99), 3)),
            mandate_id=agent.mandate_id,
            mandate_amount_cap=agent.mandate_amount_cap,
            mandate_category_scope=agent.mandate_category_scope,
            mandate_recurring_flag=agent.mandate_recurring_flag,
            transaction_within_mandate_envelope=within_envelope,
            account_age_days=acct.account_age_days,
            prior_transaction_count=acct.prior_transaction_count,
            historical_avg_amount=acct.historical_avg_amount,
            credit_history_months=acct.credit_history_months,
            geo_country=acct.geo_country,
            authorization_result="approved",
            is_fraud=True,
            fraud_category="C",
            fraud_vector="C2",
            narrative_tag="malicious storefront: legitimate agent deceived into paying a newly registered, low-reputation merchant",
            **_novelty_flags(rng, _NOVELTY_LEGIT),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
        )
        rows.append(row)
    return rows


def generate_c3_mandate_scope_abuse(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Transaction content diverging from the pattern of prior transactions
    executed under the same mandate; amount or merchant category outside the
    historical envelope for that agent-consumer pairing."
    """
    rows = []
    for _ in range(n):
        agent = entities.sample_agent(rng)
        merchant = entities.sample_merchant(rng)
        breach_amount = bool(rng.random() < 0.6)
        if breach_amount:
            # amount breach; category may or may not also be out of scope
            amount = agent.mandate_amount_cap * float(rng.uniform(1.3, 4.0))
            category = merchant.category
        else:
            # amount stays within the mandate's cap; the breach is purely a
            # category-scope violation, so it must actually differ from scope
            amount = min(lognormal_amount(rng, 4.5, 0.5), agent.mandate_amount_cap * float(rng.uniform(0.2, 0.9)))
            other_categories = [c for c in MERCHANT_CATEGORIES if c != agent.mandate_category_scope]
            category = str(rng.choice(other_categories))
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=agent.linked_account_id,
            customer_id=short_id("cust", rng),
            channel="agentic_checkout",
            amount=float(np.round(amount, 2)),
            merchant_id=merchant.merchant_id,
            merchant_category=category,
            merchant_registration_age_days=merchant.registration_age_days,
            merchant_reputation_score=merchant.reputation_score,
            agent_id=agent.agent_id,
            agent_attestation_present=True,
            agent_attestation_verified=True,
            # C3's agent identity isn't in question - the taxonomy frames it
            # as "a legitimate agent, or an attacker who has compromised one"
            # executing within a broad but abused mandate, not an attestation
            # mismatch - so its behavior-pattern score matches the same
            # well-matched range as a genuinely verified agent (C2/legit),
            # not the low C1-style mismatch score.
            agent_behavior_pattern_match_score=float(np.round(rng.uniform(0.7, 0.99), 3)),
            mandate_id=agent.mandate_id,
            mandate_amount_cap=agent.mandate_amount_cap,
            mandate_category_scope=agent.mandate_category_scope,
            mandate_recurring_flag=agent.mandate_recurring_flag,
            transaction_within_mandate_envelope=False,
            credit_history_months=agent.linked_account.credit_history_months,
            authorization_result="approved",
            is_fraud=True,
            fraud_category="C",
            fraud_vector="C3",
            narrative_tag="delegated mandate scope abuse: execution diverges from the mandate's amount cap or category scope",
            **_novelty_flags(rng, _NOVELTY_LEGIT),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Category D: Automated, Machine Speed Attacks
# ---------------------------------------------------------------------------

def generate_d1_card_testing(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Very low value, high velocity transaction attempts across a cluster
    of merchants or a single merchant in a short window, often with low
    authorization success rate followed by a spike in usage of successfully
    validated cards elsewhere."
    """
    rows: list[dict] = []
    while len(rows) < n:
        seq_id = short_id("seq", rng)
        attempts = int(rng.integers(5, 30))
        window_seconds = float(rng.integers(30, 600))
        card_account = short_id("acct", rng)  # stolen/generated card, not a real pool account
        # the stolen card belongs to some real cardholder the attacker knows
        # nothing about, so its credit history is drawn from the same
        # population range as any real account, not left null or forced low.
        stolen_credit_history_months = int(rng.integers(0, 240))
        base_ts = random_timestamp(rng)
        for _ in range(attempts):
            merchant = entities.sample_merchant(rng)
            approved = rng.random() < 0.08
            rows.append(
                _txn(
                    transaction_id=short_id("txn", rng),
                    timestamp=base_ts,
                    account_id=card_account,
                    customer_id=short_id("cust", rng),
                    channel="card_not_present",
                    amount=float(np.round(rng.uniform(0.5, 5.0), 2)),
                    merchant_id=merchant.merchant_id,
                    merchant_category=merchant.category,
                    merchant_registration_age_days=merchant.registration_age_days,
                    merchant_reputation_score=merchant.reputation_score,
                    account_age_days=0,
                    prior_transaction_count=0,
                    historical_avg_amount=0.0,
                    credit_history_months=stolen_credit_history_months,
                    attempt_sequence_id=seq_id,
                    attempts_in_window=attempts,
                    window_seconds=window_seconds,
                    authorization_result="approved" if approved else "declined",
                    geo_country=str(rng.choice(GEOS)),
                    is_fraud=True,
                    fraud_category="D",
                    fraud_vector="D1",
                    narrative_tag="AI-accelerated card testing: low-value high-velocity attempts, low success rate",
                    **_novelty_flags(rng, _NOVELTY_MODERATE),
                    **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
                    **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
                )
            )
            if len(rows) >= n:
                break
    return rows[:n]


def generate_d2_credential_stuffing_ato(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Login from a new device or geography immediately followed by a
    payment or transfer, especially one that changes account contact details
    or payout destinations first."
    """
    rows = []
    for _ in range(n):
        acct = entities.sample_account(rng)
        channel = str(rng.choice(["p2p", "ach", "card_not_present"]))
        row = _txn(
            transaction_id=short_id("txn", rng),
            timestamp=random_timestamp(rng),
            account_id=acct.account_id,
            customer_id=acct.customer_id,
            channel=channel,
            amount=max(lognormal_amount(rng, 6.0, 0.7), acct.historical_avg_amount * float(rng.uniform(2, 8))),
            account_age_days=acct.account_age_days,
            prior_transaction_count=acct.prior_transaction_count,
            historical_avg_amount=acct.historical_avg_amount,
            credit_history_months=acct.credit_history_months,
            geo_country=str(rng.choice(GEOS)),
            device_id=short_id("dev", rng),
            login_event_flag=True,
            account_detail_changed_before_txn=bool(rng.random() < 0.75),
            authorization_result="approved",
            is_fraud=True,
            fraud_category="D",
            fraud_vector="D2",
            narrative_tag="AI-scaled credential stuffing ATO: new device/geo login, payout details changed, then a transfer",
            **_novelty_flags(rng, _NOVELTY_ELEVATED),
            **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
            **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
            **_sample_payee(rng, channel, first_time_bias=0.5),
            **_sample_approval_context(rng, channel, _URGENCY_OTHER, _HIERARCHY_OTHER),
            **_sample_support_call_context(rng, channel, _SUPPORT_CALL_OTHER),
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Category E: Post Transaction and Dispute Fraud
# ---------------------------------------------------------------------------

_DISPUTE_REASONS = ["item_not_as_described", "damaged_on_arrival", "not_delivered", "unauthorized_charge"]


def generate_e1_fabricated_dispute_evidence(
    rng: np.random.Generator, entities: EntityPool, n: int
) -> tuple[list[dict], list[dict]]:
    """"Refund or dispute filed with supporting media, correlated across
    multiple claims with subtle generative artifacts, unusually high dispute
    rate from an account with an otherwise clean transaction history."

    Returns (linked_transactions, disputes): the underlying purchase is a
    genuine transaction (is_fraud False on that row); the fraud is in the
    dispute filed against it.
    """
    txns, disputes = [], []
    for _ in range(n):
        acct = entities.sample_account(rng)
        merchant = entities.sample_merchant(rng)
        txn_id = short_id("txn", rng)
        purchase_ts = random_timestamp(rng, window_days=200)
        txns.append(
            _txn(
                transaction_id=txn_id,
                timestamp=purchase_ts,
                account_id=acct.account_id,
                customer_id=acct.customer_id,
                channel="card_not_present",
                amount=lognormal_amount(rng, 4.4, 0.8),
                merchant_id=merchant.merchant_id,
                merchant_category=merchant.category,
                merchant_registration_age_days=merchant.registration_age_days,
                merchant_reputation_score=merchant.reputation_score,
                account_age_days=acct.account_age_days,
                prior_transaction_count=acct.prior_transaction_count,
                historical_avg_amount=acct.historical_avg_amount,
                credit_history_months=acct.credit_history_months,
                geo_country=acct.geo_country,
                authorization_result="approved",
                is_fraud=False,
                credential_exposure_window_flag=bool(rng.random() < _CRED_EXPOSURE_LEGIT),
                claimed_employment_tenure_months=int(rng.integers(0, 480)),
                identity_cross_source_correlation=float(np.round(rng.uniform(0.75, 0.99), 3)),
                **_novelty_flags(rng, _NOVELTY_LEGIT),
            )
        )
        disputes.append(
            _dispute(
                dispute_id=short_id("dsp", rng),
                original_transaction_id=txn_id,
                account_id=acct.account_id,
                dispute_filed_at=random_timestamp(rng, window_days=30),
                dispute_reason=str(rng.choice(_DISPUTE_REASONS)),
                evidence_type=str(rng.choice(["photo", "receipt"])),
                evidence_generated_flag=True,
                evidence_generative_artifact_score=float(np.round(rng.uniform(0.55, 0.96), 3)),
                claimant_dispute_rate_30d=float(np.round(rng.uniform(0.3, 0.9), 3)),
                correlated_claim_count=int(rng.integers(2, 9)),
                is_fraud=True,
                fraud_category="E",
                fraud_vector="E1",
                narrative_tag="GenAI fabricated dispute evidence: generative artifacts in submitted media, correlated across claims",
            )
        )
    return txns, disputes


def generate_legit_disputes(rng: np.random.Generator, entities: EntityPool, n: int) -> tuple[list[dict], list[dict]]:
    txns, disputes = [], []
    for _ in range(n):
        acct = entities.sample_account(rng)
        merchant = entities.sample_merchant(rng)
        txn_id = short_id("txn", rng)
        txns.append(
            _txn(
                transaction_id=txn_id,
                timestamp=random_timestamp(rng, window_days=200),
                account_id=acct.account_id,
                customer_id=acct.customer_id,
                channel="card_not_present",
                amount=lognormal_amount(rng, 4.2, 0.8),
                merchant_id=merchant.merchant_id,
                merchant_category=merchant.category,
                merchant_registration_age_days=merchant.registration_age_days,
                merchant_reputation_score=merchant.reputation_score,
                account_age_days=acct.account_age_days,
                prior_transaction_count=acct.prior_transaction_count,
                historical_avg_amount=acct.historical_avg_amount,
                credit_history_months=acct.credit_history_months,
                geo_country=acct.geo_country,
                authorization_result="approved",
                is_fraud=False,
                credential_exposure_window_flag=bool(rng.random() < _CRED_EXPOSURE_LEGIT),
                claimed_employment_tenure_months=int(rng.integers(0, 480)),
                identity_cross_source_correlation=float(np.round(rng.uniform(0.75, 0.99), 3)),
                **_novelty_flags(rng, _NOVELTY_LEGIT),
            )
        )
        disputes.append(
            _dispute(
                dispute_id=short_id("dsp", rng),
                original_transaction_id=txn_id,
                account_id=acct.account_id,
                dispute_filed_at=random_timestamp(rng, window_days=30),
                dispute_reason=str(rng.choice(_DISPUTE_REASONS)),
                evidence_type=str(rng.choice(["photo", "receipt", "none"])),
                evidence_generated_flag=False,
                evidence_generative_artifact_score=float(np.round(rng.uniform(0.0, 0.15), 3)),
                claimant_dispute_rate_30d=float(np.round(rng.uniform(0.0, 0.08), 3)),
                correlated_claim_count=int(rng.integers(0, 2)),
                is_fraud=False,
            )
        )
    return txns, disputes


def generate_e2_promo_abuse(rng: np.random.Generator, entities: EntityPool, n: int) -> list[dict]:
    """"Cluster of accounts with distinct identities but correlated device,
    network, or behavioral fingerprints, each making a single low value
    promotional transaction shortly after account creation."
    """
    rows: list[dict] = []
    while len(rows) < n:
        cluster_id = short_id("clu", rng)
        cluster_size = int(rng.integers(5, 16))
        shared_device = short_id("dev", rng)
        shared_geo = str(rng.choice(GEOS))
        for _ in range(cluster_size):
            merchant = entities.sample_merchant(rng)
            rows.append(
                _txn(
                    transaction_id=short_id("txn", rng),
                    timestamp=random_timestamp(rng),
                    account_id=short_id("acct", rng),
                    customer_id=short_id("cust", rng),
                    channel="promo_redemption",
                    amount=float(np.round(rng.uniform(1, 25), 2)),
                    merchant_id=merchant.merchant_id,
                    merchant_category=merchant.category,
                    merchant_registration_age_days=merchant.registration_age_days,
                    merchant_reputation_score=merchant.reputation_score,
                    account_age_days=int(rng.integers(0, 2)),
                    prior_transaction_count=0,
                    historical_avg_amount=0.0,
                    credit_history_months=int(rng.integers(0, 6)),
                    device_id=shared_device,
                    geo_country=shared_geo,
                    promo_code=str(rng.choice(_PROMO_CODES)),
                    account_cluster_id=cluster_id,
                    authorization_result="approved",
                    is_fraud=True,
                    fraud_category="E",
                    fraud_vector="E2",
                    narrative_tag="GenAI-scaled promo abuse: cluster of fresh accounts sharing device/network fingerprint",
                    **_novelty_flags(rng, _NOVELTY_ELEVATED),
                    **_sample_credential_exposure(rng, _CRED_EXPOSURE_LEGIT),
                    **_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL),
                )
            )
            if len(rows) >= n:
                break
    return rows[:n]


VECTOR_GENERATORS = {
    "A1": generate_a1_synthetic_identity,
    "A2": generate_a2_deepfake_kyc,
    "B1": generate_b1_bec_voice_deepfake,
    "B2": generate_b2_deepfake_video_app,
    "B3": generate_b3_multivector_phishing,
    "C1": generate_c1_agent_impersonation,
    "C2": generate_c2_malicious_storefront,
    "C3": generate_c3_mandate_scope_abuse,
    "D1": generate_d1_card_testing,
    "D2": generate_d2_credential_stuffing_ato,
    "E2": generate_e2_promo_abuse,
}
