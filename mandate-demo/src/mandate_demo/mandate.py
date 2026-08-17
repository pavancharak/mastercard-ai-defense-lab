"""The mandate-envelope check: a deterministic, non-ML structural layer.

This is the local reimplementation of Parmana's real pattern: a consumer
delegates a bounded mandate to an agent (amount cap, category scope,
recurring permission), and every proposed execution is checked against
that mandate's actual terms before it's trusted -- not against whatever
the agent (or a compromised version of it) claims about itself. Nothing
here is machine-learned; every verdict is a plain comparison against the
mandate record, so it can't be talked out of a decision the way a
classifier's soft score sometimes can.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mandate:
    """What the consumer actually delegated to the agent. Authoritative --
    never mutated by a proposed transaction, only compared against."""

    mandate_id: str
    principal_id: str  # the consumer who delegated
    agent_id: str  # the agent authorized to act
    category_scope: tuple[str, ...]  # merchant categories the mandate covers
    per_transaction_cap: float
    monthly_cap: float
    recurring_allowed: bool
    currency: str = "USD"


@dataclass(frozen=True)
class ProposedPurchase:
    """What the agent actually attempted, as reported by the mock merchant
    API after a successful checkout -- the ground truth of what would be
    charged, not what the agent claims it's about to do."""

    product_id: str
    product_name: str
    merchant_id: str
    merchant_name: str
    category: str
    amount: float
    recurring: bool
    currency: str = "USD"


@dataclass
class MandateLedger:
    """Tracks cumulative approved spend under one mandate within the
    current billing period. A plain running total -- this is exactly the
    kind of state a per-transaction feature vector can never see, which is
    why the monthly-cap case in this demo is structurally invisible to a
    row-at-a-time classifier regardless of how well-trained it is."""

    mandate_id: str
    period_label: str
    approved_total: float = 0.0

    def project(self, amount: float) -> float:
        return self.approved_total + amount

    def record_approved(self, amount: float) -> None:
        self.approved_total += amount


@dataclass(frozen=True)
class BindingViolation:
    """One field-level mismatch between what the mandate authorized and
    what was actually proposed. Mirrors the shape of Parmana's real
    RefusalBindingViolation (signal_key/intent_path/signal_value/
    intent_value) -- see refusal.py."""

    field: str
    mandate_authorized: object
    actually_proposed: object
    explanation: str


@dataclass(frozen=True)
class EnvelopeVerdict:
    allowed: bool
    violations: list[BindingViolation] = field(default_factory=list)


def check_envelope(
    mandate: Mandate, proposed: ProposedPurchase, ledger: MandateLedger
) -> EnvelopeVerdict:
    """Deterministic allow/refuse. Every check is a direct comparison
    against the mandate record -- no scoring, no threshold, no model."""

    violations: list[BindingViolation] = []

    if proposed.category not in mandate.category_scope:
        violations.append(
            BindingViolation(
                field="merchant_category",
                mandate_authorized=list(mandate.category_scope),
                actually_proposed=proposed.category,
                explanation=(
                    f"mandate authorizes {list(mandate.category_scope)}; "
                    f"proposed purchase is in category '{proposed.category}'"
                ),
            )
        )

    if proposed.amount > mandate.per_transaction_cap:
        violations.append(
            BindingViolation(
                field="amount",
                mandate_authorized=mandate.per_transaction_cap,
                actually_proposed=proposed.amount,
                explanation=(
                    f"mandate's per-transaction cap is {mandate.per_transaction_cap} "
                    f"{mandate.currency}; proposed amount is {proposed.amount} "
                    f"{proposed.currency}"
                ),
            )
        )

    projected_total = ledger.project(proposed.amount)
    if projected_total > mandate.monthly_cap:
        violations.append(
            BindingViolation(
                field="cumulative_period_spend",
                mandate_authorized=mandate.monthly_cap,
                actually_proposed=round(projected_total, 2),
                explanation=(
                    f"mandate's monthly cap is {mandate.monthly_cap} {mandate.currency}; "
                    f"approved spend so far this period is {ledger.approved_total} "
                    f"{mandate.currency}, and this purchase would bring the running "
                    f"total to {round(projected_total, 2)} {mandate.currency}"
                ),
            )
        )

    if proposed.recurring and not mandate.recurring_allowed:
        violations.append(
            BindingViolation(
                field="recurring",
                mandate_authorized=mandate.recurring_allowed,
                actually_proposed=proposed.recurring,
                explanation="mandate does not authorize recurring/subscription charges",
            )
        )

    return EnvelopeVerdict(allowed=len(violations) == 0, violations=violations)
