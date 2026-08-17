"""Deterministic tests for the mandate-envelope check -- no live API calls,
no OpenAI, no network. This is the layer the whole demo depends on being
correct and non-flaky."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mandate_demo.mandate import (  # noqa: E402
    Mandate,
    MandateLedger,
    ProposedPurchase,
    check_envelope,
)

MANDATE = Mandate(
    mandate_id="mnd_test",
    principal_id="alex",
    agent_id="agent_test",
    category_scope=("subscription",),
    per_transaction_cap=200.0,
    monthly_cap=200.0,
    recurring_allowed=True,
)


def _purchase(**overrides) -> ProposedPurchase:
    defaults = dict(
        product_id="p1",
        product_name="Test Product",
        merchant_id="m1",
        merchant_name="Test Merchant",
        category="subscription",
        amount=10.0,
        recurring=True,
    )
    defaults.update(overrides)
    return ProposedPurchase(**defaults)


def test_within_mandate_is_allowed():
    ledger = MandateLedger(mandate_id=MANDATE.mandate_id, period_label="test")
    verdict = check_envelope(MANDATE, _purchase(amount=15.99), ledger)
    assert verdict.allowed
    assert verdict.violations == []


def test_category_scope_violation_is_refused():
    ledger = MandateLedger(mandate_id=MANDATE.mandate_id, period_label="test")
    verdict = check_envelope(MANDATE, _purchase(category="electronics", amount=165.0), ledger)
    assert not verdict.allowed
    assert [v.field for v in verdict.violations] == ["merchant_category"]


def test_per_transaction_amount_violation_is_refused():
    ledger = MandateLedger(mandate_id=MANDATE.mandate_id, period_label="test")
    verdict = check_envelope(MANDATE, _purchase(amount=250.0), ledger)
    assert not verdict.allowed
    fields = [v.field for v in verdict.violations]
    assert "amount" in fields
    assert "cumulative_period_spend" in fields  # 250 alone also exceeds the 200 monthly cap


def test_cumulative_monthly_cap_violation_with_otherwise_valid_transaction():
    ledger = MandateLedger(mandate_id=MANDATE.mandate_id, period_label="test")
    ledger.record_approved(150.0)  # already spent this period
    verdict = check_envelope(MANDATE, _purchase(amount=60.0), ledger)  # under per-tx cap alone
    assert not verdict.allowed
    assert [v.field for v in verdict.violations] == ["cumulative_period_spend"]


def test_recurring_not_allowed_is_refused():
    strict_mandate = Mandate(
        mandate_id="mnd_strict",
        principal_id="alex",
        agent_id="agent_test",
        category_scope=("subscription",),
        per_transaction_cap=200.0,
        monthly_cap=200.0,
        recurring_allowed=False,
    )
    ledger = MandateLedger(mandate_id=strict_mandate.mandate_id, period_label="test")
    verdict = check_envelope(strict_mandate, _purchase(recurring=True), ledger)
    assert not verdict.allowed
    assert [v.field for v in verdict.violations] == ["recurring"]


def test_ledger_only_accumulates_when_caller_records_it():
    ledger = MandateLedger(mandate_id=MANDATE.mandate_id, period_label="test")
    assert ledger.approved_total == 0.0
    assert ledger.project(15.99) == 15.99
    assert ledger.approved_total == 0.0  # project() must not mutate state
    ledger.record_approved(15.99)
    assert ledger.approved_total == 15.99
