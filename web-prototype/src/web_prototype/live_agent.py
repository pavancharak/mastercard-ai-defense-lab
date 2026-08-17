"""Triggers a real, live run of mandate-demo's agent from the web UI.

This module does NOT reimplement mandate-demo's agent loop, classifier
scoring, or mandate-envelope check. It imports and calls the real
functions from mandate_demo.agent / mandate_demo.server /
mandate_demo.mandate / mandate_demo.refusal / mandate_demo.runner /
mandate_demo.scenarios, in the same sequence mandate_demo.runner.main()
already uses, and only adds two things neither of those needed before:
(1) usage/cost tracking, via a thin object passed in place of a raw
OpenAI() client -- mandate_demo.agent.run_agent_purchase() already takes
a `client` parameter and only ever calls `client.chat.completions.create`
on it, so this satisfies that same interface without agent.py itself
being touched; (2) returning structured data instead of printing to
stdout, since this runs inside a web request instead of a CLI.

OPENAI_API_KEY is read server-side only (by the OpenAI() client
constructor, from the server process's own environment) and never sent
to, or readable by, the browser -- see app.py's live-run endpoint, which
returns only this module's structured result, never any request/response
object that could carry the key.
"""

from __future__ import annotations

from dataclasses import asdict

from openai import OpenAI

from mandate_demo.agent import MODEL, run_agent_purchase
from mandate_demo.mandate import MandateLedger
from mandate_demo.refusal import build_refusal_record, refusal_record_to_dict
from mandate_demo.runner import _assert_local_only, run_three_checks
from mandate_demo.scenarios import MANDATE, PROFILE, SCRIPTED_INTENTS
from mandate_demo.server import LocalMerchantServer

from .scoring import get_classifier

# Checked live via WebFetch against https://developers.openai.com/api/docs/pricing
# during this build (gpt-4o-mini standard tier). Re-check before relying on
# this for real budgeting if OpenAI's pricing has changed since.
_GPT_4O_MINI_PRICE_PER_1M_INPUT_USD = 0.15
_GPT_4O_MINI_PRICE_PER_1M_OUTPUT_USD = 0.60
_PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing (checked live during development)"


class _UsageTrackingCompletions:
    def __init__(self, real_completions, usage_sink: list[dict]) -> None:
        self._real = real_completions
        self._sink = usage_sink

    def create(self, **kwargs):
        response = self._real.create(**kwargs)
        if response.usage is not None:
            self._sink.append(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )
        return response


class _UsageTrackingChat:
    def __init__(self, real_chat, usage_sink: list[dict]) -> None:
        self.completions = _UsageTrackingCompletions(real_chat.completions, usage_sink)


class UsageTrackingOpenAIClient:
    """Duck-types as an OpenAI client for run_agent_purchase()'s
    purposes: exposes .chat.completions.create() with the identical
    signature/behavior, forwarding to the real client, and records
    token usage as a side effect. mandate_demo/agent.py is never
    modified or aware this wrapper exists."""

    def __init__(self, real_client: OpenAI, usage_sink: list[dict]) -> None:
        self.chat = _UsageTrackingChat(real_client.chat, usage_sink)


def estimate_cost(usage_records: list[dict]) -> dict:
    prompt_tokens = sum(u["prompt_tokens"] for u in usage_records)
    completion_tokens = sum(u["completion_tokens"] for u in usage_records)
    total_tokens = sum(u["total_tokens"] for u in usage_records)
    cost_usd = (
        (prompt_tokens / 1_000_000) * _GPT_4O_MINI_PRICE_PER_1M_INPUT_USD
        + (completion_tokens / 1_000_000) * _GPT_4O_MINI_PRICE_PER_1M_OUTPUT_USD
    )
    return {
        "model": MODEL,
        "api_calls": len(usage_records),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost_usd, 6),
        "pricing_source": _PRICING_SOURCE,
    }


def run_live_scenarios() -> dict:
    """Mirrors mandate_demo.runner.main()'s orchestration loop exactly
    (same functions, same call sequence, same per-scenario checks), but
    returns a JSON-serializable dict instead of printing to stdout, and
    tracks real OpenAI token usage/cost alongside it."""

    real_client = OpenAI()
    usage_records: list[dict] = []
    tracked_client = UsageTrackingOpenAIClient(real_client, usage_records)

    server = LocalMerchantServer()
    server.start()
    try:
        classifier = get_classifier()
        ledger = MandateLedger(mandate_id=MANDATE.mandate_id, period_label="live-web-demo")

        all_urls_called: list[str] = [f"{server.base_url}/health"]
        scenario_reports: list[dict] = []
        demonstrated_cases: list[str] = []

        for intent in SCRIPTED_INTENTS:
            proposed, call_log, _transcript = run_agent_purchase(
                tracked_client, server.base_url, intent.text, MANDATE
            )
            for entry in call_log:
                all_urls_called.append(entry.url)

            if proposed is None:
                scenario_reports.append(
                    {"intent_id": intent.intent_id, "intent_text": intent.text, "agent_completed": False}
                )
                continue

            checks_call_log: list[dict] = []
            merchant_result, classifier_result, mandate_result = run_three_checks(
                server, classifier, ledger, proposed, checks_call_log
            )
            for entry in checks_call_log:
                all_urls_called.append(entry["url"])

            refusal_record = None
            if mandate_result.allowed:
                ledger.record_approved(proposed.amount)
            else:
                refusal_record = build_refusal_record(
                    refusal_record_id=f"live_refusal_{intent.intent_id}_{len(scenario_reports)}",
                    purchase_attempt_id=f"live_attempt_{intent.intent_id}",
                    target=f"{proposed.merchant_name}:{proposed.product_id}",
                    parameters={
                        "amount": proposed.amount,
                        "currency": proposed.currency,
                        "category": proposed.category,
                        "recurring": proposed.recurring,
                    },
                    violations=mandate_result.violations,
                    submitted_by=MANDATE.agent_id,
                )

            classifier_missed = (not mandate_result.allowed) and classifier_result.risk_label == "LOW_RISK"
            if classifier_missed:
                demonstrated_cases.append(intent.intent_id)

            scenario_reports.append(
                {
                    "intent_id": intent.intent_id,
                    "intent_text": intent.text,
                    "note": intent.note,
                    "agent_completed": True,
                    "proposed_purchase": asdict(proposed),
                    "check_a_merchant": merchant_result,
                    "check_b_classifier": asdict(classifier_result),
                    "check_c_mandate": {
                        "allowed": mandate_result.allowed,
                        "violations": [asdict(v) for v in mandate_result.violations],
                    },
                    "refusal_record": refusal_record_to_dict(refusal_record) if refusal_record else None,
                    "classifier_missed_mandate_caught": classifier_missed,
                }
            )

        _assert_local_only(all_urls_called)

        return {
            "mandate": asdict(MANDATE),
            "profile": asdict(PROFILE),
            "scenarios": scenario_reports,
            "all_urls_called": all_urls_called,
            "demonstrated_cases": demonstrated_cases,
            "usage": estimate_cost(usage_records),
            "source": "live",
        }
    finally:
        server.stop()
