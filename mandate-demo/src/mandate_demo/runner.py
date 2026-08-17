"""Orchestrates the full demo: starts the local mock merchant API, runs
the live OpenAI agent against each scripted purchase intent, then runs
the three independent checks (merchant recheck / Defend classifier /
mandate-envelope) in parallel for each resulting proposed purchase.

Run with:  .venv/Scripts/python -m mandate_demo.runner
"""

from __future__ import annotations

import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI

from .agent import run_agent_purchase
from .catalog import get_product
from .classifier import ClassifierResult, DefendClassifier
from .mandate import EnvelopeVerdict, MandateLedger, ProposedPurchase, check_envelope
from .refusal import build_refusal_record, refusal_record_to_dict
from .scenarios import MANDATE, PROFILE, SCRIPTED_INTENTS
from .server import LocalMerchantServer

# Loads mastercard-ai-defense-lab/.env (the repo root, three levels above
# this file: mandate_demo/ -> src/ -> mandate-demo/ -> repo root) into
# the process environment before anything below reads OPENAI_API_KEY --
# an explicit path, not reliant on the current working directory
# matching wherever .env happens to sit. Runs at import time (module
# level), so this also covers callers that import this module without
# calling main() directly (e.g. web-prototype's live_agent.py). A
# missing .env is not an error here -- load_dotenv() just no-ops. override=True
# so this project's own .env always wins over a stray/stale environment
# variable already present in the shell (e.g. left over from earlier manual
# testing, or a machine-wide variable) -- the whole point of this fix is that
# .env is the single, trustworthy source, not "whatever happened to be set."
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"

ALLOWED_HOSTS = {"127.0.0.1"}


def _assert_local_only(urls: list[str]) -> None:
    from urllib.parse import urlparse

    for url in urls:
        host = urlparse(url).hostname
        if host not in ALLOWED_HOSTS:
            raise RuntimeError(
                f"HARD CONSTRAINT VIOLATION: a call was made to '{url}' "
                f"(host '{host}'), which is not the local mock merchant API."
            )


def build_classifier_features(proposed: ProposedPurchase) -> dict:
    """Maps the agent's proposed purchase + the fixed agent/consumer
    profile onto Defend's real feature schema. Deliberately omits
    `transaction_within_mandate_envelope` -- see classifier.py's
    module docstring for why."""

    product = get_product(proposed.product_id)
    return {
        "amount": proposed.amount,
        "merchant_registration_age_days": product.merchant_registration_age_days if product else None,
        "merchant_reputation_score": product.merchant_reputation_score if product else None,
        "account_age_days": PROFILE.account_age_days,
        "prior_transaction_count": PROFILE.prior_transaction_count,
        "historical_avg_amount": PROFILE.historical_avg_amount,
        "credit_history_months": PROFILE.credit_history_months,
        "claimed_employment_tenure_months": PROFILE.claimed_employment_tenure_months,
        "identity_cross_source_correlation": PROFILE.identity_cross_source_correlation,
        "session_video_artifact_score": None,
        "onboarding_velocity_seconds": None,
        "payee_prior_interaction_count": None,
        "time_since_support_call_minutes": None,
        "agent_behavior_pattern_match_score": PROFILE.agent_behavior_pattern_match_score,
        "mandate_amount_cap": MANDATE.per_transaction_cap,
        "attempts_in_window": None,
        "window_seconds": None,
        "txn_hour": PROFILE.txn_hour,
        "channel": PROFILE.channel,
        "currency": proposed.currency,
        "merchant_category": proposed.category,
        "geo_country": PROFILE.geo_country,
        "mandate_category_scope": MANDATE.category_scope[0],
        "authorization_result": "approved",
        "device_is_new": PROFILE.device_is_new,
        "geo_is_new": PROFILE.geo_is_new,
        "session_frame_rate_anomaly": None,
        "login_event_flag": None,
        "account_detail_changed_before_txn": None,
        "is_first_time_payee": None,
        "request_urgency_flag": None,
        "approval_outside_hierarchy": None,
        "preceded_by_support_call": None,
        "credential_exposure_window_flag": PROFILE.credential_exposure_window_flag,
        "agent_attestation_present": PROFILE.agent_attestation_present,
        "agent_attestation_verified": PROFILE.agent_attestation_verified,
        "mandate_recurring_flag": MANDATE.recurring_allowed,
        # transaction_within_mandate_envelope: intentionally absent.
    }


def run_three_checks(
    server: LocalMerchantServer,
    classifier: DefendClassifier,
    ledger: MandateLedger,
    proposed: ProposedPurchase,
    call_log: list,
) -> tuple[dict, ClassifierResult, EnvelopeVerdict]:
    def merchant_recheck() -> dict:
        url = f"{server.base_url}/products/{proposed.product_id}"
        resp = requests.get(url, timeout=5)
        call_log.append({"tool": "merchant_recheck", "http_method": "GET", "url": url, "status_code": resp.status_code})
        if resp.status_code == 200:
            body = resp.json()
            return {"success": True, "in_stock": body["in_stock"], "detail": body}
        return {"success": False, "detail": resp.json()}

    def classifier_check() -> ClassifierResult:
        features = build_classifier_features(proposed)
        return classifier.score(features)

    def mandate_check() -> EnvelopeVerdict:
        return check_envelope(MANDATE, proposed, ledger)

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_merchant = pool.submit(merchant_recheck)
        f_classifier = pool.submit(classifier_check)
        f_mandate = pool.submit(mandate_check)
        merchant_result = f_merchant.result()
        classifier_result = f_classifier.result()
        mandate_result = f_mandate.result()

    return merchant_result, classifier_result, mandate_result


def main() -> int:
    print("=" * 78)
    print("C3 (Delegated Mandate Scope Abuse) structural-prevention demo")
    print("Local reimplementation of Parmana's mandate-binding pattern.")
    print("NOT the real @parmana/sdk. NOT connected to any live Parmana system.")
    print("=" * 78)

    server = LocalMerchantServer()
    server.start()
    print(f"\n[setup] local mock merchant API up at {server.base_url} (127.0.0.1 only)")

    try:
        classifier = DefendClassifier()
        print(f"[setup] loaded Defend's real trained model from {classifier.metadata['target_column']!r} "
              f"target, decision_threshold={classifier.metadata['decision_threshold']}")
    except FileNotFoundError as e:
        print(f"[setup] FATAL: {e}")
        server.stop()
        return 1

    client = OpenAI()
    ledger = MandateLedger(mandate_id=MANDATE.mandate_id, period_label="2026-08")

    all_urls_called: list[str] = [f"{server.base_url}/health"]
    scenario_reports = []
    demonstrated_cases = []

    for intent in SCRIPTED_INTENTS:
        print(f"\n{'-' * 78}\n[{intent.intent_id}] intent: {intent.text}\n  ({intent.note})")

        proposed, call_log, transcript = run_agent_purchase(client, server.base_url, intent.text, MANDATE)
        for entry in call_log:
            all_urls_called.append(entry.url)

        if proposed is None:
            print("  [agent] FAILED to complete a purchase within the turn budget -- skipping checks.")
            scenario_reports.append({"intent_id": intent.intent_id, "intent_text": intent.text, "agent_completed": False})
            continue

        print(f"  [agent] proposed: {proposed.product_name} ({proposed.category}), "
              f"${proposed.amount:.2f} {proposed.currency}, recurring={proposed.recurring}")

        checks_call_log: list[dict] = []
        merchant_result, classifier_result, mandate_result = run_three_checks(
            server, classifier, ledger, proposed, checks_call_log
        )
        for entry in checks_call_log:
            all_urls_called.append(entry["url"])

        print(f"  [check a: merchant]   success={merchant_result['success']}, in_stock={merchant_result.get('in_stock')}")
        print(f"  [check b: classifier] fraud_probability={classifier_result.fraud_probability:.4f} "
              f"(threshold={classifier_result.decision_threshold}) -> {classifier_result.risk_label}")
        print(f"  [check c: mandate]    allowed={mandate_result.allowed}"
              + ("" if mandate_result.allowed else f", violations={[v.field for v in mandate_result.violations]}"))

        refusal_record = None
        if mandate_result.allowed:
            ledger.record_approved(proposed.amount)
            print(f"  [ledger] approved -- running total this period: ${ledger.approved_total:.2f}")
        else:
            refusal_record = build_refusal_record(
                refusal_record_id=f"refusal_{uuid.uuid4().hex[:12]}",
                purchase_attempt_id=f"attempt_{intent.intent_id}_{uuid.uuid4().hex[:8]}",
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
            print(f"  [refusal record {refusal_record.refusal_record_id}]")
            for v in mandate_result.violations:
                print(f"      - {v.field}: mandate authorized {v.mandate_authorized!r}, "
                      f"actually proposed {v.actually_proposed!r} -- {v.explanation}")

        classifier_missed = (not mandate_result.allowed) and classifier_result.risk_label == "LOW_RISK"
        if classifier_missed:
            print("  *** DEMONSTRATED: classifier scored this LOW_RISK; mandate check refused it anyway. ***")
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

    print(f"\n{'=' * 78}\nVerifying nothing was called outside the local mock merchant API...")
    _assert_local_only(all_urls_called)
    print(f"[verified] all {len(all_urls_called)} HTTP calls made during this run resolved to "
          f"127.0.0.1 only (host allowlist: {sorted(ALLOWED_HOSTS)}).")

    print(f"\n{'=' * 78}\nSUMMARY")
    print(f"  scenarios run: {len(scenario_reports)}")
    allowed_count = sum(1 for r in scenario_reports if r.get("check_c_mandate", {}).get("allowed"))
    refused_count = sum(1 for r in scenario_reports if r.get("agent_completed") and not r["check_c_mandate"]["allowed"])
    print(f"  mandate check: {allowed_count} allowed, {refused_count} refused")
    if demonstrated_cases:
        print(f"  classifier-missed-it/mandate-caught-it cases: {', '.join(demonstrated_cases)}")
    else:
        print("  no classifier-missed-it case occurred this run (see README for the structural-invisibility "
              "case, which doesn't depend on the classifier's actual score).")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mandate": asdict(MANDATE),
                "profile": asdict(PROFILE),
                "scenarios": scenario_reports,
                "all_urls_called": all_urls_called,
                "demonstrated_cases": demonstrated_cases,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n[output] full captured run written to {out_path}")

    server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
