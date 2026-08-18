"""FastAPI app for the Mastercard Innovation Challenge web prototype.

Every route either renders a template from data_sources.py's read-only
loaders, or (for scoring / the live mandate-demo run) calls into
scoring.py / live_agent.py, which themselves only call mandate-demo's
real code. Nothing under identify/, generate/, defend/, or mandate-demo/
is written to by this app.
"""

from __future__ import annotations

import concurrent.futures
import os
import secrets
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent

# Loads mastercard-ai-defense-lab/.env (the repo root, three levels above
# BASE_DIR: web_prototype/ -> src/ -> web-prototype/ -> repo root) into
# the process environment on app startup, before any request can trigger
# a live mandate-demo run -- an explicit path, not reliant on whatever
# directory uvicorn happens to be launched from. Must run before the
# `.live_agent` import below: mandate_demo.runner (imported transitively
# by live_agent.py) also calls load_dotenv() at its own module level, so
# whichever import happens first wins the race to populate os.environ
# first; doing it here first keeps this app's own startup order explicit
# rather than depending on that transitive side effect. A missing .env
# is not an error -- load_dotenv() just no-ops. override=True so this
# project's own .env always wins over a stray/stale environment variable
# already present in the shell -- see mandate_demo/runner.py's identical
# comment for the full reasoning.
load_dotenv(BASE_DIR.parents[2] / ".env", override=True)

from . import data_sources, scoring  # noqa: E402
from .live_agent import run_live_scenarios  # noqa: E402
from .rate_limit import live_run_limiter  # noqa: E402

LIVE_RUN_TIMEOUT_SECONDS = float(os.environ.get("MANDATE_DEMO_LIVE_RUN_TIMEOUT_SECONDS", "90"))

app = FastAPI(title="AI Defense Lab -- Web Prototype")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

_live_run_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="live-mandate-run")

SESSION_COOKIE_NAME = "ai_defense_lab_session"


def _get_session_id(request: Request) -> str:
    return request.cookies.get(SESSION_COOKIE_NAME) or secrets.token_hex(16)


def _set_session_cookie(response, session_id: str) -> None:
    response.set_cookie(SESSION_COOKIE_NAME, session_id, httponly=True, samesite="lax")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/about-parmana", response_class=HTMLResponse)
def about_parmana(request: Request):
    return templates.TemplateResponse(request, "about_parmana.html", {})


@app.get("/cases", response_class=HTMLResponse)
def cases(request: Request):
    sample = data_sources.sample_transactions(limit_per_vector=3)
    return templates.TemplateResponse(
        request,
        "cases.html",
        {
            "sample_transactions": sample,
            "free_form_fields": scoring.FREE_FORM_FIELDS,
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "aggregate": data_sources.load_metrics_aggregate(),
            "per_category": data_sources.load_metrics_per_category(),
            "per_vector": data_sources.load_metrics_per_vector(),
        },
    )


@app.get("/taxonomy", response_class=HTMLResponse)
def taxonomy(request: Request):
    return templates.TemplateResponse(
        request,
        "taxonomy.html",
        {"categories": data_sources.load_taxonomy()},
    )


@app.get("/mandate-demo", response_class=HTMLResponse)
def mandate_demo_page(request: Request):
    session_id = _get_session_id(request)
    captured = data_sources.load_captured_run()
    rate_status = live_run_limiter.status(session_id)
    response = templates.TemplateResponse(
        request,
        "mandate_demo.html",
        {
            "captured": captured,
            "rate_status": rate_status,
            "max_per_session": live_run_limiter.max_per_session,
            "max_per_minute": live_run_limiter.max_per_minute,
        },
    )
    _set_session_cookie(response, session_id)
    return response


# ---------------------------------------------------------------------------
# JSON API: scoring
# ---------------------------------------------------------------------------


@app.get("/api/transaction/{transaction_id}")
def api_get_transaction(transaction_id: str):
    row = data_sources.get_transaction_by_id(transaction_id)
    if row is None:
        return JSONResponse({"error": f"no transaction with id '{transaction_id}'"}, status_code=404)
    return row


@app.post("/api/score/dataset")
def api_score_dataset(transaction_id: str = Form(...)):
    row = data_sources.get_transaction_by_id(transaction_id)
    if row is None:
        return JSONResponse({"error": f"no transaction with id '{transaction_id}'"}, status_code=404)

    features = scoring.feature_values_from_csv_row(row)
    result = scoring.score(features)
    return {
        "transaction_id": transaction_id,
        "ground_truth": {
            "is_fraud": row.get("is_fraud"),
            "fraud_category": row.get("fraud_category") or None,
            "fraud_vector": row.get("fraud_vector") or None,
            "narrative_tag": row.get("narrative_tag") or None,
        },
        "fraud_probability": result.fraud_probability,
        "decision_threshold": result.decision_threshold,
        "risk_label": result.risk_label,
        "features_sent": result.features_sent,
    }


@app.post("/api/score/custom")
async def api_score_custom(request: Request):
    form = dict((await request.form()))
    features = scoring.feature_values_from_form(form)
    result = scoring.score(features)
    return {
        "fraud_probability": result.fraud_probability,
        "decision_threshold": result.decision_threshold,
        "risk_label": result.risk_label,
        "features_sent": result.features_sent,
    }


# ---------------------------------------------------------------------------
# JSON API: mandate-demo live run
# ---------------------------------------------------------------------------


@app.post("/api/mandate-demo/live-run")
def api_live_run(request: Request):
    session_id = _get_session_id(request)
    decision = live_run_limiter.check_and_reserve(session_id)

    if not decision.allowed:
        response = JSONResponse(
            {
                "status": "rate_limited",
                "reason": decision.reason,
                "fallback": data_sources.load_captured_run(),
            },
            status_code=429,
        )
        _set_session_cookie(response, session_id)
        return response

    future = _live_run_executor.submit(run_live_scenarios)
    try:
        result = future.result(timeout=LIVE_RUN_TIMEOUT_SECONDS)
        response = JSONResponse({"status": "ok", "result": result})
    except concurrent.futures.TimeoutError:
        response = JSONResponse(
            {
                "status": "timeout",
                "reason": f"Live run did not complete within {LIVE_RUN_TIMEOUT_SECONDS:.0f}s.",
                "fallback": data_sources.load_captured_run(),
            },
            status_code=504,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any live-run failure must fall back cleanly, not 500 during a live demo
        response = JSONResponse(
            {
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "fallback": data_sources.load_captured_run(),
            },
            status_code=502,
        )

    _set_session_cookie(response, session_id)
    return response


@app.get("/api/mandate-demo/rate-limit-status")
def api_rate_limit_status(request: Request):
    session_id = _get_session_id(request)
    decision = live_run_limiter.status(session_id)
    response = JSONResponse(
        {
            "allowed": decision.allowed,
            "session_runs_used": decision.session_runs_used,
            "session_runs_remaining": decision.session_runs_remaining,
            "global_runs_in_last_minute": decision.global_runs_in_last_minute,
            "max_per_session": live_run_limiter.max_per_session,
            "max_per_minute": live_run_limiter.max_per_minute,
        }
    )
    _set_session_cookie(response, session_id)
    return response
