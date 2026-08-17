# Web Prototype

The "working web-based prototype with a presentable UI that demonstrates the closed-loop
system in action," required by the Mastercard Innovation Challenge submission. A read-only UI
layer on top of `identify/`, `generate/`, `defend/`, and `mandate-demo/` -- **nothing in those
four directories is modified, and none of their logic is reimplemented here.**

## Honest framing

- Every number, score, and record shown in the UI is read live from a file already committed
  in this repo, or computed by calling `defend/`'s actual trained model or `mandate-demo/`'s
  actual code -- never hardcoded.
- The live-agent trigger calls `mandate-demo`'s real, unmodified `agent.py` /
  `mock_merchant.py` / `mandate.py` / `classifier.py` / `refusal.py`, installed as a local
  editable Python package (`pip install -e ../mandate-demo`), not copied or reimplemented.
- `OPENAI_API_KEY` is read server-side only, by the `OpenAI()` client constructor in
  `live_agent.py`, from the server process's own environment -- populated automatically at
  startup from a project-local, gitignored `.env` file (see "How to run it locally" below), not
  a machine-wide environment variable. It is never sent to, embedded in, or readable from any
  template, static file, or API response served to the browser --
  confirmed by grepping the entire `templates/` and `static/` trees (everything actually
  served to a browser; there's no separate JS bundler/build step) for the key and for the
  string `OPENAI_API_KEY` before finishing this build. Zero matches.

## Structure

```
web-prototype/
  pyproject.toml
  src/web_prototype/
    app.py           FastAPI routes (pages + JSON API)
    data_sources.py    read-only loaders: taxonomy.md parser, transactions.csv sampler,
                         defend/results/*.{json,csv,md} loaders, mandate-demo captured-run loader
    scoring.py           wraps mandate_demo.classifier.DefendClassifier for the case browser
                          (real dataset rows or a judge's free-form input)
    live_agent.py          triggers a real live mandate-demo run by calling mandate_demo's
                            actual agent/server/mandate/classifier/refusal functions in the
                            same sequence mandate_demo.runner.main() already uses; adds only
                            usage/cost tracking and structured-JSON return (see its docstring)
    rate_limit.py            per-session + per-minute hard caps on live runs, backend-enforced
    templates/                 Jinja2: base, index, cases, dashboard, taxonomy, mandate_demo
    static/                     style.css, app.js (vanilla JS, no external CDN, no build step)
  tests/                  9 tests, no network: taxonomy parsing, metrics loading, captured-run
                          shape, classifier scoring (real model, real rows)
```

## How to run it locally

Requires Python >=3.11, and `defend/model/` already trained (it is, as of the committed state).

```bash
cd web-prototype
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"       # Windows; use bin/ not Scripts/ on macOS/Linux
.venv/Scripts/pip install -e ../mandate-demo  # local editable install -- not on any index
.venv/Scripts/python -m pytest -q             # 9 tests, no network, ~2s
.venv/Scripts/python -m uvicorn web_prototype.app:app --app-dir src --host 127.0.0.1 --port 8420
```

Then open `http://127.0.0.1:8420`.

**No manual export step for `OPENAI_API_KEY` needed** -- only required at all if you intend to
click "Run live now" on the Mandate Demo page (every other page works without it), and even then
it's picked up automatically. `app.py` calls `load_dotenv()` at module import time (before the
FastAPI app object or any route is even defined), pointed at an explicit path
(`../.env` relative to this project -- i.e. `mastercard-ai-defense-lab/.env`, the repo root, not
wherever `uvicorn` happens to be launched from), with `override=True` so this file always wins
over any stray environment variable already present in the shell. Copy `../.env.example` to
`../.env` and fill in a real key there; the server picks it up on every startup, automatically,
regardless of what directory the run command above is executed from. Verified end-to-end in a
shell with `OPENAI_API_KEY` explicitly unset before starting the server, using only the
`uvicorn` command above -- see "Live-run verification" below.

## What each view shows

- **`/` (Overview)** -- four cards linking to the taxonomy, dashboard, case browser, and the
  mandate-demo centerpiece, each with a one-line description of what it demonstrates.
- **`/taxonomy`** -- all 5 categories / 12 vectors from `identify/attack-taxonomy.md`, parsed
  from the file's own (escaped-markdown) format, each as a collapsible `<details>` panel with
  Mechanism / Channel / Transaction signal / Plausibility / Severity, exactly as written in the
  source file. A `muted` note flags that the source doc's own intro text says "fourteen" while
  only 12 are actually defined -- surfaced honestly rather than silently repeated as 14
  (the same discrepancy already found and corrected for when drafting this repo's initial
  commit message; `identify/` itself is not touched here either).
- **`/dashboard`** -- Defend's real aggregate precision/recall/F1/ROC-AUC as large stat tiles,
  a per-category table, and a per-vector recall table with a plain CSS width-bar (no chart
  library) -- all read straight from `defend/results/metrics_aggregate.json` /
  `metrics_per_category.csv` / `metrics_per_vector.csv`.
- **`/cases`** -- two ways to score a transaction:
  1. A sample table (3 rows per fraud vector + legit, 36 rows total) from
     `generate/data/transactions.csv`, each with a "Score" button that calls
     `defend/model/xgboost_model.json`'s real `predict_proba` via `/api/score/dataset` and
     shows the fraud probability/risk label next to the row's real ground-truth label
     (`is_fraud`/`fraud_category`/`fraud_vector`) for direct comparison -- plus a free-text
     `transaction_id` lookup for any of the ~21k rows, not just the sample.
  2. A free-form form (19 of the most decision-relevant fields, out of Defend's 38) a judge
     can fill in directly; every field left blank is sent to the classifier as genuinely
     missing (`NaN`), not a fabricated default -- exactly how Defend's own training data
     already represents a field that doesn't apply to a given channel/vector.

  Both paths show the full 38-feature vector actually sent to the model, collapsed under a
  `<details>` toggle, so a judge can verify nothing is hidden.

- **`/mandate-demo` (the centerpiece)** -- the mandate under test (`"up to $200/month on
  subscription services"`), then every one of the 6 scripted scenarios from the real captured
  run (`mandate-demo/output/run_20260817T030135Z.json`) as a card: the intent, the agent's
  actual proposed purchase, all three checks side by side (merchant / Defend classifier /
  mandate envelope), and the full refusal record (with every mandate-binding violation) where
  refused. The 4 "classifier said LOW_RISK, mandate refused anyway" cases are visually
  highlighted (amber border + badge), including S3 -- the $165 electronics purchase that's the
  strongest single piece of evidence in the whole submission. A collapsible panel at the bottom
  lists every one of the 19 HTTP calls the captured run made, all resolving to `127.0.0.1`.

  **"Run live now"** button triggers a real, live re-run of all 6 scenarios through the actual
  agent, server-side, right now -- see "Live-run verification" below for its actual status in
  this environment.

## Live-run verification

**Confirmed working end-to-end with a real, successful live run**, after the `OPENAI_API_KEY`
issue noted in earlier development (an invalid/rejected key, unrelated to this app's code --
see the repo's own change history) was resolved by switching to automatic `.env` loading (see
"How to run it locally" above). The test that matters: the server was started in a shell with
`OPENAI_API_KEY` explicitly unset, using only the documented `uvicorn` run command -- no manual
export, no environment variable of any kind present in that process -- and a live run through
`POST /api/mandate-demo/live-run` still returned `200 status: "ok"` in 13.7s, because
`load_dotenv()` picked up the real key from `mastercard-ai-defense-lab/.env` automatically.
`mandate-demo`'s own standalone entry point (`python -m mandate_demo.runner`, run independently,
not via this app) was verified the same way, in its own shell with the key unset first.

**Real measured cost for that run** (not an estimate): 12 OpenAI API calls (2 per scenario x 6
scenarios), 7,172 prompt tokens + 187 completion tokens = 7,359 total tokens, at `gpt-4o-mini`
pricing ($0.15/1M input, $0.60/1M output) = **$0.001188** for the full 6-scenario run. Same
`demonstrated_cases` result as the pre-captured transcript (S3, S4, S5, S6), confirming
behavioral consistency between the live path and the captured one.

What else is verified, with real requests against the running server:

- **Rate limiting** (`rate_limit.py`): confirmed via a real session-cookie-persisted request
  sequence that a live-run attempt reserves a slot *before* attempting the call (so retry
  storms are capped even when every attempt would fail) -- `session_runs_used` went `0 -> 1`
  and `global_runs_in_last_minute` went `0 -> 1` after one POST to
  `/api/mandate-demo/live-run`, both correctly scoped per-session (cookie-based) and globally
  (in-process sliding window).
- **Graceful failure** (`app.py`'s `api_live_run`): earlier in development, before the `.env`
  fix, a real `401 Incorrect API key provided` from OpenAI was caught cleanly, returned as a
  structured `502` JSON response (`status: "error"`, a plain-English `reason`, and `fallback`
  set to the real captured-run JSON) -- no crash, no bare 500, and the frontend (`app.js`'s
  `renderFallback`) is wired to show that fallback and point back at the already-rendered
  captured transcript, unaffected. The same fallback path also covers a timeout
  (`concurrent.futures.TimeoutError`, `LIVE_RUN_TIMEOUT_SECONDS=90` by default) and a
  rate-limit rejection (`429`, tested directly) -- both still real, live-tested behavior, not
  retired by the `.env` fix.
- **Model choice / cost control**: `mandate-demo/agent.py`'s existing `MODEL = "gpt-4o-mini"`
  is unmodified and reused via `from mandate_demo.agent import MODEL` (single source of truth,
  not a second hardcoded string). Checked live against
  `https://developers.openai.com/api/docs/pricing` during this build: gpt-4o-mini is
  **already the cheapest tool-calling-capable model currently offered** ($0.15/1M input,
  $0.60/1M output) -- no cheaper reliable option exists to switch to. Real measured per-run
  cost is in "Live-run verification" above.

## Scope deliberately cut, and why

- **No "which category does this resemble" classifier output for free-form submissions.**
  Defend's model is a single binary `is_fraud` detector, trained and evaluated per-category by
  slicing the test set (see `defend/README.md`'s own design rationale) -- it has no multiclass
  category output to surface honestly. For dataset rows, the real ground-truth
  `fraud_category`/`fraud_vector` label is shown alongside the score instead, clearly marked
  as the labeled ground truth, not a model prediction. Inventing a category-resemblance number
  would have violated the "never hardcode or fake" rule.
- **No screenshots captured.** The Chrome browser extension used for this session's browser
  automation tool wasn't connected in this environment (checked directly). Detailed written
  descriptions of every view are given above instead, and the server is left running at
  `http://127.0.0.1:8420` for direct viewing.
- **No authentication, database, or persistence layer.** Judges are trusted viewers of a local
  demo; every data source is a file already on disk. Adding auth/a DB would be unjustified
  complexity for what this needs to do.
- **Free-form scoring form covers 19 of 38 features**, not all of them -- the ones most likely
  to change a judge's intuition about a score. A full 38-field form would be an intimidating
  wall of inputs for a timed demo; every field not shown is genuinely sent as missing, not
  defaulted, so the model still sees a real (if sparser) row, not a fabricated one.
- **Case browser's dataset sample is 36 rows** (3 per vector + legit), not the full ~21k-row
  dataset, to keep the default page fast and legible; the free-text `transaction_id` lookup
  covers the rest.
