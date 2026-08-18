# Mastercard AI Defense Lab

Mastercard Innovation Challenge submission. A closed loop system for GenAI era payment fraud:
a taxonomy of the actual attack vectors (Identify), a synthetic labeled dataset built from that
taxonomy (Generate), a trained classifier evaluated against it with the leaks found and fixed
in the open (Defend), and a second, structurally different layer of protection for the one
category where a classifier alone genuinely is not enough (the Mandate Demo). A working web
prototype ties all four together and is deployed live for direct use, not just described here.

## Live demo

**https://mastercard-ai-defense-lab.fly.dev/**

Open it and go straight to **Mandate Demo** in the top navigation. That page is the strongest
single piece of evidence in this submission: a real captured run showing four cases where
Defend's actual trained classifier scored a mandate violation LOW RISK and an independent,
deterministic check refused it anyway. Look for:

* The rate limit indicator near the top ("X of Y runs left this session, max Z per minute
  globally"), proving the live agent trigger is capped server side, not just documented as
  capped.
* The **Run live now** button. Clicking it makes a real OpenAI call, right now, against the
  live deployment, and returns a fresh result in ten to fifteen seconds, not a replay.
* The four highlighted scenario cards (amber border, "CLASSIFIER MISSED IT, MANDATE CAUGHT IT"
  badge), especially the one hundred sixty five dollar electronics purchase: comfortably
  inside the mandate's dollar cap, from a profile that looks completely established, wrong only
  on merchant category. Defend's classifier scores it well under one percent fraud probability.
  The mandate check refuses it outright.

Also worth a look: **Case Browser** (score any transaction from Generate's real dataset, or
your own free form input, against Defend's actual trained model live) and **Dashboard**
(Defend's real, non fabricated precision, recall, and F1 by category and by vector).

## Why this approach

Most fraud detection submissions stop at a classifier: label the data, train a model, report
recall. This submission does that too, honestly, including the categories where the model is
genuinely weak. But it also adds something most submissions will not: a second layer that is
structurally different in kind, not just a second model, for the one taxonomy category
(C3, delegated mandate scope abuse) where a carefully crafted violation can look statistically
normal to any classifier trained on transaction features alone.

The mandate demo proves this is not a hypothetical. A transaction that is within the dollar
cap, in a merchant category the classifier has little reason to flag, from an account profile
that looks established rather than freshly opened, scores well under one percent fraud
probability from Defend's real trained model. It is still a clear violation of what the
customer's mandate actually authorized. A classifier trained on statistical patterns cannot
structurally see that; a deterministic check that compares the proposed action against the
authoritative mandate record can, every time, with no training data and no threshold to tune.
One of the four demonstrated cases (the monthly cumulative cap breach) is not just empirically
missed by the classifier, it is structurally invisible to it: no feature in Defend's entire
thirty eight column schema encodes running spend under a mandate at all, regardless of how well
the model is trained.

This is the honest differentiator: not a bigger model, a different kind of check, added exactly
where the taxonomy itself says a transaction level classifier should struggle.

## Architecture

The four pillars form a single pipeline, each one consuming the last:

* **Identify** grounds everything. `identify/attack-taxonomy.md` defines every attack vector,
  its mechanism, its channel, and the transaction level signal it should leave behind. Generate
  and the mandate demo's C3 scenario both trace directly back to this document.
* **Generate** reads Identify's taxonomy as its specification and produces one synthetic
  generator per vector plus a legitimate baseline, all sharing a common schema so the two
  populations are genuinely comparable, not two different shapes of data.
* **Defend** reads Generate's output only, trains a single binary classifier, and evaluates it
  broken out by category and by vector so a strong aggregate number cannot hide a weak
  category.
* **Mandate Demo** reads Defend's actual saved model file directly (not a copy, not a
  reimplementation) to score proposed transactions, and adds a second, independent,
  deterministic mandate check alongside it for category C3.
* **Web Prototype** is the only piece that touches all four: it reads Identify's taxonomy file,
  Generate's dataset, Defend's saved model and results, and calls Mandate Demo's real code,
  live, to render the whole system as one browsable, clickable application.

## A note on a discrepancy found while writing this document

`identify/attack-taxonomy.md`'s own introduction states "fourteen attack vectors," and
`generate/README.md`'s opening line repeats "every one of the 14 attack vectors." The taxonomy
file as actually written defines twelve: A1, A2, B1, B2, B3, C1, C2, C3, D1, D2, E1, E2, across
five categories. This document uses twelve throughout, since that is what the source file
actually contains, and flags the mismatch here rather than silently choosing a number. Neither
`identify/` nor `generate/` was modified to write this README; a judge counting entries in the
taxonomy file will get twelve, not fourteen.

## Identify

**What it does.** `identify/attack-taxonomy.md` is a single markdown document defining twelve
GenAI era payment fraud attack vectors across five categories (A: Identity and Onboarding
Fraud, B: Social Engineering and Authorization Fraud, C: Agentic Commerce Fraud, D: Automated
Machine Speed Attacks, E: Post Transaction and Loyalty Fraud). Each vector states its mechanism,
the channel it targets, the transaction level signal a detector should learn from, and a
plausibility and severity rating grounded in a documented real world source. This document is
the shared specification the other pillars build against, not a summary written after the fact.

**How to use it.** No code, just read `identify/attack-taxonomy.md` directly, or browse it
through the web prototype's Taxonomy page for a formatted, collapsible view of every vector.

**Status.** Complete and stable; nothing in this pillar changed after the earlier build rounds
that shaped Generate and Defend.

## Generate

**What it does.** Produces a synthetic, labeled transaction dataset: one generator per taxonomy
vector plus a legitimate baseline, sharing a single schema so fraud and legitimate rows are
genuinely comparable rather than structurally distinguishable by construction. Twelve vectors
are point of sale adjacent and share a `TransactionRecord` shape; E1 (fabricated dispute
evidence) is post transaction and lives in a linked `DisputeRecord` table instead.

**How to run it standalone.**

```bash
cd generate
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m generate.cli --format csv
```

Output: `data/transactions.csv`, `data/disputes.csv`, `data/generation_summary.csv`.

**Status, real and current.** Twenty one thousand nine hundred sixty total rows, seven hundred
fraud rows, across the eleven transaction shaped vectors Defend currently trains on plus the
legitimate baseline. Multiple rounds of leakage were found and fixed in the open, each one the
same underlying shape: a field populated for the fraud vector that names it in the taxonomy text
and left null everywhere else, which let a classifier learn "which generator wrote this row"
instead of real signal. Every instance found is now fixed and guarded by a dedicated test in
`tests/test_dataset.py`. Two items remain open and are reported, not hidden: an account age
floor in the shared entity pool that is fully disjoint from four vectors representing brand new
or throwaway identities (a mechanical fix, not yet made), and two vectors (A2, D1) whose
liveness and burst detection fields have no legitimate counterpart row to attach to at all,
which needs a design decision before it can be a one line fix. See `generate/README.md`'s
"Known but not yet fixed" section for the full evidence behind both.

## Defend

**What it does.** An XGBoost gradient boosted classifier, binary `is_fraud` target, trained on
Generate's output only, with results broken out by category and by vector so the aggregate
number cannot hide a weak spot. Categorical columns use native category splits, not one hot
encoding; the train and test split is stratified on the twelve way vector key, not the binary
label, so even a twenty row vector stays represented in both halves.

**How to run it standalone.**

```bash
cd defend
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m defend.train
```

**Status, real and current, from `defend/results/report.md`.** On the held out test set (five
thousand four hundred ninety rows, one hundred seventy five actual fraud, decision threshold
zero point five):

| Metric | Value |
|---|---|
| Precision | 0.944 |
| Recall | 0.966 |
| F1 | 0.955 |
| ROC AUC | 0.999 |
| PR AUC | 0.985 |

By category:

| Category | Recall | Note |
|---|---|---|
| A, Identity and Onboarding | 1.000 | |
| B, Social Engineering and Authorization | 0.891 | genuinely hard by taxonomy design, see below |
| C, Agentic Commerce | 1.000 | on Defend's own held out split, which includes the mandate compliance field; see the Mandate Demo section for the harder case built specifically without it |
| D, Automated Machine Speed | 1.000 | |
| E, this is E2 only, not E1 plus E2 | 1.000 | E1 excluded, see below |

Category B's zero point eight nine one recall is an honest, verified result, not a residual
leak and not a shortfall to explain away. The taxonomy itself predicts this category should be
hard: B2 in particular is genuinely authorized by the real accountholder, so there is no failed
authorization signal for a transaction level classifier to key on. This was checked, not
assumed: three separate diagnostics (feature dominance, amount range overlap against the same
channel legitimate population, and prediction confidence spread) confirmed no single feature
dominates, every vector's amount range genuinely overlaps its legitimate counterpart, and
predicted fraud probability spans a real range rather than clustering near certainty.

Category E here reflects E2 (promotion and coupon abuse) only. E1 (fabricated dispute evidence)
is excluded from this evaluation pass, not because of a current blocker (the identifier
generation bug that once prevented the required join has been fixed and verified) but as a
deliberate choice to keep this run's metrics comparable to the earlier baseline. Re adding the
join is a clean next step, not attempted in this pass.

Per vector recall, eleven vectors:

| Vector | Recall |
|---|---|
| A1 | 1.000 |
| A2 | 1.000 |
| B1 | 0.867 |
| B2 | 0.800 |
| B3 | 0.960 |
| C1 | 1.000 |
| C2 | 1.000 |
| C3 | 1.000 |
| D1 | 1.000 |
| D2 | 1.000 |
| E2 | 1.000 |

Every number above traces to a real held out test set evaluation, not a training set number and
not a hand picked example. The full investigation, including how a fake one point zero zero
zero result was caught and fixed on two separate occasions during this build, is in
`defend/results/report.md`.

## Mandate Demo

**What it does.** A local, honestly labeled reimplementation of the architectural pattern used
in a production mandate binding and execution authorization system: independent verification of
a proposed action against the authoritative record of what was actually authorized, rather than
trusting the acting agent's own judgment. This is explicitly not any real third party SDK and
never calls any live third party infrastructure; every merchant, product, and mandate involved
is synthetic, invented for this demo. Built for taxonomy category C3, delegated mandate scope
abuse: a well authorized agent executes a transaction that technically falls within a broad
mandate but violates what was actually intended.

An OpenAI powered purchasing agent, operating under a fixed mandate ("up to two hundred dollars
per month on subscription services"), attempts six scripted purchase intents against a local
synthetic merchant API. Every proposed purchase runs through three independent checks: does the
merchant itself accept it, what does Defend's real trained classifier score it, and does a
purely deterministic check find it inside the mandate's actual terms. When the mandate check
refuses, a structured refusal record is produced, shaped to mirror a real refusal record's
field structure: what was requested, what the mandate actually authorized, and why it was
refused.

**How to run it standalone.**

```bash
cd mandate-demo
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m mandate_demo.runner
```

Needs `OPENAI_API_KEY`. Copy `.env.example` at the repository root to `.env` and fill in a real
key there; both this pillar and the web prototype load it automatically at startup, no manual
export step required, verified in a shell with the variable explicitly unset before running.

**Status, real and current.** All six scripted scenarios were run live, not simulated; the
scripted example fallback described in the original task specification was not needed. Of six
scenarios, two are correctly allowed and four are correctly refused by the mandate check, and
in all four of those refused cases Defend's real classifier independently scored the same
proposed transaction as low risk. One of the four (a monthly cumulative cap breach across two
prior approved purchases) cannot be caught by the classifier under any amount of training, since
no feature in Defend's schema encodes running spend under a mandate at all. The mechanism
behind the other three was diagnosed directly, not asserted: the identical proposed transaction
scores well under one percent fraud probability when scored the honest way (mandate compliance
withheld, established account profile) and ninety nine point six seven percent when scored
against Defend's own training distribution for this category (a freshly opened account, the
shape every training example for this vector actually uses). Full transcript and diagnostic
numbers are in `mandate-demo/README.md` and the real captured run at
`mandate-demo/output/run_20260817T030135Z.json`.

## Web Prototype

**What it does.** The working web based prototype the challenge requires: a FastAPI backend and
a server rendered frontend, with no separate build step, sitting on top of all four other
pillars as a read only consumer. It reuses Defend's actual saved model and Mandate Demo's actual
code (installed as a local editable package, not copied) rather than reimplementing either.
Five views: an overview, the full taxonomy browser, Defend's results dashboard, a transaction
case browser that scores real dataset rows or free form input live against Defend's real model,
and the Mandate Demo centerpiece with its live trigger.

**How to run it standalone.**

```bash
cd web-prototype
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pip install -e ../mandate-demo
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m uvicorn web_prototype.app:app --app-dir src --host 127.0.0.1 --port 8420
```

Then open `http://127.0.0.1:8420`. The same `OPENAI_API_KEY` setup as Mandate Demo above applies
here for the live run trigger; every other page works without it.

**Status, real and current.** Deployed live at `mastercard-ai-defense-lab.fly.dev`, region iad,
built from `mastercard/master` at commit `7fcdbc2`. Verified end to end against the live public
URL, not just a local run and not just a successful deploy exit code: the homepage and the
Mandate Demo page render correctly, a real click on Run Live Now against the deployed instance
returned a genuine result in about ten seconds (twelve OpenAI API calls, seven thousand three
hundred fifty nine tokens, one tenth of one cent), and the rate limit indicator correctly
reflected the live server's own state after that run. `fly logs` after that test run show the
API key in no log line and no errors beyond two expected, self resolving health check messages
during each machine's first several seconds of startup.

## What is real, and what is a clearly labeled scope cut

Real, verified, traceable to a file or a live request in this repository:

* Every metric in the Defend section above: read directly from `defend/results/report.md`,
  computed on a genuine held out test split, cross checked with dedicated leakage diagnostics
  across multiple investigation rounds rather than accepted at face value the first time a
  number looked good.
* The mandate demo's four demonstrated cases: a real captured run, real OpenAI tool calls, no
  fallback or scripted substitute code path exists anywhere in that pillar to have produced it
  another way.
* The web prototype's live deployment result quoted above: an actual HTTP request against the
  actual public URL, inspected directly, not assumed from a deploy log.
* Every number the web prototype displays: read from a file already in this repository or
  computed by calling Defend's or Mandate Demo's real code at request time. Confirmed by
  grepping the entire served frontend for hardcoded figures and for the API key; neither
  appears.

Deliberate scope cuts, stated plainly rather than left implicit:

* Category E1 (fabricated dispute evidence) is excluded from Defend's current evaluation. The
  bug that once blocked it is fixed; it is out for build comparability, a clean next step, not
  a current limitation of the approach.
* Two items in Generate remain open: an unrealistic account age floor affecting four vectors,
  and two vectors whose fields have no legitimate counterpart row to attach to, needing a design
  decision rather than a mechanical fix.
* The web prototype's free form scoring form exposes nineteen of Defend's thirty eight features,
  chosen as the most decision relevant; every field left blank is sent to the model as genuinely
  missing, not filled with an invented default.
* The web prototype does not offer a "which category does this resemble" output for free form
  input. Defend's model is a single binary detector, evaluated per category by slicing the test
  set, not a multiclass classifier; inventing a category resemblance number would not have been
  honest.
* The case browser's default table shows a thirty six row sample, not the full dataset; a free
  text transaction identifier lookup covers the rest.
* No authentication, database, or persistence layer in the web prototype. Every data source is a
  file already on disk; judges are trusted viewers of a local or deployed demo, not a
  multi tenant service.

## Configuration

Environment variables this repository reads, and what each one is for. No actual key values
appear anywhere in this document; see `.env.example` at the repository root for the local setup
template.

**Required for local development**

* `OPENAI_API_KEY` — used by `mandate-demo/src/mandate_demo/runner.py` (the live purchasing
  agent) and `web-prototype/src/web_prototype/live_agent.py` (the **Run live now** button on
  `/mandate-demo`). Only required for that one feature: if it is unset, the app falls back to a
  pre-captured transcript for that demo. Every other feature (dashboard, taxonomy, cases, dataset
  scoring) works without it.

  Copy `.env.example` to `.env` and fill in a real key there; both this pillar and the web
  prototype load it automatically at startup, no manual export step required.

**In production (fly.dev)**

`OPENAI_API_KEY` is not read from `.env` in the deployed image at all. `.dockerignore`
deliberately excludes `.env` from the build context, so the key is set directly through Fly's own
secrets manager instead:

```bash
fly secrets set OPENAI_API_KEY=<value> -a mastercard-ai-defense-lab
```

**Optional tuning variables**

Not secrets, and safe to leave unset: each one falls back to the default shown below if it is not
present in the environment.

* `MANDATE_DEMO_LIVE_RUN_TIMEOUT_SECONDS` (default: 90) — how long the live agent run on
  `/mandate-demo` is allowed to take before the web prototype times it out.
* `MANDATE_DEMO_MAX_RUNS_PER_SESSION` (default: 5) — live run cap per browser session.
* `MANDATE_DEMO_MAX_RUNS_PER_MINUTE` (default: 3) — live run cap across all sessions, globally,
  per minute.

## How to run everything locally

Requires Python three point eleven or newer. Commands below are written for Windows
(`Scripts/`); substitute `bin/` on macOS or Linux. Each pillar has its own virtual environment
and its own test suite; none require each other except Mandate Demo and the Web Prototype, which
need Defend's model already trained.

```bash
# 1. Identify: nothing to run, read identify/attack-taxonomy.md directly

# 2. Generate: produces the dataset Defend trains on
cd generate
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m generate.cli --format csv
cd ..

# 3. Defend: trains and evaluates the classifier against Generate's output
cd defend
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m defend.train
cd ..

# 4. Set up the API key both remaining pillars need for their live agent
cp .env.example .env
# then edit .env and fill in a real OPENAI_API_KEY

# 5. Mandate Demo: standalone live run
cd mandate-demo
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m mandate_demo.runner
cd ..

# 6. Web Prototype: the full UI over all four pillars
cd web-prototype
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pip install -e ../mandate-demo
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m uvicorn web_prototype.app:app --app-dir src --host 127.0.0.1 --port 8420
```

Or skip all of that and use the live deployment: **https://mastercard-ai-defense-lab.fly.dev/**
