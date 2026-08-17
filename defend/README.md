# Defend

Gradient-boosted fraud classifier trained on Generate's synthetic dataset,
evaluated against `identify/attack-taxonomy.md`'s taxonomy. Reads
`generate/data/transactions.csv` read-only - this pillar never writes into
`generate/`.

**Status: Category B and C fixed and verified; a same-session identity-
verification regression was self-caught and fixed; A1's account-age floor
plus A2/D1/E2 have confirmed, unfixed issues.** Every leakage pattern found
so far has been traced to a root cause in `generate/vectors.py`, fixed there
(never worked around in Defend's own code), and verified via the diagnostics
described below. Category B - the taxonomy's own example of a category that
should be hard to detect from transaction data alone - now scores a genuine
~0.89 recall with real prediction-confidence spread. Category C, after an
initial fix regressed it back to a fake 1.000 unnoticed, is now confirmed
genuine at ~1.00 the same way (see `results/report.md` for why that's not
itself suspicious for this category). Generalizing A1's identity-verification
fields to the rest of the vectors (build 5, see Changelog) briefly
reintroduced a fake 1.000 across every metric when the first step (A2 only)
was retrained before the remaining 9 generators were updated - caught before
being reported, via the same diagnostics, and fixed same-session. The same
three diagnostics, run against A2/D1/E2 (and A1's remaining account-age
issue) on request, found real issues in all of them - **not yet fixed**,
pending a scope decision since two need more than a mechanical fix. See the
Changelog for the full investigation history and `results/report.md`'s
"Diagnostics run on A1, A2, D1, E2" section for the complete findings.

**Scope: 11 of 12 transaction-shaped vectors (A1-A2, B1-B3, C1-C3, D1-D2,
E2), not the full 14.** E1 (fabricated dispute evidence) is deliberately
excluded - see "Known limitations" below for why, and that it's a scope
choice, not a blocker, as of this pass.

## Why it's built this way

- **Categorical columns use pandas `category` dtype + XGBoost's native
  categorical splits (`enable_categorical=True`)**, not one-hot encoding.
  The category->code mapping is fixed once on the training data and saved to
  `model/feature_metadata.json` so the web prototype (or the held-out test
  split) can reproduce it exactly rather than re-deriving categories from
  whatever rows happen to be present.
- **Stratified split on `fraud_vector`, not on the binary label.** A plain
  stratified split on `is_fraud` can starve a 20-row vector (C1/C2/C3) down
  to 0-2 test rows by chance. Splitting on the 12-way key (11 vectors +
  "LEGIT") keeps every vector proportionally represented in both halves; see
  `split.py`.
- **Per-category metrics are computed on a category-vs-legit slice of the
  test set**, not derived from a multiclass model. Precision/recall/F1/AUC
  are only well-defined for a binary problem, so each category's numbers
  come from scoring that category's fraud rows plus every legit test row
  with the same trained model and threshold - see `evaluate.py`'s docstring
  for the full reasoning.

## Changelog

- **2026-08-16, build 1**: first working pipeline. Scored a suspicious 1.000
  precision/recall/F1/AUC on every category. Traced to
  `generate/entities.py`'s `short_id()` off-by-slice bug (see below) plus a
  second, independent issue: `credit_history_months` and
  `payee_prior_interaction_count` were populated for only the one or two
  vectors whose taxonomy text names them and left `NaN` everywhere else,
  which let the model fingerprint "which generator wrote this row" from
  missingness alone instead of learning real signal. Reported to the user
  rather than silently worked around.
- **2026-08-16, build 2**: `short_id()` fixed
  (`generate/entities.py:22`, `.hex[:10]` -> `.hex[-10:]` - the value only
  ever occupies the low 63 of 128 bits, so the leading hex digits were
  always zero) and the full dataset regenerated with real, unique IDs.
  Verified the missingness bug was a **separate** issue - identical pattern
  before and after the ID fix - then fixed it directly in
  `generate/vectors.py` (populate every vector realistically instead of
  leaving fields null; see `generate/README.md`'s anti-leakage notes for the
  full per-vector reasoning). Retrained: `credit_history_months`/
  `payee_prior_interaction_count` importance dropped to ~0.0002-0.0004 each
  (confirmed fixed), but Category B recall was *still* 1.00. Dug into why
  again: `geo_is_new`/`device_is_new` had the identical
  populated-for-some-vectors/null-for-others shape (not part of the
  originally reported finding, found while investigating why B was still
  perfect) - fixed the same way, replacing deterministic `True`/unset with a
  shared `_novelty_flags()` helper using taxonomy-grounded probability tiers
  per vector instead of a hard split.
- **2026-08-16, build 3, round 1**: retrained. Feature importance is now
  spread across many features (top feature 25%, not 79-82%), and Category
  C's recall finally moved off 1.000 (0.933) - real evidence the fixes
  above removed genuine leakage. **Category B recall is still 1.00.**
  Investigated: `amount` is now the top feature, and B1's amount range does
  not overlap AT ALL with legit wire transfers ($3,398-$65,611 vs legit's
  $3.07-$298.51) - a straightforward disjoint-threshold separator. Per an
  explicit instruction not to shrink B1/B2's fraud amounts (real wire fraud
  is genuinely large-dollar - shrinking the fraud side to look detectable
  would be dishonest), the fix was to widen Generate's legit `wire`/`ach`/
  `rtp`/`p2p` population instead: researched real-world ranges (Nacha ACH
  network ~$2,642 average in 2025; RTP network's per-transaction cap raised
  to $10M in Feb 2025 with bank-set P2P limits commonly $2.5k-$15k; median
  US home closing ~$400-430k as a proxy for wire-transfer scale) and
  widened each channel's legit lognormal parameters accordingly - see
  `generate/vectors.py`'s `_WIRE_LIKE_AMOUNT_PARAMS`. Also gave B1/B2's
  named behavioral signals (`request_urgency_flag`,
  `approval_outside_hierarchy`, `preceded_by_support_call`) real
  Bernoulli-drawn variance on every payee-channel row instead of being
  deterministic-or-null, the same fix pattern as build 2. Retrained -
  **Category B recall was still 1.00.**
- **2026-08-16, build 3, round 2**: investigated again rather than declaring
  it fixed: `device_is_new` had jumped to the top feature, and
  `merchant_registration_age_days` (plus the other `merchant_*` fields)
  were populated for legit on *every* channel including wire/rtp/p2p -
  which have a payee, not a merchant - while B1/B2/D2 correctly never set
  them. Same null-vs-populated shape as every earlier fix, on a field pair
  nobody had scoped to channel yet. Fixed by scoping legit's merchant
  sampling to non-payee channels only, and adding merchant fields to B3
  (genuinely a card-not-present purchase that had never had them). Retrained
  - **Category B recall was still 1.00.**
- **2026-08-16, build 3, round 3**: investigated again: `credential_exposure_window_flag`
  was now the single dominant feature (46%) - round 2's fix had only added
  it to legit and B3 directly, leaving it null for every other vector
  including B1/B2, reintroducing the exact same bug shape by omission.
  Generalized into a `_sample_credential_exposure()` helper (mirroring the
  existing `_novelty_flags()` pattern) applied to every generator. Retrained
  - **Category B recall: 0.78** (B1 0.87, B2 0.73, B3 0.76 individually),
  aggregate precision/recall/F1 all in the 0.93-0.95 range. Verified this is
  genuine, not a smaller leak: no feature or pair dominates (top feature
  23%, top 3 combined 50%), every vector's amount range now genuinely
  overlaps its same-channel legit population (`amount_range_overlap.csv`),
  and B1/B2/B3's predicted-fraud probabilities span a real range instead of
  clustering at >0.999 confidence (`prediction_confidence_spread.csv`,
  25th-percentile confidence tops out at 0.953). Category B is genuinely
  fixed as of this build - see `results/report.md`'s investigation section
  for the full table and evidence.
- **2026-08-16, build 4**: asked to check whether build 3's
  `credential_exposure_window_flag` fix (which touched every generator, not
  just B) had incidentally affected Category C - it hadn't (the field
  doesn't even appear in a C-vs-legit model's top 15 features), but running
  the same three diagnostics scoped to C surfaced a real, **pre-existing**
  bug unrelated to any of this session's changes: `generate_c1_agent_impersonation`/
  `generate_c2_malicious_storefront` set `mandate_id` from the agent they
  use but never propagated `mandate_amount_cap`/`mandate_category_scope`/
  `mandate_recurring_flag`/envelope-compliance - null for C1/C2, populated
  for C3/legit, the same missingness shape as everything above. Category C
  had drifted from an unverified 0.933 (build 2) to a fake 1.000 (build 3)
  without anyone noticing, since only B was under scrutiny. Fixed by
  deriving envelope compliance from each vector's own already-sampled
  amount/merchant-category against the real agent's mandate (C1/C2), and
  giving C3 a well-matched `agent_behavior_pattern_match_score` (C3's
  taxonomy signal is scope abuse, not identity mismatch, unlike C1).
  Retrained - **Category C recall: 0.933** (C1 1.0, C2 0.8, C3 1.0),
  confirmed genuine via the same three diagnostics post-fix (both fields
  100% populated everywhere now; `channel` dropped out of the C-vs-legit
  model's top 10 entirely; C2's prediction confidence now spans 0.28-0.989).
  C1/C3 remain highly confident, but that traces to their own literal
  taxonomy-named signals (failed attestation; deterministic envelope
  breach), not to missingness - taxonomy doesn't predict C should be hard,
  unlike B. **Aggregate: precision 0.941, recall 0.914, F1 0.928 - this is
  the current final result.** Also ran the same three diagnostics against
  A1, A2, D1, E2 on request (not fixed, per instruction to report only):
  all four show the identical clustered-near-1.000-confidence pattern that
  turned out to be real bugs in B and C, and indeed all four have at least
  one confirmed real issue - a `generate/entities.py` account-age floor
  (`rng.integers(30, 4000)`) that's completely disjoint from these vectors'
  brand-new/throwaway account ages, plus vector-specific missingness (A2's
  session/liveness fields, D1's burst-sequence fields, E2's `promo_code`
  never set on legit's own promo rows). The account-age floor is a
  mechanical fix (same pattern as everything above); A2's and D1's
  vector-specific fields are not - they're null because Generate's schema
  has no legit onboarding-event or legit transaction-burst row to attach
  them to, not because of a copyable omission, so fixing those needs a
  design decision first. See `results/report.md`'s "Diagnostics run on A1,
  A2, D1, E2" section for the full per-vector table.
- **2026-08-16, build 5**: acted on build 4's own "Known limitations" note
  that A1's `claimed_employment_tenure_months`/`identity_cross_source_correlation`
  "plausibly apply to legit accounts too... and could likely be added
  mechanically." First step: added them to A2 only, via a new
  `_sample_identity_verification()` helper, and retrained before touching the
  other 9 generators - to check the pattern in isolation before generalizing
  it. That retrain scored a suspicious **1.000 precision/recall/F1/AUC across
  every category and vector again** - the identical fake-perfect signature as
  builds 1 and 3. Same root cause as every fix in this changelog: legit, A1,
  and now A2 had real values for the two fields; every other fraud vector
  (B1, B2, B3, C1, C2, C3, D1, D2, E2) still read `NaN`, and
  `identity_cross_source_correlation` alone carried 53.1% of feature
  importance with `claimed_employment_tenure_months` carrying another 15.9%
  (69.0% combined - on par with build 1's original two-field leak). Caught
  before being reported as a result, using the same diagnostics as every
  build above, not trusted at face value. Archived under
  `defend/results_regression_snapshot/` before being overwritten. Fixed by
  applying the identical `_sample_identity_verification(rng,
  _IDENTITY_VERIFICATION_NORMAL)` call to the remaining 9 generators in
  `generate/vectors.py`. Regenerated the dataset and retrained: both fields
  dropped to 0.2% and 0.1% importance respectively (noise level, not merely
  smaller), and **aggregate metrics settled at precision 0.944, recall
  0.966, F1 0.955** - Category B recall 0.891 (up slightly from build 4's
  0.782, consistent with B's amount/behavioral-field distributions being
  unrelated to this fix), Category C recall 1.000 (unaffected, as expected -
  this fix never touched any of C's own fields). See `results/report.md`'s
  "Investigation: identity-verification field regression (self-caught, same
  session)" section for the full before/after evidence table.

## Known limitations

- **E1 (GenAI Fabricated Refund and Chargeback Evidence) is still not
  trained or evaluated this pass**, even though the ID bug that originally
  blocked it is fixed (see Changelog build 2). Kept out deliberately so
  every run's metrics stay comparable to the build-1 baseline in
  `results_before_fix/` across this whole investigation; re-adding the
  `transactions.csv` + `disputes.csv` join is a clean next step whenever
  that comparison is no longer needed (the join design itself - left-join
  on transaction id, override the transaction's label only where the linked
  dispute is fraudulent - was always sound; it was blocked by the ID bug,
  not by a design flaw).
- **A1, A2, D1, E2 have confirmed, unfixed leaks - checked but not yet
  acted on.** Category C's near-miss (a real bug hiding behind a
  plausible-looking recall number) prompted checking the other categories
  that still show 1.000/near-1.000 recall with the same three diagnostics.
  All four have at least one real issue. (A1's identity-verification field
  gap - listed as an open item as of build 4 - was fixed in build 5, see
  Changelog; its account-age floor below remains open, same as the other
  three.)
  - **Mechanical, shared by all four**: `generate/entities.py`'s account
    pool hardcodes `account_age_days=int(rng.integers(30, 4000))` for
    every legit account - a hard floor with zero overlap against A1
    (0-13 days), A2 (0-2), D1 (always 0), and E2 (0-1), all of which
    represent brand-new/throwaway identities. `historical_avg_amount` is
    correspondingly always exactly 0.0 for these four vs. legit's $4.24
    minimum - also fully disjoint. Real accounts get used within days of
    opening constantly, so this floor is itself the unrealistic
    assumption, the same class of problem as B1's original wire-amount
    ceiling. E2 additionally never gets `promo_code` populated on legit's
    own `promo_redemption` rows (0% of 1,007 legit promo rows).
  - **Architectural, A2 and D1 only**: A2's `session_frame_rate_anomaly`/
    `session_video_artifact_score`/`onboarding_velocity_seconds` and D1's
    `attempt_sequence_id`/`attempts_in_window`/`window_seconds` are null
    for legit not because of a copyable omission but because Generate has
    no legit onboarding-event or legit transaction-burst row to attach
    them to at all. Fixing these means designing and building an
    analogous legit sub-population, not a one-line field addition - the
    same category of judgment call that the B1/B2 amount-distribution fix
    required stopping and asking about, rather than being improvised.

  A1's `claimed_employment_tenure_months`/`identity_cross_source_correlation`
  gap (previously listed here as a plausibly-mechanical fix) was closed in
  build 5 - see Changelog. A1's remaining open issue is the same
  account-age floor shared by A2/D1/E2 above, not a separate identity-field
  gap.

  **Account-age floor and A2/D1's architectural fields not fixed as of
  this pass** - see `results/report.md`'s "Diagnostics
  run on A1, A2, D1, E2" section for the full per-vector table and
  reasoning. The tools that found this are reusable:
  `missingness_profile()`, `amount_range_overlap()`, and
  `prediction_confidence_spread()` in `diagnostics.py` run on every
  training run and write their evidence to `results/` regardless of which
  category is being checked - use them the same way for any future vector
  under suspicion.
- **No cross-transaction graph** (inherited from Generate's own documented
  non-goal): per-row summary stats (`prior_transaction_count`,
  `historical_avg_amount`) are used as-is; no sequence/graph modeling.

## Anti-leakage

Declared once in `config.py` rather than inferred, so the exclusion list is
auditable:

- **ID columns** (`transaction_id`, `account_id`, `merchant_id`, `agent_id`,
  `mandate_id`, ...) - unique identifiers, no generalizable signal, and a
  tree would happily overfit to them if given the chance. These are real,
  unique values as of the Changelog's build 2 fix (previously collapsed to
  a constant string by a Generate-side bug); the exclusion rule was already
  in place before that fix and stays in place regardless.
- **`narrative_tag`** - free text that literally names the fraud vector
  (e.g. `"synthetic identity: thin file, mature first-transaction spend..."`).
  The single most dangerous column in this dataset if left in the feature
  matrix; excluded outright.
- **`fraud_category` / `fraud_vector`** - label-adjacent metadata, not
  observable before the label itself. Used only for stratification and
  per-category evaluation slicing, never as a model input.
- **Raw `timestamp`** - dropped in favor of derived `txn_hour`, the only
  part of a timestamp that generalizes past this dataset's specific date
  window.

This list was written independently of Generate's own anti-leakage notes
(agentic-checkout metadata on legit rows, C3's forced real envelope breach)
but is consistent with them - those fixes are what make it *safe* for this
pillar to use `agent_*`/`mandate_*` fields as real signal instead of a
leakage risk in disguise.

## Layout

```
defend/
  src/defend/
    config.py     paths, seed, column-role declarations (ID/leakage/target)
    data.py        loads transactions.csv (see module docstring re: E1 gap)
    features.py     derives txn_hour, builds X/y + category encoding
    split.py          stratified train/test split on fraud_vector
    model.py           XGBoost training + feature importances
    evaluate.py          aggregate + per-category + per-vector metrics
    diagnostics.py         missingness/amount-overlap/confidence-spread evidence tools
    report.py             renders results/ (json/csv/markdown)
    persist.py              saves/loads model + feature metadata for reuse
    train.py                 entrypoint: python -m defend.train
  tests/test_pipeline.py   leakage/split/train smoke tests on real data
  model/                    saved model + feature_metadata.json (for the web prototype)
  results/                  metrics + report.md from the last training run
  results_before_fix/       build-1 (leaked) baseline, kept for comparison
  results_regression_snapshot/  build 4.5 (A2-only identity-verification fix,
                                 briefly leaked again), kept for comparison
```

## Running it

```bash
cd defend
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m defend.train
```

## Results

See `results/report.md` for the current (build 5) breakdown: aggregate
metrics, the Category B investigation (full build 1->5 comparison table
plus the three-round investigation within build 3), the Category C
investigation (how a fix aimed at B incidentally regressed C, and how that
was caught and fixed), the identity-verification field regression
investigation (how generalizing A1's fields to A2 alone briefly reintroduced
a fake 1.000 across the board, and how that was self-caught and fixed
same-session), the "Diagnostics run on A1, A2, D1, E2" section (checked on
request, confirmed real issues; A1's identity-field portion is now resolved,
its account-age floor remains open alongside A2/D1/E2), per-category A-E
table, per-vector recall, worst-performing vectors for a future Generate
feedback loop, and the evidence tables (`amount_range_overlap.csv`,
`prediction_confidence_spread.csv`, `missingness_profile.csv`) that back
every "genuine, not declared" claim above. `results_before_fix/` holds the
original build-1 (leaked) results, and `results_regression_snapshot/` holds
the build 4.5 (A2-only, briefly re-leaked) results, both for direct
before/after comparison.
