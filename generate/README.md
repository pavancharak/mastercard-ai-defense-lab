# Generate

Synthetic dataset for the Mastercard Innovation Challenge AI Defense Lab. This
pillar takes `identify/attack-taxonomy.md` as its spec: every one of the 14
attack vectors (A1-A2, B1-B3, C1-C3, D1-D2, E1-E2) gets a dedicated generator
that samples the *"Transaction signal"* the taxonomy describes for that
vector, plus a legitimate baseline population for contrast. Defend trains and
evaluates against this output.

## Why it's built this way

- **One row-shape per vector's actual data shape, not per category.** Twelve
  vectors (A-D) are point-of-sale-adjacent and share `TransactionRecord`.
  Category E is post-transaction: `E1` (fabricated dispute evidence) needs
  transaction *and* dispute metadata, so it lives in a linked `DisputeRecord`
  table instead of being forced into the transaction schema. This follows the
  taxonomy's own "Notes for the Generate and Defend pillars" section directly.
- **Fraud rows are distribution shifts, not flags.** Every fraud generator
  samples the *same* fields a legitimate row has, just from parameters shifted
  toward the anomaly the taxonomy names (e.g. A1's `credit_history_months`
  near zero combined with a high first-transaction `amount`). There is no
  hidden label field a model could shortcut on - see "Anti-leakage" below.
- **A1 (identity fraud) is simulated at its downstream payment signature**,
  not at account-opening, because that's what a payment-level detector could
  actually observe. This mirrors the taxonomy's explicit guidance that
  Category A "sits mostly upstream of the transaction level."
- **B2 (deepfake video APP fraud) sets `authorization_result="approved"`
  deliberately** - the taxonomy calls this out as the hardest category to
  catch on transaction data alone, since the payment is genuinely authorized
  by the real accountholder. The signal is payee novelty + call proximity,
  not a failed auth check. Don't "fix" this into a detectable auth failure;
  that would misrepresent the vector.
- **Per-vector volumes are derived from the taxonomy's Plausibility rating**
  (Very high=5x, High=3x, Emerging=1x unit weight, see `config.py`), so the
  dataset's class balance is a traceable function of the taxonomy document,
  not an arbitrary choice made in code.

## Anti-leakage notes

A couple of fields are easy to get wrong in a way that hands a classifier a
free shortcut instead of the real signal. Both are handled deliberately here:

- **Agentic checkout metadata.** `agent_id` / `mandate_*` fields are populated
  for legitimate `agentic_checkout` transactions too (with attestation
  verified and the transaction inside its mandate envelope), not just for
  C1-C3. Otherwise "agent fields are non-null" would trivially mean fraud.
- **C3 mandate-scope abuse always breaches something real.** Every C3 row
  either exceeds `mandate_amount_cap` or has a `merchant_category` different
  from `mandate_category_scope` (checked in
  `tests/test_dataset.py`-adjacent logic in `vectors.py`); it's never labeled
  fraud while both dimensions are actually in-scope.
- **`short_id()` produces real, unique IDs** (fixed 2026-08-16). It used to
  slice the *leading* 10 hex characters of a UUID built from a value that
  only ever occupied the low 63 of 128 bits - always-zero digits - which
  collapsed every ID (`transaction_id`, `account_id`, everything) to the same
  constant string across the whole dataset. Found downstream in Defend: a
  join on the constant `transaction_id` key silently became a Cartesian
  product. Fixed by slicing the *trailing* 10 characters instead, where the
  actual randomness lives. `tests/test_dataset.py::test_ids_unique_per_row`
  guards this.
- **`credit_history_months` and `payee_prior_interaction_count` carry real
  signal, not a "which generator wrote this row" fingerprint** (fixed
  2026-08-16). These used to be populated for only the one or two vectors
  whose taxonomy text mentions them and left `NaN` for every other vector
  (including legit, for `credit_history_months`), which let a classifier
  learn missingness itself as a near-perfect fraud/vector tell - this is
  what caused Defend's first training run to score a suspicious 1.000 on
  every category, including Category B, which the taxonomy explicitly says
  should be hard. Fixed by: (a) populating `credit_history_months` for every
  vector - pulled from the same pooled `Account` entity as legit rows for
  vectors that use an established account (B1-C3, D2, E1/legit-dispute
  purchases), independently sampled with a deliberate-but-overlapping low
  skew for vectors whose real signal *is* thin/no credit file (A1, A2, E2),
  and drawn from the same population range as any real account for D1
  (a stolen card belongs to some real, unknown cardholder - there's no
  reason its credit history should differ from the general population); and
  (b) making `payee_prior_interaction_count`/`is_first_time_payee`
  presence depend on the transaction's `channel` (does this transaction
  even have a payee - wire/ach/rtp/p2p) instead of on which generator wrote
  the row, with legit and fraud (B1/B2/D2) drawing from the same {0} u
  uniform[1,50) support and only the *mixture weight* toward "first time"
  differing - never a hard 0-vs-populated split.
  `tests/test_dataset.py::test_credit_history_months_populated_for_every_vector`
  and `::test_payee_fields_driven_by_channel_not_by_fraud_label` guard this.
- **B1/B2's legit-population comparison is now realistic, not artificially
  narrow** (fixed 2026-08-16). `wire`/`ach`/`rtp`/`p2p` legit rows used to
  share the same ~$3-$300 retail-purchase amount distribution as a
  card-not-present coffee purchase, so B1's fraud amounts (a real BEC wire,
  correctly left untouched - FBI IC3 puts the average BEC loss around
  $122k-$137k) fell in a completely disjoint range and were trivially
  separable on `amount` alone. Fixed by widening each channel's legit
  distribution to its real-world population instead of shrinking the fraud
  side: `wire` mean=$3.6k median with a tail into the hundreds of thousands
  (real estate closings, business payments), `ach` centered near Nacha's
  reported ~$2,642 average, `rtp`/`p2p` with a P2P-typical few-hundred-dollar
  median and a tail into the low thousands matching real bank-set transfer
  limits. See `_WIRE_LIKE_AMOUNT_PARAMS`'s comment in `vectors.py` for full
  sourcing. The fraud side was never touched - shrinking it to make
  detection easier would have been dishonest fidelity, not a fix.
- **Merchant fields only apply to merchant-relevant channels** (fixed
  2026-08-16). Legit used to sample and populate `merchant_id`/
  `merchant_category`/`merchant_registration_age_days`/
  `merchant_reputation_score` on every channel, including `wire`/`ach`/
  `rtp`/`p2p` - which have a payee, not a merchant. B1/B2/D2 (correctly)
  never set these fields on those channels, so the null-vs-populated split
  became a clean legit-vs-B1/B2/D2 tell once other leaks were fixed. Fixed
  by scoping legit's merchant sampling to non-payee channels (mirroring how
  payee fields are already scoped to payee channels); B3, which is
  genuinely a card-not-present purchase, had never had merchant fields set
  at all and got them added.
- **Every behavioral-signal field taxonomy names for B1/B2/B3 now has real,
  non-deterministic variance on every applicable row, not just the one
  vector it names** (fixed 2026-08-16): `request_urgency_flag`/
  `approval_outside_hierarchy` (B1), `preceded_by_support_call`/
  `time_since_support_call_minutes` (B2), `credential_exposure_window_flag`
  (B3). Each used to be deterministic-or-unset for exactly one vector and
  null for every other row including legit - legitimate urgent wires,
  occasional non-standard approvals, pre-transfer verification calls, and
  false-positive credential-breach flags all happen in the real world too,
  so every payee-channel row (legit and fraud alike) now draws these from a
  real Bernoulli distribution, with only the *bias* elevated for the vector
  the taxonomy names it as a signal for. `transaction_hour_local` (B1) was
  removed outright rather than fixed - it duplicated Defend's own derived
  `txn_hour` feature (parsed from every row's timestamp) while only ever
  being populated for B1, making it a pure redundant leak with no
  independent signal to preserve.

This took three rounds to fully resolve - fixing the first instance of this
disease (`credit_history_months`/`payee_prior_interaction_count`) revealed a
second (`geo_is_new`/`device_is_new`), and fixing the amount distributions
above revealed a third (merchant fields, then `credential_exposure_window_flag`).
Each was found the same way: an unexpectedly perfect Defend result was
treated as a bug report, not a win, until `defend/results/report.md`'s
prediction-confidence and amount-overlap diagnostics showed genuine,
non-trivial separability. See `defend/README.md`'s changelog for the full
investigation history.

- **C1/C2 now propagate the real mandate, not just `mandate_id`** (fixed
  2026-08-16). Both generators sampled a real `Agent` (with real
  `mandate_amount_cap`/`mandate_category_scope`/`mandate_recurring_flag`
  attributes) and set `mandate_id` from it, but never propagated the rest -
  `transaction_within_mandate_envelope` stayed null for 100% of C1/C2 while
  legit and C3 always populated it, a missingness-fingerprint bug that
  wasn't part of the round-3 fix above (it's unrelated to
  `credential_exposure_window_flag`) and predates this whole investigation.
  It surfaced only when Category C's result was checked directly after
  round 3 incidentally moved it from an unverified 0.933 back to a fake
  1.000. Fixed by deriving `transaction_within_mandate_envelope` from each
  transaction's own already-sampled `amount`/`merchant.category` against
  the real agent's mandate (a genuine computed consequence of existing
  values, not a new arbitrary bias) - honest by construction, since the
  attacker in C1/C2 isn't targeting the mandate scope, so whether they
  happen to land inside it is exactly what it looks like: mostly chance.
  C3 also got `agent_behavior_pattern_match_score` added (0.7-0.99, the
  same well-matched range as C2/legit - C3's signal is scope abuse, not
  attestation/identity mismatch).
- **`claimed_employment_tenure_months`/`identity_cross_source_correlation`
  are now populated for every vector, not just legit/A1/A2** (fixed
  2026-08-16, same session). Generalizing these two fields - previously
  flagged below as "likely closer to a mechanical fix" for A1 - was done in
  two steps on purpose: first added to A2 alone via a new
  `_sample_identity_verification()` helper, retrained to check the pattern
  in isolation, and it immediately reproduced the exact same
  null-vs-populated bug shape as every fix above, just on this field pair:
  legit/A1/A2 had real values, the other 9 fraud vectors (B1-D2, E2) read
  `NaN`, and `identity_cross_source_correlation` alone carried 53.1% of
  Defend's feature importance (another 15.9% on `claimed_employment_tenure_months`,
  69.0% combined - on par with the original build-1 leak). Caught before
  being reported as a result, using the same diagnostics as every fix in
  this document. Fixed by applying the identical
  `_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL)` call
  to the remaining 9 generators (B1, B2, B3, C1, C2, C3, D1, D2, E2); A1
  keeps its own narrower `_IDENTITY_VERIFICATION_A1` range (0.05-0.35,
  its actual signal: thin file, low cross-source correlation) rather than
  the normal range every other vector/legit uses (0.75-0.99). Retrained:
  both fields dropped to 0.2%/0.1% importance (noise level).
  `tests/test_dataset.py::test_identity_verification_fields_populated_for_every_vector`
  guards this. See `defend/README.md`'s changelog (build 5) and
  `defend/results/report.md`'s identity-verification investigation section
  for the full before/after evidence.

## Known but not yet fixed (found 2026-08-16, checked on request)

Checking whether the round-3 fix above had incidentally affected Category C
prompted checking A1, A2, D1, and E2 too, since they show the same
clustered-near-1.000-confidence pattern that turned out to be real bugs in
both B and C. All four had at least one confirmed issue; A1's
identity-verification-field gap has since been fixed (see the
`identity_cross_source_correlation` anti-leakage note above) - A1 still
shares the account-age floor below with A2/D1/E2, which remain open for a
scope decision:

- **`entities.py`'s account-age floor is unrealistic and fully disjoint
  from four vectors.** `build_entity_pool()` hardcodes
  `account_age_days=int(rng.integers(30, 4000))` for every legit account,
  so no legit row is ever younger than 30 days - but A1 (0-13 days), A2
  (0-2), D1 (always 0), and E2 (0-1) all represent brand-new or throwaway
  identities below that floor, and `historical_avg_amount` is
  correspondingly always exactly `0.0` for these four versus legit's $4.24
  minimum. Real accounts get used within days of opening constantly; this
  floor is the same class of problem as the pre-fix wire-amount ceiling -
  a mechanical fix (widen the floor), just not made yet.
- **E2 never gets `promo_code` on legit's own `promo_redemption` rows**
  (0% of 1,007 legit promo rows) - the same channel-scoping gap already
  fixed for merchant/payee fields, just not yet applied here.
- **A2's session/liveness fields and D1's burst fields have no legit
  counterpart to attach to.** `session_frame_rate_anomaly`/
  `session_video_artifact_score`/`onboarding_velocity_seconds` (A2) and
  `attempt_sequence_id`/`attempts_in_window`/`window_seconds` (D1) are null
  for legit not because of a copyable omission but because this dataset has
  no "legit onboarding event" or "legit burst of small purchases" row to
  populate them on. Unlike the account-age floor, this needs a design
  decision (what should a legit onboarding event or legit purchase burst
  look like?) before it's a one-line fix.

See `defend/results/report.md`'s "Diagnostics run on A1, A2, D1, E2"
section for the full evidence (feature dominance, value-range overlap,
confidence spread) behind each of these.

## Layout

```
generate/
  src/generate/
    config.py     seed, volumes, per-vector weight table, shared constants
    entities.py   shared population: accounts, merchants, agents (for ID/history coherence)
    schema.py     TransactionRecord / DisputeRecord field definitions
    utils.py      timestamp / amount sampling helpers
    vectors.py    one generator function per taxonomy vector + legit baseline
    dataset.py    orchestrates generators into the full labeled dataset
    cli.py        `python -m generate.cli`
  tests/test_dataset.py   coverage + anti-leakage smoke tests
  data/                   generated output (gitignored)
```

## Running it

```bash
cd generate
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m generate.cli --format csv
```

Useful flags: `--scale` (multiplier on every vector's volume, default 1.0),
`--n-legit`, `--n-dispute-legit`, `--seed`, `--out-dir`, `--format csv|parquet`.

Output: `data/transactions.csv`, `data/disputes.csv`, and
`data/generation_summary.csv` (row count per vector, traceable back to the
taxonomy's weight table).

## Data dictionary

### `transactions.csv`

| Field | Meaning | Populated for |
|---|---|---|
| `transaction_id`, `timestamp`, `account_id`, `customer_id`, `channel`, `amount`, `currency` | Core identity/time/value | all |
| `merchant_id`, `merchant_category`, `merchant_registration_age_days`, `merchant_reputation_score` | Merchant context | non-payee channels only (card_not_present, card_present, agentic_checkout, promo_redemption) - never wire/ach/rtp/p2p, which have a payee instead; central to C2 |
| `account_age_days`, `prior_transaction_count`, `historical_avg_amount`, `credit_history_months`, `claimed_employment_tenure_months`, `identity_cross_source_correlation` | Account/identity history | all; central to A1 |
| `device_id`, `device_is_new`, `geo_country`, `geo_is_new` | Device/session | all; central to A2, B3, D2 |
| `session_frame_rate_anomaly`, `session_video_artifact_score`, `onboarding_velocity_seconds` | Liveness/capture artifacts | A2 |
| `login_event_flag`, `account_detail_changed_before_txn` | Takeover context | D2 |
| `payee_id`, `is_first_time_payee`, `payee_prior_interaction_count`, `request_urgency_flag`, `approval_outside_hierarchy` | Payee/authorization behavior | payee channels only (wire, ach, rtp, p2p) on every vector that lands there, not just B1 - elevated bias for B1 specifically |
| `preceded_by_support_call`, `time_since_support_call_minutes` | Call-proximity context | payee channels only, elevated bias for B2 specifically |
| `credential_exposure_window_flag` | Phishing-exposure proximity | all; elevated bias for B3 specifically |
| `agent_id`, `agent_attestation_present`, `agent_attestation_verified`, `agent_behavior_pattern_match_score`, `mandate_id`, `mandate_amount_cap`, `mandate_category_scope`, `mandate_recurring_flag`, `transaction_within_mandate_envelope` | Agentic commerce trust/scope | C1, C2, C3, and legitimate agentic_checkout rows |
| `attempt_sequence_id`, `attempts_in_window`, `window_seconds`, `authorization_result` | Velocity/automation | D1 |
| `promo_code`, `account_cluster_id` | Promo abuse clustering | E2 |
| `is_fraud`, `fraud_category`, `fraud_vector`, `narrative_tag` | Labels | all |

### `disputes.csv`

| Field | Meaning |
|---|---|
| `dispute_id`, `original_transaction_id`, `account_id`, `dispute_filed_at`, `dispute_reason` | Dispute identity, linked back to `transactions.csv` |
| `evidence_type`, `evidence_generated_flag`, `evidence_generative_artifact_score` | Submitted evidence and its generative-artifact score |
| `claimant_dispute_rate_30d`, `correlated_claim_count` | Account-level and cross-claim dispute-rate signal |
| `is_fraud`, `fraud_category`, `fraud_vector`, `narrative_tag` | Labels (`E1` when fraudulent, otherwise a genuine dispute) |

## Known gaps / deliberate non-goals

- **Category B is intentionally hard to separate on transaction fields
  alone**, per the taxonomy's own limitation note - and this is now verified,
  not just asserted. After every missingness/distribution leak found in
  Defend's build was fixed (see "Anti-leakage notes" above), Category B
  recall settled at a genuine ~0.89 with prediction confidence spanning a
  real range, not the 1.000 every earlier, leakier version of this dataset
  produced. Don't read that recall as a bug in Generate - it's the taxonomy's
  predicted reality of the vector, confirmed rather than assumed.
- **No cross-transaction graph.** Entities in `entities.py` give ID/history
  coherence, but there's no full relational transaction history per account
  beyond the summary stats (`prior_transaction_count`, `historical_avg_amount`).
  That's sufficient for the per-row signals the taxonomy names; a
  graph/sequence model is out of scope here.
