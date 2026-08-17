# Defend Pillar: Evaluation Report

Primary classifier: XGBoost gradient-boosted trees, binary is_fraud target, trained on generate/data/transactions.csv (11 of 12 transaction-shaped vectors - E1 excluded this pass, see 'Category E: scope note' below). See defend/README.md for the full design writeup and 'Known limitations' section.

## Dataset

- **Total rows**: 21960
- **Total fraud rows**: 700
- **Train rows**: 16470
- **Test rows**: 5490
- **Test fraud rows**: 175
- **Feature count**: 39
- **Random seed**: 42
- **Test size**: 0.25

## Aggregate metrics (test set, threshold = 0.5)

| Metric | Value |
|---|---|
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| ROC-AUC | 1.000 |
| PR-AUC | 1.000 |
| Test rows | 5490 (175 actual fraud) |
| Confusion | TP=175 FP=0 TN=5315 FN=0 |

## Why the aggregate scores are near-perfect (read this before the rest)

The top 2 features alone account for **82%** of feature importance (`credit_history_months` and `payee_prior_interaction_count` - see the full table at the bottom). That is disproportionate for a model that's supposed to be learning fraud behavior, and the near-1.000 scores across every category below - including Category B, which the taxonomy explicitly predicts should be *hard* - are the tell that something other than genuine signal is driving this.

The cause: several optional fields in generate/vectors.py are populated for one or two specific fraud vectors and left NaN for every other vector, while the legit generator populates a broader, more consistent set of fields on every row. XGBoost's native missing-value handling - the same property that makes it a good fit for this schema - also makes it extremely good at learning 'which generator produced this row' from missingness alone. That is a different, much easier task than 'is this transaction fraudulent', and it does not reflect how a real fraud/legit population would behave (real legitimate transactions don't uniformly populate a field that real fraud uniformly omits).

Evidence - fraction of rows with a non-null value, by vector:

| Feature | LEGIT | A1 | A2 | B1 | B2 | B3 | C1 | C2 | C3 | D1 | D2 | E2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `credit_history_months` | 0.94 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `payee_prior_interaction_count` | 0.94 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `attempts_in_window` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| `account_age_days` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `geo_is_new` | 0.94 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| `historical_avg_amount` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

`payee_prior_interaction_count` is the clearest example: legit rows always have it (uniform 1-49, never 0), B1/B2 fraud always set it to exactly 0, and every other fraud vector leaves it null. That field alone nearly perfectly separates legit / B1-or-B2-fraud / everything-else-fraud into three buckets before any genuinely behavioral signal comes into play - which is specifically why Category B scores as trivially detectable here despite the taxonomy calling it out as inherently hard. This is flagged as the #1 item for the Generate feedback loop (see defend/README.md) - not acted on in this pass.

## Per-category metrics (category-vs-legit slice of the test set)

Each row's precision/recall/F1/AUC is computed on that category's fraud rows plus every legit test row (same pool, same 0.5 threshold, same model), so categories stay comparable to each other. See evaluate.py docstring for why this slicing is necessary rather than optional.

| Category | Test fraud rows | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| A | 40 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B | 55 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| D | 50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| E | 15 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Category B: honest read

Category B (Social Engineering/Authorization Fraud) recall: 1.00 over 55 test cases. This is NOT the result identify/attack-taxonomy.md predicts - it explicitly calls B2 genuinely-authorized-by-the-accountholder and therefore hard to catch on transaction data alone. A near-perfect score here is a red flag, not a win: see results/report.md's 'Why the aggregate scores are near-perfect' section - B1/B2's is_first_time_payee/payee_prior_interaction_count are set deterministically and from a value range that never overlaps with legit rows, which makes this synthetic slice of Category B trivially separable rather than genuinely hard. Do not read this score as evidence the model would perform this well on real B-category fraud.

## Category E: scope note

Category E here reflects E2 (promo abuse) ONLY, not E1+E2. E1 (GenAI fabricated dispute evidence) is excluded from this training run: it lives in disputes.csv, and the transaction<->dispute join needed to recover its label is blocked by an ID-generation bug upstream in generate/entities.py (every short_id() output collapsed to a constant string - see data.py's module docstring for the exact root cause). This is a known, reported gap, not a silently dropped category. Category E recall shown below (1.00) should be read as 'recall on E2', not 'recall on category E'.

## Per-vector recall (11 vectors)

| Vector | Test rows | Caught | Recall |
|---|---|---|---|
| A1 | 25 | 25 | 1.000 |
| A2 | 15 | 15 | 1.000 |
| B1 | 15 | 15 | 1.000 |
| B2 | 15 | 15 | 1.000 |
| B3 | 25 | 25 | 1.000 |
| C1 | 5 | 5 | 1.000 |
| C2 | 5 | 5 | 1.000 |
| C3 | 5 | 5 | 1.000 |
| D1 | 25 | 25 | 1.000 |
| D2 | 25 | 25 | 1.000 |
| E2 | 15 | 15 | 1.000 |

## Worst-performing vectors (lowest recall)

In this run every vector scores at or near 1.000 recall (see the missingness diagnostic above for why), so this list isn't yet a meaningful differentiator between vectors - it will become useful once the missingness-fingerprinting issue is addressed and scores spread out. Included for completeness and to establish the reporting format.

| Vector | Test rows | Caught | Recall |
|---|---|---|---|
| A1 | 25 | 25 | 1.000 |
| A2 | 15 | 15 | 1.000 |
| B1 | 15 | 15 | 1.000 |
| B2 | 15 | 15 | 1.000 |
| B3 | 25 | 25 | 1.000 |

## Top feature importances

| Feature | Importance |
|---|---|
| credit_history_months | 0.4147 |
| payee_prior_interaction_count | 0.4070 |
| attempts_in_window | 0.0581 |
| account_age_days | 0.0479 |
| geo_is_new | 0.0167 |
| historical_avg_amount | 0.0157 |
| amount | 0.0123 |
| channel | 0.0111 |
| device_is_new | 0.0059 |
| agent_behavior_pattern_match_score | 0.0051 |
| merchant_registration_age_days | 0.0027 |
| prior_transaction_count | 0.0012 |
| authorization_result | 0.0006 |
| merchant_reputation_score | 0.0006 |
| window_seconds | 0.0003 |
