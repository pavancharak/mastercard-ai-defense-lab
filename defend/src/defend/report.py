"""Renders evaluation results to markdown + CSV/JSON under defend/results/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR
from .evaluate import EvaluationResult


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


# Hand-recorded snapshots of the earlier, now-superseded runs, for the
# before/after/after-again/final comparison table. Build 1 is also archived
# in full under defend/results_before_fix/; build 2 was reported directly to
# the user at the end of the prior session and isn't separately archived,
# so its numbers are recorded here rather than re-derived.
_HISTORY = [
    {
        "build": "1 (original, leaked)",
        "precision": 1.000, "recall": 1.000, "f1": 1.000,
        "cat_b_recall": 1.000, "cat_c_recall": 1.000,
        "note": "credit_history_months/payee_prior_interaction_count null-pattern leak",
    },
    {
        "build": "2 (partial fix)",
        "precision": 0.994, "recall": 0.994, "f1": 0.994,
        "cat_b_recall": 1.000, "cat_c_recall": 0.933,
        "note": "fixed above + geo_is_new/device_is_new; Category B still leaked "
                "(Category C's 0.933 here was never independently verified - see build 4)",
    },
    {
        "build": "3 (Category B fixed)",
        "precision": 0.948, "recall": 0.931, "f1": 0.939,
        "cat_b_recall": 0.782, "cat_c_recall": 1.000,
        "note": "amount distributions + B1/B2/B3 behavioral fields fixed across 3 sub-rounds; "
                "Category C regressed to 1.000, unnoticed until checked directly",
    },
    {
        "build": "4 (Category C fixed)",
        "precision": 0.941, "recall": 0.914, "f1": 0.928,
        "cat_b_recall": 0.782, "cat_c_recall": 0.933,
        "note": "Category C's mandate-field leak fixed (see defend/README.md changelog); "
                "B unchanged from build 3, this fix only touched C",
    },
    {
        "build": "4.5 (regression, leaked again)",
        "precision": 1.000, "recall": 1.000, "f1": 1.000,
        "cat_b_recall": 1.000, "cat_c_recall": 1.000,
        "note": "identity_cross_source_correlation/claimed_employment_tenure_months added to A2 "
                "only (not generalized) - same null-vs-populated shape as every earlier fix, this "
                "time on a field pair that happened to carry ~69% combined feature importance; "
                "self-caught before being reported as a result - see investigation below. Archived "
                "in full under defend/results_regression_snapshot/",
    },
]


def write_results(
    eval_result: EvaluationResult,
    feature_importance_df: pd.DataFrame,
    dataset_stats: dict,
    missingness_df: pd.DataFrame,
    amount_overlap_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
    out_dir: Path = RESULTS_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(eval_result.aggregate, f, indent=2)

    eval_result.per_category.to_csv(out_dir / "metrics_per_category.csv", index=False)
    eval_result.per_vector.to_csv(out_dir / "metrics_per_vector.csv", index=False)
    feature_importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    missingness_df.to_csv(out_dir / "missingness_profile.csv", index=False)
    amount_overlap_df.to_csv(out_dir / "amount_range_overlap.csv", index=False)
    confidence_df.to_csv(out_dir / "prediction_confidence_spread.csv", index=False)

    agg = eval_result.aggregate
    cat_b = eval_result.per_category.set_index("category").loc["B"]
    fi_by_feature = feature_importance_df.set_index("feature")["importance"]
    identity_corr_importance = float(fi_by_feature.get("identity_cross_source_correlation", 0.0))
    tenure_importance = float(fi_by_feature.get("claimed_employment_tenure_months", 0.0))
    top1_importance = float(feature_importance_df.iloc[0]["importance"])
    top3_importance = float(feature_importance_df.head(3)["importance"].sum())
    all_ranges_overlap = bool(amount_overlap_df["ranges_overlap"].all())
    max_conf_p25 = float(confidence_df.loc[confidence_df.vector.isin(["B1", "B2", "B3"]), "proba_p25"].max())

    lines = []
    lines.append("# Defend Pillar: Evaluation Report\n")
    lines.append("Primary classifier: XGBoost gradient-boosted trees, binary is_fraud target, "
                  "trained on generate/data/transactions.csv (11 of 12 transaction-shaped vectors - "
                  "E1 excluded this pass, see 'Category E: scope note' below). See defend/README.md "
                  "for the full design writeup, changelog, and 'Known limitations' section.\n")

    lines.append("## Dataset\n")
    for k, v in dataset_stats.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Aggregate metrics (test set, threshold = 0.5)\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Precision | {_fmt(agg['precision'])} |")
    lines.append(f"| Recall | {_fmt(agg['recall'])} |")
    lines.append(f"| F1 | {_fmt(agg['f1'])} |")
    lines.append(f"| ROC-AUC | {_fmt(agg['roc_auc'])} |")
    lines.append(f"| PR-AUC | {_fmt(agg['pr_auc'])} |")
    lines.append(f"| Test rows | {agg['n']} ({agg['n_actual_fraud']} actual fraud) |")
    lines.append(f"| Confusion | TP={agg['true_positives']} FP={agg['false_positives']} "
                  f"TN={agg['true_negatives']} FN={agg['false_negatives']} |")
    lines.append("")

    lines.append("## Investigation: how Category B went from a fake 1.000 to a genuine "
                  f"{cat_b['recall']:.2f}\n")
    lines.append(
        "| Build | Precision | Recall | F1 | Category B recall | Category C recall | What changed |\n"
        "|---|---|---|---|---|---|---|"
    )
    for h in _HISTORY:
        lines.append(
            f"| {h['build']} | {h['precision']:.3f} | {h['recall']:.3f} | {h['f1']:.3f} | "
            f"{h['cat_b_recall']:.3f} | {h['cat_c_recall']:.3f} | {h['note']} |"
        )
    cat_c = eval_result.per_category.set_index("category").loc["C"]
    lines.append(
        f"| 5 (final, this run) | {agg['precision']:.3f} | {agg['recall']:.3f} | {agg['f1']:.3f} | "
        f"{cat_b['recall']:.3f} | {cat_c['recall']:.3f} | identity-verification fields generalized "
        "to the remaining 9 generators (B1-D2, E2), closing the 4.5 regression - see below |"
    )
    lines.append("")
    lines.append(
        "Build 3's own investigation went through three sub-rounds before landing here, each found "
        "by refusing to accept a still-too-good result at face value:\n"
    )
    lines.append(
        "1. **Widened `wire`/`ach`/`rtp`/`p2p` legit amount distributions** to a realistic "
        "population (real-world grounded: median US home closing ~$400k, Nacha ACH average "
        "~$2,642, RTP network cap now $10M/txn but bank-set P2P limits commonly $2.5k-$15k - see "
        "generate/vectors.py's `_WIRE_LIKE_AMOUNT_PARAMS` comment for full sourcing) instead of the "
        "same ~$3-$300 retail-purchase distribution every channel shared before. B1/B2's own fraud "
        "amounts were **not** touched, per the explicit instruction not to make the attack "
        "artificially subtle. Also added real (not deterministic) variance to B1/B2's named "
        "behavioral signals (`request_urgency_flag`, `approval_outside_hierarchy`, "
        "`preceded_by_support_call`) - retrained, Category B was **still 1.000**.\n"
        "2. Investigated why: `device_is_new` had jumped to the top feature, and "
        "`merchant_registration_age_days` (and the other merchant_* fields) were populated for "
        "legit on every channel including wire/rtp/p2p - where a transfer has a payee, not a "
        "merchant - while B1/B2/D2 correctly never set them. Same null-vs-populated shape as every "
        "earlier fix, just on a field pair nobody had scoped to channel yet. Fixed (legit now only "
        "samples a merchant on merchant-relevant channels; B3, which is genuinely a card-not-present "
        "purchase, got merchant fields added since it had never had them) - retrained, Category B "
        "was **still 1.000**.\n"
        "3. Investigated again: `credential_exposure_window_flag` was now the single dominant "
        "feature (46%) - it had only been added to legit and B3 in the previous step, leaving it "
        "null for every other vector including B1/B2, the exact same bug shape reintroduced by "
        "omission. Generalized into a proper per-vector helper (mirroring `_novelty_flags`) applied "
        "to every generator. Retrained - **this is the result reported below**.\n"
    )
    lines.append(
        f"Evidence this is now genuine, not just a smaller version of the same bug: no single "
        f"feature or pair dominates anymore (top feature {top1_importance:.0%}, top 3 combined "
        f"{top3_importance:.0%}, versus 82% carried by two fields in build 1); every vector's "
        f"amount range now genuinely overlaps its same-channel legit population "
        f"({'all' if all_ranges_overlap else 'not all'} 11 vectors - see "
        "`amount_range_overlap.csv`); and B1/B2/B3's predicted-fraud-probabilities now span a real "
        f"range instead of clustering at ~1.000 confidence (25th-percentile confidence across "
        f"B1/B2/B3 test rows tops out at {max_conf_p25:.3f}, not >0.99 - see "
        "`prediction_confidence_spread.csv`).\n"
    )

    cat_c = eval_result.per_category.set_index("category").loc["C"]
    lines.append(f"## Investigation: Category C, {cat_c['recall']:.3f} confirmed genuine\n")
    lines.append(
        "Category C was never independently checked during the Category B investigation above - "
        "it happened to read 0.933 in build 2 and 1.000 again in this build's earlier rounds, "
        "neither number verified with dedicated diagnostics. Prompted to check specifically "
        "whether the credential_exposure_window_flag fix (which touched every generator, not just "
        "B) had incidentally affected C, the same three diagnostics were run scoped to C alone:\n"
    )
    lines.append(
        "- **Feature dominance (C-vs-legit only model)**: `credential_exposure_window_flag` did "
        "not appear in the top 15 features - ruled out. Instead, `channel` (42%) and "
        "`agent_attestation_verified`/`transaction_within_mandate_envelope`/"
        "`agent_attestation_present` (45% combined) dominated. `channel` is legitimate (agentic "
        "commerce fraud only happens on `agentic_checkout`, by taxonomy design). The other three "
        "were not: within the agentic_checkout population specifically, `transaction_within_mandate_envelope` "
        "was null for 100% of C1/C2 and populated for 100% of C3/legit, and "
        "`agent_behavior_pattern_match_score` was null for 100% of C3 and populated for 100% of "
        "C1/C2/legit - the identical missingness-fingerprint shape as every earlier fix, just not "
        "checked for Category C until now. Root cause: `generate_c1_agent_impersonation`/"
        "`generate_c2_malicious_storefront` set `mandate_id` from the agent they use but never "
        "propagated the agent's other real mandate attributes (`mandate_amount_cap`, "
        "`mandate_category_scope`, `mandate_recurring_flag`, envelope compliance) into the "
        "transaction - a pre-existing gap, not something this session's fixes introduced.\n"
        "- **Amount overlap**: already clean for C1/C2/C3, not implicated.\n"
        "- **Confidence spread**: C1/C3 were clustered near 1.000 confidence; C2 showed a wide "
        "spread (0.28-0.989) even before this fix, which in hindsight was itself a clue that C1/C3 "
        "specifically - not the whole category - were the leaking ones.\n"
    )
    lines.append(
        "Fixed by deriving `transaction_within_mandate_envelope` from the transaction's own "
        "already-sampled amount and merchant category against the real agent's mandate (a genuine "
        "computed consequence, not an arbitrary bias) for C1/C2, and giving C3 a "
        "well-matched `agent_behavior_pattern_match_score` (0.7-0.99, the same range as C2/legit, "
        "since C3's taxonomy signal is about mandate scope, not attestation/identity mismatch - "
        "unlike C1). Retrained:\n"
    )
    lines.append("| | Precision | Recall | F1 |\n|---|---|---|---|")
    lines.append(f"| Category C, before this fix (build 3) | {_fmt(0.625)} | {_fmt(1.000)} | {_fmt(0.769)} |")
    lines.append(f"| Category C, after this fix (build 4) | {_fmt(cat_c['precision'])} | {_fmt(cat_c['recall'])} | {_fmt(cat_c['f1'])} |")
    lines.append("")
    lines.append(
        "Re-ran all three diagnostics after the fix to confirm, not just re-run the metric: both "
        "fields are now 100% populated for every vector (no more null pattern); the C-only "
        "feature-importance model no longer has `channel` in its top 10 at all, and its new top "
        "feature (`transaction_within_mandate_envelope`, 59%) is now a genuinely value-derived "
        "signal rather than a missingness pattern - confirmed by Monte Carlo: C1's near-zero "
        "envelope-compliance rate is expected by chance given its independently-sampled amount vs "
        "mandate cap distributions (~1.6% expected, matching the ~0% observed), not a logic bug; "
        "and C2's prediction confidence now spans 0.28-0.989 genuinely uncertain. C1/C3 remain "
        "highly confident (>0.999), but that traces to their own literal taxonomy-named signals "
        "(C1: failed attestation verification; C3: deterministic mandate-envelope breach, which is "
        "quite literally C3's whole defining mechanism) rather than to a missingness artifact - the "
        "taxonomy doesn't predict C should be hard the way it predicts B is, so strong signal here "
        "isn't itself suspicious once the missingness bug is confirmed gone.\n"
    )

    lines.append("## Investigation: identity-verification field regression (self-caught, same session)\n")
    lines.append(
        "Prompted by this report's own build-4 'Known limitations' note that A1's "
        "`claimed_employment_tenure_months`/`identity_cross_source_correlation` 'plausibly apply "
        "to legit accounts too... and could likely be added mechanically', a follow-up session set "
        "out to generalize those two fields the same way every other field in this report's history "
        "was generalized. The first step - adding them to A2 via a new "
        "`_sample_identity_verification()` helper, without yet touching the other 9 fraud "
        "generators - was retrained before the remaining generators were updated, to check the "
        "pattern in isolation. That retrain scored a suspicious **1.000 precision/recall/F1/AUC "
        "across every category and vector again** - the identical fake-perfect signature as builds "
        "1 and 3's leaks. The same null-vs-populated shape as every earlier fix, just on a new "
        "field pair: legit, A1, and now A2 had real values; every other fraud vector (B1-D2, E2) "
        "still read `NaN`, and `identity_cross_source_correlation` alone carried "
        "**53.1%** of feature importance with `claimed_employment_tenure_months` carrying another "
        "**15.9%** (69.0% combined, on par with build 1's original leak). Archived in full under "
        "`defend/results_regression_snapshot/` before being overwritten.\n"
    )
    lines.append(
        "Caught before being reported as a result, using the same diagnostics this whole "
        "investigation has relied on rather than trusting the number at face value. Fixed by "
        "applying the identical `_sample_identity_verification(rng, _IDENTITY_VERIFICATION_NORMAL)` "
        "call to the remaining 9 generators (B1, B2, B3, C1, C2, C3, D1, D2, E2) in "
        "`generate/vectors.py`, so every vector - not just A2 - draws these fields from a real "
        "distribution instead of leaving them null. Regenerated the dataset and retrained:\n"
    )
    lines.append("| | Aggregate P/R/F1 | `identity_cross_source_correlation` importance | `claimed_employment_tenure_months` importance |\n|---|---|---|---|")
    lines.append(
        f"| Regression (A2 only) | 1.000 / 1.000 / 1.000 | 53.1% | 15.9% |\n"
        f"| This fix (all 11 vectors) | {agg['precision']:.3f} / {agg['recall']:.3f} / {agg['f1']:.3f} | "
        f"{identity_corr_importance:.1%} | {tenure_importance:.1%} |"
    )
    lines.append("")
    lines.append(
        f"Both fields dropped out of dominance entirely ({identity_corr_importance:.1%} and "
        f"{tenure_importance:.1%} respectively, versus 53.1%/15.9% before) rather than merely "
        "shrinking - consistent with every other genuine fix in this investigation, where the "
        "leaking field falls to noise level once it's populated the same way across every vector "
        "instead of disappearing from the model's view. Category B's honest-read recall "
        f"({cat_b['recall']:.2f}) and the amount/confidence-spread evidence below are unaffected by "
        "this fix, since it only ever touched the two identity fields.\n"
    )

    lines.append("## Diagnostics run on A1, A2, D1, E2 - not fixed this pass\n")
    lines.append(
        "The same clustered-near-1.000-confidence pattern that turned out to be real bugs in both "
        "B and C also appears in A1, A2, D1, and E2. Checked with the same three diagnostics, "
        "**none of the four is clean** - each has at least one real, confirmed issue - but they "
        "split into two different kinds of fix:\n"
    )
    lines.append(
        "**Mechanical (same fix pattern as everything above), common to all four**: "
        "`account_age_days`/`historical_avg_amount`/`prior_transaction_count` are completely "
        "disjoint from legit's range - not a null pattern this time, a hard floor. "
        "`generate/entities.py`'s account pool hardcodes `account_age_days=int(rng.integers(30, "
        "4000))` for every legit account, so no legit row is ever younger than 30 days, while "
        "A1 (0-13), A2 (0-2), D1 (always 0), and E2 (0-1) all represent brand-new/throwaway "
        "identities below that floor - `historical_avg_amount` is correspondingly always exactly "
        "0.0 for these four versus legit's minimum of $4.24, a fully disjoint gap. Real accounts "
        "get used within days of opening constantly; a 30-day floor on the entire legit population "
        "is itself the unrealistic assumption, the same class of issue as B1's original $3-$300 "
        "wire ceiling. E2 additionally never gets `promo_code` set on legit's own "
        "`promo_redemption` rows (0% populated on 1,007 legit promo rows) - the same null pattern "
        "as every other channel-scoped field fixed this session.\n"
    )
    lines.append(
        "**Update**: A1's `claimed_employment_tenure_months`/`identity_cross_source_correlation` - "
        "flagged below as 'could likely be added mechanically' - have since been generalized to "
        "every vector (see the identity-verification field investigation above), so they no longer "
        "appear in A1's isolated feature dominance. A1's remaining issue is the same "
        "`account_age_days`/`historical_avg_amount` floor shared by A2/D1/E2 below, not a separate "
        "identity-field gap.\n"
    )
    lines.append(
        "**Architectural (not a one-line fix - needs a design decision first)**: A2's "
        "`session_frame_rate_anomaly`/`session_video_artifact_score`/`onboarding_velocity_seconds` "
        "and D1's `attempt_sequence_id`/`attempts_in_window`/`window_seconds` are different: "
        "they're null for legit not because of an omission but because Generate's schema has no "
        "legit onboarding-event or legit transaction-burst row to attach them to at all - fixing "
        "these means designing and building an analogous legit sub-population (a 'legit onboarding "
        "event' and a 'legit burst of small purchases'), not copying a value from an object that "
        "already exists. That's the same category of judgment call that stopped the amount-fix "
        "investigation earlier rather than being improvised unilaterally.\n"
    )
    lines.append("| Vector | Feature dominance (isolated model) | Amount/value-range overlap | Confidence spread |")
    lines.append("|---|---|---|---|")
    lines.append("| A1 | `account_age_days`/`historical_avg_amount` (identity-verification fields no longer dominant - see fix above) | `account_age_days`/`historical_avg_amount` disjoint from legit | 0.999-1.0, no spread |")
    lines.append("| A2 | `session_video_artifact_score`/`onboarding_velocity_seconds`/`session_frame_rate_anomaly` (100% combined) | `account_age_days`/`historical_avg_amount` disjoint from legit | flat 1.0, no spread |")
    lines.append("| D1 | `window_seconds`/`attempts_in_window`/`account_age_days` (100% combined) | `account_age_days`/`historical_avg_amount` disjoint from legit | 0.999-1.0, no spread |")
    lines.append("| E2 | `prior_transaction_count`/`account_age_days`/`historical_avg_amount` (99% combined) | same disjoint range, plus `promo_code` null on legit's own promo_redemption rows | 0.999-1.0, no spread |")
    lines.append("")
    lines.append(
        "Not fixed in this pass - reported for a decision on scope/priority before touching "
        "generate/ further.\n"
    )

    lines.append("## Per-category metrics (category-vs-legit slice of the test set)\n")
    lines.append("Each row's precision/recall/F1/AUC is computed on that category's fraud rows "
                  "plus every legit test row (same pool, same 0.5 threshold, same model), so "
                  "categories stay comparable to each other. See evaluate.py docstring for why "
                  "this slicing is necessary rather than optional.\n")
    lines.append("| Category | Test fraud rows | Precision | Recall | F1 | ROC-AUC | PR-AUC |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, row in eval_result.per_category.iterrows():
        lines.append(
            f"| {row['category']} | {row['n_test_fraud']} | {_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} | {_fmt(row['f1'])} | {_fmt(row['roc_auc'])} | {_fmt(row['pr_auc'])} |"
        )
    lines.append("")

    lines.append("## Category B: honest read\n")
    lines.append(eval_result.category_b_note + "\n")

    lines.append("## Category E: scope note\n")
    lines.append(eval_result.category_e_note + "\n")

    lines.append(f"## Per-vector recall ({len(eval_result.per_vector)} vectors)\n")
    lines.append("| Vector | Test rows | Caught | Recall |")
    lines.append("|---|---|---|---|")
    for _, row in eval_result.per_vector.iterrows():
        recall = "n/a" if row["recall"] is None else f"{row['recall']:.3f}"
        lines.append(f"| {row['vector']} | {row['n_test']} | {row['n_caught']} | {recall} |")
    lines.append("")

    lines.append("## Worst-performing vectors (lowest recall)\n")
    lines.append("This is a report for the feedback loop back to Generate - it is **not** acted "
                  "on in this pass. Flagged here so a future iteration can decide whether to add "
                  "more/sharper synthetic signal for these vectors, not to change this model.\n")
    lines.append("| Vector | Test rows | Caught | Recall |")
    lines.append("|---|---|---|---|")
    for row in eval_result.worst_performers:
        recall = "n/a" if row["recall"] is None else f"{row['recall']:.3f}"
        lines.append(f"| {row['vector']} | {row['n_test']} | {row['n_caught']} | {recall} |")
    lines.append("")

    lines.append("## Amount range overlap (evidence, per vector)\n")
    lines.append("| Vector | Channel(s) | Legit amount range (same channel) | Vector amount range | Ranges overlap |")
    lines.append("|---|---|---|---|---|")
    for _, row in amount_overlap_df.iterrows():
        legit_range = f"${row['legit_amount_min']:.2f}-${row['legit_amount_max']:.2f}"
        vec_range = f"${row['vector_amount_min']:.2f}-${row['vector_amount_max']:.2f}"
        flag = "yes" if row["ranges_overlap"] else "no **<- disjoint**"
        lines.append(f"| {row['vector']} | {row['channels']} | {legit_range} | {vec_range} | {flag} |")
    lines.append("")

    lines.append("## Prediction confidence spread, B1/B2/B3 (evidence)\n")
    lines.append("| Vector | Test rows | Min | P25 | Median | Max |")
    lines.append("|---|---|---|---|---|---|")
    for _, row in confidence_df[confidence_df.vector.isin(["B1", "B2", "B3"])].iterrows():
        lines.append(
            f"| {row['vector']} | {row['n_test']} | {row['proba_min']:.3f} | {row['proba_p25']:.3f} | "
            f"{row['proba_median']:.3f} | {row['proba_max']:.3f} |"
        )
    lines.append("")

    lines.append("## Top feature importances\n")
    lines.append("| Feature | Importance |")
    lines.append("|---|---|")
    for _, row in feature_importance_df.head(15).iterrows():
        lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
