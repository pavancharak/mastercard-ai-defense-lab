"""Diagnoses *why* the model scores as well as it does, rather than taking a
high score at face value.

This pipeline's first real training run scored a suspicious 1.000 across
every metric, every category, every vector - including Category B, which
identify/attack-taxonomy.md explicitly says should be hard to separate on
transaction data alone. That contradiction was the signal that something
other than genuine behavioral signal was driving the score, and across
several rounds of investigation it turned out to be real - a family of bugs
in generate/vectors.py, all the same shape: an optional field populated for
legit (or for one specific vector) and left null/deterministic everywhere
else, letting the model fingerprint "which generator wrote this row" instead
of learning real fraud behavior. See defend/README.md's changelog and
results/report.md's investigation section for the full history of what was
found and fixed (credit_history_months/payee_prior_interaction_count,
geo_is_new/device_is_new, merchant_* fields on non-merchant channels,
credential_exposure_window_flag, and B1/B2's amount distributions).

The three functions below are the reusable evidence-gathering tools that
made each of those findings falsifiable rather than asserted, and now double
as regression checks: run them again on any future training run to confirm
a score is genuine before trusting it.
"""

from __future__ import annotations

import pandas as pd


def missingness_profile(df: pd.DataFrame, columns: list[str], vector_order: list[str]) -> pd.DataFrame:
    """For each column, the fraction of rows with a non-null value, broken
    out by fraud_vector (+ LEGIT). A column that's ~100% populated for
    exactly one vector and ~0% for every other fraud vector is acting as a
    row-source fingerprint, not a behavioral feature."""
    key = df["fraud_vector"].fillna("LEGIT")
    rows = []
    for col in columns:
        non_null = df[col].notna()
        by_vector = non_null.groupby(key).mean()
        row = {"feature": col}
        row["LEGIT"] = round(float(by_vector.get("LEGIT", float("nan"))), 3)
        for v in vector_order:
            row[v] = round(float(by_vector.get(v, float("nan"))), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_confidence_spread(test_df: pd.DataFrame, y_proba, vectors: list[str]) -> pd.DataFrame:
    """Per-vector distribution of the model's predicted fraud probability on
    the test set. A vector the model is trivially/mechanically separating
    (a leftover missingness or disjoint-range bug) shows every row clustered
    at ~1.000 confidence with essentially zero spread; a vector that's
    genuinely-hard-but-mostly-caught shows real spread, including some rows
    the model is genuinely unsure about or gets wrong. This is the check
    used to confirm Category B's fix was real rather than declared on faith
    - see results/report.md's investigation section."""
    import numpy as np

    proba = np.asarray(y_proba)
    rows = []
    for vector in vectors:
        mask = (test_df["fraud_vector"] == vector).to_numpy()
        if mask.sum() == 0:
            continue
        sub = proba[mask]
        rows.append({
            "vector": vector,
            "n_test": int(mask.sum()),
            "proba_min": round(float(sub.min()), 3),
            "proba_p25": round(float(pd.Series(sub).quantile(0.25)), 3),
            "proba_median": round(float(pd.Series(sub).median()), 3),
            "proba_max": round(float(sub.max()), 3),
        })
    return pd.DataFrame(rows)


def amount_range_overlap(df: pd.DataFrame, vectors: list[str]) -> pd.DataFrame:
    """For each vector, compares its amount range to LEGIT rows on the same
    channel (the fair comparison - a wire transfer should be compared to
    other wire transfers, not to card-present coffee purchases). A vector
    whose amount range doesn't overlap AT ALL with same-channel legit rows
    is separable by a single threshold on amount alone, independent of any
    other feature - which is exactly as strong a tell as the missingness
    bug this function is modeled after, just expressed as a value range
    instead of a null pattern. See report.py's rendering of this table for
    the read on which vectors that's true for."""
    rows = []
    for vector in vectors:
        vec_rows = df[df["fraud_vector"] == vector]
        if vec_rows.empty:
            continue
        channels = set(vec_rows["channel"].unique())
        legit_same_channel = df[(df["fraud_vector"].isna()) & (df["channel"].isin(channels))]["amount"]
        vec_amount = vec_rows["amount"]
        overlap = not (vec_amount.min() > legit_same_channel.max() or vec_amount.max() < legit_same_channel.min())
        rows.append({
            "vector": vector,
            "channels": ",".join(sorted(channels)),
            "legit_amount_min": round(float(legit_same_channel.min()), 2) if len(legit_same_channel) else None,
            "legit_amount_max": round(float(legit_same_channel.max()), 2) if len(legit_same_channel) else None,
            "vector_amount_min": round(float(vec_amount.min()), 2),
            "vector_amount_max": round(float(vec_amount.max()), 2),
            "ranges_overlap": overlap,
        })
    return pd.DataFrame(rows)
