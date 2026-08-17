"""Loads Generate's transaction table.

Scope note: this pass trains only on transactions.csv (categories A-D, and
category E via E2). disputes.csv / E1 (fabricated dispute evidence) is
excluded this pass, not by design but because of an upstream bug in Generate:

    generate/src/generate/entities.py:22, short_id():
        uuid.UUID(int=int(rng.integers(0, 2**63))).hex[:10]

rng.integers(0, 2**63) only ever fills the low 63 bits of the 128-bit UUID
integer; .hex is big-endian, so .hex[:10] reads the leading, always-zero
digits instead of the random ones. Every ID short_id() produces (including
transaction_id and disputes.csv's original_transaction_id) collapsed to the
same constant string. Feature columns are unaffected (verified: amount,
account_age_days, etc. are independently sampled and vary normally per row),
but the transaction<->dispute link is unrecoverable from the CSVs as they
stand - the join key is constant on both sides, and transactions.csv and
disputes.csv are also shuffled independently in dataset.py, so row order
doesn't recover it either.

Per explicit instruction, generate/ is not modified in this pass. Once
short_id() is fixed and the dataset regenerated, disputes.csv can be
reintroduced as a genuine transaction_id -> dispute_id left join (the
approach was validated conceptually, just blocked on real IDs to join with).
See defend/README.md "Known limitations" for the full writeup.
"""

from __future__ import annotations

import pandas as pd

from .config import TRANSACTIONS_CSV


def load_transactions() -> pd.DataFrame:
    return pd.read_csv(TRANSACTIONS_CSV)


def build_dataset() -> pd.DataFrame:
    return load_transactions()
