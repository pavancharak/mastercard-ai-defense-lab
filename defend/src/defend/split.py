"""Stratified train/test split.

Stratifies on fraud_vector (with legit rows collapsed to one "LEGIT" bucket)
rather than on the binary is_fraud label, so that a 25% test split still
holds back a proportional slice of every vector - including the rarest ones
(C1/C2/C3 at 20 rows each). Stratifying on the binary label alone would let
sklearn's random split starve a 20-row vector down to 0-2 test examples by
chance, which would make that vector's per-category metrics meaningless.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_SEED, TEST_SIZE


def stratify_key(df: pd.DataFrame) -> pd.Series:
    return df["fraud_vector"].fillna("LEGIT")


def split_dataset(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = stratify_key(df)
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=key
    )
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    return train_df, test_df
