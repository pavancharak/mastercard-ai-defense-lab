"""Assembles the full labeled dataset: legit baseline + one block per vector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import vectors
from .config import (
    DEFAULT_N_DISPUTE_LEGIT,
    DEFAULT_N_LEGIT,
    DEFAULT_SEED,
    DISPUTE_VECTORS,
    UNITS_PER_WEIGHT,
    VECTOR_CATEGORY,
    VECTOR_NAME,
    VECTOR_WEIGHTS,
)
from .entities import EntityPool, build_entity_pool
from .schema import DISPUTE_COLUMNS, TRANSACTION_COLUMNS


def vector_count(vector: str, scale: float) -> int:
    return max(1, round(VECTOR_WEIGHTS[vector] * UNITS_PER_WEIGHT * scale))


@dataclass
class BuildResult:
    transactions: pd.DataFrame
    disputes: pd.DataFrame
    summary: pd.DataFrame


def build_dataset(
    seed: int = DEFAULT_SEED,
    scale: float = 1.0,
    n_legit: int = DEFAULT_N_LEGIT,
    n_dispute_legit: int = DEFAULT_N_DISPUTE_LEGIT,
) -> BuildResult:
    rng = np.random.default_rng(seed)
    entities: EntityPool = build_entity_pool(rng)

    txn_rows: list[dict] = vectors.generate_legit(rng, entities, n_legit)
    dispute_rows: list[dict] = []

    counts: dict[str, int] = {}

    for vector, generator in vectors.VECTOR_GENERATORS.items():
        n = vector_count(vector, scale)
        counts[vector] = n
        txn_rows.extend(generator(rng, entities, n))

    # Category E1 is dispute-shaped: it emits linked (transaction, dispute) pairs.
    n_e1 = vector_count("E1", scale)
    counts["E1"] = n_e1
    e1_txns, e1_disputes = vectors.generate_e1_fabricated_dispute_evidence(rng, entities, n_e1)
    txn_rows.extend(e1_txns)
    dispute_rows.extend(e1_disputes)

    legit_dispute_txns, legit_disputes = vectors.generate_legit_disputes(rng, entities, n_dispute_legit)
    txn_rows.extend(legit_dispute_txns)
    dispute_rows.extend(legit_disputes)

    transactions = pd.DataFrame(txn_rows, columns=TRANSACTION_COLUMNS)
    transactions = transactions.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    disputes = pd.DataFrame(dispute_rows, columns=DISPUTE_COLUMNS)
    disputes = disputes.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    summary_rows = [
        {
            "vector": "LEGIT",
            "category": None,
            "name": "Legitimate baseline",
            "row_count": n_legit,
            "table": "transactions",
        },
        {
            "vector": "LEGIT_DISPUTE",
            "category": None,
            "name": "Legitimate dispute baseline",
            "row_count": n_dispute_legit,
            "table": "disputes",
        },
    ]
    for vector, n in counts.items():
        summary_rows.append(
            {
                "vector": vector,
                "category": VECTOR_CATEGORY[vector],
                "name": VECTOR_NAME[vector],
                "row_count": n,
                "table": "disputes" if vector in DISPUTE_VECTORS else "transactions",
            }
        )
    summary = pd.DataFrame(summary_rows)

    return BuildResult(transactions=transactions, disputes=disputes, summary=summary)


def write_dataset(result: BuildResult, out_dir: Path, fmt: str = "csv") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        result.transactions.to_csv(out_dir / "transactions.csv", index=False)
        result.disputes.to_csv(out_dir / "disputes.csv", index=False)
    elif fmt == "parquet":
        result.transactions.to_parquet(out_dir / "transactions.parquet", index=False)
        result.disputes.to_parquet(out_dir / "disputes.parquet", index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    result.summary.to_csv(out_dir / "generation_summary.csv", index=False)
