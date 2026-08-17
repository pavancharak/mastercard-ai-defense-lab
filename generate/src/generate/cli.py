"""CLI entrypoint: python -m generate.cli [options]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_N_DISPUTE_LEGIT, DEFAULT_N_LEGIT, DEFAULT_SEED
from .dataset import build_dataset, write_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the GenAI payment fraud dataset.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale", type=float, default=1.0, help="Multiplier on per-vector volumes.")
    parser.add_argument("--n-legit", type=int, default=DEFAULT_N_LEGIT)
    parser.add_argument("--n-dispute-legit", type=int, default=DEFAULT_N_DISPUTE_LEGIT)
    args = parser.parse_args(argv)

    result = build_dataset(
        seed=args.seed,
        scale=args.scale,
        n_legit=args.n_legit,
        n_dispute_legit=args.n_dispute_legit,
    )
    write_dataset(result, args.out_dir, fmt=args.format)

    print(f"Wrote {len(result.transactions):,} transactions and {len(result.disputes):,} disputes to {args.out_dir}")
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
