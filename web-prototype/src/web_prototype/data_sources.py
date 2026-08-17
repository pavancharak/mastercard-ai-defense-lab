"""Read-only loaders for everything the UI displays. Every function here
reads an existing file under identify/, generate/, defend/, or
mandate-demo/ and returns it as structured data -- nothing is computed,
inferred, or hardcoded here that isn't already in one of those files.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTIFY_DIR = REPO_ROOT / "identify"
GENERATE_DIR = REPO_ROOT / "generate"
DEFEND_DIR = REPO_ROOT / "defend"
MANDATE_DEMO_DIR = REPO_ROOT / "mandate-demo"

TAXONOMY_PATH = IDENTIFY_DIR / "attack-taxonomy.md"
TRANSACTIONS_CSV = GENERATE_DIR / "data" / "transactions.csv"
REPORT_MD_PATH = DEFEND_DIR / "results" / "report.md"
METRICS_AGGREGATE_PATH = DEFEND_DIR / "results" / "metrics_aggregate.json"
METRICS_PER_CATEGORY_PATH = DEFEND_DIR / "results" / "metrics_per_category.csv"
METRICS_PER_VECTOR_PATH = DEFEND_DIR / "results" / "metrics_per_vector.csv"


# ---------------------------------------------------------------------------
# Taxonomy (identify/)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaxonomyVector:
    vector_id: str  # e.g. "C3"
    name: str
    category_id: str  # e.g. "C"
    mechanism: str
    channel: str
    transaction_signal: str
    plausibility: str
    severity: str


@dataclass(frozen=True)
class TaxonomyCategory:
    category_id: str
    name: str
    vectors: list[TaxonomyVector]


_CATEGORY_RE = re.compile(r"^\\## Category ([A-E]): (.+)$")
_VECTOR_RE = re.compile(r"^\\### ([A-E]\d)\. (.+)$")
_FIELD_RE = re.compile(r"^\\\*\\\*(Mechanism|Channel|Transaction signal|Plausibility|Severity)\\\*\\\*: (.+)$")


def load_taxonomy() -> list[TaxonomyCategory]:
    """Parses identify/attack-taxonomy.md's own escaped-markdown format
    (headers are literally written as '\\## ...' / '\\### ...' /
    '\\*\\*Field\\*\\*: ...' in the source file itself -- this parser
    matches that exact format, it doesn't alter it)."""

    text = TAXONOMY_PATH.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines()]

    categories: list[TaxonomyCategory] = []
    current_category: dict | None = None
    current_vector: dict | None = None

    def flush_vector():
        if current_vector is not None and current_category is not None:
            current_category["vectors"].append(
                TaxonomyVector(
                    vector_id=current_vector["id"],
                    name=current_vector["name"],
                    category_id=current_category["id"],
                    mechanism=current_vector.get("Mechanism", ""),
                    channel=current_vector.get("Channel", ""),
                    transaction_signal=current_vector.get("Transaction signal", ""),
                    plausibility=current_vector.get("Plausibility", ""),
                    severity=current_vector.get("Severity", ""),
                )
            )

    def flush_category():
        flush_vector()
        if current_category is not None:
            categories.append(
                TaxonomyCategory(
                    category_id=current_category["id"],
                    name=current_category["name"],
                    vectors=current_category["vectors"],
                )
            )

    for line in lines:
        cat_match = _CATEGORY_RE.match(line)
        if cat_match:
            flush_category()
            current_category = {"id": cat_match.group(1), "name": cat_match.group(2), "vectors": []}
            current_vector = None
            continue

        vec_match = _VECTOR_RE.match(line)
        if vec_match:
            flush_vector()
            current_vector = {"id": vec_match.group(1), "name": vec_match.group(2)}
            continue

        field_match = _FIELD_RE.match(line)
        if field_match and current_vector is not None:
            current_vector[field_match.group(1)] = field_match.group(2)

    flush_category()
    return categories


# ---------------------------------------------------------------------------
# Generate (generate/data/transactions.csv)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS_FOR_DISPLAY = [
    "transaction_id", "timestamp", "channel", "amount", "currency",
    "merchant_category", "geo_country", "is_fraud", "fraud_category",
    "fraud_vector", "narrative_tag",
]


def sample_transactions(limit_per_vector: int = 3) -> list[dict]:
    """A small, legible sample: a few legit rows plus a few rows from
    every fraud vector, for the case browser's default listing -- not
    the full ~21k-row dataset (that's what the free-text transaction_id
    lookup and free-form scoring form are for)."""

    with open(TRANSACTIONS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_vector: dict[str, list[dict]] = {}
    for row in rows:
        key = row.get("fraud_vector") or "LEGIT"
        by_vector.setdefault(key, []).append(row)

    sample: list[dict] = []
    for key in sorted(by_vector):
        sample.extend(by_vector[key][:limit_per_vector])
    return sample


def get_transaction_by_id(transaction_id: str) -> dict | None:
    with open(TRANSACTIONS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["transaction_id"] == transaction_id:
                return row
    return None


# ---------------------------------------------------------------------------
# Defend (defend/results/)
# ---------------------------------------------------------------------------


def load_report_markdown() -> str:
    return REPORT_MD_PATH.read_text(encoding="utf-8")


def load_metrics_aggregate() -> dict:
    with open(METRICS_AGGREGATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_metrics_per_category() -> list[dict]:
    with open(METRICS_PER_CATEGORY_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_metrics_per_vector() -> list[dict]:
    with open(METRICS_PER_VECTOR_PATH, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# mandate-demo (mandate-demo/output/)
# ---------------------------------------------------------------------------


def latest_captured_run_path() -> Path | None:
    output_dir = MANDATE_DEMO_DIR / "output"
    runs = sorted(output_dir.glob("run_*.json"))
    return runs[-1] if runs else None


def load_captured_run() -> dict | None:
    path = latest_captured_run_path()
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["_source_file"] = path.name
    return data
