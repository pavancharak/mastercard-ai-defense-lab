"""Paths, seed, and column-role constants for the Defend pillar.

Column roles are declared here (not inferred at runtime) so the exclusion
list is auditable in one place: anything that identifies a row (an ID),
restates the label in disguise (narrative_tag literally names the vector),
or *is* the label doesn't belong in the feature matrix. See defend/README.md
"Anti-leakage" section for the reasoning behind each exclusion.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATE_DATA_DIR = REPO_ROOT / "generate" / "data"
TRANSACTIONS_CSV = GENERATE_DATA_DIR / "transactions.csv"
DISPUTES_CSV = GENERATE_DATA_DIR / "disputes.csv"

DEFEND_DIR = REPO_ROOT / "defend"
MODEL_DIR = DEFEND_DIR / "model"
RESULTS_DIR = DEFEND_DIR / "results"

RANDOM_SEED = 42
TEST_SIZE = 0.25
DECISION_THRESHOLD = 0.5

CATEGORY_ORDER = ["A", "B", "C", "D", "E"]
# E1 (fabricated dispute evidence) is excluded this pass: it lives only in
# disputes.csv and the join needed to recover its label is blocked by an
# upstream ID-generation bug in generate/entities.py. See data.py and
# defend/README.md "Known limitations" for the full explanation. Category E
# here is therefore E2-only, not E1+E2.
VECTOR_ORDER = ["A1", "A2", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "E2"]

# Row-identity fields: unique per row (or unique per entity), carry no
# generalizable signal, and would let a tree split on an ID instead of on
# behavior.
ID_COLUMNS = [
    "transaction_id", "account_id", "customer_id", "merchant_id", "device_id",
    "payee_id", "agent_id", "mandate_id", "attempt_sequence_id",
    "account_cluster_id", "promo_code",
    "dispute_id", "original_transaction_id",
]

# Raw timestamps are dropped in favor of the derived txn_hour feature (see
# features.py); the raw string itself isn't a usable numeric/categorical
# feature and doesn't generalize past this dataset's date window.
TIMESTAMP_COLUMNS = ["timestamp", "dispute_filed_at"]

# narrative_tag is a free-text description that literally names the fraud
# vector (e.g. "synthetic identity: thin file..."); fraud_category/fraud_vector
# are label-adjacent metadata, not observable pre-label. All three are the
# single biggest leakage risk in this dataset if included as features.
LABEL_LEAKAGE_COLUMNS = ["narrative_tag", "fraud_category", "fraud_vector"]

TARGET_COLUMN = "is_fraud"

EXCLUDED_COLUMNS = set(ID_COLUMNS) | set(TIMESTAMP_COLUMNS) | set(LABEL_LEAKAGE_COLUMNS) | {TARGET_COLUMN}
