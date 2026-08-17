"""Sanity tests for the read-only data loaders. No network, no OpenAI,
no writes to identify/generate/defend/mandate-demo -- just confirms the
parsers/loaders correctly read what's actually committed there."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web_prototype import data_sources  # noqa: E402


def test_taxonomy_loads_all_categories_and_vectors():
    categories = data_sources.load_taxonomy()
    assert [c.category_id for c in categories] == ["A", "B", "C", "D", "E"]
    total_vectors = sum(len(c.vectors) for c in categories)
    assert total_vectors == 12  # see mandate-demo's initial-commit discussion: 12, not 14

    c_category = next(c for c in categories if c.category_id == "C")
    vector_ids = [v.vector_id for v in c_category.vectors]
    assert vector_ids == ["C1", "C2", "C3"]

    c3 = next(v for v in c_category.vectors if v.vector_id == "C3")
    assert "Delegated Mandate Scope Abuse" in c3.name
    assert c3.mechanism  # non-empty
    assert c3.channel
    assert c3.transaction_signal


def test_sample_transactions_covers_every_vector():
    sample = data_sources.sample_transactions(limit_per_vector=2)
    vectors_seen = {row.get("fraud_vector") or "LEGIT" for row in sample}
    assert "LEGIT" in vectors_seen
    assert "C3" in vectors_seen
    assert len(sample) == 2 * 12  # 11 fraud vectors + LEGIT


def test_metrics_aggregate_has_expected_shape():
    aggregate = data_sources.load_metrics_aggregate()
    for key in ("precision", "recall", "f1", "roc_auc", "n"):
        assert key in aggregate


def test_metrics_per_category_and_per_vector_load():
    per_category = data_sources.load_metrics_per_category()
    assert {row["category"] for row in per_category} == {"A", "B", "C", "D", "E"}

    per_vector = data_sources.load_metrics_per_vector()
    assert "C3" in {row["vector"] for row in per_vector}


def test_captured_mandate_demo_run_loads_and_has_a_demonstrated_case():
    captured = data_sources.load_captured_run()
    assert captured is not None
    assert len(captured["scenarios"]) == 6
    assert len(captured["demonstrated_cases"]) >= 1
    s3 = next(s for s in captured["scenarios"] if s["intent_id"] == "S3")
    assert s3["classifier_missed_mandate_caught"] is True
    assert s3["check_b_classifier"]["risk_label"] == "LOW_RISK"
    assert s3["check_c_mandate"]["allowed"] is False
