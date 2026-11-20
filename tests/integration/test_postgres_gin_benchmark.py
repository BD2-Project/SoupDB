"""Opt-in smoke test against an already running PostgreSQL instance."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.scripts.postgres_gin import (
    _judged_relevant,
    benchmark_gin,
    parse_memory_mib,
    precision_recall_at_10,
)

DSN = os.environ.get("SOUPDB_PG_DSN")
CONTAINER_ID = os.environ.get("SOUPDB_PG_CONTAINER_ID")


def test_memory_and_relevance_helpers() -> None:
    assert parse_memory_mib("1.5GiB") == 1536
    assert parse_memory_mib("512MiB") == 512
    metrics = precision_recall_at_10(["relevant", "other"], relevant_ids={"relevant"}, k=2)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k_conditional"] == 1.0
    assert _judged_relevant({"q1": frozenset({"relevant"})}, "q1", {"relevant", "other"}) == {
        "relevant"
    }


@pytest.mark.skipif(not DSN, reason="SOUPDB_PG_DSN is not configured")
def test_postgres_gin_build_query_and_relevance(tmp_path: Path) -> None:
    pytest.importorskip("psycopg2", reason="psycopg2-binary is not installed")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"_id": "doc-a", "title": "climate change", "text": "ocean warming study"},
                {"_id": "doc-b", "title": "database systems", "text": "query indexing"},
            )
        ),
        encoding="utf-8",
    )

    report = benchmark_gin(
        corpus,
        [{"query_id": "q1", "text": "climate ocean", "length_words": 7}],
        {"q1": {"doc-a": 2, "doc-b": 0}},
        dsn=DSN or "",
        repeats=1,
        docker_container_id=CONTAINER_ID,
        force_gin=True,
    )

    assert report["documents"] == 2
    assert report["gin_index_bytes"] > 0
    assert report["setup_seconds"] == (
        report["insert_and_vectorize_seconds"]
        + report["gin_build_seconds"]
        + report["analyze_seconds"]
    )
    result = report["queries"][0]
    assert result["hits"][0]["document_id"] == "doc-a"
    assert result["query_length_words"] == 7
    assert result["precision_at_k"] == 0.1
    assert result["recall_at_k_conditional"] == 1.0
    assert result["used_gin_plan"] is True
    if CONTAINER_ID:
        memory = report["container_memory_observed"]
        assert memory["baseline_mib"] is not None
        assert memory["peak_observed_mib"] is not None
        assert memory["delta_peak_from_baseline_mib"] is not None
