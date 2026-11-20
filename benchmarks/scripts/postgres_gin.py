"""Benchmark PostgreSQL English full-text search backed by a GIN index.

The runner creates a session-local temporary table, so it does not alter the
container's databases or persist corpus data. GIN supplies candidates;
``ts_rank_cd`` is the separate ranking function used for this baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from engine.indexes.inverted.corpus_reader import read_corpus

_UNIT_MB = {"b": 1 / 1048576, "kb": 1 / 1024, "mb": 1, "gb": 1024, "tb": 1048576}
_MEMORY = re.compile(r"^\s*([\d.]+)\s*([KMGT]?i?B)\s*$", re.IGNORECASE)


def parse_memory_mib(value: str) -> float:
    """Parse Docker's displayed memory size into MiB."""
    match = _MEMORY.match(value)
    if not match:
        raise ValueError(f"unrecognized Docker memory value: {value!r}")
    unit = match.group(2).lower().replace("i", "")
    return float(match.group(1)) * _UNIT_MB[unit]


class DockerMemorySampler:
    """Sample one existing container's total memory without modifying it."""

    def __init__(self, container_id: str, *, interval_seconds: float = 1.0) -> None:
        if not container_id or interval_seconds <= 0:
            raise ValueError("container ID and positive sampling interval are required")
        self.container_id = container_id
        self.interval_seconds = interval_seconds
        self.samples_mib: list[float] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.container_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            used = result.stdout.strip().split(" / ", maxsplit=1)[0]
            self.samples_mib.append(parse_memory_mib(used))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.errors.append(str(exc))

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> DockerMemorySampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=12)
        self._sample()

    def summary(self) -> dict[str, float | int | None]:
        baseline = self.samples_mib[0] if self.samples_mib else None
        peak = max(self.samples_mib, default=None)
        return {
            "sample_interval_seconds": self.interval_seconds,
            "samples": len(self.samples_mib),
            "scope": "whole Docker container; sampled, not a guaranteed peak",
            "baseline_mib": baseline,
            "end_mib": self.samples_mib[-1] if self.samples_mib else None,
            "peak_observed_mib": peak,
            "delta_peak_from_baseline_mib": (
                peak - baseline if peak is not None and baseline is not None else None
            ),
        }


def _judged_relevant(
    qrels: Mapping[str, Mapping[str, int] | Sequence[str] | set[str] | frozenset[str]],
    query_id: str,
    sample_ids: set[str],
) -> set[str]:
    """Accept either grades or the positive-ID sets from retrieval_metrics."""
    judgments = qrels.get(str(query_id), {})
    if isinstance(judgments, Mapping):
        relevant = (str(doc_id) for doc_id, grade in judgments.items() if grade >= 1)
    else:
        relevant = (str(doc_id) for doc_id in judgments)
    return {document_id for document_id in relevant if document_id in sample_ids}


def precision_recall_at_10(
    retrieved_ids: Sequence[str], *, relevant_ids: set[str], k: int = 10
) -> dict[str, float | int | None]:
    """Return precision@k and sample-conditional recall@k for judged qrels."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    top = list(retrieved_ids[:k])
    found = sum(document_id in relevant_ids for document_id in top)
    return {
        "precision_at_k": found / k,
        "recall_at_k_conditional": found / len(relevant_ids) if relevant_ids else None,
        "relevant_in_sample": len(relevant_ids),
        "relevant_retrieved": found,
    }


def benchmark_gin(
    corpus_path: str | Path,
    queries: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, int] | Sequence[str] | set[str] | frozenset[str]],
    *,
    dsn: str,
    k: int = 10,
    repeats: int = 5,
    docker_container_id: str | None = None,
    force_gin: bool = False,
) -> dict[str, Any]:
    """Build a session-local GIN index, rank queries and collect resource data.

    Corpus, qrels and queries must refer to the exact same nested sample and
    query IDs as the other rankers. If Docker is supplied, memory is sampled at
    one-second intervals; this is container-wide observed memory, not a peak
    guarantee. PostgreSQL persists no benchmark table after disconnect.
    """
    if not dsn.strip():
        raise ValueError("PostgreSQL DSN is required")
    if type(k) is not int or k <= 0 or type(repeats) is not int or repeats < 1:
        raise ValueError("k must be positive and repeats must be at least one")
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError as exc:
        raise RuntimeError(
            "psycopg2-binary is required to run the PostgreSQL GIN benchmark"
        ) from exc
    documents = list(read_corpus(corpus_path))
    if len({document.source_id for document in documents}) != len(documents):
        raise ValueError("corpus contains duplicate source IDs")
    sample_ids = {document.source_id for document in documents}
    memory = DockerMemorySampler(docker_container_id) if docker_container_id else None
    connection = psycopg2.connect(dsn)
    connection.autocommit = False
    results: list[dict[str, Any]] = []
    if memory is not None:
        memory.__enter__()
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL default_text_search_config = 'pg_catalog.english'")
            cursor.execute("SELECT version()")
            postgres_version = cursor.fetchone()[0]
            cursor.execute("""
                CREATE TEMP TABLE gin_benchmark_documents (
                    document_id text NOT NULL,
                    document_vector tsvector NOT NULL
                ) ON COMMIT PRESERVE ROWS
            """)
            connection.commit()
            insert_start = time.perf_counter()
            # Generate tsvector server-side in batches; body+title use the same
            # concatenation convention as the SoupDB inverted provider.
            for start in range(0, len(documents), 1000):
                batch = documents[start : start + 1000]
                with connection.cursor() as cursor:
                    execute_values(
                        cursor,
                        """INSERT INTO gin_benchmark_documents(document_id, document_vector)
                           VALUES %s""",
                        [(doc.source_id, doc.text) for doc in batch],
                        template="(%s, to_tsvector('english', %s))",
                        page_size=1000,
                    )
                connection.commit()
            insert_seconds = time.perf_counter() - insert_start

            index_start = time.perf_counter()
            with connection.cursor() as cursor:
                cursor.execute(
                    "CREATE INDEX gin_benchmark_documents_gin "
                    "ON gin_benchmark_documents USING GIN (document_vector)"
                )
            connection.commit()
            gin_build_seconds = time.perf_counter() - index_start
            analyze_start = time.perf_counter()
            with connection.cursor() as cursor:
                cursor.execute("ANALYZE gin_benchmark_documents")
                cursor.execute(
                    "SELECT pg_relation_size('gin_benchmark_documents'), "
                    "pg_relation_size('gin_benchmark_documents_gin'), "
                    "pg_total_relation_size('gin_benchmark_documents')"
                )
                heap_bytes, index_bytes, total_bytes = cursor.fetchone()
            connection.commit()
            analyze_seconds = time.perf_counter() - analyze_start

            if force_gin:
                with connection.cursor() as cursor:
                    # Session-local to this short-lived benchmark connection.
                    cursor.execute("SET enable_seqscan = off")
            for query in queries:
                query_id, query_text = str(query["query_id"]), query["text"]
                length_words = query.get("length_words", len(query_text.split()))
                if type(length_words) is not int or length_words < 0:
                    raise ValueError("query length_words must be a nonnegative integer")
                relevant = _judged_relevant(qrels, query_id, sample_ids)
                sql = """SELECT document_id,
                                 ts_rank_cd(document_vector,
                                     plainto_tsquery('english', %s)) AS score
                          FROM gin_benchmark_documents
                          WHERE document_vector @@ plainto_tsquery('english', %s)
                          ORDER BY score DESC, document_id ASC LIMIT %s"""
                with connection.cursor() as cursor:
                    cursor.execute("EXPLAIN (FORMAT TEXT) " + sql, (query_text, query_text, k))
                    plan = "\n".join(row[0] for row in cursor.fetchall())
                latency: list[float] = []
                top_ids: list[str] = []
                for repeat in range(repeats + 1):
                    start = time.perf_counter()
                    with connection.cursor() as cursor:
                        cursor.execute(sql, (query_text, query_text, k))
                        rows = cursor.fetchall()
                    elapsed = time.perf_counter() - start
                    if repeat:
                        latency.append(elapsed)
                    top_ids = [row[0] for row in rows]
                results.append(
                    {
                        "query_id": query_id,
                        "query": query_text,
                        "query_length_words": length_words,
                        "latency_seconds": latency,
                        "latency_median_seconds": statistics.median(latency),
                        "hits": [
                            {"document_id": doc_id, "score": float(score)} for doc_id, score in rows
                        ],
                        **precision_recall_at_10(top_ids, relevant_ids=relevant, k=k),
                        "used_gin_plan": "gin_benchmark_documents_gin" in plan,
                        "explain_plan": plan,
                    }
                )
    finally:
        connection.rollback()
        connection.close()
        if memory is not None:
            memory.__exit__(None, None, None)

    if memory is not None:
        # The table is session-local and gone after the connection closes;
        # sample container memory through the DDL/query phases inside session.
        # A separate context is intentionally not used to avoid changing it.
        memory_summary = memory.summary()
    else:
        memory_summary = None
    return {
        "engine": "postgresql_gin_ts_rank_cd",
        "postgresql_version": postgres_version,
        "documents": len(documents),
        "k": k,
        "repeats": repeats,
        "insert_and_vectorize_seconds": insert_seconds,
        "gin_build_seconds": gin_build_seconds,
        "analyze_seconds": analyze_seconds,
        "setup_seconds": insert_seconds + gin_build_seconds + analyze_seconds,
        "heap_bytes": heap_bytes,
        "gin_index_bytes": index_bytes,
        "total_relation_bytes": total_bytes,
        "container_memory_observed": memory_summary,
        "queries": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("queries", type=Path, help="JSON array of {query_id,text}")
    parser.add_argument("qrels", type=Path, help="JSON object mapping query_id to document grades")
    parser.add_argument("--dsn", default=os.environ.get("SOUPDB_PG_DSN"))
    parser.add_argument("--docker-container-id", default=os.environ.get("SOUPDB_PG_CONTAINER_ID"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--force-gin", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.dsn:
        parser.error("pass --dsn or set SOUPDB_PG_DSN")
    report = benchmark_gin(
        args.corpus,
        json.loads(args.queries.read_text(encoding="utf-8")),
        json.loads(args.qrels.read_text(encoding="utf-8")),
        dsn=args.dsn,
        k=args.k,
        repeats=args.repeats,
        docker_container_id=args.docker_container_id,
        force_gin=args.force_gin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "queries"}, indent=2))


if __name__ == "__main__":
    main()
