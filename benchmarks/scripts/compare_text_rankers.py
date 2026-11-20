"""Compare SoupDB TF-IDF/BM25 with PostgreSQL GIN on nested TREC-COVID sets."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from benchmarks.retrieval_metrics import (
    build_query_length_ablation,
    parse_qrels,
    restrict_qrels_to_corpus,
)
from benchmarks.scripts.postgres_gin import benchmark_gin
from engine.common.rid import RID
from engine.indexes.inverted.corpus_reader import read_corpus
from engine.indexes.inverted.index import InvertedIndex
from engine.indexes.inverted.tokenizer import DEFAULT_TOKENIZER


def _query_rows(queries_path: Path) -> dict[str, list[dict[str, Any]]]:
    with queries_path.open(encoding="utf-8") as file:
        buckets = build_query_length_ablation(file)
    return {
        bucket: [
            {"query_id": query_id, "text": text, "length_words": len(DEFAULT_TOKENIZER(text))}
            for query_id, text in sorted(values.items())
        ]
        for bucket, values in buckets.items()
    }


def _qrels(path: Path) -> dict[str, frozenset[str]]:
    with path.open(encoding="utf-8") as file:
        return parse_qrels(file)


def _source_ids(items: list[tuple[str, RID]], documents: list[Any]) -> dict[RID, str]:
    return {rid: doc.source_id for (rid, _text), doc in zip(items, documents, strict=True)}


def _macro(rows: list[dict[str, Any]]) -> dict[str, Any]:
    p = [row["precision_at_10"] for row in rows]
    recalls = [
        row["recall_at_10_conditional"]
        for row in rows
        if row["recall_at_10_conditional"] is not None
    ]
    latencies = [row["latency_median_seconds"] for row in rows]
    return {
        "query_count": len(rows),
        "precision_at_10_macro": statistics.fmean(p) if p else None,
        "recall_at_10_conditional_macro": statistics.fmean(recalls) if recalls else None,
        "queries_with_relevant_in_sample": len(recalls),
        "queries_without_relevant_in_sample": len(rows) - len(recalls),
        "latency_median_seconds_macro": statistics.fmean(latencies) if latencies else None,
        "latency_p95_seconds": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
        if latencies
        else None,
    }


class _ProcessMemorySampler:
    """Sample this process' Windows working set without tracing allocations."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.samples_bytes: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _working_set_bytes() -> int | None:
        if sys.platform != "win32":
            return None

        class _CountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = _CountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        process = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)

    def _sample(self) -> None:
        value = self._working_set_bytes()
        if value is not None:
            self.samples_bytes.append(value)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> _ProcessMemorySampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._sample()

    def summary(self) -> dict[str, Any]:
        baseline = self.samples_bytes[0] if self.samples_bytes else None
        peak = max(self.samples_bytes, default=None)
        return {
            "type": "current process working set sampled on Windows",
            "sample_interval_seconds": self.interval_seconds,
            "samples": len(self.samples_bytes),
            "baseline_bytes": baseline,
            "peak_observed_bytes": peak,
            "delta_peak_from_baseline_bytes": peak - baseline
            if peak is not None and baseline is not None
            else None,
        }


def _run_soupdb(
    corpus_path: Path,
    queries_by_bucket: dict[str, list[dict[str, Any]]],
    qrels: dict[str, frozenset[str]],
    *,
    memory_limit_bytes: int,
    repeats: int,
) -> dict[str, Any]:
    documents = list(read_corpus(corpus_path))
    if not documents:
        raise ValueError(f"empty corpus: {corpus_path}")
    if len({doc.source_id for doc in documents}) != len(documents):
        raise ValueError("duplicate corpus IDs")
    items: list[tuple[str, RID]] = []
    source_ids: dict[RID, str] = {}
    for ordinal, document in enumerate(documents):
        rid = RID(ordinal // 256, ordinal % 256)
        items.append((document.text, rid))
        source_ids[rid] = document.source_id

    with tempfile.TemporaryDirectory(prefix="soupdb-ranker-compare-") as temporary:
        index_path = Path(temporary) / "index"
        with _ProcessMemorySampler() as process_memory:
            started = time.perf_counter()
            index = InvertedIndex.build(index_path, items, memory_limit_bytes=memory_limit_bytes)
            build_seconds = time.perf_counter() - started
        artifact_bytes = sum(path.stat().st_size for path in index_path.iterdir())
        results: dict[str, list[dict[str, Any]]] = {"tf_idf": [], "bm25": []}
        try:
            available_qrels = restrict_qrels_to_corpus(qrels, (doc.source_id for doc in documents))
            for bucket, queries in queries_by_bucket.items():
                for method in ("tf_idf", "bm25"):
                    for query in queries:
                        query_text, query_id = query["text"], query["query_id"]
                        latencies: list[float] = []
                        hits: list[tuple[RID, float]] = []
                        for repeat in range(repeats + 1):
                            started = time.perf_counter()
                            hits = index.match_top_k(query_text, 10, method=method)
                            elapsed = time.perf_counter() - started
                            if repeat:
                                latencies.append(elapsed)
                        hit_ids = [source_ids[rid] for rid, _score in hits]
                        relevant = available_qrels.get(query_id, frozenset())
                        found = sum(doc_id in relevant for doc_id in hit_ids[:10])
                        results[method].append(
                            {
                                "query_id": query_id,
                                "length_bucket": bucket,
                                "query_length_words": query["length_words"],
                                "query": query_text,
                                "latency_seconds": latencies,
                                "latency_median_seconds": statistics.median(latencies),
                                "hits": [
                                    {
                                        "document_id": source_ids[rid],
                                        "rid": list(rid),
                                        "score": score,
                                    }
                                    for rid, score in hits
                                ],
                                "precision_at_10": found / 10,
                                "recall_at_10_conditional": found / len(relevant)
                                if relevant
                                else None,
                                "relevant_in_sample": len(relevant),
                            }
                        )
        finally:
            index.close()
    return {
        "build_seconds": build_seconds,
        "build_memory_observed": process_memory.summary(),
        "artifact_bytes": artifact_bytes,
        "queries": results,
    }


def compare(
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
    *,
    dsn: str,
    docker_container_id: str | None = None,
    memory_limit_bytes: int = 8 * 1024 * 1024,
    repeats: int = 5,
) -> dict[str, Any]:
    """Run all rankers against one exact sample and shared topic IDs."""
    corpus_path, queries_path, qrels_path = map(Path, (corpus_path, queries_path, qrels_path))
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    queries = _query_rows(queries_path)
    qrels = _qrels(qrels_path)
    if not queries["1"] or len({q["query_id"] for q in queries["1"]}) != 48:
        raise ValueError("expected the 48 shared TREC-COVID topics with original length >= 5")
    soupdb = _run_soupdb(
        corpus_path, queries, qrels, memory_limit_bytes=memory_limit_bytes, repeats=repeats
    )
    # The same query IDs, exact query strings, and class metadata go to GIN.
    flat_queries = [query for bucket in ("1", "3", "5+") for query in queries[bucket]]
    gin = benchmark_gin(
        corpus_path,
        flat_queries,
        qrels,
        dsn=dsn,
        k=10,
        repeats=repeats,
        docker_container_id=docker_container_id,
        force_gin=True,
    )
    if any(not row["used_gin_plan"] for row in gin["queries"]):
        raise RuntimeError("PostgreSQL did not use the GIN index for every query")
    gin_rows = {
        "postgresql_gin_ts_rank_cd": [
            {
                **row,
                "length_bucket": (
                    "1"
                    if row["query_length_words"] == 1
                    else "3"
                    if row["query_length_words"] == 3
                    else "5+"
                ),
                "precision_at_10": row["precision_at_k"],
                "recall_at_10_conditional": row["recall_at_k_conditional"],
            }
            for row in gin["queries"]
        ]
    }
    systems: dict[str, Any] = {
        "tf_idf_cosine": {
            "build_seconds": soupdb["build_seconds"],
            "space_bytes": soupdb["artifact_bytes"],
            "memory_metric": soupdb["build_memory_observed"],
            "queries": soupdb["queries"]["tf_idf"],
        },
        "bm25": {
            "build_seconds": soupdb["build_seconds"],
            "space_bytes": soupdb["artifact_bytes"],
            "memory_metric": soupdb["build_memory_observed"],
            "queries": soupdb["queries"]["bm25"],
        },
        "postgresql_gin_ts_rank_cd": {
            "build_seconds": gin["setup_seconds"],
            "space_bytes": gin["total_relation_bytes"],
            "memory_metric": gin["container_memory_observed"],
            "queries": gin_rows["postgresql_gin_ts_rank_cd"],
            "build_phases_seconds": {
                "insert_and_vectorize": gin["insert_and_vectorize_seconds"],
                "gin_build": gin["gin_build_seconds"],
                "analyze": gin["analyze_seconds"],
            },
        },
    }
    summary: dict[str, Any] = {}
    for system, values in systems.items():
        summary[system] = {}
        for bucket in ("1", "3", "5+"):
            summary[system][bucket] = _macro(
                [row for row in values["queries"] if row["length_bucket"] == bucket]
            )
    return {
        "dataset": "TREC-COVID",
        "corpus_path": str(corpus_path),
        "documents": len(list(read_corpus(corpus_path))),
        "query_source": "MTEB TREC-COVID queries; 1/3-token prefix ablations reuse parent qrels",
        "query_ids_per_length_bucket": 48,
        "qrels_rule": (
            "grades 1/2 relevant; 0/-1 not relevant; recall conditional on relevant docs in sample"
        ),
        "postgresql": {key: value for key, value in gin.items() if key != "queries"},
        "systems": systems,
        "summary_by_system_and_length": summary,
        "tokenizer_note": (
            "SoupDB Spanish/English heuristic tokenizer; "
            "PostgreSQL uses English text search configuration."
        ),
        "memory_note": (
            "SoupDB process working set and PostgreSQL whole-container memory are sampled. "
            "Their scopes differ and sample intervals can miss short peaks; "
            "do not treat as directly equivalent exact peaks."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("queries", type=Path)
    parser.add_argument("qrels", type=Path)
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--docker-container-id", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import os

    dsn = args.dsn or os.environ.get("SOUPDB_PG_DSN")
    if not dsn:
        parser.error("pass --dsn or set SOUPDB_PG_DSN")
    report = compare(
        args.corpus,
        args.queries,
        args.qrels,
        dsn=dsn,
        docker_container_id=args.docker_container_id,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary_by_system_and_length"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
