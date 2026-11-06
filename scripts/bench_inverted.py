"""Measure SPIMI build/reopen and TF-IDF/BM25 MATCH queries on a JSONL corpus."""

import argparse
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

from engine.common.rid import RID
from engine.indexes.inverted.corpus_reader import read_corpus
from engine.indexes.inverted.index import InvertedIndex


def benchmark(
    corpus: str | Path,
    *,
    queries: tuple[str, ...] = ("coronavirus infection", "clinical trial"),
    k: int = 10,
    memory_limit_bytes: int = 8 * 1024 * 1024,
) -> dict:
    """Use temporary artifacts and verify identical hits after close/reopen.

    Synthetic RIDs are benchmark-only references into the source-ID map; SQL
    factory integration tests separately verify real storage RIDs.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")
    corpus = Path(corpus)
    documents = list(read_corpus(corpus))
    source_ids = {
        RID(i // 256, i % 256): document.source_id for i, document in enumerate(documents)
    }
    items = [(document.text, rid) for document, rid in zip(documents, source_ids, strict=True)]
    measurements = []
    with tempfile.TemporaryDirectory(prefix="soupdb-inverted-bench-") as temporary:
        index_path = Path(temporary) / "index"
        start = time.perf_counter()
        index = InvertedIndex.build(index_path, items, memory_limit_bytes=memory_limit_bytes)
        build_seconds = time.perf_counter() - start
        try:
            for method in ("tf_idf", "bm25"):
                for query in queries:
                    start = time.perf_counter()
                    hits = index.match_top_k(query, k, method=method)
                    seconds = time.perf_counter() - start
                    if any(
                        rid not in source_ids or not math.isfinite(score) for rid, score in hits
                    ) or hits != sorted(hits, key=lambda hit: (-hit[1], hit[0])):
                        raise AssertionError("invalid benchmark ranking")
                    measurements.append(
                        {"method": method, "query": query, "query_seconds": seconds, "hits": hits}
                    )
        finally:
            index.close()
        start = time.perf_counter()
        reopened = InvertedIndex.load(index_path)
        load_seconds = time.perf_counter() - start
        try:
            for measurement in measurements:
                after = reopened.match_top_k(measurement["query"], k, method=measurement["method"])
                before = measurement["hits"]
                if [rid for rid, _ in before] != [rid for rid, _ in after] or any(
                    not math.isclose(a[1], b[1], rel_tol=1e-12, abs_tol=1e-12)
                    for a, b in zip(before, after, strict=True)
                ):
                    raise AssertionError("ranking changed after reopening")
                measurement["hits"] = [
                    {"source_id": source_ids[rid], "rid": list(rid), "score": score}
                    for rid, score in before
                ]
        finally:
            reopened.close()
        artifact_bytes = sum(path.stat().st_size for path in index_path.iterdir())
    return {
        "corpus": str(corpus),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "documents": len(documents),
        "k": k,
        "memory_limit_bytes": memory_limit_bytes,
        "build_seconds": build_seconds,
        "load_seconds": load_seconds,
        "artifact_bytes": artifact_bytes,
        "reopen_verified": True,
        "queries": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--memory-limit-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(
        args.corpus,
        queries=tuple(
            args.queries or ("coronavirus infection", "clinical trial", "transmisión virus")
        ),
        k=args.k,
        memory_limit_bytes=args.memory_limit_bytes,
    )
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
