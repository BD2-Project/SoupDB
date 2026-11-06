"""Merge sorted SPIMI runs through the external-sort public API."""

import json
from collections.abc import Iterable, Iterator
from itertools import groupby
from pathlib import Path

from engine.algorithms.external_sort import merge_sorted_runs
from engine.common.record import Record


def encode_posting(term: str, doc_id: int, tf: int) -> Record:
    """Encode one term/document/frequency tuple inside the run framing."""
    return Record(data=json.dumps([term, doc_id, tf], ensure_ascii=False).encode("utf-8"))


def decode_posting(record: Record) -> tuple[str, int, int]:
    term, doc_id, tf = json.loads(record.data)
    if not isinstance(term, str) or not term or type(doc_id) is not int or doc_id < 0:
        raise ValueError("invalid SPIMI term or document ID")
    if type(tf) is not int or tf <= 0:
        raise ValueError("SPIMI frequency must be a positive integer")
    return term, doc_id, tf


def compare_postings(left: Record, right: Record) -> int:
    a, b = decode_posting(left)[:2], decode_posting(right)[:2]
    return (a > b) - (a < b)


def merged_terms(
    paths: Iterable[str | Path], *, fan_in: int = 8, temp_dir: str | Path | None = None
) -> Iterator[tuple[str, list[tuple[int, int]]]]:
    """Yield terms and ordered postings, combining pairs split across blocks.

    A single term's complete posting list is materialized. Call close() when
    abandoning iteration so the underlying merge releases temporary files.
    """
    records = merge_sorted_runs(paths, compare_postings, fan_in=fan_in, temp_dir=temp_dir)
    try:
        decoded = (decode_posting(record) for record in records)
        for term, group in groupby(decoded, key=lambda item: item[0]):
            postings: list[tuple[int, int]] = []
            for _, doc_id, tf in group:
                if postings and postings[-1][0] == doc_id:
                    postings[-1] = (doc_id, postings[-1][1] + tf)
                else:
                    postings.append((doc_id, tf))
            yield term, postings
    finally:
        records.close()
