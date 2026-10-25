"""Exact k-NN by scanning every stored vector.

It is the sequential baseline of the comparison 2.4.4 and the oracle the
recall of IVF and HNSW is measured against.
"""

import heapq
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from engine.indexes.vector.constants import DEFAULT_METRIC, VECTOR_BUFFER_CAPACITY
from engine.indexes.vector.metrics import norm, score_fn
from engine.indexes.vector.store import VectorStore


class BruteForceIndex(RankedIndex):
    """RankedIndex that scores every vector; memory is O(k), not O(N)."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    @classmethod
    def open(
        cls,
        path: str | Path,
        dim: int | None = None,
        *,
        page_size: int = 4096,
        buffer_capacity: int = VECTOR_BUFFER_CAPACITY,
    ) -> "BruteForceIndex":
        return cls(
            VectorStore.open(path, dim, page_size=page_size, buffer_capacity=buffer_capacity)
        )

    @property
    def store(self) -> VectorStore:
        return self._store

    def __len__(self) -> int:
        return len(self._store)

    def insert(self, vector: Sequence[float], rid: RID) -> None:
        self._store.append(vector, rid)

    def top_k(
        self, query: Sequence[float], k: int, *, metric: str = DEFAULT_METRIC
    ) -> list[tuple[RID, float]]:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self._store.check_dim(query)
        scorer = score_fn(metric)
        query_norm = norm(query)
        scored = (
            (scorer(query, query_norm, vector, vector_norm), rid)
            for _vid, rid, vector, vector_norm in self._store.scan()
        )
        best = heapq.nlargest(k, scored, key=_by_score)
        return [(rid, value) for value, rid in best]

    def close(self) -> None:
        self._store.close()


def _by_score(pair: tuple[float, Any]) -> float:
    return pair[0]
