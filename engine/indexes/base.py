"""Frozen contract for every index structure.

All indexes point to RIDs. Every implementation of :class:`Index` must pass
the conformance suite in ``tests/unit/indexes/test_index_contract.py``.
"""

from abc import ABC, abstractmethod
from typing import Any

from engine.common.rid import RID

#: Key type supported by indexes (int, str and composite keys).
Key = Any


class Index(ABC):
    """Base interface for all index structures."""

    @abstractmethod
    def insert(self, key: Key, rid: RID) -> None:
        """Insert a RID under a key."""

    @abstractmethod
    def search(self, key: Key) -> list[RID]:
        """Return all RIDs matching the key."""

    @abstractmethod
    def range_search(self, lo: Key, hi: Key) -> list[RID]:
        """Return RIDs whose key is in the inclusive range [lo, hi].

        Raises UnsupportedOperation on indexes that do not support ranges.
        """

    @abstractmethod
    def remove(self, key: Key, rid: RID | None = None) -> int:
        """Remove a RID (or all RIDs) under a key; returns the count removed."""

    @property
    @abstractmethod
    def supports_range(self) -> bool:
        """Whether the index supports range searches."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release resources."""


#: Consulta de un índice rankeado: texto para el invertido, vector para IVF/HNSW.
Query = Any


class RankedIndex(ABC):
    """Base interface for indexes that answer top-k queries with a score.

    ``Index.search`` returns bare RIDs, but text and vector retrieval need the
    ranking itself. Every implementation must pass the conformance suite in
    ``tests/unit/indexes/test_ranked_index_contract.py``.

    ``query`` depends on the index:

    - inverted index (Parte 3): a ``str`` with the query text; options
      ``method="tf_idf" | "bm25"``.
    - IVF and HNSW (Parte 4): a ``Sequence[float]`` of the index dimension;
      options ``metric="euclidean" | "cosine" | "inner_product"``, plus
      ``nprobe`` (IVF) or ``ef`` (HNSW).

    The score is always "higher is better", so results are sorted in
    descending order no matter the metric. Unknown options raise
    ``ValueError``.
    """

    @abstractmethod
    def top_k(self, query: Query, k: int, **opts: Any) -> list[tuple[RID, float]]:
        """Return up to ``k`` (RID, score) pairs sorted by descending score."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release resources."""
