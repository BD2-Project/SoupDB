"""Deterministic RankedIndex fake for SQL integration tests.

Scripted responses model provider behavior without calculating ranking scores.
Every response is tied to an exact query and set of search options.
"""

from typing import Any

from engine.common.rid import RID
from engine.indexes.base import Query, RankedIndex


class ScriptedRankedIndex(RankedIndex):
    """Return explicitly registered RID/score hits and record search calls."""

    _OPTIONS = {
        "text": frozenset({"method"}),
        "ivf": frozenset({"metric", "nprobe"}),
        "hnsw": frozenset({"metric", "ef"}),
    }
    _METRICS = frozenset({"cosine", "euclidean", "inner_product"})

    def __init__(self, algorithm: str) -> None:
        if algorithm not in self._OPTIONS:
            raise ValueError(f"unsupported algorithm: {algorithm!r}")
        self.algorithm = algorithm
        self.calls: list[tuple[object, int, dict[str, Any]]] = []
        self.closed = False
        self._responses: dict[
            tuple[object, tuple[tuple[str, Any], ...]],
            list[tuple[RID, float]] | Exception,
        ] = {}

    def _key(
        self, query: Query, opts: dict[str, Any]
    ) -> tuple[object, tuple[tuple[str, Any], ...]]:
        unknown = opts.keys() - self._OPTIONS[self.algorithm]
        if unknown:
            raise ValueError(f"unsupported search options: {sorted(unknown)}")

        if self.algorithm == "text":
            if not isinstance(query, str):
                raise TypeError("text search requires a string query")
            if opts.get("method") not in {"tf_idf", "bm25"}:
                raise ValueError("text search requires method='tf_idf' or 'bm25'")
            normalized_query: object = query
        else:
            if isinstance(query, (str, bytes)):
                raise TypeError("vector search requires a numeric sequence")
            try:
                normalized_query = tuple(query)
            except TypeError as exc:
                raise TypeError("vector search requires a sequence") from exc
            if opts.get("metric") not in self._METRICS:
                raise ValueError("vector search requires a supported metric")
            parameter = "nprobe" if self.algorithm == "ivf" else "ef"
            if parameter in opts and (type(opts[parameter]) is not int or opts[parameter] <= 0):
                raise ValueError(f"{parameter} must be a positive integer")

        return normalized_query, tuple(sorted(opts.items()))

    def set_results(self, query: Query, results: list[tuple[RID, float]], **opts: Any) -> None:
        """Register hits verbatim, including invalid hits for negative tests."""
        self._responses[self._key(query, opts)] = list(results)

    def set_error(self, query: Query, error: Exception, **opts: Any) -> None:
        """Register a deterministic provider failure."""
        self._responses[self._key(query, opts)] = error

    def top_k(self, query: Query, k: int, **opts: Any) -> list[tuple[RID, float]]:
        if self.closed:
            raise RuntimeError("ranked index is closed")
        if type(k) is not int or k < 0:
            raise ValueError("k must be a non-negative integer")

        key = self._key(query, opts)
        self.calls.append((key[0], k, dict(opts)))
        if key not in self._responses:
            raise AssertionError(f"no scripted response for {self.algorithm}: {key!r}")

        response = self._responses[key]
        if isinstance(response, Exception):
            raise response
        return list(response[:k])

    def close(self) -> None:
        self.closed = True
