"""In-memory ranked index oracle implementing the RankedIndex contract.

Scores every stored item against the query by brute force, so it is the
reference the fuzz tests compare IVF, HNSW and the inverted index against.
The scorers are written independently from ``engine/`` on purpose: an oracle
that reuses the code under test cannot catch its bugs.
"""

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from engine.common.rid import RID
from engine.indexes.base import Query, RankedIndex

Scorer = Callable[..., float]


def vector_scorer(
    query: Sequence[float], item: Sequence[float], metric: str = "euclidean"
) -> float:
    """Score two vectors so that higher is better, for the three metrics of 2.4.2."""
    if len(query) != len(item):
        raise ValueError(f"dimension mismatch: {len(query)} != {len(item)}")
    dot = sum(a * b for a, b in zip(query, item, strict=True))
    if metric == "euclidean":
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(query, item, strict=True)))
        return 1.0 / (1.0 + distance)
    if metric == "inner_product":
        return dot
    if metric == "cosine":
        norms = math.sqrt(sum(a * a for a in query)) * math.sqrt(sum(b * b for b in item))
        return dot / norms if norms else 0.0
    raise ValueError(f"unknown metric {metric!r}")


def text_scorer(query: str, item: str, method: str = "tf_idf") -> float:
    """Count shared lowercase terms; enough to exercise the contract, not a real ranking."""
    if method not in ("tf_idf", "bm25"):
        raise ValueError(f"unknown method {method!r}")
    terms = set(query.lower().split())
    return float(sum(1 for word in item.lower().split() if word in terms))


class FakeRankedIndex(RankedIndex):
    """Brute-force ranked index; persists to a JSON file when given a path."""

    def __init__(self, scorer: Scorer = vector_scorer, path: Path | None = None) -> None:
        self._scorer = scorer
        self._path = Path(path) if path is not None else None
        self._items: list[tuple[Any, RID]] = []
        if self._path is not None and self._path.exists():
            stored = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = [(item, RID(*rid)) for item, rid in stored]

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: Any, rid: RID) -> None:
        self._items.append((item, rid))

    def top_k(self, query: Query, k: int, **opts: Any) -> list[tuple[RID, float]]:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        scored = [(rid, self._scorer(query, item, **opts)) for item, rid in self._items]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]

    def close(self) -> None:
        if self._path is None:
            return
        payload = [[item, list(rid)] for item, rid in self._items]
        self._path.write_text(json.dumps(payload), encoding="utf-8")
