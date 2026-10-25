"""Hierarchical Navigable Small World graph persisted on pages (Malkov & Yashunin).

The index is a directory with three files:

- ``vectors.db``: a :class:`VectorStore`; the vid of a vector is its node id.
- ``layer0.db``: one fixed record per node with its top level, where its upper
  layers start in ``upper.db`` and its up to ``2*M`` neighbors in layer 0.
- ``upper.db``: one fixed record per (node, layer >= 1) with up to ``M``
  neighbors. A node of level ``L`` owns ``L`` consecutive records.

Fixed-size records make every neighbor list addressable by arithmetic, so a
hop of the greedy search costs one page pin.
"""

import heapq
import math
import random
import struct
from collections.abc import Sequence
from pathlib import Path

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from engine.indexes.vector._record_file import FixedRecordFile, peek_meta
from engine.indexes.vector.constants import (
    DEFAULT_METRIC,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    HNSW_MAX_LEVEL,
    VECTOR_BUFFER_CAPACITY,
)
from engine.indexes.vector.metrics import METRICS, ScoreFn, norm, score_fn
from engine.indexes.vector.store import VectorStore
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_LAYER0_MAGIC = b"SHN0"
_UPPER_MAGIC = b"SHNU"
#: M, ef_construction, metric, seed, entry point, max level.
_META = struct.Struct("<HHBQqi")
_NODE_HEAD = struct.Struct("<BqH")
_UPPER_HEAD = struct.Struct("<H")
_NO_NODE = -1


class HNSWIndex(RankedIndex):
    """HNSW over pages; every node, vector and neighbor list lives on disk."""

    def __init__(
        self,
        path: str | Path,
        dim: int | None = None,
        *,
        m: int = HNSW_M,
        ef_construction: int = HNSW_EF_CONSTRUCTION,
        metric: str = DEFAULT_METRIC,
        seed: int = 0,
        page_size: int = 4096,
        buffer_capacity: int = VECTOR_BUFFER_CAPACITY,
    ) -> None:
        """Open (or create) the index stored in the directory ``path``.

        ``m``, ``ef_construction``, ``metric`` and ``seed`` only apply when the
        index is created; reopening uses the values stored on disk.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._disks: list[DiskManager] = []

        vectors_disk = self._disk(directory / "vectors.db", page_size)
        self._store = VectorStore(
            vectors_disk, BufferManager(vectors_disk, capacity=buffer_capacity), dim
        )
        layer0_disk = self._disk(directory / "layer0.db", page_size)
        layer0_buffer = BufferManager(layer0_disk, capacity=buffer_capacity)

        if layer0_disk.page_count > 0:
            m, ef_construction, metric_code, seed, entry, max_level = peek_meta(
                layer0_buffer, _LAYER0_MAGIC, _META
            )
            metric = METRICS[metric_code]
        else:
            if m < 2:
                raise ValueError(f"m must be at least 2, got {m}")
            score_fn(metric)
            entry, max_level = _NO_NODE, -1

        self._m = m
        self._m0 = 2 * m
        self._ef_construction = ef_construction
        self._metric = metric
        self._seed = seed
        self._entry = entry
        self._max_level = max_level
        self._insert_cache: dict[int, tuple[tuple[float, ...], float]] | None = None
        # mL = 1/ln(M): así cada nivel tiene ~1/M de los nodos del nivel de abajo.
        self._level_mult = 1.0 / math.log(m)

        self._layer0_record = struct.Struct(f"{_NODE_HEAD.format}{self._m0}q")
        self._upper_record = struct.Struct(f"{_UPPER_HEAD.format}{m}q")
        self._layer0 = FixedRecordFile(
            layer0_disk,
            layer0_buffer,
            record_size=self._layer0_record.size,
            magic=_LAYER0_MAGIC,
        )
        upper_disk = self._disk(directory / "upper.db", page_size)
        self._upper = FixedRecordFile(
            upper_disk,
            BufferManager(upper_disk, capacity=buffer_capacity),
            record_size=self._upper_record.size,
            magic=_UPPER_MAGIC,
        )
        self._write_meta()

    @property
    def dim(self) -> int:
        return self._store.dim

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def max_level(self) -> int:
        return self._max_level

    @property
    def disk_managers(self) -> tuple[DiskManager, ...]:
        return tuple(self._disks)

    def __len__(self) -> int:
        return len(self._store)

    def insert(self, vector: Sequence[float], rid: RID) -> None:
        # Una inserción vuelve a puntuar los mismos vecinos al elegir y al podar listas;
        # la caché vive solo durante esta inserción, así que la memoria no crece con N.
        self._insert_cache = {}
        try:
            self._insert(vector, rid)
        finally:
            self._insert_cache = None

    def _insert(self, vector: Sequence[float], rid: RID) -> None:
        node = self._store.append(vector, rid)
        query, query_norm = self._vector(node)
        level = self._random_level(node)

        upper_start = len(self._upper)
        for _ in range(level):
            self._upper.append(self._upper_record, 0, *self._padded([], self._m))
        self._layer0.append(self._layer0_record, level, upper_start, 0, *self._padded([], self._m0))

        if self._entry == _NO_NODE:
            self._entry, self._max_level = node, level
            self._write_meta()
            return

        scorer = score_fn(self._metric)
        entry_points = [(self._score(scorer, query, query_norm, self._entry), self._entry)]
        for layer in range(self._max_level, level, -1):
            entry_points = self._search_layer(scorer, query, query_norm, entry_points, 1, layer)

        for layer in range(min(level, self._max_level), -1, -1):
            candidates = self._search_layer(
                scorer, query, query_norm, entry_points, self._ef_construction, layer
            )
            capacity = self._capacity(layer)
            chosen = self._select_neighbors(scorer, candidates, capacity)
            self._write_neighbors(node, layer, [neighbor for _s, neighbor in chosen])
            for link_score, neighbor in chosen:
                self._link(scorer, neighbor, node, link_score, layer)
            entry_points = candidates

        if level > self._max_level:
            self._entry, self._max_level = node, level
            self._write_meta()

    def top_k(
        self,
        query: Sequence[float],
        k: int,
        *,
        metric: str | None = None,
        ef: int = HNSW_EF_SEARCH,
    ) -> list[tuple[RID, float]]:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if ef <= 0:
            raise ValueError(f"ef must be positive, got {ef}")
        self._store.check_dim(query)
        scorer = score_fn(metric or self._metric)
        if k == 0 or self._entry == _NO_NODE:
            return []

        query_norm = norm(query)
        entry_points = [(self._score(scorer, query, query_norm, self._entry), self._entry)]
        for layer in range(self._max_level, 0, -1):
            entry_points = self._search_layer(scorer, query, query_norm, entry_points, 1, layer)
        found = self._search_layer(scorer, query, query_norm, entry_points, max(ef, k), 0)
        return [(self._store.rid(node), value) for value, node in found[:k]]

    def neighbors(self, node: int, layer: int) -> list[int]:
        """Neighbor ids of a node in a layer; exposed for tests and the report."""
        if layer == 0:
            values = self._layer0.unpack(node, self._layer0_record)
            return list(values[3 : 3 + values[2]])
        values = self._upper.unpack(self._upper_index(node, layer), self._upper_record)
        return list(values[1 : 1 + values[0]])

    def level(self, node: int) -> int:
        return self._layer0.unpack(node, _NODE_HEAD)[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._write_meta()
        self._store.flush()
        self._layer0.flush()
        self._upper.flush()
        for disk in self._disks:
            disk.close()

    # --- internals ---------------------------------------------------------

    def _disk(self, path: Path, page_size: int) -> DiskManager:
        disk = DiskManager(path, page_size=page_size)
        self._disks.append(disk)
        return disk

    def _write_meta(self) -> None:
        self._layer0.write_meta(
            _META,
            self._m,
            self._ef_construction,
            METRICS.index(self._metric),
            self._seed,
            self._entry,
            self._max_level,
        )

    def _random_level(self, node: int) -> int:
        # Semilla derivada del nodo: el nivel es reproducible aunque se cierre y reabra.
        uniform = random.Random(self._seed * 1_000_003 + node).random()
        return min(int(-math.log(1.0 - uniform) * self._level_mult), HNSW_MAX_LEVEL)

    def _capacity(self, layer: int) -> int:
        return self._m0 if layer == 0 else self._m

    def _upper_index(self, node: int, layer: int) -> int:
        level, start, _count = self._layer0.unpack(node, _NODE_HEAD)
        if not 1 <= layer <= level:
            raise ValueError(f"node {node} has no layer {layer}")
        return start + layer - 1

    def _padded(self, neighbors: list[int], capacity: int) -> list[int]:
        return neighbors + [_NO_NODE] * (capacity - len(neighbors))

    def _write_neighbors(self, node: int, layer: int, neighbors: list[int]) -> None:
        if layer == 0:
            level, start, _count = self._layer0.unpack(node, _NODE_HEAD)
            self._layer0.pack(
                node,
                self._layer0_record,
                level,
                start,
                len(neighbors),
                *self._padded(neighbors, self._m0),
            )
            return
        record = self._upper_index(node, layer)
        self._upper.pack(
            record, self._upper_record, len(neighbors), *self._padded(neighbors, self._m)
        )

    def _vector(self, node: int) -> tuple[tuple[float, ...], float]:
        cache = self._insert_cache
        if cache is None:
            return self._store.vector(node)
        cached = cache.get(node)
        if cached is None:
            cached = cache[node] = self._store.vector(node)
        return cached

    def _score(
        self, scorer: ScoreFn, query: Sequence[float], query_norm: float, node: int
    ) -> float:
        vector, vector_norm = self._vector(node)
        return scorer(query, query_norm, vector, vector_norm)

    def _search_layer(
        self,
        scorer: ScoreFn,
        query: Sequence[float],
        query_norm: float,
        entry_points: list[tuple[float, int]],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Best-first search in one layer; returns up to ef (score, node), best first."""
        visited = {node for _score, node in entry_points}
        # candidates: max-heap por score (negado); results: min-heap con los ef mejores.
        candidates = [(-value, node) for value, node in entry_points]
        heapq.heapify(candidates)
        results = list(entry_points)
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            negated, node = heapq.heappop(candidates)
            if -negated < results[0][0] and len(results) >= ef:
                break
            for neighbor in self.neighbors(node, layer):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                value = self._score(scorer, query, query_norm, neighbor)
                if len(results) < ef or value > results[0][0]:
                    heapq.heappush(candidates, (-value, neighbor))
                    heapq.heappush(results, (value, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted(results, reverse=True)

    def _select_neighbors(
        self, scorer: ScoreFn, candidates: list[tuple[float, int]], capacity: int
    ) -> list[tuple[float, int]]:
        """Neighbor-selection heuristic (algorithm 4) with keepPrunedConnections.

        A candidate is kept only if it is closer to the base node than to every
        neighbor already kept; this favors links in different directions, which
        is what keeps clustered data navigable. Pruned ones fill the remaining slots.
        """
        if len(candidates) <= capacity:
            return list(candidates)
        kept: list[tuple[float, int]] = []
        kept_vectors: list[tuple[Sequence[float], float]] = []
        pruned: list[tuple[float, int]] = []
        for value, node in candidates:
            if len(kept) >= capacity:
                break
            vector, vector_norm = self._vector(node)
            if all(scorer(vector, vector_norm, o, o_norm) < value for o, o_norm in kept_vectors):
                kept.append((value, node))
                kept_vectors.append((vector, vector_norm))
            else:
                pruned.append((value, node))
        for pair in pruned:
            if len(kept) >= capacity:
                break
            kept.append(pair)
        return kept

    def _link(
        self, scorer: ScoreFn, node: int, new_neighbor: int, link_score: float, layer: int
    ) -> None:
        """Add a back-link node -> new_neighbor, shrinking the list if it overflows."""
        current = self.neighbors(node, layer)
        capacity = self._capacity(layer)
        if len(current) < capacity:
            self._write_neighbors(node, layer, current + [new_neighbor])
            return
        base, base_norm = self._vector(node)
        scored = [(self._score(scorer, base, base_norm, other), other) for other in current]
        scored.append((link_score, new_neighbor))
        scored.sort(reverse=True)
        chosen = self._select_neighbors(scorer, scored, capacity)
        self._write_neighbors(node, layer, [other for _s, other in chosen])
