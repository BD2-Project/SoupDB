"""Inverted File index (IVF-Flat) persisted on pages.

Layout of the single file:

- page 0: header (dim, nlist, count and where the centroids and the list
  directory start).
- centroid pages: ``(norm, c_0 .. c_{K-1})`` per centroid.
- directory pages: ``(head, tail, size)`` of each inverted list.
- list pages: ``(next, count)`` followed by ``(rid, norm, vector)`` records.

The vectors live inside their list, so a query reads only the pages of the
``nprobe`` lists it visits. The centroids are kept in memory once loaded: they
are the coarse quantizer, ``nlist`` vectors, and every query needs all of them.
"""

import heapq
import random
import struct
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from engine.indexes.vector.constants import (
    DEFAULT_METRIC,
    IVF_KMEANS_ITERATIONS,
    IVF_NLIST,
    IVF_NPROBE,
    IVF_TRAIN_SAMPLE,
    VECTOR_BUFFER_CAPACITY,
)
from engine.indexes.vector.kmeans import kmeans, nearest, reservoir_sample
from engine.indexes.vector.metrics import norm, score_fn
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_MAGIC = b"SIVF"
_VERSION = 1
_HEADER = struct.Struct("<4sHIIQII")
_DIRECTORY_ENTRY = struct.Struct("<qqQ")
_LIST_HEADER = struct.Struct("<qH")
_NO_PAGE = -1

VectorSource = Iterable[tuple[Sequence[float], RID]]


class IVFIndex(RankedIndex):
    """IVF over pages: K-Means centroids plus one paged inverted list per centroid."""

    def __init__(
        self,
        disk_manager: DiskManager,
        buffer_manager: BufferManager,
        dim: int | None = None,
    ) -> None:
        self._disk = disk_manager
        self._buffer = buffer_manager
        self._page_size = disk_manager.page_size
        self._owns_managers = False
        self._closed = False

        if disk_manager.page_count == 0:
            if dim is None or dim <= 0:
                raise ValueError("a new IVF index needs a positive dim")
            self._init_layout(dim)
            self._nlist = 0
            self._count = 0
            self._centroid_start = 0
            self._directory_start = 0
            self._centroids: list[tuple[float, ...]] = []
            self._centroid_norms: list[float] = []
            self._disk.allocate_page()
            self._write_header()
        else:
            self._load(dim)

    @classmethod
    def open(
        cls,
        path: str | Path,
        dim: int | None = None,
        *,
        page_size: int = 4096,
        buffer_capacity: int = VECTOR_BUFFER_CAPACITY,
    ) -> "IVFIndex":
        """Open (or create) an index at path, owning its DiskManager and BufferManager."""
        disk_manager = DiskManager(path, page_size=page_size)
        buffer_manager = BufferManager(disk_manager, capacity=buffer_capacity)
        try:
            index = cls(disk_manager, buffer_manager, dim)
        except ValueError:
            disk_manager.close()
            raise
        index._owns_managers = True
        return index

    @classmethod
    def build(
        cls,
        path: str | Path,
        source: VectorSource,
        *,
        dim: int,
        nlist: int = IVF_NLIST,
        sample_size: int = IVF_TRAIN_SAMPLE,
        iterations: int = IVF_KMEANS_ITERATIONS,
        seed: int = 0,
        page_size: int = 4096,
        buffer_capacity: int = VECTOR_BUFFER_CAPACITY,
    ) -> "IVFIndex":
        """Train on a sample of ``source`` and insert all of it, in two streaming passes.

        ``source`` must be re-iterable (a list or an object whose ``__iter__``
        restarts the reading), not a one-shot generator.
        """
        if iter(source) is source:
            raise ValueError(
                "source must be re-iterable; a generator would be consumed by training"
            )
        index = cls.open(path, dim, page_size=page_size, buffer_capacity=buffer_capacity)
        rng = random.Random(seed)
        sample = reservoir_sample((vector for vector, _rid in source), sample_size, rng)
        index.train(sample, nlist=nlist, iterations=iterations, seed=seed)
        for vector, rid in source:
            index.insert(vector, rid)
        return index

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def nlist(self) -> int:
        return self._nlist

    @property
    def is_trained(self) -> bool:
        return self._nlist > 0

    @property
    def disk_manager(self) -> DiskManager:
        return self._disk

    def __len__(self) -> int:
        return self._count

    def list_sizes(self) -> list[int]:
        return [self._directory_entry(cluster)[2] for cluster in range(self._nlist)]

    def train(
        self,
        sample: Sequence[Sequence[float]],
        *,
        nlist: int = IVF_NLIST,
        iterations: int = IVF_KMEANS_ITERATIONS,
        seed: int = 0,
    ) -> None:
        """Compute the centroids and lay out their pages; only allowed once."""
        if self.is_trained:
            raise ValueError("IVF index is already trained")
        for vector in sample:
            self._check_dim(vector)
        centroids = kmeans(
            [tuple(vector) for vector in sample],
            nlist,
            iterations=iterations,
            rng=random.Random(seed),
        )
        # Se redondean a float32 aquí para que en memoria estén igual que al reabrir.
        self._centroids = [self._vector.unpack(self._vector.pack(*c)) for c in centroids]
        self._centroid_norms = [norm(c) for c in self._centroids]
        self._nlist = len(self._centroids)

        centroid_pages = -(-self._nlist // self._centroids_per_page)
        directory_pages = -(-self._nlist // self._entries_per_page)
        self._centroid_start = self._disk.page_count
        for _ in range(centroid_pages):
            self._disk.allocate_page()
        self._directory_start = self._disk.page_count
        for _ in range(directory_pages):
            self._disk.allocate_page()

        for cluster, centroid in enumerate(self._centroids):
            page_id, offset = self._centroid_slot(cluster)
            frame = self._buffer.pin(page_id)
            self._centroid.pack_into(frame, offset, self._centroid_norms[cluster], *centroid)
            self._buffer.unpin(page_id, dirty=True)
            self._write_directory_entry(cluster, _NO_PAGE, _NO_PAGE, 0)
        self._write_header()

    def insert(self, vector: Sequence[float], rid: RID) -> None:
        if not self.is_trained:
            raise ValueError("train the IVF index before inserting")
        self._check_dim(vector)
        rounded = self._vector.unpack(self._vector.pack(*vector))
        cluster = nearest(rounded, self._centroids)
        head, tail, size = self._directory_entry(cluster)

        if tail != _NO_PAGE:
            frame = self._buffer.pin(tail)
            next_page, count = _LIST_HEADER.unpack_from(frame, 0)
            if count < self._records_per_page:
                self._record.pack_into(
                    frame,
                    self._record_offset(count),
                    rid.page_id,
                    rid.slot,
                    norm(rounded),
                    *rounded,
                )
                _LIST_HEADER.pack_into(frame, 0, next_page, count + 1)
                self._buffer.unpin(tail, dirty=True)
                self._write_directory_entry(cluster, head, tail, size + 1)
                self._bump_count()
                return
            self._buffer.unpin(tail)

        new_page = self._disk.allocate_page()
        frame = self._buffer.pin(new_page)
        _LIST_HEADER.pack_into(frame, 0, _NO_PAGE, 1)
        self._record.pack_into(
            frame, self._record_offset(0), rid.page_id, rid.slot, norm(rounded), *rounded
        )
        self._buffer.unpin(new_page, dirty=True)
        if tail != _NO_PAGE:
            frame = self._buffer.pin(tail)
            count = _LIST_HEADER.unpack_from(frame, 0)[1]
            _LIST_HEADER.pack_into(frame, 0, new_page, count)
            self._buffer.unpin(tail, dirty=True)
        self._write_directory_entry(
            cluster, head if head != _NO_PAGE else new_page, new_page, size + 1
        )
        self._bump_count()

    def top_k(
        self,
        query: Sequence[float],
        k: int,
        *,
        metric: str = DEFAULT_METRIC,
        nprobe: int = IVF_NPROBE,
    ) -> list[tuple[RID, float]]:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if nprobe <= 0:
            raise ValueError(f"nprobe must be positive, got {nprobe}")
        self._check_dim(query)
        scorer = score_fn(metric)
        if k == 0 or not self.is_trained:
            return []

        query_norm = norm(query)
        # Los centroides se rankean con la misma métrica de la consulta: es la región
        # donde esa métrica espera encontrar a los vecinos.
        ranked_lists = heapq.nlargest(
            min(nprobe, self._nlist),
            range(self._nlist),
            key=lambda c: scorer(query, query_norm, self._centroids[c], self._centroid_norms[c]),
        )
        scored = (
            (scorer(query, query_norm, vector, vector_norm), rid)
            for cluster in ranked_lists
            for rid, vector, vector_norm in self._scan_list(cluster)
        )
        return [(rid, value) for value, rid in heapq.nlargest(k, scored, key=_by_score)]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer.flush_all()
        if self._owns_managers:
            self._disk.close()

    # --- internals ---------------------------------------------------------

    def _init_layout(self, dim: int) -> None:
        self._dim = dim
        self._vector = struct.Struct(f"<{dim}f")
        self._centroid = struct.Struct(f"<d{dim}f")
        self._record = struct.Struct(f"<qqd{dim}f")
        self._centroids_per_page = self._page_size // self._centroid.size
        self._entries_per_page = self._page_size // _DIRECTORY_ENTRY.size
        self._records_per_page = (self._page_size - _LIST_HEADER.size) // self._record.size
        if self._records_per_page == 0:
            raise ValueError(f"dim {dim} does not fit in a {self._page_size}-byte list page")

    def _load(self, dim: int | None) -> None:
        frame = self._buffer.pin(0)
        try:
            magic, version, stored_dim, nlist, count, centroid_start, directory_start = (
                _HEADER.unpack_from(frame, 0)
            )
        finally:
            self._buffer.unpin(0)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError(f"{self._disk.path} is not an IVF index")
        if dim is not None and dim != stored_dim:
            raise ValueError(f"index has dim {stored_dim}, requested {dim}")
        self._init_layout(stored_dim)
        self._nlist = nlist
        self._count = count
        self._centroid_start = centroid_start
        self._directory_start = directory_start
        self._centroids = []
        self._centroid_norms = []
        for cluster in range(nlist):
            page_id, offset = self._centroid_slot(cluster)
            frame = self._buffer.pin(page_id)
            values = self._centroid.unpack_from(frame, offset)
            self._buffer.unpin(page_id)
            self._centroid_norms.append(values[0])
            self._centroids.append(values[1:])

    def _write_header(self) -> None:
        frame = self._buffer.pin(0)
        _HEADER.pack_into(
            frame,
            0,
            _MAGIC,
            _VERSION,
            self._dim,
            self._nlist,
            self._count,
            self._centroid_start,
            self._directory_start,
        )
        self._buffer.unpin(0, dirty=True)

    def _bump_count(self) -> None:
        self._count += 1
        self._write_header()

    def _centroid_slot(self, cluster: int) -> tuple[int, int]:
        page = self._centroid_start + cluster // self._centroids_per_page
        return page, (cluster % self._centroids_per_page) * self._centroid.size

    def _directory_slot(self, cluster: int) -> tuple[int, int]:
        page = self._directory_start + cluster // self._entries_per_page
        return page, (cluster % self._entries_per_page) * _DIRECTORY_ENTRY.size

    def _directory_entry(self, cluster: int) -> tuple[int, int, int]:
        page_id, offset = self._directory_slot(cluster)
        frame = self._buffer.pin(page_id)
        try:
            return _DIRECTORY_ENTRY.unpack_from(frame, offset)
        finally:
            self._buffer.unpin(page_id)

    def _write_directory_entry(self, cluster: int, head: int, tail: int, size: int) -> None:
        page_id, offset = self._directory_slot(cluster)
        frame = self._buffer.pin(page_id)
        _DIRECTORY_ENTRY.pack_into(frame, offset, head, tail, size)
        self._buffer.unpin(page_id, dirty=True)

    def _record_offset(self, position: int) -> int:
        return _LIST_HEADER.size + position * self._record.size

    def _scan_list(self, cluster: int) -> Iterator[tuple[RID, tuple[float, ...], float]]:
        page_id = self._directory_entry(cluster)[0]
        while page_id != _NO_PAGE:
            frame = self._buffer.pin(page_id)
            try:
                next_page, count = _LIST_HEADER.unpack_from(frame, 0)
                for position in range(count):
                    values = self._record.unpack_from(frame, self._record_offset(position))
                    yield RID(values[0], values[1]), values[3:], values[2]
            finally:
                self._buffer.unpin(page_id)
            page_id = next_page

    def _check_dim(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dim:
            raise ValueError(f"expected a vector of dim {self._dim}, got {len(vector)}")


def _by_score(pair: tuple[float, RID]) -> float:
    return pair[0]
