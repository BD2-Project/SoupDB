"""Paged storage of float32 vectors with the RID they point to.

Each record is ``(rid.page_id, rid.slot, norm, v_0 .. v_{K-1})``. The norm is
stored so cosine needs a single dot product per candidate. Vectors are
addressed by a dense internal id (``vid``) that IVF and HNSW use as pointer.
"""

import struct
from collections.abc import Iterator, Sequence
from pathlib import Path

from engine.common.rid import RID
from engine.indexes.vector._record_file import FixedRecordFile, peek_meta
from engine.indexes.vector.constants import VECTOR_BUFFER_CAPACITY
from engine.indexes.vector.metrics import norm
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_MAGIC = b"SVEC"
_HEAD = struct.Struct("<qqd")
_META = struct.Struct("<I")


def record_size(dim: int) -> int:
    return _HEAD.size + 4 * dim


def max_dim(page_size: int) -> int:
    """Largest dimension whose record fits in one page."""
    return (page_size - _HEAD.size) // 4


class VectorStore:
    """Append-only array of vectors on pages, read through the buffer pool."""

    def __init__(
        self,
        disk_manager: DiskManager,
        buffer_manager: BufferManager,
        dim: int | None = None,
    ) -> None:
        self._disk = disk_manager
        self._owns_managers = False
        self._closed = False

        if disk_manager.page_count == 0:
            if dim is None or dim <= 0:
                raise ValueError("a new vector store needs a positive dim")
        else:
            dim = self._stored_dim(buffer_manager, dim)
        if dim > max_dim(disk_manager.page_size):
            raise ValueError(
                f"dim {dim} does not fit in a {disk_manager.page_size}-byte page "
                f"(max {max_dim(disk_manager.page_size)})"
            )

        self._dim = dim
        self._vector = struct.Struct(f"<{dim}f")
        self._record = struct.Struct(f"<qqd{dim}f")
        self._file = FixedRecordFile(
            disk_manager, buffer_manager, record_size=record_size(dim), magic=_MAGIC
        )
        self._file.write_meta(_META, dim)

    @classmethod
    def open(
        cls,
        path: str | Path,
        dim: int | None = None,
        *,
        page_size: int = 4096,
        buffer_capacity: int = VECTOR_BUFFER_CAPACITY,
    ) -> "VectorStore":
        """Open (or create) a store at path, owning its DiskManager and BufferManager."""
        disk_manager = DiskManager(path, page_size=page_size)
        buffer_manager = BufferManager(disk_manager, capacity=buffer_capacity)
        try:
            store = cls(disk_manager, buffer_manager, dim)
        except ValueError:
            disk_manager.close()
            raise
        store._owns_managers = True
        return store

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def disk_manager(self) -> DiskManager:
        return self._disk

    def __len__(self) -> int:
        return len(self._file)

    def append(self, vector: Sequence[float], rid: RID) -> int:
        """Store a vector and return its vid."""
        self._check_dim(vector)
        # La norma sale del vector ya redondeado a float32, tal como lo leerá la búsqueda.
        rounded = self._vector.unpack(self._vector.pack(*vector))
        return self._file.append(self._record, rid.page_id, rid.slot, norm(rounded), *rounded)

    def rid(self, vid: int) -> RID:
        page_id, slot, _norm = self._file.unpack(vid, _HEAD)
        return RID(page_id, slot)

    def vector(self, vid: int) -> tuple[tuple[float, ...], float]:
        """Return (vector, norm) of a vid."""
        values = self._file.unpack(vid, self._record)
        return values[3:], values[2]

    def get(self, vid: int) -> tuple[RID, tuple[float, ...], float]:
        values = self._file.unpack(vid, self._record)
        return RID(values[0], values[1]), values[3:], values[2]

    def scan(self) -> Iterator[tuple[int, RID, tuple[float, ...], float]]:
        """Yield (vid, rid, vector, norm) in vid order, one page pin per page."""
        for vid, values in self._file.scan(self._record):
            yield vid, RID(values[0], values[1]), values[3:], values[2]

    def check_dim(self, vector: Sequence[float]) -> None:
        self._check_dim(vector)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._file.flush()
        if self._owns_managers:
            self._disk.close()

    def _check_dim(self, vector: Sequence[float]) -> None:
        if len(vector) != self._dim:
            raise ValueError(f"expected a vector of dim {self._dim}, got {len(vector)}")

    @staticmethod
    def _stored_dim(buffer_manager: BufferManager, requested: int | None) -> int:
        (stored,) = peek_meta(buffer_manager, _MAGIC, _META)
        if requested is not None and requested != stored:
            raise ValueError(f"store has dim {stored}, requested {requested}")
        return stored
