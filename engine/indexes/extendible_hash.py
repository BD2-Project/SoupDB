"""Disk-backed extendible hashing index."""

import hashlib
import struct
from pathlib import Path

from engine.common.errors import UnsupportedOperation
from engine.common.rid import RID
from engine.indexes import _hash_page as codec
from engine.indexes.base import Index, Key
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

#: Tope de duplicaciones del directorio; al alcanzarlo se encadenan buckets de overflow.
HASH_GLOBAL_DEPTH_MAX = 16

_INT_TAG = 0
_STR_TAG = 1
_INT_FORMAT = "<q"
_MASK64 = (1 << 64) - 1


def _encode_key(key: Key) -> bytes:
    """Encode a key as tag + payload so int and str keys can share one index."""
    if isinstance(key, bool):
        raise ValueError("bool is not a valid key type")
    if isinstance(key, int):
        return bytes([_INT_TAG]) + struct.pack(_INT_FORMAT, key)
    if isinstance(key, str):
        return bytes([_STR_TAG]) + key.encode("utf-8")
    raise ValueError(f"unsupported key type {type(key).__name__}")


def _hash_encoded(encoded: bytes) -> int:
    """Hash an encoded key; its low bits are the bucket label read as a suffix."""
    if encoded[0] == _INT_TAG:
        (value,) = struct.unpack_from(_INT_FORMAT, encoded, 1)
        return value & _MASK64
    # blake2b y no hash(): hash() cambia entre procesos y rompería la persistencia.
    return int.from_bytes(hashlib.blake2b(encoded[1:], digest_size=8).digest(), "little")


class ExtendibleHash(Index):
    """Extendible hashing over pages, with the directory and buckets kept on disk."""

    def __init__(
        self,
        disk_manager: DiskManager,
        buffer_manager: BufferManager,
        *,
        max_global_depth: int = HASH_GLOBAL_DEPTH_MAX,
    ) -> None:
        self._disk_manager = disk_manager
        self._buffer_manager = buffer_manager
        self._owns_managers = False
        self._closed = False
        self._page_size = disk_manager.page_size
        self._slots_per_page = codec.directory_slots_per_page(self._page_size)

        if disk_manager.page_count == 0:
            self._create(max_global_depth)
        else:
            self._load()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        page_size: int = 4096,
        buffer_capacity: int = 64,
        max_global_depth: int = HASH_GLOBAL_DEPTH_MAX,
    ) -> "ExtendibleHash":
        """Open (or create) an index at path, owning its DiskManager and BufferManager."""
        disk_manager = DiskManager(path, page_size=page_size)
        buffer_manager = BufferManager(disk_manager, capacity=buffer_capacity)
        index = cls(disk_manager, buffer_manager, max_global_depth=max_global_depth)
        index._owns_managers = True
        return index

    @property
    def supports_range(self) -> bool:
        return False

    @property
    def global_depth(self) -> int:
        return self._global_depth

    def insert(self, key: Key, rid: RID) -> None:
        encoded = _encode_key(key)
        if len(encoded) > codec.max_key_size(self._page_size):
            raise ValueError(
                f"key of {len(encoded)} bytes does not fit in a page of {self._page_size} bytes"
            )
        digest = _hash_encoded(encoded)

        while True:
            slot = digest & self._directory_mask
            bucket_id = self._directory_slot(slot)

            frame = self._buffer_manager.pin(bucket_id)
            added = codec.add_entry(frame, encoded, rid)
            local_depth = codec.local_depth(frame)
            self._buffer_manager.unpin(bucket_id, dirty=added)
            if added:
                return

            # Sin margen para dividir: solo queda encadenar overflow.
            if local_depth >= self._max_depth:
                self._insert_into_overflow(bucket_id, encoded, rid)
                return

            self._split_bucket(slot, bucket_id, local_depth)

    def search(self, key: Key) -> list[RID]:
        encoded = _encode_key(key)
        rids: list[RID] = []
        for page_id in self._chain(self._bucket_for(encoded)):
            frame = self._buffer_manager.pin(page_id)
            try:
                rids.extend(rid for stored, rid in codec.iter_entries(frame) if stored == encoded)
            finally:
                self._buffer_manager.unpin(page_id, dirty=False)
        return rids

    def range_search(self, lo: Key, hi: Key) -> list[RID]:
        # El planner debe elegir B+ o secuencial para un BETWEEN: simular un scan aquí
        # escondería la diferencia que mide la comparación experimental.
        raise UnsupportedOperation("ExtendibleHash does not support range searches")

    def remove(self, key: Key, rid: RID | None = None) -> int:
        encoded = _encode_key(key)
        removed = 0
        for page_id in self._chain(self._bucket_for(encoded)):
            frame = self._buffer_manager.pin(page_id)
            page_removed = codec.remove_entries(frame, encoded, rid)
            self._buffer_manager.unpin(page_id, dirty=page_removed > 0)
            removed += page_removed
        return removed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._buffer_manager.flush_all()
        if self._owns_managers:
            self._disk_manager.close()

    # --- internals ---------------------------------------------------------

    @property
    def _directory_mask(self) -> int:
        return (1 << self._global_depth) - 1

    def _create(self, max_global_depth: int) -> None:
        if max_global_depth < 0:
            raise ValueError(f"max_global_depth must be non-negative, got {max_global_depth}")

        self._global_depth = 0
        self._max_depth = max_global_depth

        metadata_id = self._disk_manager.allocate_page()
        frame = self._buffer_manager.pin(metadata_id)
        frame[:] = codec.new_metadata(self._page_size, global_depth=0, max_depth=max_global_depth)
        self._buffer_manager.unpin(metadata_id, dirty=True)

        self._directory_pages = [self._new_directory_page()]
        self._write_directory_slot(0, self._new_bucket(local_depth=0))
        self._flush_metadata()

    def _load(self) -> None:
        frame = self._buffer_manager.pin(0)
        try:
            self._global_depth, self._max_depth, self._directory_pages = codec.read_metadata(frame)
        finally:
            self._buffer_manager.unpin(0, dirty=False)

    def _flush_metadata(self) -> None:
        frame = self._buffer_manager.pin(0)
        codec.write_metadata(
            frame,
            global_depth=self._global_depth,
            max_depth=self._max_depth,
            directory_pages=self._directory_pages,
        )
        self._buffer_manager.unpin(0, dirty=True)

    def _new_directory_page(self) -> int:
        page_id = self._disk_manager.allocate_page()
        frame = self._buffer_manager.pin(page_id)
        frame[:] = bytearray(self._page_size)
        self._buffer_manager.unpin(page_id, dirty=True)
        return page_id

    def _new_bucket(self, *, local_depth: int) -> int:
        page_id = self._disk_manager.allocate_page()
        frame = self._buffer_manager.pin(page_id)
        frame[:] = codec.new_bucket(self._page_size, local_depth=local_depth)
        self._buffer_manager.unpin(page_id, dirty=True)
        return page_id

    def _directory_slot(self, slot: int) -> int:
        page_id = self._directory_pages[slot // self._slots_per_page]
        frame = self._buffer_manager.pin(page_id)
        try:
            return codec.read_directory_slot(frame, slot % self._slots_per_page)
        finally:
            self._buffer_manager.unpin(page_id, dirty=False)

    def _write_directory_slot(self, slot: int, bucket_id: int) -> None:
        page_id = self._directory_pages[slot // self._slots_per_page]
        frame = self._buffer_manager.pin(page_id)
        codec.write_directory_slot(frame, slot % self._slots_per_page, bucket_id)
        self._buffer_manager.unpin(page_id, dirty=True)

    def _bucket_for(self, encoded: bytes) -> int:
        return self._directory_slot(_hash_encoded(encoded) & self._directory_mask)

    def _chain(self, bucket_id: int) -> list[int]:
        """Bucket page ids of a bucket, following its overflow chain."""
        chain = []
        page_id = bucket_id
        while page_id != codec.NO_OVERFLOW:
            chain.append(page_id)
            frame = self._buffer_manager.pin(page_id)
            try:
                next_id = codec.overflow_page_id(frame)
            finally:
                self._buffer_manager.unpin(page_id, dirty=False)
            page_id = next_id
        return chain

    def _split_bucket(self, slot: int, bucket_id: int, local_depth: int) -> None:
        if local_depth == self._global_depth:
            self._double_directory()

        new_bucket_id = self._new_bucket(local_depth=local_depth + 1)

        frame = self._buffer_manager.pin(bucket_id)
        stay: list[tuple[bytes, RID]] = []
        move: list[tuple[bytes, RID]] = []
        for stored_key, stored_rid in codec.iter_entries(frame):
            moves = (_hash_encoded(stored_key) >> local_depth) & 1
            (move if moves else stay).append((stored_key, stored_rid))
        codec.write_entries(frame, stay)
        codec.set_local_depth(frame, local_depth + 1)
        self._buffer_manager.unpin(bucket_id, dirty=True)

        new_frame = self._buffer_manager.pin(new_bucket_id)
        codec.write_entries(new_frame, move)
        self._buffer_manager.unpin(new_bucket_id, dirty=True)

        # Convención del curso: la etiqueta es sufijo; el bucket original conserva el 0
        # antepuesto y el nuevo recibe el 1, o sea las entradas con ese bit en 1.
        suffix = slot & ((1 << local_depth) - 1)
        first = suffix | (1 << local_depth)
        step = 1 << (local_depth + 1)
        for directory_slot in range(first, 1 << self._global_depth, step):
            self._write_directory_slot(directory_slot, new_bucket_id)

    def _double_directory(self) -> None:
        old_size = 1 << self._global_depth
        needed_pages = -(-(old_size * 2) // self._slots_per_page)
        if needed_pages > codec.max_directory_pages(self._page_size):
            raise ValueError("directory does not fit in the metadata page")

        while len(self._directory_pages) < needed_pages:
            self._directory_pages.append(self._new_directory_page())

        # Cada entrada nueva apunta al mismo bucket que su gemela por sufijo.
        for directory_slot in range(old_size):
            self._write_directory_slot(
                directory_slot + old_size, self._directory_slot(directory_slot)
            )

        self._global_depth += 1
        self._flush_metadata()

    def _insert_into_overflow(self, bucket_id: int, encoded: bytes, rid: RID) -> None:
        chain = self._chain(bucket_id)
        last_id = chain[-1]

        frame = self._buffer_manager.pin(last_id)
        added = codec.add_entry(frame, encoded, rid)
        self._buffer_manager.unpin(last_id, dirty=added)
        if added:
            return

        overflow_id = self._new_bucket(local_depth=self._max_depth)
        overflow_frame = self._buffer_manager.pin(overflow_id)
        codec.add_entry(overflow_frame, encoded, rid)
        self._buffer_manager.unpin(overflow_id, dirty=True)

        frame = self._buffer_manager.pin(last_id)
        codec.set_overflow_page_id(frame, overflow_id)
        self._buffer_manager.unpin(last_id, dirty=True)
