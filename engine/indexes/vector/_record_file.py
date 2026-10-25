"""Array of fixed-size records over pages of a DiskManager.

Page 0 is a header (magic, record size, record count and a free area for the
owner's metadata); record ``i`` lives at page ``1 + i // per_page``. Fixed
size lets a record be addressed by arithmetic instead of a slot directory,
which is what vectors and HNSW adjacency lists need.
"""

import struct
from collections.abc import Iterator

from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

_HEADER = struct.Struct("<4sHIQ")
_VERSION = 1


def peek_meta(buffer_manager: BufferManager, magic: bytes, layout: struct.Struct) -> tuple:
    """Read the owner's metadata of an existing file before knowing its record size."""
    frame = buffer_manager.pin(0)
    try:
        stored_magic = _HEADER.unpack_from(frame, 0)[0]
        values = layout.unpack_from(frame, _HEADER.size)
    finally:
        buffer_manager.unpin(0)
    if stored_magic != magic:
        raise ValueError(f"not a {magic!r} file")
    return values


class FixedRecordFile:
    """Append-only array of fixed-size records, read and written in place."""

    def __init__(
        self,
        disk_manager: DiskManager,
        buffer_manager: BufferManager,
        *,
        record_size: int,
        magic: bytes,
    ) -> None:
        page_size = disk_manager.page_size
        if record_size <= 0 or record_size > page_size:
            raise ValueError(
                f"record of {record_size} bytes does not fit in a {page_size}-byte page"
            )
        if len(magic) != 4:
            raise ValueError("magic must be exactly 4 bytes")

        self._disk = disk_manager
        self._buffer = buffer_manager
        self._record_size = record_size
        self._per_page = page_size // record_size
        self._magic = magic

        if disk_manager.page_count == 0:
            disk_manager.allocate_page()
            self._count = 0
            self._write_header()
        else:
            frame = buffer_manager.pin(0)
            try:
                stored_magic, version, stored_size, count = _HEADER.unpack_from(frame, 0)
            finally:
                buffer_manager.unpin(0)
            if stored_magic != magic or version != _VERSION:
                raise ValueError(f"{disk_manager.path} is not a {magic!r} file")
            if stored_size != record_size:
                raise ValueError(f"record size on disk is {stored_size}, expected {record_size}")
            self._count = count

    def __len__(self) -> int:
        return self._count

    @property
    def per_page(self) -> int:
        return self._per_page

    @property
    def meta_capacity(self) -> int:
        return self._disk.page_size - _HEADER.size

    def read_meta(self, layout: struct.Struct) -> tuple:
        frame = self._buffer.pin(0)
        try:
            return layout.unpack_from(frame, _HEADER.size)
        finally:
            self._buffer.unpin(0)

    def write_meta(self, layout: struct.Struct, *values: object) -> None:
        if layout.size > self.meta_capacity:
            raise ValueError("metadata does not fit in the header page")
        frame = self._buffer.pin(0)
        layout.pack_into(frame, _HEADER.size, *values)
        self._buffer.unpin(0, dirty=True)

    def append(self, layout: struct.Struct, *values: object) -> int:
        index = self._count
        page_id, offset = self._locate(index)
        if page_id == self._disk.page_count:
            self._disk.allocate_page()
        frame = self._buffer.pin(page_id)
        layout.pack_into(frame, offset, *values)
        self._buffer.unpin(page_id, dirty=True)
        self._count += 1
        self._write_header()
        return index

    def unpack(self, index: int, layout: struct.Struct, extra: int = 0) -> tuple:
        """Unpack ``layout`` at byte ``extra`` of record ``index``."""
        page_id, offset = self._locate(self._check(index))
        frame = self._buffer.pin(page_id)
        try:
            return layout.unpack_from(frame, offset + extra)
        finally:
            self._buffer.unpin(page_id)

    def pack(self, index: int, layout: struct.Struct, *values: object, extra: int = 0) -> None:
        page_id, offset = self._locate(self._check(index))
        frame = self._buffer.pin(page_id)
        layout.pack_into(frame, offset + extra, *values)
        self._buffer.unpin(page_id, dirty=True)

    def scan(self, layout: struct.Struct) -> Iterator[tuple[int, tuple]]:
        """Yield (index, values) for every record, pinning each page once."""
        index = 0
        page_id = 1
        while index < self._count:
            frame = self._buffer.pin(page_id)
            try:
                in_page = min(self._per_page, self._count - index)
                for slot in range(in_page):
                    yield index + slot, layout.unpack_from(frame, slot * self._record_size)
            finally:
                self._buffer.unpin(page_id)
            index += in_page
            page_id += 1

    def flush(self) -> None:
        self._buffer.flush_all()

    def _locate(self, index: int) -> tuple[int, int]:
        return 1 + index // self._per_page, (index % self._per_page) * self._record_size

    def _check(self, index: int) -> int:
        if not 0 <= index < self._count:
            raise IndexError(f"record {index} out of range (count {self._count})")
        return index

    def _write_header(self) -> None:
        frame = self._buffer.pin(0)
        _HEADER.pack_into(frame, 0, self._magic, _VERSION, self._record_size, self._count)
        self._buffer.unpin(0, dirty=True)
