"""Sole gateway to the filesystem, tracks reads and writes."""

import os
from pathlib import Path


class DiskManager:
    """Manages a single database file at page granularity."""

    def __init__(self, path: str | Path, page_size: int) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")

        self._page_size = page_size
        self.reads = 0
        self.writes = 0

        self._path = Path(path)
        self._path.touch(exist_ok=True)
        self._file = open(self._path, "r+b")

        self._file.seek(0, os.SEEK_END)
        size = self._file.tell()
        if size % page_size != 0:
            self._file.close()
            raise ValueError(f"file size {size} is not a multiple of page_size {page_size}")

        self._next_page_id = size // page_size

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def page_count(self) -> int:
        return self._next_page_id

    def _validate_page_id(self, page_id: int) -> None:
        if page_id < 0 or page_id >= self._next_page_id:
            raise ValueError(f"page_id {page_id} does not reference an allocated page")

    def allocate_page(self) -> int:
        """Physically extend the file by one zero-filled page."""
        page_id = self._next_page_id

        self._file.seek(0, os.SEEK_END)
        self._file.write(bytes(self._page_size))
        self._file.flush()

        self.writes += 1
        self._next_page_id += 1
        return page_id

    def write_page(self, page_id: int, data: bytes) -> None:
        """Overwrite an existing page with exactly page_size bytes."""
        self._validate_page_id(page_id)
        if len(data) != self._page_size:
            raise ValueError(f"data must be exactly {self._page_size} bytes, got {len(data)}")

        self._file.seek(page_id * self._page_size)
        self._file.write(data)
        self._file.flush()

        self.writes += 1

    def read_page(self, page_id: int) -> bytes:
        """Read exactly one page's worth of bytes from disk."""
        self._validate_page_id(page_id)

        self._file.seek(page_id * self._page_size)
        data = self._file.read(self._page_size)
        if len(data) != self._page_size:
            raise ValueError(f"short read: expected {self._page_size} bytes, got {len(data)}")

        self.reads += 1
        return data

    def close(self) -> None:
        """Close the managed file. Safe to call more than once."""
        self._file.close()
