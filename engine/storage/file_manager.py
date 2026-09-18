"""Wrapper over storage managers: one physical file per logical name.

A :class:`FileManager` owns a root directory and opens or creates a
``(DiskManager, BufferManager)`` pair per logical file name. Callers ask for a
name and receive the cached managers; closing the manager flushes every file.
"""

from pathlib import Path

from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager


class FileManager:
    """Opens or creates one (DiskManager, BufferManager) pair per file name."""

    def __init__(
        self,
        root: Path,
        page_size: int = 4096,
        buffer_capacity: int = 16,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._page_size = page_size
        self._buffer_capacity = buffer_capacity
        self._storages: dict[str, tuple[DiskManager, BufferManager]] = {}

    @property
    def page_size(self) -> int:
        return self._page_size

    def storage(self, file_id: str) -> tuple[DiskManager, BufferManager]:
        """Return (DiskManager, BufferManager) for a logical file, cached per name."""
        cached = self._storages.get(file_id)
        if cached is not None:
            return cached
        disk_manager = DiskManager(self._root / f"{file_id}.db", self._page_size)
        buffer_manager = BufferManager(disk_manager, capacity=self._buffer_capacity)
        self._storages[file_id] = (disk_manager, buffer_manager)
        return disk_manager, buffer_manager

    def drop(self, file_id: str) -> None:
        """Close and delete a logical file, forgetting the cached managers."""
        storage = self._storages.pop(file_id, None)
        if storage is None:
            return
        disk_manager, buffer_manager = storage
        buffer_manager.flush_all()
        disk_manager.delete()

    def close(self) -> None:
        """Flush and close every open file. Safe to call more than once."""
        for disk_manager, buffer_manager in self._storages.values():
            buffer_manager.flush_all()
            disk_manager.close()
