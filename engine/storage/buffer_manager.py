"""Buffer pool with pin/unpin and LRU policy."""

from collections import OrderedDict

from engine.storage.disk_manager import DiskManager


class BufferManager:
    """Fixed-capacity page cache in front of a DiskManager, with LRU eviction."""

    def __init__(self, disk_manager: DiskManager, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self._disk_manager = disk_manager
        self._capacity = capacity
        self._frames: OrderedDict[int, bytearray] = OrderedDict()
        self._pin_counts: dict[int, int] = {}
        self._dirty: set[int] = set()

    @property
    def capacity(self) -> int:
        return self._capacity

    def pin(self, page_id: int) -> bytearray:
        """Load a page into the pool (if needed), pin it, and return its frame."""
        if page_id in self._frames:
            self._frames.move_to_end(page_id)
        else:
            victim = None
            if len(self._frames) >= self._capacity:
                victim = self._select_victim()

            data = self._disk_manager.read_page(page_id)

            if victim is not None:
                self.flush_page(victim)
                del self._frames[victim]
            self._frames[page_id] = bytearray(data)

        self._pin_counts[page_id] = self._pin_counts.get(page_id, 0) + 1
        return self._frames[page_id]

    def unpin(self, page_id: int, dirty: bool = False) -> None:
        """Release one pin on a page, optionally flagging it as dirty."""
        if self._pin_counts.get(page_id, 0) <= 0:
            raise ValueError(f"page_id {page_id} is not pinned")

        if dirty:
            self._dirty.add(page_id)

        self._pin_counts[page_id] -= 1
        if self._pin_counts[page_id] == 0:
            del self._pin_counts[page_id]

    def is_pinned(self, page_id: int) -> bool:
        return self._pin_counts.get(page_id, 0) > 0

    def flush_page(self, page_id: int) -> None:
        """Write a page back to disk if it is dirty; a no-op otherwise."""
        if page_id not in self._dirty:
            return

        self._disk_manager.write_page(page_id, bytes(self._frames[page_id]))
        self._dirty.discard(page_id)

    def flush_all(self) -> None:
        """Flush every currently dirty page."""
        for page_id in list(self._dirty):
            self.flush_page(page_id)

    def _select_victim(self) -> int:
        """Pick the least-recently-used unpinned page id, without touching the cache."""
        for candidate in self._frames:
            if self._pin_counts.get(candidate, 0) == 0:
                return candidate
        raise RuntimeError("cannot evict: every page in the buffer pool is pinned")
