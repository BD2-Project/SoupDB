"""Heap file organization with free-space reuse."""

import heapq
from collections.abc import Iterator

from engine.common.record import Record
from engine.common.rid import RID
from engine.storage import _slotted_page as codec
from engine.storage.base import FileOrganization
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager


class HeapFile(FileOrganization):
    """Heap file with free-space reuse and physical page/slot scanning."""

    def __init__(self, disk_manager: DiskManager, buffer_manager: BufferManager) -> None:
        self._disk_manager = disk_manager
        self._buffer_manager = buffer_manager
        self._page_capacity: dict[int, int] = {}
        self._capacity_heap: list[tuple[int, int]] = []
        self._unmeasured_pages: set[int] = set()

        if disk_manager.page_count == 0:
            self._active_page_id = self._new_page()
        else:
            self._unmeasured_pages = set(range(disk_manager.page_count))
            self._active_page_id = disk_manager.page_count - 1

    def insert(self, record: Record) -> RID:
        data = record.data
        page_size = self._disk_manager.page_size
        if len(data) > codec.max_record_size(page_size):
            raise ValueError(
                f"record of {len(data)} bytes can never fit in a page of {page_size} bytes"
            )

        page_id = self._find_page_with_room_for(data)
        if page_id is None:
            page_id = self._new_page()
        self._active_page_id = page_id

        frame = self._buffer_manager.pin(page_id)
        slot = codec.insert_record(frame, data)
        if slot is None:
            # cabia solo tras compactar (ya lo confirmo _find_page_with_room_for)
            codec.compact(frame)
            slot = codec.insert_record(frame, data)
        assert slot is not None

        self._set_capacity(page_id, self._frame_capacity(frame))
        self._buffer_manager.unpin(page_id, dirty=True)
        return RID(page_id=page_id, slot=slot)

    def fetch(self, rid: RID) -> Record | None:
        if not self._page_exists(rid.page_id):
            return None

        frame = self._buffer_manager.pin(rid.page_id)
        try:
            data = codec.read_record(frame, rid.slot)
        finally:
            self._buffer_manager.unpin(rid.page_id, dirty=False)

        return None if data is None else Record(data=data)

    def remove(self, rid: RID) -> bool:
        if not self._page_exists(rid.page_id):
            return False

        frame = self._buffer_manager.pin(rid.page_id)
        removed = codec.delete_record(frame, rid.slot)
        if removed:
            self._set_capacity(rid.page_id, self._frame_capacity(frame))
        self._buffer_manager.unpin(rid.page_id, dirty=removed)
        return removed

    def scan(self) -> Iterator[tuple[RID, Record]]:
        for page_id in range(self._disk_manager.page_count):
            frame = self._buffer_manager.pin(page_id)
            try:
                for slot in range(codec.slot_count(frame)):
                    data = codec.read_record(frame, slot)
                    if data is not None:
                        yield RID(page_id=page_id, slot=slot), Record(data=data)
            finally:
                self._buffer_manager.unpin(page_id, dirty=False)

    def _page_exists(self, page_id: int) -> bool:
        return 0 <= page_id < self._disk_manager.page_count

    def _find_page_with_room_for(self, data: bytes) -> int | None:
        """Find a page with room for data: known capacity first, then lazy old pages."""
        required = len(data) + codec.SLOT_SIZE

        active_capacity = self._page_capacity.get(self._active_page_id)
        if active_capacity is not None and active_capacity >= required:
            return self._active_page_id

        while self._capacity_heap:
            neg_capacity, page_id = self._capacity_heap[0]
            if self._page_capacity.get(page_id) != -neg_capacity:
                heapq.heappop(self._capacity_heap)  # entrada obsoleta
                continue
            if -neg_capacity >= required:
                return page_id
            break  # la mayor capacidad conocida no alcanza

        if self._active_page_id in self._unmeasured_pages:
            if self._measure_page(self._active_page_id) >= required:
                return self._active_page_id

        while self._unmeasured_pages:
            page_id = self._unmeasured_pages.pop()
            if self._measure_page(page_id) >= required:
                return page_id

        return None

    def _measure_page(self, page_id: int) -> int:
        frame = self._buffer_manager.pin(page_id)
        capacity = self._frame_capacity(frame)
        self._buffer_manager.unpin(page_id, dirty=False)
        self._set_capacity(page_id, capacity)
        return capacity

    def _set_capacity(self, page_id: int, capacity: int) -> None:
        self._unmeasured_pages.discard(page_id)
        self._page_capacity[page_id] = capacity
        heapq.heappush(self._capacity_heap, (-capacity, page_id))

    @staticmethod
    def _frame_capacity(frame: bytes) -> int:
        return codec.free_space(frame) + codec.reclaimable_space(frame)

    def _new_page(self) -> int:
        page_id = self._disk_manager.allocate_page()
        frame = self._buffer_manager.pin(page_id)
        frame[:] = codec.new_page(self._disk_manager.page_size)
        self._set_capacity(page_id, self._frame_capacity(frame))
        self._buffer_manager.unpin(page_id, dirty=True)
        return page_id
