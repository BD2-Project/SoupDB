"""Heap file organization with free-space reuse."""

from collections.abc import Iterator

from engine.common.record import Record
from engine.common.rid import RID
from engine.storage import _slotted_page as codec
from engine.storage.base import FileOrganization
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager


class HeapFile(FileOrganization):
    """Records stored in arrival order across slotted pages, addressed by RID."""

    def __init__(self, disk_manager: DiskManager, buffer_manager: BufferManager) -> None:
        self._disk_manager = disk_manager
        self._buffer_manager = buffer_manager

        if disk_manager.page_count == 0:
            self._active_page_id = self._new_page()
        else:
            self._active_page_id = disk_manager.page_count - 1

    def insert(self, record: Record) -> RID:
        data = record.data
        page_size = self._disk_manager.page_size
        if len(data) > codec.max_record_size(page_size):
            raise ValueError(
                f"record of {len(data)} bytes can never fit in a page of {page_size} bytes"
            )

        page_id = self._active_page_id
        frame = self._buffer_manager.pin(page_id)
        slot = codec.insert_record(frame, data)

        if slot is None:
            self._buffer_manager.unpin(page_id, dirty=False)
            page_id = self._new_page()
            self._active_page_id = page_id
            frame = self._buffer_manager.pin(page_id)
            slot = codec.insert_record(frame, data)
            assert slot is not None  # ya validamos que cabe en una pagina vacia

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

    def _new_page(self) -> int:
        page_id = self._disk_manager.allocate_page()
        frame = self._buffer_manager.pin(page_id)
        frame[:] = codec.new_page(self._disk_manager.page_size)
        self._buffer_manager.unpin(page_id, dirty=True)
        return page_id
