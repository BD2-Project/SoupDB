"""Paged sequential file sorted by a configurable key.

Records are grouped into logical buckets: a main page plus its overflow
chain. Overflow never moves a live record -- insert only ever writes a
brand-new slot, so a record's RID never changes once assigned. Page order
is tracked via persistent chain pointers (page 0 is always the first main
page), not physical page_id, since overflow pages break that assumption.
"""

from collections.abc import Callable, Iterator
from typing import Any

from engine.common.record import Record
from engine.common.rid import RID
from engine.storage import _sequential_page as seqpage
from engine.storage import _slotted_page as slotted
from engine.storage.base import FileOrganization
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

KeyFn = Callable[[Record], Any]

_MAIN_HEAD = 0


class SequentialFile(FileOrganization):
    """Sequential file organization: records kept in ascending key order."""

    def __init__(
        self, disk_manager: DiskManager, buffer_manager: BufferManager, key_fn: KeyFn
    ) -> None:
        if not callable(key_fn):
            raise TypeError("key_fn must be callable")

        self._disk_manager = disk_manager
        self._buffer_manager = buffer_manager
        self._key_fn = key_fn
        self._page_max_key: dict[int, Any] = {}
        self._main_next: dict[int, int | None] = {}
        self._page_owner_main: dict[int, int] = {}

        if disk_manager.page_count == 0:
            self._new_page(seqpage.MAIN)
        else:
            self._rebuild_chain_metadata()

    def insert(self, record: Record) -> RID:
        data = record.data
        page_size = self._disk_manager.page_size
        max_payload = slotted.max_record_size(seqpage.body_size(page_size))
        if len(data) > max_payload:
            raise ValueError(
                f"record of {len(data)} bytes can never fit in a {page_size}-byte page "
                f"(maximum record payload is {max_payload} bytes)"
            )

        key = self._key_fn(record)
        main_id, in_range = self._target_main(key)

        frame = self._buffer_manager.pin(main_id)
        body = seqpage.body(frame)
        slot = slotted.insert_record(body, data)
        if slot is None:
            slotted.compact(body)
            slot = slotted.insert_record(body, data)

        if slot is not None:
            rid = RID(page_id=main_id, slot=slot)
            self._bump_max(main_id, key)
            self._buffer_manager.unpin(main_id, dirty=True)
            return rid

        self._buffer_manager.unpin(main_id, dirty=False)

        if not in_range:
            new_main = self._new_page(seqpage.MAIN)
            self._link_main(main_id, new_main)
            return self._insert_fresh(new_main, new_main, key, data)

        return self._insert_into_overflow(main_id, key, data)

    def fetch(self, rid: RID) -> Record | None:
        if not self._page_exists(rid.page_id):
            return None

        frame = self._buffer_manager.pin(rid.page_id)
        try:
            data = slotted.read_record(seqpage.body(frame), rid.slot)
        finally:
            self._buffer_manager.unpin(rid.page_id, dirty=False)

        return None if data is None else Record(data=data)

    def remove(self, rid: RID) -> bool:
        if not self._page_exists(rid.page_id):
            return False

        frame = self._buffer_manager.pin(rid.page_id)
        removed = slotted.delete_record(seqpage.body(frame), rid.slot)
        self._buffer_manager.unpin(rid.page_id, dirty=removed)

        if removed:
            owner_main_id = self._page_owner_main[rid.page_id]
            self._page_max_key[owner_main_id] = self._bucket_max(owner_main_id)

        return removed

    def scan(self) -> Iterator[tuple[RID, Record]]:
        main_id: int | None = _MAIN_HEAD
        while main_id is not None:
            frame = self._buffer_manager.pin(main_id)
            entries = self._entries(seqpage.body(frame), main_id)
            overflow_id = seqpage.first_overflow_page_id(frame)
            next_main = seqpage.next_page_id(frame)
            self._buffer_manager.unpin(main_id, dirty=False)

            while overflow_id is not None:
                frame = self._buffer_manager.pin(overflow_id)
                entries += self._entries(seqpage.body(frame), overflow_id)
                next_overflow = seqpage.next_page_id(frame)
                self._buffer_manager.unpin(overflow_id, dirty=False)
                overflow_id = next_overflow

            entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
            for _, page_id, slot, data in entries:
                yield RID(page_id=page_id, slot=slot), Record(data=data)

            main_id = next_main

    def _target_main(self, key: Any) -> tuple[int, bool]:
        main_id = _MAIN_HEAD
        while True:
            max_key = self._page_max_key.get(main_id)
            if max_key is not None and max_key >= key:
                return main_id, True
            next_id = self._main_next.get(main_id)
            if next_id is None:
                return main_id, False
            main_id = next_id

    def _insert_into_overflow(self, main_id: int, key: Any, data: bytes) -> RID:
        frame = self._buffer_manager.pin(main_id)
        current = seqpage.first_overflow_page_id(frame)
        self._buffer_manager.unpin(main_id, dirty=False)

        if current is None:
            new_overflow = self._new_page(seqpage.OVERFLOW)
            self._page_owner_main[new_overflow] = main_id
            frame = self._buffer_manager.pin(main_id)
            seqpage.set_first_overflow_page_id(frame, new_overflow)
            self._buffer_manager.unpin(main_id, dirty=True)
            return self._insert_fresh(new_overflow, main_id, key, data)

        while True:
            frame = self._buffer_manager.pin(current)
            body = seqpage.body(frame)
            slot = slotted.insert_record(body, data)
            if slot is None:
                slotted.compact(body)
                slot = slotted.insert_record(body, data)

            if slot is not None:
                rid = RID(page_id=current, slot=slot)
                self._bump_max(main_id, key)
                self._buffer_manager.unpin(current, dirty=True)
                return rid

            next_overflow = seqpage.next_page_id(frame)
            self._buffer_manager.unpin(current, dirty=False)

            if next_overflow is None:
                new_overflow = self._new_page(seqpage.OVERFLOW)
                self._page_owner_main[new_overflow] = main_id
                frame = self._buffer_manager.pin(current)
                seqpage.set_next_page_id(frame, new_overflow)
                self._buffer_manager.unpin(current, dirty=True)
                return self._insert_fresh(new_overflow, main_id, key, data)

            current = next_overflow

    def _insert_fresh(self, page_id: int, owner_main_id: int, key: Any, data: bytes) -> RID:
        frame = self._buffer_manager.pin(page_id)
        slot = slotted.insert_record(seqpage.body(frame), data)
        assert slot is not None  # pagina recien creada: siempre cabe (tamano ya validado)
        self._bump_max(owner_main_id, key)
        self._buffer_manager.unpin(page_id, dirty=True)
        return RID(page_id=page_id, slot=slot)

    def _bump_max(self, main_id: int, key: Any) -> None:
        current = self._page_max_key.get(main_id)
        self._page_max_key[main_id] = key if current is None else max(current, key)

    def _link_main(self, from_id: int, to_id: int) -> None:
        frame = self._buffer_manager.pin(from_id)
        seqpage.set_next_page_id(frame, to_id)
        self._buffer_manager.unpin(from_id, dirty=True)
        self._main_next[from_id] = to_id

    def _new_page(self, kind: int) -> int:
        page_id = self._disk_manager.allocate_page()
        frame = self._buffer_manager.pin(page_id)
        frame[:] = seqpage.new_page(self._disk_manager.page_size, kind)
        self._buffer_manager.unpin(page_id, dirty=True)
        if kind == seqpage.MAIN:
            self._page_max_key[page_id] = None
            self._main_next[page_id] = None
            self._page_owner_main[page_id] = page_id
        return page_id

    def _bucket_max(self, main_id: int) -> Any:
        frame = self._buffer_manager.pin(main_id)
        keys = self._body_keys(seqpage.body(frame))
        overflow_id = seqpage.first_overflow_page_id(frame)
        self._buffer_manager.unpin(main_id, dirty=False)

        while overflow_id is not None:
            frame = self._buffer_manager.pin(overflow_id)
            keys += self._body_keys(seqpage.body(frame))
            next_overflow = seqpage.next_page_id(frame)
            self._buffer_manager.unpin(overflow_id, dirty=False)
            overflow_id = next_overflow

        return max(keys) if keys else None

    def _rebuild_chain_metadata(self) -> None:
        main_id: int | None = _MAIN_HEAD
        while main_id is not None:
            self._page_owner_main[main_id] = main_id
            frame = self._buffer_manager.pin(main_id)
            overflow_id = seqpage.first_overflow_page_id(frame)
            next_main = seqpage.next_page_id(frame)
            self._buffer_manager.unpin(main_id, dirty=False)

            while overflow_id is not None:
                self._page_owner_main[overflow_id] = main_id
                frame = self._buffer_manager.pin(overflow_id)
                next_overflow = seqpage.next_page_id(frame)
                self._buffer_manager.unpin(overflow_id, dirty=False)
                overflow_id = next_overflow

            self._page_max_key[main_id] = self._bucket_max(main_id)
            self._main_next[main_id] = next_main
            main_id = next_main

    def _page_exists(self, page_id: int) -> bool:
        return 0 <= page_id < self._disk_manager.page_count

    def _entries(self, body: memoryview, page_id: int) -> list[tuple[Any, int, int, bytes]]:
        return [
            (self._key_fn(Record(data=data)), page_id, slot, data)
            for slot, data in self._live_records(body)
        ]

    def _body_keys(self, body: memoryview) -> list[Any]:
        return [self._key_fn(Record(data=data)) for _, data in self._live_records(body)]

    def _live_records(self, body: memoryview) -> list[tuple[int, bytes]]:
        result = []
        for slot in range(slotted.slot_count(body)):
            data = slotted.read_record(body, slot)
            if data is not None:
                result.append((slot, data))
        return result
