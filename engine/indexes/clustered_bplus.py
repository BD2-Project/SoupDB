"""Clustered B+ tree backed by a key-ordered SequentialFile."""

from collections.abc import Iterator

from engine.common.record import Record
from engine.common.rid import RID
from engine.indexes.base import Key
from engine.indexes.bplus_tree import BPlusTree
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.sequential_file import KeyFn, SequentialFile


class ClusteredBPlusTree(BPlusTree):
    """B+ tree whose RIDs point to records stored in key order."""

    def __init__(
        self,
        index_disk_manager: DiskManager,
        index_buffer_manager: BufferManager,
        data_disk_manager: DiskManager,
        data_buffer_manager: BufferManager,
        key_fn: KeyFn,
    ) -> None:
        self._data_buffer_manager = data_buffer_manager
        self._key_fn = key_fn
        self._data_file = SequentialFile(
            data_disk_manager,
            data_buffer_manager,
            key_fn,
        )

        super().__init__(
            index_disk_manager,
            index_buffer_manager,
        )

    @property
    def is_clustered(self) -> bool:
        return True

    def insert_record(self, record: Record) -> RID:
        """Store a record in clustered order and index its resulting RID."""
        key = self._key_fn(record)
        rid = self._data_file.insert(record)

        try:
            super().insert(key, rid)
        except Exception:
            self._data_file.remove(rid)
            raise

        return rid

    def fetch_record(self, rid: RID) -> Record | None:
        return self._data_file.fetch(rid)

    def search_records(self, key: Key) -> list[Record]:
        return self._fetch_rids(super().search(key))

    def range_records(self, lo: Key, hi: Key) -> list[Record]:
        return self._fetch_rids(super().range_search(lo, hi))

    def scan_records(self) -> Iterator[tuple[RID, Record]]:
        return self._data_file.scan()

    def remove_record(self, key: Key, rid: RID) -> bool:
        """Remove one record from both the B+ tree and clustered storage."""
        record = self._data_file.fetch(rid)

        if record is None or self._key_fn(record) != key:
            return False

        if super().remove(key, rid) != 1:
            return False

        if self._data_file.remove(rid):
            return True

        super().insert(key, rid)
        return False

    def close(self) -> None:
        self._data_buffer_manager.flush_all()
        super().close()

    def _fetch_rids(self, rids: list[RID]) -> list[Record]:
        records: list[Record] = []

        for rid in rids:
            record = self._data_file.fetch(rid)

            if record is None:
                raise ValueError(f"clustered B+ contains stale RID {rid}")

            records.append(record)

        return records
