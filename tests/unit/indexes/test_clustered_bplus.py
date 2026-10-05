from pathlib import Path

from engine.common.record import Record
from engine.indexes.clustered_bplus import ClusteredBPlusTree
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 256


def _key(record: Record) -> int:
    return int(record.data.split(b":", 1)[0])


def _record(key: int, value: str = "value") -> Record:
    return Record(data=f"{key}:{value}".encode())


def _make_index(
    tmp_path: Path,
) -> tuple[ClusteredBPlusTree, DiskManager, DiskManager]:
    index_dm = DiskManager(
        tmp_path / "clustered-index.db",
        page_size=PAGE_SIZE,
    )
    index_bm = BufferManager(index_dm, capacity=2)

    data_dm = DiskManager(
        tmp_path / "clustered-data.db",
        page_size=PAGE_SIZE,
    )
    data_bm = BufferManager(data_dm, capacity=2)

    index = ClusteredBPlusTree(
        index_dm,
        index_bm,
        data_dm,
        data_bm,
        _key,
    )

    return index, index_dm, data_dm


def test_insert_record_indexes_returned_rid(tmp_path: Path) -> None:
    index, _index_dm, _data_dm = _make_index(tmp_path)

    record = _record(10)
    rid = index.insert_record(record)

    assert index.search(10) == [rid]
    assert index.fetch_record(rid) == record


def test_search_records_returns_clustered_records(tmp_path: Path) -> None:
    index, _index_dm, _data_dm = _make_index(tmp_path)

    index.insert_record(_record(7, "a"))
    index.insert_record(_record(7, "b"))

    assert index.search_records(7) == [
        _record(7, "a"),
        _record(7, "b"),
    ]


def test_sequential_storage_scans_in_key_order(tmp_path: Path) -> None:
    index, _index_dm, _data_dm = _make_index(tmp_path)

    for key in (40, 10, 30, 20, 5, 35):
        index.insert_record(_record(key))

    scanned_keys = [_key(record) for _rid, record in index.scan_records()]

    assert scanned_keys == [5, 10, 20, 30, 35, 40]


def test_range_records_follow_key_order(tmp_path: Path) -> None:
    index, _index_dm, _data_dm = _make_index(tmp_path)

    for key in (50, 10, 40, 20, 30):
        index.insert_record(_record(key))

    assert [_key(record) for record in index.range_records(20, 40)] == [20, 30, 40]


def test_remove_record_updates_index_and_storage(tmp_path: Path) -> None:
    index, _index_dm, _data_dm = _make_index(tmp_path)

    rid = index.insert_record(_record(15))

    assert index.remove_record(15, rid) is True
    assert index.search(15) == []
    assert index.fetch_record(rid) is None


def test_clustered_index_persists_after_reopen(tmp_path: Path) -> None:
    index_path = tmp_path / "clustered-index.db"
    data_path = tmp_path / "clustered-data.db"

    index_dm = DiskManager(index_path, page_size=PAGE_SIZE)
    index_bm = BufferManager(index_dm, capacity=2)

    data_dm = DiskManager(data_path, page_size=PAGE_SIZE)
    data_bm = BufferManager(data_dm, capacity=2)

    index = ClusteredBPlusTree(
        index_dm,
        index_bm,
        data_dm,
        data_bm,
        _key,
    )

    for key in (30, 10, 20):
        index.insert_record(_record(key))

    index.close()
    index_dm.close()
    data_dm.close()

    index_dm2 = DiskManager(index_path, page_size=PAGE_SIZE)
    index_bm2 = BufferManager(index_dm2, capacity=2)

    data_dm2 = DiskManager(data_path, page_size=PAGE_SIZE)
    data_bm2 = BufferManager(data_dm2, capacity=2)

    reopened = ClusteredBPlusTree(
        index_dm2,
        index_bm2,
        data_dm2,
        data_bm2,
        _key,
    )

    assert [_key(record) for record in reopened.range_records(10, 30)] == [10, 20, 30]


def test_clustered_flag_is_true(tmp_path: Path) -> None:
    index, _index_dm, _data_dm = _make_index(tmp_path)

    assert index.is_clustered is True
