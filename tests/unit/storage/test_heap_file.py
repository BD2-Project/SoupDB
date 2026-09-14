from pathlib import Path

import pytest

from engine.common.record import Record
from engine.common.rid import RID
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.heap_file import HeapFile

PAGE_SIZE = 64


def _make_heap_file(
    tmp_path: Path, buffer_capacity: int = 2
) -> tuple[HeapFile, DiskManager, BufferManager]:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=buffer_capacity)
    return HeapFile(dm, bm), dm, bm


def test_insert_and_fetch_basic(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    rid = hf.insert(Record(b"hello"))
    assert hf.fetch(rid) == Record(b"hello")


def test_fetch_unknown_page_returns_none(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    assert hf.fetch(RID(page_id=99, slot=0)) is None


def test_fetch_out_of_range_slot_returns_none(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    hf.insert(Record(b"x"))
    assert hf.fetch(RID(page_id=0, slot=99)) is None


def test_insert_multiple_records_in_same_page(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    r0 = hf.insert(Record(b"aa"))
    r1 = hf.insert(Record(b"bb"))

    assert r0.page_id == r1.page_id
    assert (r0.slot, r1.slot) == (0, 1)
    assert hf.fetch(r0) == Record(b"aa")
    assert hf.fetch(r1) == Record(b"bb")


def test_insert_allocates_new_page_when_active_page_is_full(tmp_path: Path) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path)
    data = b"x" * 20  # con page_size=64, caben 2 por página, no 3

    r0 = hf.insert(Record(data))
    r1 = hf.insert(Record(data))
    r2 = hf.insert(Record(data))

    assert r0.page_id == r1.page_id
    assert r2.page_id != r0.page_id
    assert dm.page_count == 2
    assert hf.fetch(r2) == Record(data)


def test_remove_then_fetch_returns_none(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    rid = hf.insert(Record(b"data"))

    assert hf.remove(rid) is True
    assert hf.fetch(rid) is None


def test_double_remove_returns_false(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    rid = hf.insert(Record(b"data"))
    hf.remove(rid)
    assert hf.remove(rid) is False


def test_remove_unknown_page_returns_false(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    assert hf.remove(RID(page_id=99, slot=0)) is False


def test_remove_unknown_slot_returns_false(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    hf.insert(Record(b"x"))
    assert hf.remove(RID(page_id=0, slot=99)) is False


def test_scan_yields_all_live_records_in_insertion_order(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    inserted = [hf.insert(Record(bytes([i]) * 5)) for i in range(3)]

    result = list(hf.scan())

    assert [rid for rid, _ in result] == inserted
    assert [record for _, record in result] == [Record(bytes([i]) * 5) for i in range(3)]


def test_scan_skips_removed_records(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    r0 = hf.insert(Record(b"a"))
    r1 = hf.insert(Record(b"b"))
    hf.remove(r0)

    assert list(hf.scan()) == [(r1, Record(b"b"))]


def test_scan_across_multiple_pages(tmp_path: Path) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path)
    data = b"x" * 20
    rids = [hf.insert(Record(data)) for _ in range(3)]

    assert dm.page_count == 2
    assert [rid for rid, _ in hf.scan()] == rids


def test_insert_record_too_large_for_any_page_raises_value_error(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    too_big = b"x" * PAGE_SIZE  # nunca cabria en ninguna pagina, ni vacia

    with pytest.raises(ValueError):
        hf.insert(Record(too_big))


def test_persistence_after_flush_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    hf = HeapFile(dm, bm)

    rid = hf.insert(Record(b"persisted"))
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    hf2 = HeapFile(dm2, bm2)

    assert hf2.fetch(rid) == Record(b"persisted")


def test_insert_after_reopen_reuses_existing_page_with_space(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    hf = HeapFile(dm, bm)
    hf.insert(Record(b"a"))
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    hf2 = HeapFile(dm2, bm2)
    rid = hf2.insert(Record(b"b"))

    assert rid.page_id == 0  # reutiliza la pagina existente, no crea una nueva
    assert dm2.page_count == 1
