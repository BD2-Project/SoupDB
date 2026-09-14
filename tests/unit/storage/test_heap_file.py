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


def test_scan_yields_all_live_records(tmp_path: Path) -> None:
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


def test_insert_after_remove_reuses_same_page_via_compaction(tmp_path: Path) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path)
    data = b"x" * 20  # con page_size=64 caben 2 directo, no 3

    r0 = hf.insert(Record(data))
    r1 = hf.insert(Record(data))
    hf.remove(r0)
    hf.remove(r1)

    r2 = hf.insert(Record(data))

    assert r2.page_id == 0
    assert dm.page_count == 1  # no debio crear una pagina nueva
    assert hf.fetch(r2) == Record(data)


def test_insert_after_remove_does_not_increase_page_count_when_space_is_recoverable(
    tmp_path: Path,
) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path)
    data = b"x" * 20
    r0 = hf.insert(Record(data))
    hf.insert(Record(data))
    hf.remove(r0)

    before = dm.page_count
    hf.insert(Record(data))  # requiere compactar para caber

    assert dm.page_count == before


def test_rid_of_live_record_still_resolves_after_compaction_triggered_by_insert(
    tmp_path: Path,
) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    r0 = hf.insert(Record(b"x" * 20))
    r1 = hf.insert(Record(bytes([9]) * 20))
    hf.remove(r0)  # r1 sigue viva

    hf.insert(Record(b"z" * 20))  # dispara compactacion de la pagina 0

    assert hf.fetch(r1) == Record(bytes([9]) * 20)  # el RID original de r1 sigue sirviendo
    assert (r1.page_id, r1.slot) == (0, 1)  # su slot no cambio, solo su offset interno


def test_scan_correct_after_compaction(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    r0 = hf.insert(Record(b"x" * 20))
    r1 = hf.insert(Record(bytes([9]) * 20))
    hf.remove(r0)

    r2 = hf.insert(Record(bytes([7]) * 20))  # dispara compactacion

    result = dict(hf.scan())
    assert result == {r1: Record(bytes([9]) * 20), r2: Record(bytes([7]) * 20)}
    assert r0 not in result


def test_multiple_tombstones_then_reuse(tmp_path: Path) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path)
    data = b"x" * 20
    r0 = hf.insert(Record(data))
    r1 = hf.insert(Record(data))
    hf.remove(r0)
    hf.remove(r1)

    r2 = hf.insert(Record(data))
    r3 = hf.insert(Record(b"y" * 20))

    assert dm.page_count == 1  # ambos reutilizan el espacio liberado en la misma pagina
    assert (r2.page_id, r3.page_id) == (0, 0)
    assert hf.fetch(r0) is None
    assert hf.fetch(r1) is None
    assert hf.fetch(r2) == Record(data)
    assert hf.fetch(r3) == Record(b"y" * 20)


def test_reused_space_persists_after_flush_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    hf = HeapFile(dm, bm)

    data = b"x" * 20
    r0 = hf.insert(Record(data))
    r1 = hf.insert(Record(data))
    hf.remove(r0)
    r2 = hf.insert(Record(data))  # reutiliza via compactacion

    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    hf2 = HeapFile(dm2, bm2)

    assert dm2.page_count == 1
    assert hf2.fetch(r0) is None
    assert hf2.fetch(r1) == Record(data)
    assert hf2.fetch(r2) == Record(data)


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


def test_remove_updates_cached_page_capacity(tmp_path: Path) -> None:
    hf, _dm, _bm = _make_heap_file(tmp_path)
    data = b"x" * 20
    r0 = hf.insert(Record(data))
    hf.insert(Record(data))

    hf.remove(r0)

    assert hf._page_capacity[0] == 32


def test_reopen_does_not_read_any_page_in_constructor(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    hf = HeapFile(dm, bm)
    data = b"x" * 20
    for _ in range(5):
        hf.insert(Record(data))
        hf.insert(Record(data))
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)

    reads_before = dm2.reads
    hf2 = HeapFile(dm2, bm2)

    assert dm2.reads == reads_before
    assert hf2._page_capacity == {}
    assert hf2._unmeasured_pages == set(range(5))


def test_page_capacity_is_measured_lazily_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    hf = HeapFile(dm, bm)

    data = b"x" * 20
    r0 = hf.insert(Record(data))
    hf.insert(Record(data))
    hf.remove(r0)

    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=2)
    hf2 = HeapFile(dm2, bm2)

    assert hf2._page_capacity == {}
    assert hf2._unmeasured_pages == {0}

    r2 = hf2.insert(Record(data))  # descubre y reutiliza el espacio de forma perezosa

    assert r2.page_id == 0
    assert dm2.page_count == 1
    assert hf2._page_capacity == {0: 8}  # 32 libres - 24 consumidos por r2 (20 + SLOT_SIZE)
    assert hf2._unmeasured_pages == set()


def test_measured_old_pages_are_not_reread_on_later_inserts(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=3)
    hf = HeapFile(dm, bm)
    data = b"x" * 20
    for _ in range(3):
        hf.insert(Record(data))
        hf.insert(Record(data))  # 3 paginas llenas, sin espacio recuperable
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=3)
    hf2 = HeapFile(dm2, bm2)

    hf2.insert(Record(data))  # mide las 3 antiguas (no caben) y crea una pagina nueva
    assert hf2._unmeasured_pages == set()

    reads_before = dm2.reads
    hf2.insert(Record(data))  # debe usar la pagina activa recien creada, no repasar las 3 viejas

    assert dm2.reads - reads_before <= 2


def test_selects_known_candidate_among_many_full_pages_without_extra_reads(
    tmp_path: Path,
) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path, buffer_capacity=4)
    data = b"x" * 20

    for _ in range(20):
        hf.insert(Record(data))
        hf.insert(Record(data))  # 20 paginas llenas, todas medidas al crearse

    hf.remove(RID(page_id=10, slot=0))  # deja espacio recuperable en una pagina conocida
    assert dm.page_count == 20

    reads_before = dm.reads
    rid = hf.insert(Record(data))

    assert rid.page_id == 10
    assert dm.page_count == 20
    assert dm.reads - reads_before <= 1


def test_insert_into_new_page_does_not_read_every_existing_page(tmp_path: Path) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path, buffer_capacity=2)
    data = b"x" * 20  # sin remove: cada pagina llena no tiene espacio recuperable

    for _ in range(5):
        hf.insert(Record(data))
        hf.insert(Record(data))

    assert dm.page_count == 5

    reads_before = dm.reads
    hf.insert(Record(data))  # debe crear la pagina 5 sin repasar las 5 anteriores

    assert dm.page_count == 6
    assert dm.reads - reads_before <= 2


def test_scan_after_cross_page_reuse_yields_all_live_records(tmp_path: Path) -> None:
    hf, dm, _bm = _make_heap_file(tmp_path)
    data = b"x" * 20  # 2 caben directo por pagina de 64 bytes

    r0 = hf.insert(Record(data))
    r1 = hf.insert(Record(data))
    r2 = hf.insert(Record(data))  # llena page 0, crea page 1
    r3 = hf.insert(Record(data))  # llena page 1
    hf.remove(r0)  # libera espacio en page 0

    r4 = hf.insert(Record(data))  # reutiliza page 0 via compactacion

    assert dm.page_count == 2
    result = dict(hf.scan())
    assert set(result) == {r1, r2, r3, r4}
    assert r0 not in result
    assert all(record == Record(data) for record in result.values())
