import random
from pathlib import Path

import pytest

from engine.common.record import Record
from engine.common.rid import RID
from engine.storage import _sequential_page as seqpage
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.sequential_file import SequentialFile

PAGE_SIZE = 64  # cuerpo util = 55 bytes; registros de 10 bytes: 3 caben directo


def _key_fn(record: Record) -> int:
    return record.data[0]


def _record(key: int, size: int = 10) -> Record:
    return Record(bytes([key]) + b"\x00" * (size - 1))


def _overflow_chain_length(bm: BufferManager, main_id: int) -> int:
    frame = bm.pin(main_id)
    current = seqpage.first_overflow_page_id(frame)
    bm.unpin(main_id, dirty=False)

    length = 0
    while current is not None:
        length += 1
        frame = bm.pin(current)
        next_id = seqpage.next_page_id(frame)
        bm.unpin(current, dirty=False)
        current = next_id
    return length


def _make_sequential_file(
    tmp_path: Path, buffer_capacity: int = 4, page_size: int = PAGE_SIZE
) -> tuple[SequentialFile, DiskManager, BufferManager]:
    dm = DiskManager(tmp_path / "data.db", page_size=page_size)
    bm = BufferManager(dm, capacity=buffer_capacity)
    return SequentialFile(dm, bm, _key_fn), dm, bm


def test_rejects_non_callable_key_fn(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    with pytest.raises(TypeError):
        SequentialFile(dm, bm, None)


def test_construct_allocates_first_page(tmp_path: Path) -> None:
    _sf, dm, _bm = _make_sequential_file(tmp_path)
    assert dm.page_count == 1


def test_insert_single_record(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    rid = sf.insert(_record(5))
    assert sf.fetch(rid) == _record(5)


def test_scan_orders_out_of_order_inserts(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path, page_size=256)
    for key in (5, 1, 4, 2, 3):
        sf.insert(_record(key))

    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [1, 2, 3, 4, 5]


def test_insert_at_start_middle_and_end(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path, page_size=256)
    sf.insert(_record(10))
    sf.insert(_record(20))
    sf.insert(_record(5))  # inicio
    sf.insert(_record(15))  # medio
    sf.insert(_record(30))  # final

    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [5, 10, 15, 20, 30]


def test_duplicate_keys_all_appear(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path, page_size=256)
    sf.insert(_record(7))
    sf.insert(_record(7))
    sf.insert(_record(7))

    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [7, 7, 7]


def test_duplicate_keys_scan_order_is_deterministic(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    sf.insert(_record(7))
    sf.insert(_record(7))

    assert list(sf.scan()) == list(sf.scan())


def test_multiple_records_fit_in_one_page(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path, page_size=256)
    for key in range(5):
        sf.insert(_record(key))

    assert dm.page_count == 1
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [0, 1, 2, 3, 4]


def test_insert_creates_overflow_when_bucket_full(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))  # llena la pagina principal 0

    rid = sf.insert(_record(15))  # dentro del rango [10,30]: overflow, no pagina principal nueva

    assert dm.page_count == 2
    assert rid.page_id == 1
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [10, 15, 20, 30]


def test_insert_extends_main_chain_when_key_exceeds_everything(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))

    rid = sf.insert(_record(40))  # excede todo lo existente

    assert rid.page_id == 1
    assert dm.page_count == 2
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [10, 20, 30, 40]


def test_fetch_unknown_page_returns_none(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    assert sf.fetch(RID(page_id=99, slot=0)) is None


def test_fetch_out_of_range_slot_returns_none(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    sf.insert(_record(1))
    assert sf.fetch(RID(page_id=0, slot=99)) is None


def test_remove_then_fetch_returns_none(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    rid = sf.insert(_record(1))
    assert sf.remove(rid) is True
    assert sf.fetch(rid) is None


def test_double_remove_returns_false(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    rid = sf.insert(_record(1))
    sf.remove(rid)
    assert sf.remove(rid) is False


def test_remove_unknown_rid_returns_false(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    assert sf.remove(RID(page_id=99, slot=0)) is False


def test_scan_skips_removed_records(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    r1 = sf.insert(_record(1))
    sf.insert(_record(2))
    sf.remove(r1)

    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [2]


def test_persistence_after_flush_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=4)
    sf = SequentialFile(dm, bm, _key_fn)
    for key in (5, 1, 3):
        sf.insert(_record(key))
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=4)
    sf2 = SequentialFile(dm2, bm2, _key_fn)

    keys = [_key_fn(record) for _, record in sf2.scan()]
    assert keys == [1, 3, 5]


def test_insert_after_reopen_keeps_order(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=4)
    sf = SequentialFile(dm, bm, _key_fn)
    sf.insert(_record(1))
    sf.insert(_record(5))
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=4)
    sf2 = SequentialFile(dm2, bm2, _key_fn)
    sf2.insert(_record(3))

    keys = [_key_fn(record) for _, record in sf2.scan()]
    assert keys == [1, 3, 5]


def test_insert_record_too_large_raises_value_error(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    too_big = Record(b"\x00" * PAGE_SIZE)
    with pytest.raises(ValueError):
        sf.insert(too_big)


def test_empty_file_scan_is_empty(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    assert list(sf.scan()) == []


def test_empty_file_fetch_returns_none(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    assert sf.fetch(RID(page_id=0, slot=0)) is None


# --- invariante de estabilidad de RID (obligatorios) ---


def test_rid_stable_after_many_inserts(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    record_a = _record(50)
    rid_a = sf.insert(record_a)

    for key in range(60, 160):
        sf.insert(_record(key))

    assert sf.fetch(rid_a) == record_a


def test_rid_stable_when_overflow_is_created(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    rid_first = sf.insert(_record(10))
    sf.insert(_record(20))
    sf.insert(_record(30))  # llena la pagina 0
    sf.insert(_record(15))  # crea overflow

    assert dm.page_count == 2
    assert sf.fetch(rid_first) == _record(10)


def test_rid_stable_after_flush_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=4)
    sf = SequentialFile(dm, bm, _key_fn)
    rid_a = sf.insert(_record(10))
    sf.insert(_record(20))
    sf.insert(_record(30))
    sf.insert(_record(15))  # overflow
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=4)
    sf2 = SequentialFile(dm2, bm2, _key_fn)

    assert sf2.fetch(rid_a) == _record(10)


def test_scan_globally_ordered_with_overflow_in_middle_page(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    for key in (0, 1, 2):
        sf.insert(_record(key))
    sf.insert(_record(1))  # duplicado dentro de [0,2]: overflow de la pagina 0

    for key in (10, 11, 12):
        sf.insert(_record(key))
    sf.insert(_record(13))  # excede todo: nueva pagina principal

    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == sorted(keys)
    assert dm.page_count >= 3


def test_duplicate_keys_split_between_main_and_overflow(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    for _ in range(3):
        sf.insert(_record(7))  # llena la pagina 0 con 3 copias
    sf.insert(_record(7))  # 4ta copia: overflow

    assert dm.page_count == 2
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [7, 7, 7, 7]


def test_remove_record_in_overflow_page(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    rid_overflow = sf.insert(_record(15))
    assert dm.page_count == 2

    assert sf.remove(rid_overflow) is True
    assert sf.fetch(rid_overflow) is None
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [10, 20, 30]


def test_reopen_preserves_chain_structure_and_order(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=4)
    sf = SequentialFile(dm, bm, _key_fn)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    sf.insert(_record(15))  # overflow de la pagina 0 (pagina 1)
    sf.insert(_record(40))  # nueva pagina principal (pagina 2)
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=4)
    sf2 = SequentialFile(dm2, bm2, _key_fn)

    assert sf2._main_next == {0: 2, 2: None}
    assert sf2._page_max_key == {0: 30, 2: 40}
    keys = [_key_fn(record) for _, record in sf2.scan()]
    assert keys == [10, 15, 20, 30, 40]


def test_scan_never_yields_control_records(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30, 15, 40):
        sf.insert(_record(key))

    assert dm.page_count >= 2
    result = list(sf.scan())
    assert len(result) == 5
    assert all(len(record.data) == 10 for _, record in result)


def test_overflow_chain_can_grow_to_two_pages(tmp_path: Path) -> None:
    sf, dm, bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    for key in (11, 12, 13, 14):
        sf.insert(_record(key))  # todos dentro de [10,30]: fuerzan 2 paginas de overflow

    assert _overflow_chain_length(bm, 0) == 2
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == sorted(keys)


def test_chain_main_overflow_overflow_then_next_main(tmp_path: Path) -> None:
    sf, dm, bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    for key in (11, 12, 13, 14):
        sf.insert(_record(key))

    rid_last = sf.insert(_record(40))  # excede todo: nueva pagina principal

    assert dm.page_count == 4  # main0, overflow, overflow, main1
    assert _overflow_chain_length(bm, 0) == 2
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == sorted(keys)
    assert sf.fetch(rid_last) == _record(40)


def test_reopen_preserves_two_level_overflow_chain(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=4)
    sf = SequentialFile(dm, bm, _key_fn)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    for key in (11, 12, 13, 14):
        sf.insert(_record(key))
    sf.insert(_record(40))
    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=4)
    sf2 = SequentialFile(dm2, bm2, _key_fn)

    assert _overflow_chain_length(bm2, 0) == 2
    keys = [_key_fn(record) for _, record in sf2.scan()]
    assert keys == sorted(keys)


def test_insert_after_emptying_only_bucket_reuses_it(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    rids = [sf.insert(_record(k)) for k in (10, 20, 30)]
    for rid in rids:
        sf.remove(rid)

    assert list(sf.scan()) == []

    new_rid = sf.insert(_record(5))
    assert new_rid.page_id == 0
    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [5]


def test_emptying_intermediate_bucket_is_skipped_without_error(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    bucket0_rids = [sf.insert(_record(k)) for k in (10, 20, 30)]
    overflow_rid = sf.insert(_record(15))
    for k in (40, 50, 60):
        sf.insert(_record(k))  # bucket principal nuevo

    for rid in [*bucket0_rids, overflow_rid]:
        sf.remove(rid)  # vacia el bucket 0 por completo

    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [40, 50, 60]

    new_rid = sf.insert(_record(12))  # "hubiera" pertenecido al bucket 0, ahora vacio
    assert sf.fetch(new_rid) == _record(12)
    keys_after = sorted(_key_fn(record) for _, record in sf.scan())
    assert keys_after == [12, 40, 50, 60]


def test_duplicate_key_at_bucket_boundary_stays_consistent(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    sf.insert(_record(30))  # duplicado exacto del limite: overflow del bucket 0
    sf.insert(_record(31))  # recien mayor: nueva pagina principal
    sf.insert(_record(30))  # otro duplicado del limite, sigue perteneciendo al bucket 0

    keys = [_key_fn(record) for _, record in sf.scan()]
    assert keys == [10, 20, 30, 30, 30, 31]


def test_removing_other_record_does_not_change_surviving_rid(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    rid_a = sf.insert(_record(10))
    rid_b = sf.insert(_record(20))

    sf.remove(rid_b)

    assert sf.fetch(rid_a) == _record(10)
    assert rid_a.page_id == 0


def test_compaction_inside_page_preserves_surviving_rids(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    rid_a = sf.insert(_record(10))
    rid_b = sf.insert(_record(20))
    rid_c = sf.insert(_record(30))  # llena la pagina principal 0
    sf.remove(rid_b)  # libera espacio interno, no contiguo todavia

    sf.insert(_record(15))  # cabe solo si insert() compacta la pagina 0 primero

    assert sf.fetch(rid_a) == _record(10)
    assert sf.fetch(rid_c) == _record(30)
    assert (rid_a.page_id, rid_c.page_id) == (0, 0)


def test_stress_random_inserts_preserve_order_and_all_rids(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    keys = list(range(80))
    random.Random(7).shuffle(keys)

    inserted = []
    for key in keys:
        rid = sf.insert(_record(key))
        inserted.append((rid, _record(key)))

    scanned = [_key_fn(record) for _, record in sf.scan()]
    assert scanned == sorted(keys)

    for rid, expected in inserted:
        assert sf.fetch(rid) == expected
