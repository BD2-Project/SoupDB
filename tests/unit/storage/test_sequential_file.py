import random
from collections.abc import Callable
from pathlib import Path

import pytest

from engine.common.record import Record
from engine.common.rid import RID
from engine.storage import _sequential_page as seqpage
from engine.storage import _slotted_page as slotted
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


def test_insert_record_too_large_error_reports_real_page_size_and_max_payload(
    tmp_path: Path,
) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    too_big = Record(b"\x00" * PAGE_SIZE)  # 64 bytes: mayor al payload maximo real (47)
    with pytest.raises(ValueError) as exc_info:
        sf.insert(too_big)

    message = str(exc_info.value)
    assert "64" in message  # tamano fisico real de la pagina
    assert "55" not in message  # nunca mostrar body_size como si fuera el tamano de pagina
    assert "47" in message  # payload maximo real (body_size - header/slot de _slotted_page)


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


# --- _page_max_key debe reflejar el maximo VIVO actual, no un high-water mark ---


def test_max_key_updates_immediately_after_removing_bucket_max(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    sf.insert(_record(10))
    sf.insert(_record(20))
    r30 = sf.insert(_record(30))  # llena la pagina principal 0
    sf.insert(_record(40))  # bucket principal siguiente

    sf.remove(r30)

    assert sf._page_max_key[0] == 20  # no debe seguir siendo 30


def test_max_key_recomputed_when_max_lives_in_overflow(tmp_path: Path) -> None:
    sf, dm, _bm = _make_sequential_file(tmp_path)
    sf.insert(_record(10))
    sf.insert(_record(20))
    r30 = sf.insert(_record(30))  # llena la pagina principal, max=30
    r28 = sf.insert(_record(28))  # dentro del rango: va a overflow

    assert dm.page_count == 2

    sf.remove(r30)  # el maximo real pasa a ser 28, que vive en overflow
    assert sf._page_max_key[0] == 28

    sf.remove(r28)  # r28 esta en una pagina overflow: prueba resolucion del main propietario
    assert sf._page_max_key[0] == 20


def test_max_key_becomes_none_when_bucket_fully_emptied(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    rids = [sf.insert(_record(k)) for k in (10, 20, 30)]
    for rid in rids:
        sf.remove(rid)

    assert sf._page_max_key[0] is None


def test_rids_remain_stable_when_remove_triggers_max_recompute(tmp_path: Path) -> None:
    sf, _dm, _bm = _make_sequential_file(tmp_path)
    r10 = sf.insert(_record(10))
    r20 = sf.insert(_record(20))
    r30 = sf.insert(_record(30))

    sf.remove(r30)

    assert sf.fetch(r10) == _record(10)
    assert sf.fetch(r20) == _record(20)
    assert (r10.page_id, r20.page_id) == (0, 0)


def test_routing_of_key_is_consistent_before_and_after_reopen(tmp_path: Path) -> None:
    # sesion A: nunca se reabre
    dm_a = DiskManager(tmp_path / "a.db", page_size=PAGE_SIZE)
    bm_a = BufferManager(dm_a, capacity=4)
    sf_a = SequentialFile(dm_a, bm_a, _key_fn)
    sf_a.insert(_record(10))
    sf_a.insert(_record(20))
    r30_a = sf_a.insert(_record(30))
    sf_a.insert(_record(40))
    sf_a.remove(r30_a)
    rid_a = sf_a.insert(_record(25))
    owner_a = sf_a._page_owner_main[rid_a.page_id]

    # sesion B: operaciones identicas, pero se reabre justo antes de insertar 25
    path_b = tmp_path / "b.db"
    dm_b = DiskManager(path_b, page_size=PAGE_SIZE)
    bm_b = BufferManager(dm_b, capacity=4)
    sf_b = SequentialFile(dm_b, bm_b, _key_fn)
    sf_b.insert(_record(10))
    sf_b.insert(_record(20))
    r30_b = sf_b.insert(_record(30))
    sf_b.insert(_record(40))
    sf_b.remove(r30_b)
    bm_b.flush_all()
    dm_b.close()

    dm_b2 = DiskManager(path_b, page_size=PAGE_SIZE)
    bm_b2 = BufferManager(dm_b2, capacity=4)
    sf_b2 = SequentialFile(dm_b2, bm_b2, _key_fn)
    rid_b = sf_b2.insert(_record(25))
    owner_b = sf_b2._page_owner_main[rid_b.page_id]

    assert owner_a == owner_b


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


# --- exception safety: ningun pin debe filtrarse si key_fn lanza ---


def test_scan_unpins_main_page_when_key_fn_raises(tmp_path: Path) -> None:
    sf, _dm, bm = _make_sequential_file(tmp_path)
    sf.insert(_record(10))

    sf._key_fn = _raise_boom

    with pytest.raises(RuntimeError):
        list(sf.scan())

    assert bm.is_pinned(0) is False


def test_scan_unpins_overflow_page_when_key_fn_raises(tmp_path: Path) -> None:
    sf, dm, bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    sf.insert(_record(15))  # crea overflow (pagina 1)
    assert dm.page_count == 2

    sf._key_fn = _boom_on_key(15)

    with pytest.raises(RuntimeError):
        list(sf.scan())

    assert bm.is_pinned(0) is False
    assert bm.is_pinned(1) is False


def test_bucket_max_unpins_main_page_when_key_fn_raises(tmp_path: Path) -> None:
    sf, _dm, bm = _make_sequential_file(tmp_path)
    sf.insert(_record(10))

    sf._key_fn = _raise_boom

    with pytest.raises(RuntimeError):
        sf._bucket_max(0)

    assert bm.is_pinned(0) is False


def test_bucket_max_unpins_overflow_page_when_key_fn_raises(tmp_path: Path) -> None:
    sf, dm, bm = _make_sequential_file(tmp_path)
    for key in (10, 20, 30):
        sf.insert(_record(key))
    sf.insert(_record(15))  # overflow en pagina 1
    assert dm.page_count == 2

    sf._key_fn = _boom_on_key(15)

    with pytest.raises(RuntimeError):
        sf._bucket_max(0)

    assert bm.is_pinned(0) is False
    assert bm.is_pinned(1) is False


def test_remove_does_not_leak_pin_when_key_fn_raises_during_recompute(tmp_path: Path) -> None:
    sf, _dm, bm = _make_sequential_file(tmp_path)
    sf.insert(_record(10))
    r20 = sf.insert(_record(20))

    sf._key_fn = _boom_on_key(10)

    with pytest.raises(RuntimeError):
        sf.remove(r20)

    assert bm.is_pinned(0) is False


def _raise_boom(record: Record) -> int:
    raise RuntimeError("boom")


def _boom_on_key(trigger_key: int) -> Callable[[Record], int]:
    def _key_fn_that_booms(record: Record) -> int:
        if record.data[0] == trigger_key:
            raise RuntimeError("boom")
        return _key_fn(record)

    return _key_fn_that_booms


# --- una pagina compactada, pero que sigue sin tener espacio, debe persistir ---


def test_main_compaction_persists_when_record_still_does_not_fit(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=4)
    sf = SequentialFile(dm, bm, _key_fn)

    r10 = sf.insert(_record(10))
    r20 = sf.insert(_record(20))
    r30 = sf.insert(_record(30))  # llena la pagina principal 0
    sf.insert(_record(15))  # crea overflow1 (pagina 1), ya enlazada desde main0
    sf.remove(r20)  # tombstone en main0, con espacio recuperable
    bm.flush_all()  # limpia dirty: el bug de persistencia importa de verdad ahora

    rid_new = sf.insert(_record(16, size=16))  # no cabe en main0 ni compactando

    assert rid_new.page_id != 0

    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=4)
    frame = bm2.pin(0)
    reclaimable = slotted.reclaimable_space(seqpage.body(frame))
    bm2.unpin(0, dirty=False)
    sf2 = SequentialFile(dm2, bm2, _key_fn)

    assert reclaimable == 0
    assert sf2.fetch(r10) == _record(10)
    assert sf2.fetch(r30) == _record(30)


def test_overflow_compaction_persists_when_record_still_does_not_fit(tmp_path: Path) -> None:
    path = tmp_path / "data.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=4)
    sf = SequentialFile(dm, bm, _key_fn)

    sf.insert(_record(10))
    sf.insert(_record(20))
    sf.insert(_record(30))  # llena main0
    sf.insert(_record(11))
    r12 = sf.insert(_record(12))
    sf.insert(_record(13))  # llena overflow1 (pagina 1)
    sf.insert(_record(14))  # crea overflow2 (pagina 2), enlazada desde overflow1

    sf.remove(r12)  # tombstone en overflow1
    bm.flush_all()  # limpia dirty: el bug de persistencia importa de verdad ahora

    rid_new = sf.insert(_record(16, size=16))  # no cabe en overflow1 ni compactando

    assert rid_new.page_id == 2  # termina en la siguiente overflow ya existente

    bm.flush_all()
    dm.close()

    dm2 = DiskManager(path, page_size=PAGE_SIZE)
    bm2 = BufferManager(dm2, capacity=4)
    frame = bm2.pin(1)
    reclaimable = slotted.reclaimable_space(seqpage.body(frame))
    bm2.unpin(1, dirty=False)
    dm2.close()

    assert reclaimable == 0
