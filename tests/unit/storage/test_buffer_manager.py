from pathlib import Path

import pytest

from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager

PAGE_SIZE = 16


def _disk_manager_with_pages(
    tmp_path: Path, num_pages: int, page_size: int = PAGE_SIZE
) -> tuple[DiskManager, list[int]]:
    dm = DiskManager(tmp_path / "data.db", page_size=page_size)
    page_ids = [dm.allocate_page() for _ in range(num_pages)]
    return dm, page_ids


def test_rejects_zero_capacity(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    with pytest.raises(ValueError):
        BufferManager(dm, capacity=0)


def test_rejects_negative_capacity(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    with pytest.raises(ValueError):
        BufferManager(dm, capacity=-1)


def test_capacity_property(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=3)
    assert bm.capacity == 3


def test_pin_reads_page_from_disk_on_first_access(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    dm.write_page(page_id, b"\xaa" * PAGE_SIZE)
    reads_before = dm.reads

    bm = BufferManager(dm, capacity=2)
    frame = bm.pin(page_id)

    assert frame == b"\xaa" * PAGE_SIZE
    assert dm.reads == reads_before + 1


def test_pin_returns_mutable_bytearray(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)
    frame = bm.pin(page_id)
    assert isinstance(frame, bytearray)


def test_pin_does_not_reread_disk_on_cache_hit(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)

    bm.pin(page_id)
    reads_after_first = dm.reads
    bm.pin(page_id)

    assert dm.reads == reads_after_first


def test_pin_marks_page_as_pinned(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)

    assert bm.is_pinned(page_id) is False
    bm.pin(page_id)
    assert bm.is_pinned(page_id) is True


def test_is_pinned_false_for_unknown_page(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    assert bm.is_pinned(0) is False


def test_unpin_requires_all_matching_pins_before_becoming_unpinned(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)

    bm.pin(page_id)
    bm.pin(page_id)
    bm.unpin(page_id)
    assert bm.is_pinned(page_id) is True

    bm.unpin(page_id)
    assert bm.is_pinned(page_id) is False


def test_unpin_never_pinned_page_raises_value_error(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)
    with pytest.raises(ValueError):
        bm.unpin(0)


def test_unpin_already_unpinned_page_raises_value_error(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)

    bm.pin(page_id)
    bm.unpin(page_id)
    with pytest.raises(ValueError):
        bm.unpin(page_id)


def test_flush_page_writes_dirty_page_to_disk(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)

    frame = bm.pin(page_id)
    frame[:] = b"\xbb" * PAGE_SIZE
    bm.unpin(page_id, dirty=True)

    writes_before = dm.writes
    bm.flush_page(page_id)

    assert dm.writes == writes_before + 1
    assert dm.read_page(page_id) == b"\xbb" * PAGE_SIZE


def test_flush_page_on_clean_page_does_not_write(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)

    bm.pin(page_id)
    bm.unpin(page_id)

    writes_before = dm.writes
    bm.flush_page(page_id)
    assert dm.writes == writes_before


def test_flush_page_unknown_page_id_is_noop(tmp_path: Path) -> None:
    dm = DiskManager(tmp_path / "data.db", page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=2)

    writes_before = dm.writes
    bm.flush_page(999)
    assert dm.writes == writes_before


def test_flush_all_writes_every_dirty_page(tmp_path: Path) -> None:
    dm, page_ids = _disk_manager_with_pages(tmp_path, 3)
    bm = BufferManager(dm, capacity=3)

    for i, page_id in enumerate(page_ids):
        frame = bm.pin(page_id)
        frame[:] = bytes([i]) * PAGE_SIZE
        bm.unpin(page_id, dirty=True)

    bm.flush_all()

    for i, page_id in enumerate(page_ids):
        assert dm.read_page(page_id) == bytes([i]) * PAGE_SIZE


def test_flush_all_with_no_dirty_pages_does_not_write(tmp_path: Path) -> None:
    dm, page_ids = _disk_manager_with_pages(tmp_path, 2)
    bm = BufferManager(dm, capacity=2)

    for page_id in page_ids:
        bm.pin(page_id)
        bm.unpin(page_id)

    writes_before = dm.writes
    bm.flush_all()
    assert dm.writes == writes_before


def test_pin_evicts_least_recently_used_unpinned_page_when_full(tmp_path: Path) -> None:
    dm, page_ids = _disk_manager_with_pages(tmp_path, 3)
    p0, p1, p2 = page_ids
    bm = BufferManager(dm, capacity=2)

    bm.pin(p0)
    bm.unpin(p0)
    bm.pin(p1)
    bm.unpin(p1)
    # orden LRU hasta aquí: p0 (menos reciente), p1 (más reciente)

    bm.pin(p2)  # fuerza expulsión; p0 es la menos reciente y no está pineada
    bm.unpin(p2)

    reads_before = dm.reads
    bm.pin(p1)
    assert dm.reads == reads_before  # p1 sigue en caché
    bm.unpin(p1)

    reads_before = dm.reads
    bm.pin(p0)
    assert dm.reads == reads_before + 1  # p0 fue expulsada, se relee de disco
    bm.unpin(p0)


def test_pin_updates_recency_preventing_eviction(tmp_path: Path) -> None:
    dm, page_ids = _disk_manager_with_pages(tmp_path, 3)
    p0, p1, p2 = page_ids
    bm = BufferManager(dm, capacity=2)

    bm.pin(p0)
    bm.unpin(p0)
    bm.pin(p1)
    bm.unpin(p1)

    bm.pin(p0)  # se vuelve a tocar p0 -> ahora p1 es la menos reciente
    bm.unpin(p0)

    bm.pin(p2)  # debe expulsar a p1, no a p0
    bm.unpin(p2)

    reads_before = dm.reads
    bm.pin(p0)
    assert dm.reads == reads_before  # p0 sigue en caché
    bm.unpin(p0)

    reads_before = dm.reads
    bm.pin(p1)
    assert dm.reads == reads_before + 1  # p1 fue expulsada
    bm.unpin(p1)


def test_eviction_flushes_dirty_victim_before_reuse(tmp_path: Path) -> None:
    dm, page_ids = _disk_manager_with_pages(tmp_path, 2)
    p0, p1 = page_ids
    bm = BufferManager(dm, capacity=1)

    frame = bm.pin(p0)
    frame[:] = b"\xcc" * PAGE_SIZE
    bm.unpin(p0, dirty=True)

    bm.pin(p1)  # capacidad 1 -> debe expulsar a p0, que está dirty
    bm.unpin(p1)

    assert dm.read_page(p0) == b"\xcc" * PAGE_SIZE


def test_pin_raises_runtime_error_when_pool_full_and_all_pinned(tmp_path: Path) -> None:
    dm, page_ids = _disk_manager_with_pages(tmp_path, 3)
    p0, p1, p2 = page_ids
    bm = BufferManager(dm, capacity=2)

    bm.pin(p0)
    bm.pin(p1)
    # ninguna se libera -> el pool está lleno y nada es expulsable

    with pytest.raises(RuntimeError):
        bm.pin(p2)


def test_pin_returns_same_frame_reflecting_prior_mutations(tmp_path: Path) -> None:
    dm, [page_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=2)

    frame1 = bm.pin(page_id)
    frame1[0] = 0xFF
    bm.unpin(page_id)

    frame2 = bm.pin(page_id)
    assert frame2[0] == 0xFF


def test_pin_failed_read_does_not_evict_cached_page(tmp_path: Path) -> None:
    dm, [valid_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=1)

    bm.pin(valid_id)
    bm.unpin(valid_id)

    invalid_id = valid_id + 1  # nunca asignado en el DiskManager
    with pytest.raises(ValueError):
        bm.pin(invalid_id)

    reads_before = dm.reads
    bm.pin(valid_id)
    assert dm.reads == reads_before  # sigue en caché, no se releyó de disco
    bm.unpin(valid_id)


def test_pin_failed_read_does_not_flush_dirty_victim(tmp_path: Path) -> None:
    dm, [valid_id] = _disk_manager_with_pages(tmp_path, 1)
    bm = BufferManager(dm, capacity=1)

    frame = bm.pin(valid_id)
    frame[:] = b"\xdd" * PAGE_SIZE
    bm.unpin(valid_id, dirty=True)

    invalid_id = valid_id + 1
    writes_before = dm.writes
    with pytest.raises(ValueError):
        bm.pin(invalid_id)
    assert dm.writes == writes_before  # no se flusheó la víctima potencial

    reads_before = dm.reads
    still_cached = bm.pin(valid_id)
    assert dm.reads == reads_before  # sigue en caché, no se releyó de disco
    assert still_cached == b"\xdd" * PAGE_SIZE  # el contenido dirty sigue intacto
    bm.unpin(valid_id)


def test_pin_raises_runtime_error_before_touching_disk_when_all_pinned(tmp_path: Path) -> None:
    dm, page_ids = _disk_manager_with_pages(tmp_path, 3)
    p0, p1, p2 = page_ids
    bm = BufferManager(dm, capacity=2)

    bm.pin(p0)
    bm.pin(p1)

    reads_before = dm.reads
    with pytest.raises(RuntimeError):
        bm.pin(p2)
    assert dm.reads == reads_before  # nunca debió intentar leer disco
