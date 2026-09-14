from engine.storage import _slotted_page as codec

PAGE_SIZE = 64


def test_new_page_has_correct_size() -> None:
    page = codec.new_page(PAGE_SIZE)
    assert len(page) == PAGE_SIZE


def test_new_page_has_zero_slots() -> None:
    page = codec.new_page(PAGE_SIZE)
    assert codec.slot_count(page) == 0


def test_new_page_free_space_is_page_size_minus_header() -> None:
    page = codec.new_page(PAGE_SIZE)
    assert codec.free_space(page) == PAGE_SIZE - codec.HEADER_SIZE


def test_max_record_size() -> None:
    assert codec.max_record_size(PAGE_SIZE) == PAGE_SIZE - codec.HEADER_SIZE - codec.SLOT_SIZE


def test_insert_record_returns_slot_zero_first() -> None:
    page = codec.new_page(PAGE_SIZE)
    assert codec.insert_record(page, b"hello") == 0


def test_insert_record_returns_increasing_slots() -> None:
    page = codec.new_page(PAGE_SIZE)
    s0 = codec.insert_record(page, b"a")
    s1 = codec.insert_record(page, b"bb")
    assert (s0, s1) == (0, 1)


def test_read_record_roundtrip() -> None:
    page = codec.new_page(PAGE_SIZE)
    slot = codec.insert_record(page, b"hello world")
    assert codec.read_record(page, slot) == b"hello world"


def test_insert_record_reduces_free_space_by_data_plus_slot_size() -> None:
    page = codec.new_page(PAGE_SIZE)
    before = codec.free_space(page)
    codec.insert_record(page, b"1234")
    after = codec.free_space(page)
    assert after == before - 4 - codec.SLOT_SIZE


def test_insert_record_returns_none_when_it_does_not_fit() -> None:
    page = codec.new_page(PAGE_SIZE)
    too_big = b"x" * (codec.max_record_size(PAGE_SIZE) + 1)
    assert codec.insert_record(page, too_big) is None


def test_insert_record_that_does_not_fit_does_not_mutate_page() -> None:
    page = codec.new_page(PAGE_SIZE)
    too_big = b"x" * (codec.max_record_size(PAGE_SIZE) + 1)
    codec.insert_record(page, too_big)
    assert codec.slot_count(page) == 0
    assert codec.free_space(page) == PAGE_SIZE - codec.HEADER_SIZE


def test_insert_record_exactly_max_size_fits() -> None:
    page = codec.new_page(PAGE_SIZE)
    data = b"x" * codec.max_record_size(PAGE_SIZE)
    slot = codec.insert_record(page, data)
    assert slot == 0
    assert codec.read_record(page, slot) == data


def test_read_record_out_of_range_returns_none() -> None:
    page = codec.new_page(PAGE_SIZE)
    assert codec.read_record(page, 0) is None
    assert codec.read_record(page, -1) is None


def test_delete_record_tombstones_slot() -> None:
    page = codec.new_page(PAGE_SIZE)
    slot = codec.insert_record(page, b"data")
    assert codec.delete_record(page, slot) is True
    assert codec.read_record(page, slot) is None


def test_delete_record_twice_returns_false_the_second_time() -> None:
    page = codec.new_page(PAGE_SIZE)
    slot = codec.insert_record(page, b"data")
    codec.delete_record(page, slot)
    assert codec.delete_record(page, slot) is False


def test_delete_record_out_of_range_returns_false() -> None:
    page = codec.new_page(PAGE_SIZE)
    assert codec.delete_record(page, 0) is False


def test_delete_one_record_does_not_affect_others() -> None:
    page = codec.new_page(PAGE_SIZE)
    s0 = codec.insert_record(page, b"aaa")
    s1 = codec.insert_record(page, b"bbb")
    codec.delete_record(page, s0)
    assert codec.read_record(page, s0) is None
    assert codec.read_record(page, s1) == b"bbb"


def test_slot_count_reflects_deleted_slots_too() -> None:
    page = codec.new_page(PAGE_SIZE)
    slot = codec.insert_record(page, b"data")
    codec.delete_record(page, slot)
    assert codec.slot_count(page) == 1  # el slot sigue existiendo, solo tombstoned


def test_reclaimable_space_zero_on_fresh_page() -> None:
    page = codec.new_page(PAGE_SIZE)
    assert codec.reclaimable_space(page) == 0


def test_reclaimable_space_zero_with_no_deletions() -> None:
    page = codec.new_page(PAGE_SIZE)
    codec.insert_record(page, b"a" * 10)
    assert codec.reclaimable_space(page) == 0


def test_reclaimable_space_reflects_deleted_record_length() -> None:
    page = codec.new_page(PAGE_SIZE)
    slot = codec.insert_record(page, b"a" * 10)
    codec.delete_record(page, slot)
    assert codec.reclaimable_space(page) == 10


def test_compact_recovers_dead_space_into_free_space() -> None:
    page = codec.new_page(PAGE_SIZE)
    slot = codec.insert_record(page, b"a" * 10)
    codec.delete_record(page, slot)
    before_free = codec.free_space(page)

    codec.compact(page)

    assert codec.free_space(page) == before_free + 10
    assert codec.reclaimable_space(page) == 0


def test_compact_preserves_live_record_bytes_and_slot_number() -> None:
    page = codec.new_page(PAGE_SIZE)
    s0 = codec.insert_record(page, b"aaaaaaaaaa")
    s1 = codec.insert_record(page, b"bbbbbbbbbb")
    codec.delete_record(page, s0)

    codec.compact(page)

    assert codec.read_record(page, s1) == b"bbbbbbbbbb"
    assert codec.read_record(page, s0) is None
    assert codec.slot_count(page) == 2  # el slot del tombstone se conserva


def test_compact_keeps_tombstones_as_tombstones() -> None:
    page = codec.new_page(PAGE_SIZE)
    s0 = codec.insert_record(page, b"a" * 10)
    codec.insert_record(page, b"b" * 10)
    codec.delete_record(page, s0)

    codec.compact(page)

    assert codec.read_record(page, s0) is None
    assert codec.delete_record(page, s0) is False  # sigue sin poder re-eliminarse


def test_compact_allows_insert_that_did_not_fit_before() -> None:
    page = codec.new_page(PAGE_SIZE)
    # con page_size=64 caben 2 registros de 20 bytes de forma directa, no 3
    s0 = codec.insert_record(page, b"x" * 20)
    codec.insert_record(page, b"x" * 20)
    codec.delete_record(page, s0)

    assert codec.insert_record(page, b"y" * 20) is None  # aun fragmentado, no cabe directo

    codec.compact(page)
    slot = codec.insert_record(page, b"y" * 20)

    assert slot is not None
    assert codec.read_record(page, slot) == b"y" * 20
