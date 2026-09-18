from engine.storage import _sequential_page as seqpage
from engine.storage import _slotted_page as slotted

PAGE_SIZE = 64


def test_new_page_has_correct_size() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    assert len(page) == PAGE_SIZE


def test_new_main_page_has_no_next_and_no_overflow() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    assert seqpage.kind(page) == seqpage.MAIN
    assert seqpage.next_page_id(page) is None
    assert seqpage.first_overflow_page_id(page) is None


def test_new_overflow_page_kind() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.OVERFLOW)
    assert seqpage.kind(page) == seqpage.OVERFLOW


def test_set_next_page_id_roundtrip() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    seqpage.set_next_page_id(page, 7)
    assert seqpage.next_page_id(page) == 7


def test_set_next_page_id_back_to_none() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    seqpage.set_next_page_id(page, 7)
    seqpage.set_next_page_id(page, None)
    assert seqpage.next_page_id(page) is None


def test_set_first_overflow_page_id_roundtrip() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    seqpage.set_first_overflow_page_id(page, 3)
    assert seqpage.first_overflow_page_id(page) == 3


def test_setting_next_page_id_does_not_affect_first_overflow() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    seqpage.set_first_overflow_page_id(page, 3)
    seqpage.set_next_page_id(page, 9)
    assert seqpage.first_overflow_page_id(page) == 3
    assert seqpage.next_page_id(page) == 9


def test_body_size_accounts_for_header() -> None:
    assert seqpage.body_size(PAGE_SIZE) == PAGE_SIZE - seqpage.HEADER_SIZE


def test_body_delegates_to_slotted_page_format() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    body = seqpage.body(page)
    slot = slotted.insert_record(body, b"hello")
    assert slotted.read_record(body, slot) == b"hello"


def test_body_mutations_do_not_corrupt_header() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    seqpage.set_next_page_id(page, 5)
    seqpage.set_first_overflow_page_id(page, 2)

    body = seqpage.body(page)
    slotted.insert_record(body, b"data")

    assert seqpage.next_page_id(page) == 5
    assert seqpage.first_overflow_page_id(page) == 2
    assert seqpage.kind(page) == seqpage.MAIN


def test_header_updates_do_not_corrupt_body() -> None:
    page = seqpage.new_page(PAGE_SIZE, seqpage.MAIN)
    body = seqpage.body(page)
    slot = slotted.insert_record(body, b"data")

    seqpage.set_next_page_id(page, 5)
    seqpage.set_first_overflow_page_id(page, 2)

    assert slotted.read_record(body, slot) == b"data"
