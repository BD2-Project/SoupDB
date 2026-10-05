"""Page header for SequentialFile: main/overflow chain metadata.

The body (page[HEADER_SIZE:]) is the unmodified _slotted_page format,
accessed through a memoryview so its functions work unchanged.
"""

import struct

from engine.storage import _slotted_page as slotted

_HEADER_FORMAT = "<Bii"
HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

MAIN = 0
OVERFLOW = 1
_NONE_PAGE = -1


def body_size(page_size: int) -> int:
    return page_size - HEADER_SIZE


def new_page(page_size: int, kind: int) -> bytearray:
    page = bytearray(page_size)
    _write_header(page, kind, _NONE_PAGE, _NONE_PAGE)
    page[HEADER_SIZE:] = slotted.new_page(body_size(page_size))
    return page


def kind(page: bytes) -> int:
    return _read_header(page)[0]


def next_page_id(page: bytes) -> int | None:
    value = _read_header(page)[1]
    return None if value == _NONE_PAGE else value


def set_next_page_id(page: bytearray, value: int | None) -> None:
    k, _, first_overflow = _read_header(page)
    _write_header(page, k, _NONE_PAGE if value is None else value, first_overflow)


def first_overflow_page_id(page: bytes) -> int | None:
    value = _read_header(page)[2]
    return None if value == _NONE_PAGE else value


def set_first_overflow_page_id(page: bytearray, value: int | None) -> None:
    k, next_id, _ = _read_header(page)
    _write_header(page, k, next_id, _NONE_PAGE if value is None else value)


def body(page: bytearray) -> memoryview:
    return memoryview(page)[HEADER_SIZE:]


def _read_header(page: bytes) -> tuple[int, int, int]:
    return struct.unpack_from(_HEADER_FORMAT, page, 0)


def _write_header(page: bytearray, kind_: int, next_id: int, first_overflow: int) -> None:
    struct.pack_into(_HEADER_FORMAT, page, 0, kind_, next_id, first_overflow)
