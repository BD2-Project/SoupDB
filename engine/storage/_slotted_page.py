"""Private slotted-page codec used internally by HeapFile.

Not part of any public contract: operates on raw ``bytearray``/``bytes`` page
buffers exactly as handed out by :class:`BufferManager`.

Layout (all integers little-endian, unsigned 16-bit)::

    [0:2)                          num_slots
    [2:4)                          free_space_offset
    [4 : 4 + 4*num_slots)          slot directory: (offset, length) pairs
    ... free space ...
    [free_space_offset:page_size)  record bytes, packed backward from the end

Records grow from the end of the page backward; the slot directory grows
forward from the header. ``offset == 0`` in a slot entry marks it as
tombstoned (deleted): offset 0 always falls inside the header/directory
region, so it can never be a real record's start.
"""

import struct

_HEADER_FORMAT = "<HH"
HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

_SLOT_FORMAT = "<HH"
SLOT_SIZE = struct.calcsize(_SLOT_FORMAT)

TOMBSTONE_OFFSET = 0


def max_record_size(page_size: int) -> int:
    """Largest record that could ever fit in a freshly allocated page."""
    return page_size - HEADER_SIZE - SLOT_SIZE


def new_page(page_size: int) -> bytearray:
    """Build a fresh, empty page of exactly page_size bytes."""
    page = bytearray(page_size)
    _write_header(page, num_slots=0, free_space_offset=page_size)
    return page


def slot_count(page: bytes) -> int:
    num_slots, _ = _read_header(page)
    return num_slots


def free_space(page: bytes) -> int:
    """Bytes currently available for a brand-new slot entry plus its data."""
    num_slots, free_space_offset = _read_header(page)
    directory_end = HEADER_SIZE + num_slots * SLOT_SIZE
    return free_space_offset - directory_end


def insert_record(page: bytearray, data: bytes) -> int | None:
    """Place data in page and return its new slot index, or None if it doesn't fit."""
    if len(data) + SLOT_SIZE > free_space(page):
        return None

    num_slots, free_space_offset = _read_header(page)
    new_offset = free_space_offset - len(data)
    page[new_offset:free_space_offset] = data

    slot = num_slots
    _write_slot(page, slot, new_offset, len(data))
    _write_header(page, num_slots + 1, new_offset)
    return slot


def read_record(page: bytes, slot: int) -> bytes | None:
    """Return the bytes stored at slot, or None if it does not exist / is tombstoned."""
    num_slots, _ = _read_header(page)
    if slot < 0 or slot >= num_slots:
        return None

    offset, length = _read_slot(page, slot)
    if offset == TOMBSTONE_OFFSET:
        return None

    return bytes(page[offset : offset + length])


def delete_record(page: bytearray, slot: int) -> bool:
    """Tombstone slot; return whether it was actually removed."""
    num_slots, _ = _read_header(page)
    if slot < 0 or slot >= num_slots:
        return False

    offset, length = _read_slot(page, slot)
    if offset == TOMBSTONE_OFFSET:
        return False

    _write_slot(page, slot, TOMBSTONE_OFFSET, length)
    return True


def _read_header(page: bytes) -> tuple[int, int]:
    return struct.unpack_from(_HEADER_FORMAT, page, 0)


def _write_header(page: bytearray, num_slots: int, free_space_offset: int) -> None:
    struct.pack_into(_HEADER_FORMAT, page, 0, num_slots, free_space_offset)


def _slot_position(slot: int) -> int:
    return HEADER_SIZE + slot * SLOT_SIZE


def _read_slot(page: bytes, slot: int) -> tuple[int, int]:
    return struct.unpack_from(_SLOT_FORMAT, page, _slot_position(slot))


def _write_slot(page: bytearray, slot: int, offset: int, length: int) -> None:
    struct.pack_into(_SLOT_FORMAT, page, _slot_position(slot), offset, length)
