"""Page codecs used internally by ExtendibleHash.

Three page kinds share the same file: the metadata page (page 0), directory
pages holding bucket page ids, and bucket pages holding (key, RID) entries.
Every page is read and written through the BufferManager.
"""

import struct
from collections.abc import Iterator, Sequence

from engine.common.rid import RID

MAGIC = b"SHSH"
VERSION = 1

_META_FORMAT = "<4sBBBH"
META_HEADER_SIZE = struct.calcsize(_META_FORMAT)

_BUCKET_HEADER_FORMAT = "<BHHI"
BUCKET_HEADER_SIZE = struct.calcsize(_BUCKET_HEADER_FORMAT)

_ENTRY_HEADER_FORMAT = "<H"
ENTRY_HEADER_SIZE = struct.calcsize(_ENTRY_HEADER_FORMAT)

_RID_FORMAT = "<IH"
RID_SIZE = struct.calcsize(_RID_FORMAT)

_PAGE_ID_FORMAT = "<I"
PAGE_ID_SIZE = struct.calcsize(_PAGE_ID_FORMAT)

#: La página 0 guarda los metadatos, así que nunca es un bucket: sirve de centinela.
NO_OVERFLOW = 0


# --- metadata page ---------------------------------------------------------


def new_metadata(page_size: int, *, global_depth: int, max_depth: int) -> bytearray:
    """Build the metadata page of a brand-new index."""
    page = bytearray(page_size)
    write_metadata(page, global_depth=global_depth, max_depth=max_depth, directory_pages=[])
    return page


def write_metadata(
    page: bytearray, *, global_depth: int, max_depth: int, directory_pages: Sequence[int]
) -> None:
    """Overwrite the metadata header and the list of directory page ids."""
    struct.pack_into(
        _META_FORMAT, page, 0, MAGIC, VERSION, global_depth, max_depth, len(directory_pages)
    )
    offset = META_HEADER_SIZE
    for page_id in directory_pages:
        struct.pack_into(_PAGE_ID_FORMAT, page, offset, page_id)
        offset += PAGE_ID_SIZE


def read_metadata(page: bytes) -> tuple[int, int, list[int]]:
    """Return (global_depth, max_depth, directory_pages) from the metadata page."""
    magic, version, global_depth, max_depth, page_count = struct.unpack_from(_META_FORMAT, page, 0)
    if magic != MAGIC:
        raise ValueError(f"not an extendible hash file: magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported extendible hash version {version}")

    offset = META_HEADER_SIZE
    directory_pages = []
    for _ in range(page_count):
        (page_id,) = struct.unpack_from(_PAGE_ID_FORMAT, page, offset)
        directory_pages.append(page_id)
        offset += PAGE_ID_SIZE
    return global_depth, max_depth, directory_pages


def max_directory_pages(page_size: int) -> int:
    """How many directory page ids fit in the metadata page."""
    return (page_size - META_HEADER_SIZE) // PAGE_ID_SIZE


# --- directory pages -------------------------------------------------------


def directory_slots_per_page(page_size: int) -> int:
    """How many directory entries fit in one page."""
    return page_size // PAGE_ID_SIZE


def read_directory_slot(page: bytes, slot: int) -> int:
    (page_id,) = struct.unpack_from(_PAGE_ID_FORMAT, page, slot * PAGE_ID_SIZE)
    return page_id


def write_directory_slot(page: bytearray, slot: int, page_id: int) -> None:
    struct.pack_into(_PAGE_ID_FORMAT, page, slot * PAGE_ID_SIZE, page_id)


# --- bucket pages ----------------------------------------------------------


def entry_size(key: bytes) -> int:
    """Bytes a (key, RID) entry occupies inside a bucket page."""
    return ENTRY_HEADER_SIZE + len(key) + RID_SIZE


def max_key_size(page_size: int) -> int:
    """Largest encoded key that could ever fit in an empty bucket page."""
    return page_size - BUCKET_HEADER_SIZE - ENTRY_HEADER_SIZE - RID_SIZE


def new_bucket(page_size: int, *, local_depth: int) -> bytearray:
    """Build a fresh, empty bucket page of exactly page_size bytes."""
    page = bytearray(page_size)
    _write_bucket_header(
        page, local_depth=local_depth, count=0, free_offset=BUCKET_HEADER_SIZE, overflow=NO_OVERFLOW
    )
    return page


def local_depth(page: bytes) -> int:
    return _read_bucket_header(page)[0]


def set_local_depth(page: bytearray, depth: int) -> None:
    _, count, free_offset, overflow = _read_bucket_header(page)
    _write_bucket_header(
        page, local_depth=depth, count=count, free_offset=free_offset, overflow=overflow
    )


def entry_count(page: bytes) -> int:
    return _read_bucket_header(page)[1]


def overflow_page_id(page: bytes) -> int:
    return _read_bucket_header(page)[3]


def set_overflow_page_id(page: bytearray, page_id: int) -> None:
    depth, count, free_offset, _ = _read_bucket_header(page)
    _write_bucket_header(
        page, local_depth=depth, count=count, free_offset=free_offset, overflow=page_id
    )


def free_space(page: bytes) -> int:
    """Bytes still available for new entries."""
    return len(page) - _read_bucket_header(page)[2]


def iter_entries(page: bytes) -> Iterator[tuple[bytes, RID]]:
    """Yield every (encoded key, RID) stored in this page, in insertion order."""
    _, count, _, _ = _read_bucket_header(page)
    offset = BUCKET_HEADER_SIZE
    for _ in range(count):
        (key_size,) = struct.unpack_from(_ENTRY_HEADER_FORMAT, page, offset)
        offset += ENTRY_HEADER_SIZE
        key = bytes(page[offset : offset + key_size])
        offset += key_size
        page_id, slot = struct.unpack_from(_RID_FORMAT, page, offset)
        offset += RID_SIZE
        yield key, RID(page_id=page_id, slot=slot)


def add_entry(page: bytearray, key: bytes, rid: RID) -> bool:
    """Append an entry; returns False when the page has no room left."""
    depth, count, free_offset, overflow = _read_bucket_header(page)
    if entry_size(key) > len(page) - free_offset:
        return False

    offset = free_offset
    struct.pack_into(_ENTRY_HEADER_FORMAT, page, offset, len(key))
    offset += ENTRY_HEADER_SIZE
    page[offset : offset + len(key)] = key
    offset += len(key)
    struct.pack_into(_RID_FORMAT, page, offset, rid.page_id, rid.slot)
    offset += RID_SIZE

    _write_bucket_header(
        page, local_depth=depth, count=count + 1, free_offset=offset, overflow=overflow
    )
    return True


def write_entries(page: bytearray, entries: Sequence[tuple[bytes, RID]]) -> None:
    """Rewrite the page with exactly these entries, keeping depth and overflow."""
    depth, _, _, overflow = _read_bucket_header(page)
    _write_bucket_header(
        page, local_depth=depth, count=0, free_offset=BUCKET_HEADER_SIZE, overflow=overflow
    )
    for key, rid in entries:
        add_entry(page, key, rid)


def remove_entries(page: bytearray, key: bytes, rid: RID | None = None) -> int:
    """Drop the entries matching key (and rid when given); returns how many were removed."""
    kept: list[tuple[bytes, RID]] = []
    removed = 0
    for stored_key, stored_rid in iter_entries(page):
        matches = stored_key == key and (rid is None or stored_rid == rid)
        if matches:
            removed += 1
        else:
            kept.append((stored_key, stored_rid))

    if removed:
        write_entries(page, kept)
    return removed


def _write_bucket_header(
    page: bytearray, *, local_depth: int, count: int, free_offset: int, overflow: int
) -> None:
    struct.pack_into(_BUCKET_HEADER_FORMAT, page, 0, local_depth, count, free_offset, overflow)


def _read_bucket_header(page: bytes) -> tuple[int, int, int, int]:
    return struct.unpack_from(_BUCKET_HEADER_FORMAT, page, 0)
