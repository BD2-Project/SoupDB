"""Binary page codec for persistent extendible hashing."""

import struct
from dataclasses import dataclass

from engine.common.rid import RID
from engine.indexes._key_codec import decode_key, encode_key
from engine.indexes.base import Key

_MAGIC = b"SEH1"
_VERSION = 1

PAGE_TYPE_METADATA = 0
PAGE_TYPE_DIRECTORY = 1
PAGE_TYPE_BUCKET = 2

_NONE_PAGE_ID = -1

_HEADER_FORMAT = "<4sBBHii"
HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

_PAGE_ID_FORMAT = "<i"
_PAGE_ID_SIZE = struct.calcsize(_PAGE_ID_FORMAT)

_LENGTH_FORMAT = "<I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FORMAT)

_RID_FORMAT = "<ii"
_RID_SIZE = struct.calcsize(_RID_FORMAT)


@dataclass(frozen=True)
class Metadata:
    global_depth: int
    first_directory_page_id: int
    max_global_depth: int


@dataclass(frozen=True)
class DirectoryPage:
    next_page_id: int | None
    bucket_page_ids: tuple[int, ...]


@dataclass(frozen=True)
class BucketPage:
    local_depth: int
    overflow_page_id: int | None
    entries: tuple[tuple[Key, RID], ...]


def directory_capacity(page_size: int) -> int:
    """Maximum number of directory entries that fit in one page."""
    if page_size < HEADER_SIZE:
        raise ValueError(f"page_size must be at least {HEADER_SIZE} bytes")

    return (page_size - HEADER_SIZE) // _PAGE_ID_SIZE


def encode_metadata(
    page_size: int,
    *,
    global_depth: int,
    first_directory_page_id: int,
    max_global_depth: int,
) -> bytes:
    """Encode extendible-hash metadata into one page."""
    if global_depth < 0:
        raise ValueError("global_depth must be non-negative")
    if max_global_depth < global_depth or max_global_depth > 64:
        raise ValueError("invalid max_global_depth")

    page = _new_page(page_size)
    _write_header(
        page,
        page_type=PAGE_TYPE_METADATA,
        count=global_depth,
        first_value=first_directory_page_id,
        second_value=max_global_depth,
    )
    return bytes(page)


def decode_metadata(page: bytes) -> Metadata:
    """Decode an extendible-hash metadata page."""
    page_type, global_depth, first_directory_page_id, max_global_depth = _read_header(page)

    if page_type != PAGE_TYPE_METADATA:
        raise ValueError("page is not extendible-hash metadata")
    if first_directory_page_id < 0:
        raise ValueError("metadata contains invalid directory page id")
    if max_global_depth < global_depth or max_global_depth > 64:
        raise ValueError("metadata contains invalid depth limits")

    return Metadata(
        global_depth=global_depth,
        first_directory_page_id=first_directory_page_id,
        max_global_depth=max_global_depth,
    )


def encode_directory(
    page_size: int,
    directory: DirectoryPage,
) -> bytes:
    """Encode one paginated directory segment."""
    capacity = directory_capacity(page_size)

    if len(directory.bucket_page_ids) > capacity:
        raise ValueError("directory entries do not fit in one page")

    payload = bytearray()

    for page_id in directory.bucket_page_ids:
        if page_id < 0:
            raise ValueError("bucket page id must be non-negative")
        try:
            payload.extend(struct.pack(_PAGE_ID_FORMAT, page_id))
        except struct.error as exc:
            raise ValueError("bucket page id is outside signed 32-bit range") from exc

    page = _new_page(page_size)
    _write_header(
        page,
        page_type=PAGE_TYPE_DIRECTORY,
        count=len(directory.bucket_page_ids),
        first_value=_encode_page_id(directory.next_page_id),
        second_value=_NONE_PAGE_ID,
    )
    page[HEADER_SIZE : HEADER_SIZE + len(payload)] = payload
    return bytes(page)


def decode_directory(page: bytes) -> DirectoryPage:
    """Decode one paginated directory segment."""
    page_type, count, next_page_id, unused = _read_header(page)

    if page_type != PAGE_TYPE_DIRECTORY:
        raise ValueError("page is not an extendible-hash directory")
    if unused != _NONE_PAGE_ID:
        raise ValueError("directory page contains invalid header data")

    offset = HEADER_SIZE
    entries: list[int] = []

    for _ in range(count):
        end = offset + _PAGE_ID_SIZE

        if end > len(page):
            raise ValueError("truncated directory entry")

        page_id = struct.unpack_from(_PAGE_ID_FORMAT, page, offset)[0]
        if page_id < 0:
            raise ValueError("directory contains invalid bucket page id")

        entries.append(page_id)
        offset = end

    return DirectoryPage(
        next_page_id=_decode_page_id(next_page_id),
        bucket_page_ids=tuple(entries),
    )


def encode_bucket(
    page_size: int,
    bucket: BucketPage,
) -> bytes:
    """Encode one hash bucket page."""
    if bucket.local_depth < 0 or bucket.local_depth > 64:
        raise ValueError("local_depth must be between 0 and 64")

    payload = bytearray()

    for key, rid in bucket.entries:
        encoded_key = encode_key(key)
        payload.extend(struct.pack(_LENGTH_FORMAT, len(encoded_key)))
        payload.extend(encoded_key)

        try:
            payload.extend(struct.pack(_RID_FORMAT, rid.page_id, rid.slot))
        except struct.error as exc:
            raise ValueError("RID is outside signed 32-bit range") from exc

    _ensure_fits(page_size, len(payload))

    page = _new_page(page_size)
    _write_header(
        page,
        page_type=PAGE_TYPE_BUCKET,
        count=len(bucket.entries),
        first_value=bucket.local_depth,
        second_value=_encode_page_id(bucket.overflow_page_id),
    )
    page[HEADER_SIZE : HEADER_SIZE + len(payload)] = payload

    return bytes(page)


def decode_bucket(page: bytes) -> BucketPage:
    """Decode one hash bucket page."""
    page_type, count, local_depth, overflow_page_id = _read_header(page)

    if page_type != PAGE_TYPE_BUCKET:
        raise ValueError("page is not an extendible-hash bucket")
    if local_depth < 0 or local_depth > 64:
        raise ValueError("bucket contains invalid local depth")

    offset = HEADER_SIZE
    entries: list[tuple[Key, RID]] = []

    for _ in range(count):
        key_length, offset = _read_length(page, offset)
        key_end = offset + key_length

        if key_end > len(page):
            raise ValueError("truncated bucket key")

        key = decode_key(page[offset:key_end])
        offset = key_end

        rid_end = offset + _RID_SIZE
        if rid_end > len(page):
            raise ValueError("truncated bucket RID")

        page_id, slot = struct.unpack_from(_RID_FORMAT, page, offset)
        entries.append((key, RID(page_id=page_id, slot=slot)))
        offset = rid_end

    return BucketPage(
        local_depth=local_depth,
        overflow_page_id=_decode_page_id(overflow_page_id),
        entries=tuple(entries),
    )


def page_type(page: bytes) -> int:
    """Return the persisted extendible-hash page type."""
    return _read_header(page)[0]


def _new_page(page_size: int) -> bytearray:
    if page_size < HEADER_SIZE:
        raise ValueError(f"page_size must be at least {HEADER_SIZE} bytes")

    return bytearray(page_size)


def _ensure_fits(page_size: int, payload_size: int) -> None:
    if HEADER_SIZE + payload_size > page_size:
        raise ValueError("extendible-hash page payload does not fit")


def _write_header(
    page: bytearray,
    *,
    page_type: int,
    count: int,
    first_value: int,
    second_value: int,
) -> None:
    if count < 0 or count > 0xFFFF:
        raise ValueError("invalid extendible-hash page count")

    try:
        struct.pack_into(
            _HEADER_FORMAT,
            page,
            0,
            _MAGIC,
            _VERSION,
            page_type,
            count,
            first_value,
            second_value,
        )
    except struct.error as exc:
        raise ValueError("extendible-hash header value is out of range") from exc


def _read_header(page: bytes) -> tuple[int, int, int, int]:
    if len(page) < HEADER_SIZE:
        raise ValueError("truncated extendible-hash page header")

    magic, version, page_type_value, count, first_value, second_value = struct.unpack_from(
        _HEADER_FORMAT,
        page,
        0,
    )

    if magic != _MAGIC:
        raise ValueError("invalid extendible-hash page magic")
    if version != _VERSION:
        raise ValueError("unsupported extendible-hash page version")
    if page_type_value not in {
        PAGE_TYPE_METADATA,
        PAGE_TYPE_DIRECTORY,
        PAGE_TYPE_BUCKET,
    }:
        raise ValueError("unknown extendible-hash page type")

    return page_type_value, count, first_value, second_value


def _encode_page_id(page_id: int | None) -> int:
    if page_id is None:
        return _NONE_PAGE_ID
    if page_id < 0:
        raise ValueError("page id must be non-negative")
    return page_id


def _decode_page_id(page_id: int) -> int | None:
    if page_id == _NONE_PAGE_ID:
        return None
    if page_id < 0:
        raise ValueError("invalid negative page id")
    return page_id


def _read_length(data: bytes, offset: int) -> tuple[int, int]:
    end = offset + _LENGTH_SIZE

    if end > len(data):
        raise ValueError("truncated bucket key length")

    return struct.unpack_from(_LENGTH_FORMAT, data, offset)[0], end
