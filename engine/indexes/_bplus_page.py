"""Binary page codec for persistent B+ tree pages."""

import struct
from dataclasses import dataclass

from engine.common.rid import RID
from engine.indexes._key_codec import decode_key, encode_key
from engine.indexes.base import Key

_MAGIC = b"SBP1"
_VERSION = 1

PAGE_TYPE_METADATA = 0
PAGE_TYPE_LEAF = 1
PAGE_TYPE_INTERNAL = 2

_NONE_PAGE_ID = -1

_HEADER_FORMAT = "<4sBBHii"
HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

_LENGTH_FORMAT = "<I"
_LENGTH_SIZE = struct.calcsize(_LENGTH_FORMAT)

_RID_FORMAT = "<ii"
_RID_SIZE = struct.calcsize(_RID_FORMAT)

_PAGE_ID_FORMAT = "<i"
_PAGE_ID_SIZE = struct.calcsize(_PAGE_ID_FORMAT)


@dataclass(frozen=True)
class Metadata:
    root_page_id: int | None
    first_leaf_page_id: int | None


@dataclass(frozen=True)
class LeafNode:
    parent_page_id: int | None
    next_leaf_page_id: int | None
    entries: tuple[tuple[Key, RID], ...]


@dataclass(frozen=True)
class InternalNode:
    parent_page_id: int | None
    keys: tuple[Key, ...]
    children: tuple[int, ...]


def encode_metadata(
    page_size: int,
    root_page_id: int | None,
    first_leaf_page_id: int | None,
) -> bytes:
    """Encode B+ tree metadata into one fixed-size page."""
    page = _new_page(page_size)
    _write_header(
        page,
        page_type=PAGE_TYPE_METADATA,
        count=0,
        first_page_id=root_page_id,
        second_page_id=first_leaf_page_id,
    )
    return bytes(page)


def decode_metadata(page: bytes) -> Metadata:
    """Decode a B+ tree metadata page."""
    page_type, count, root_page_id, first_leaf_page_id = _read_header(page)

    if page_type != PAGE_TYPE_METADATA:
        raise ValueError("page is not B+ tree metadata")
    if count != 0:
        raise ValueError("metadata page must have zero entries")

    return Metadata(
        root_page_id=_decode_page_id(root_page_id),
        first_leaf_page_id=_decode_page_id(first_leaf_page_id),
    )


def encode_leaf(page_size: int, node: LeafNode) -> bytes:
    """Encode a leaf node into one fixed-size page."""
    page = _new_page(page_size)
    payload = bytearray()

    for key, rid in node.entries:
        encoded_key = encode_key(key)
        payload.extend(struct.pack(_LENGTH_FORMAT, len(encoded_key)))
        payload.extend(encoded_key)

        try:
            payload.extend(struct.pack(_RID_FORMAT, rid.page_id, rid.slot))
        except struct.error as exc:
            raise ValueError("RID is outside signed 32-bit range") from exc

    _ensure_fits(page_size, len(payload))
    _write_header(
        page,
        page_type=PAGE_TYPE_LEAF,
        count=len(node.entries),
        first_page_id=node.parent_page_id,
        second_page_id=node.next_leaf_page_id,
    )
    page[HEADER_SIZE : HEADER_SIZE + len(payload)] = payload

    return bytes(page)


def decode_leaf(page: bytes) -> LeafNode:
    """Decode a leaf node page."""
    page_type, count, parent_page_id, next_leaf_page_id = _read_header(page)

    if page_type != PAGE_TYPE_LEAF:
        raise ValueError("page is not a B+ tree leaf")

    offset = HEADER_SIZE
    entries: list[tuple[Key, RID]] = []

    for _ in range(count):
        key_length, offset = _read_length(page, offset)
        key_end = offset + key_length

        if key_end > len(page):
            raise ValueError("truncated leaf key")

        key = decode_key(page[offset:key_end])
        offset = key_end

        rid_end = offset + _RID_SIZE
        if rid_end > len(page):
            raise ValueError("truncated leaf RID")

        page_id, slot = struct.unpack_from(_RID_FORMAT, page, offset)
        offset = rid_end
        entries.append((key, RID(page_id=page_id, slot=slot)))

    return LeafNode(
        parent_page_id=_decode_page_id(parent_page_id),
        next_leaf_page_id=_decode_page_id(next_leaf_page_id),
        entries=tuple(entries),
    )


def encode_internal(page_size: int, node: InternalNode) -> bytes:
    """Encode an internal node into one fixed-size page."""
    if len(node.children) != len(node.keys) + 1:
        raise ValueError("internal node must have exactly one more child than keys")

    payload = bytearray()

    try:
        payload.extend(struct.pack(_PAGE_ID_FORMAT, node.children[0]))
    except struct.error as exc:
        raise ValueError("child page id is outside signed 32-bit range") from exc

    for key, child_page_id in zip(node.keys, node.children[1:], strict=True):
        encoded_key = encode_key(key)
        payload.extend(struct.pack(_LENGTH_FORMAT, len(encoded_key)))
        payload.extend(encoded_key)

        try:
            payload.extend(struct.pack(_PAGE_ID_FORMAT, child_page_id))
        except struct.error as exc:
            raise ValueError("child page id is outside signed 32-bit range") from exc

    _ensure_fits(page_size, len(payload))
    page = _new_page(page_size)
    _write_header(
        page,
        page_type=PAGE_TYPE_INTERNAL,
        count=len(node.keys),
        first_page_id=node.parent_page_id,
        second_page_id=None,
    )
    page[HEADER_SIZE : HEADER_SIZE + len(payload)] = payload

    return bytes(page)


def decode_internal(page: bytes) -> InternalNode:
    """Decode an internal node page."""
    page_type, count, parent_page_id, unused = _read_header(page)

    if page_type != PAGE_TYPE_INTERNAL:
        raise ValueError("page is not a B+ tree internal node")
    if unused != _NONE_PAGE_ID:
        raise ValueError("internal node contains invalid header data")

    offset = HEADER_SIZE
    child_end = offset + _PAGE_ID_SIZE

    if child_end > len(page):
        raise ValueError("truncated internal child")

    first_child = struct.unpack_from(_PAGE_ID_FORMAT, page, offset)[0]
    offset = child_end

    keys: list[Key] = []
    children = [first_child]

    for _ in range(count):
        key_length, offset = _read_length(page, offset)
        key_end = offset + key_length

        if key_end > len(page):
            raise ValueError("truncated internal key")

        keys.append(decode_key(page[offset:key_end]))
        offset = key_end

        child_end = offset + _PAGE_ID_SIZE
        if child_end > len(page):
            raise ValueError("truncated internal child")

        children.append(struct.unpack_from(_PAGE_ID_FORMAT, page, offset)[0])
        offset = child_end

    return InternalNode(
        parent_page_id=_decode_page_id(parent_page_id),
        keys=tuple(keys),
        children=tuple(children),
    )


def page_type(page: bytes) -> int:
    """Return the persisted B+ page type."""
    return _read_header(page)[0]


def _new_page(page_size: int) -> bytearray:
    if page_size < HEADER_SIZE:
        raise ValueError(f"page_size must be at least {HEADER_SIZE} bytes")

    return bytearray(page_size)


def _ensure_fits(page_size: int, payload_size: int) -> None:
    if HEADER_SIZE + payload_size > page_size:
        raise ValueError("B+ node does not fit in one page")


def _write_header(
    page: bytearray,
    *,
    page_type: int,
    count: int,
    first_page_id: int | None,
    second_page_id: int | None,
) -> None:
    if count > 0xFFFF:
        raise ValueError("too many entries for B+ page header")

    try:
        struct.pack_into(
            _HEADER_FORMAT,
            page,
            0,
            _MAGIC,
            _VERSION,
            page_type,
            count,
            _encode_page_id(first_page_id),
            _encode_page_id(second_page_id),
        )
    except struct.error as exc:
        raise ValueError("page id is outside signed 32-bit range") from exc


def _read_header(page: bytes) -> tuple[int, int, int, int]:
    if len(page) < HEADER_SIZE:
        raise ValueError("truncated B+ page header")

    magic, version, page_type_value, count, first_page_id, second_page_id = struct.unpack_from(
        _HEADER_FORMAT, page, 0
    )

    if magic != _MAGIC:
        raise ValueError("invalid B+ page magic")
    if version != _VERSION:
        raise ValueError("unsupported B+ page version")
    if page_type_value not in {PAGE_TYPE_METADATA, PAGE_TYPE_LEAF, PAGE_TYPE_INTERNAL}:
        raise ValueError("unknown B+ page type")

    return page_type_value, count, first_page_id, second_page_id


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
        raise ValueError("truncated B+ entry length")

    return struct.unpack_from(_LENGTH_FORMAT, data, offset)[0], end
