import struct

import pytest

from engine.common.rid import RID
from engine.indexes.rtree import _page_codec as codec
from engine.indexes.rtree._page_codec import DecodedRTreePage
from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point

PAGE_SIZE = 4096
MAX_ENTRIES = 8


def _leaf(entries=()) -> DecodedRTreePage:
    return DecodedRTreePage(is_leaf=True, max_entries=MAX_ENTRIES, entries=tuple(entries))


def _internal(entries=()) -> DecodedRTreePage:
    return DecodedRTreePage(is_leaf=False, max_entries=MAX_ENTRIES, entries=tuple(entries))


def _sample_leaf() -> DecodedRTreePage:
    return _leaf(
        [
            (Point(1.5, -2.25), RID(3, 7)),
            (Point(0.0, 0.0), RID(0, 0)),
            (Point(-1e9, 1e-9), RID(2**40, 12)),
        ]
    )


def _sample_internal() -> DecodedRTreePage:
    return _internal(
        [
            (MBR(0.0, 0.0, 1.0, 1.0), 1),
            (MBR(-5.5, -3.0, 2.0, 4.75), 42),
            (MBR(10.0, 10.0, 10.0, 10.0), 0),
        ]
    )


def test_capacities_for_4096() -> None:
    assert codec.HEADER_SIZE == 16
    assert codec.LEAF_ENTRY_SIZE == 32
    assert codec.INTERNAL_ENTRY_SIZE == 40
    assert codec.leaf_capacity(PAGE_SIZE) == 127
    assert codec.internal_capacity(PAGE_SIZE) == 102


def test_empty_leaf_roundtrip() -> None:
    page = _leaf()
    assert codec.decode_page(codec.encode_page(PAGE_SIZE, page), PAGE_SIZE) == page


def test_leaf_roundtrip() -> None:
    page = _sample_leaf()
    decoded = codec.decode_page(codec.encode_page(PAGE_SIZE, page), PAGE_SIZE)
    assert decoded == page
    assert all(isinstance(point, Point) and isinstance(rid, RID) for point, rid in decoded.entries)


def test_internal_roundtrip() -> None:
    page = _sample_internal()
    decoded = codec.decode_page(codec.encode_page(PAGE_SIZE, page), PAGE_SIZE)
    assert decoded == page
    assert all(isinstance(mbr, MBR) for mbr, _child in decoded.entries)


MAX_ORDER = 102


def _full_leaf(max_entries: int) -> DecodedRTreePage:
    return DecodedRTreePage(
        is_leaf=True,
        max_entries=max_entries,
        entries=tuple((Point(i, -i), RID(i, i + 1)) for i in range(max_entries)),
    )


def _full_internal(max_entries: int) -> DecodedRTreePage:
    return DecodedRTreePage(
        is_leaf=False,
        max_entries=max_entries,
        entries=tuple((MBR(i, i, i + 1, i + 1), i) for i in range(max_entries)),
    )


@pytest.mark.parametrize("build", [_full_leaf, _full_internal])
def test_global_max_order_is_valid_for_both_node_types(build) -> None:
    page = build(MAX_ORDER)
    assert codec.decode_page(codec.encode_page(PAGE_SIZE, page), PAGE_SIZE) == page


@pytest.mark.parametrize("build", [_full_leaf, _full_internal])
def test_max_order_above_global_limit_rejected_on_encode(build) -> None:
    page = DecodedRTreePage(
        is_leaf=build is _full_leaf, max_entries=MAX_ORDER + 1, entries=build(2).entries
    )
    with pytest.raises(ValueError, match="max_entries"):
        codec.encode_page(PAGE_SIZE, page)


@pytest.mark.parametrize("page", [_leaf(), _internal()])
def test_max_order_above_global_limit_rejected_on_decode(page: DecodedRTreePage) -> None:
    data = _corrupt(codec.encode_page(PAGE_SIZE, page), 6, struct.pack(">H", MAX_ORDER + 1))
    with pytest.raises(ValueError, match="max_entries"):
        codec.decode_page(data, PAGE_SIZE)


@pytest.mark.parametrize("page", [_leaf(), _sample_leaf(), _sample_internal()])
def test_encoded_length_is_page_size(page: DecodedRTreePage) -> None:
    assert len(codec.encode_page(PAGE_SIZE, page)) == PAGE_SIZE
    assert len(codec.encode_page(512, page)) == 512


def test_header_layout() -> None:
    data = codec.encode_page(PAGE_SIZE, _sample_internal())
    assert data[:16] == b"RN" + bytes((1, 0)) + struct.pack(">HHQ", 3, MAX_ENTRIES, 0)


def test_unused_space_is_zero() -> None:
    leaf = codec.encode_page(PAGE_SIZE, _sample_leaf())
    assert leaf[codec.HEADER_SIZE + 3 * codec.LEAF_ENTRY_SIZE :] == bytes(
        PAGE_SIZE - codec.HEADER_SIZE - 3 * codec.LEAF_ENTRY_SIZE
    )
    internal = codec.encode_page(PAGE_SIZE, _sample_internal())
    assert internal[codec.HEADER_SIZE + 3 * codec.INTERNAL_ENTRY_SIZE :] == bytes(
        PAGE_SIZE - codec.HEADER_SIZE - 3 * codec.INTERNAL_ENTRY_SIZE
    )


def _corrupt(data: bytes, offset: int, raw: bytes) -> bytes:
    return data[:offset] + raw + data[offset + len(raw) :]


def test_corrupt_magic_rejected() -> None:
    data = _corrupt(codec.encode_page(PAGE_SIZE, _sample_leaf()), 0, b"XX")
    with pytest.raises(ValueError, match="magic"):
        codec.decode_page(data, PAGE_SIZE)


def test_corrupt_version_rejected() -> None:
    data = _corrupt(codec.encode_page(PAGE_SIZE, _sample_leaf()), 2, bytes((2,)))
    with pytest.raises(ValueError, match="version"):
        codec.decode_page(data, PAGE_SIZE)


def test_invalid_node_type_rejected() -> None:
    data = _corrupt(codec.encode_page(PAGE_SIZE, _sample_leaf()), 3, bytes((7,)))
    with pytest.raises(ValueError, match="node type"):
        codec.decode_page(data, PAGE_SIZE)


def test_nonzero_reserved_rejected() -> None:
    data = _corrupt(codec.encode_page(PAGE_SIZE, _sample_leaf()), 8, struct.pack(">Q", 1))
    with pytest.raises(ValueError, match="reserved"):
        codec.decode_page(data, PAGE_SIZE)


def test_count_above_max_entries_rejected() -> None:
    data = _corrupt(
        codec.encode_page(PAGE_SIZE, _sample_leaf()), 4, struct.pack(">H", MAX_ENTRIES + 1)
    )
    with pytest.raises(ValueError, match="entry count"):
        codec.decode_page(data, PAGE_SIZE)


def test_count_beyond_page_capacity_rejected() -> None:
    data = _corrupt(codec.encode_page(PAGE_SIZE, _sample_leaf()), 4, struct.pack(">HH", 200, 300))
    with pytest.raises(ValueError):
        codec.decode_page(data, PAGE_SIZE)


@pytest.mark.parametrize("max_entries", [0, 1, 103, 128])
def test_invalid_max_entries_rejected_on_decode(max_entries: int) -> None:
    data = _corrupt(codec.encode_page(PAGE_SIZE, _leaf()), 6, struct.pack(">H", max_entries))
    with pytest.raises(ValueError, match="max_entries"):
        codec.decode_page(data, PAGE_SIZE)


def test_node_with_more_entries_than_max_entries_rejected() -> None:
    page = DecodedRTreePage(
        is_leaf=True,
        max_entries=2,
        entries=tuple((Point(i, i), RID(0, i)) for i in range(3)),
    )
    with pytest.raises(ValueError, match="entry count"):
        codec.encode_page(PAGE_SIZE, page)


def test_node_too_large_for_page_rejected() -> None:
    cap = codec.internal_capacity(PAGE_SIZE)
    page = DecodedRTreePage(
        is_leaf=False,
        max_entries=cap + 1,
        entries=tuple((MBR(0, 0, 1, 1), i) for i in range(cap + 1)),
    )
    with pytest.raises(ValueError, match="max_entries"):
        codec.encode_page(PAGE_SIZE, page)


def test_negative_child_page_id_rejected_on_encode() -> None:
    with pytest.raises(ValueError, match="child_page_id"):
        codec.encode_page(PAGE_SIZE, _internal([(MBR(0, 0, 1, 1), -1)]))


def test_negative_child_page_id_rejected_on_decode() -> None:
    data = codec.encode_page(PAGE_SIZE, _sample_internal())
    offset = codec.HEADER_SIZE + codec.INTERNAL_ENTRY_SIZE - 8
    data = _corrupt(data, offset, struct.pack(">q", -5))
    with pytest.raises(ValueError, match="child_page_id"):
        codec.decode_page(data, PAGE_SIZE)


@pytest.mark.parametrize("size", [0, 10, PAGE_SIZE - 1, PAGE_SIZE + 1])
def test_wrong_page_size_rejected(size: int) -> None:
    data = codec.encode_page(PAGE_SIZE, _sample_leaf())
    data = data[:size] if size <= PAGE_SIZE else data + b"\x00"
    with pytest.raises(ValueError, match="page size"):
        codec.decode_page(data, PAGE_SIZE)


def test_page_size_too_small_rejected() -> None:
    with pytest.raises(ValueError, match="page_size"):
        codec.encode_page(codec.HEADER_SIZE + codec.LEAF_ENTRY_SIZE, _leaf())
