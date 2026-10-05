import pytest

from engine.common.rid import RID
from engine.indexes import _bplus_page as codec

PAGE_SIZE = 256


def test_metadata_roundtrip() -> None:
    page = codec.encode_metadata(
        PAGE_SIZE,
        root_page_id=3,
        first_leaf_page_id=7,
    )

    assert len(page) == PAGE_SIZE
    assert codec.page_type(page) == codec.PAGE_TYPE_METADATA
    assert codec.decode_metadata(page) == codec.Metadata(
        root_page_id=3,
        first_leaf_page_id=7,
    )


def test_empty_metadata_roundtrip() -> None:
    page = codec.encode_metadata(
        PAGE_SIZE,
        root_page_id=None,
        first_leaf_page_id=None,
    )

    assert codec.decode_metadata(page) == codec.Metadata(
        root_page_id=None,
        first_leaf_page_id=None,
    )


def test_leaf_roundtrip() -> None:
    node = codec.LeafNode(
        parent_page_id=4,
        next_leaf_page_id=8,
        entries=(
            (10, RID(1, 2)),
            (20, RID(3, 4)),
        ),
    )

    page = codec.encode_leaf(PAGE_SIZE, node)

    assert len(page) == PAGE_SIZE
    assert codec.page_type(page) == codec.PAGE_TYPE_LEAF
    assert codec.decode_leaf(page) == node


def test_leaf_supports_string_and_composite_keys() -> None:
    node = codec.LeafNode(
        parent_page_id=None,
        next_leaf_page_id=None,
        entries=(
            ("árbol", RID(1, 0)),
            (("department", 42), RID(2, 1)),
        ),
    )

    assert codec.decode_leaf(codec.encode_leaf(PAGE_SIZE, node)) == node


def test_empty_leaf_roundtrip() -> None:
    node = codec.LeafNode(
        parent_page_id=None,
        next_leaf_page_id=None,
        entries=(),
    )

    assert codec.decode_leaf(codec.encode_leaf(PAGE_SIZE, node)) == node


def test_internal_roundtrip() -> None:
    node = codec.InternalNode(
        parent_page_id=9,
        keys=(10, 20),
        children=(1, 2, 3),
    )

    page = codec.encode_internal(PAGE_SIZE, node)

    assert len(page) == PAGE_SIZE
    assert codec.page_type(page) == codec.PAGE_TYPE_INTERNAL
    assert codec.decode_internal(page) == node


def test_internal_supports_composite_keys() -> None:
    node = codec.InternalNode(
        parent_page_id=None,
        keys=(("a", 1), ("b", 2)),
        children=(4, 5, 6),
    )

    assert codec.decode_internal(codec.encode_internal(PAGE_SIZE, node)) == node


def test_internal_requires_one_more_child_than_keys() -> None:
    node = codec.InternalNode(
        parent_page_id=None,
        keys=(10, 20),
        children=(1, 2),
    )

    with pytest.raises(ValueError):
        codec.encode_internal(PAGE_SIZE, node)


def test_leaf_rejects_payload_larger_than_page() -> None:
    node = codec.LeafNode(
        parent_page_id=None,
        next_leaf_page_id=None,
        entries=(("x" * PAGE_SIZE, RID(0, 0)),),
    )

    with pytest.raises(ValueError):
        codec.encode_leaf(PAGE_SIZE, node)


def test_internal_rejects_payload_larger_than_page() -> None:
    node = codec.InternalNode(
        parent_page_id=None,
        keys=("x" * PAGE_SIZE,),
        children=(1, 2),
    )

    with pytest.raises(ValueError):
        codec.encode_internal(PAGE_SIZE, node)


def test_decode_rejects_invalid_magic() -> None:
    page = bytearray(codec.encode_metadata(PAGE_SIZE, None, None))
    page[0:4] = b"BAD!"

    with pytest.raises(ValueError):
        codec.decode_metadata(bytes(page))


def test_decode_leaf_rejects_metadata_page() -> None:
    page = codec.encode_metadata(PAGE_SIZE, None, None)

    with pytest.raises(ValueError):
        codec.decode_leaf(page)


def test_decode_internal_rejects_leaf_page() -> None:
    page = codec.encode_leaf(
        PAGE_SIZE,
        codec.LeafNode(
            parent_page_id=None,
            next_leaf_page_id=None,
            entries=(),
        ),
    )

    with pytest.raises(ValueError):
        codec.decode_internal(page)


def test_negative_metadata_page_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        codec.encode_metadata(
            PAGE_SIZE,
            root_page_id=-2,
            first_leaf_page_id=None,
        )


def test_page_size_smaller_than_header_is_rejected() -> None:
    with pytest.raises(ValueError):
        codec.encode_metadata(
            codec.HEADER_SIZE - 1,
            root_page_id=None,
            first_leaf_page_id=None,
        )
