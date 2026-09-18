import pytest

from engine.common.rid import RID
from engine.indexes import _extendible_hash_page as codec

PAGE_SIZE = 128


def test_metadata_roundtrip() -> None:
    page = codec.encode_metadata(
        PAGE_SIZE,
        global_depth=3,
        first_directory_page_id=1,
        max_global_depth=16,
    )

    assert len(page) == PAGE_SIZE
    assert codec.page_type(page) == codec.PAGE_TYPE_METADATA
    assert codec.decode_metadata(page) == codec.Metadata(
        global_depth=3,
        first_directory_page_id=1,
        max_global_depth=16,
    )


def test_metadata_rejects_invalid_depth_limit() -> None:
    with pytest.raises(ValueError):
        codec.encode_metadata(
            PAGE_SIZE,
            global_depth=5,
            first_directory_page_id=1,
            max_global_depth=4,
        )


def test_directory_roundtrip() -> None:
    directory = codec.DirectoryPage(
        next_page_id=8,
        bucket_page_ids=(2, 3, 3, 4),
    )

    page = codec.encode_directory(PAGE_SIZE, directory)

    assert len(page) == PAGE_SIZE
    assert codec.page_type(page) == codec.PAGE_TYPE_DIRECTORY
    assert codec.decode_directory(page) == directory


def test_last_directory_page_has_no_next_page() -> None:
    directory = codec.DirectoryPage(
        next_page_id=None,
        bucket_page_ids=(2, 2),
    )

    assert codec.decode_directory(codec.encode_directory(PAGE_SIZE, directory)) == directory


def test_directory_capacity_matches_page_size() -> None:
    expected = (PAGE_SIZE - codec.HEADER_SIZE) // 4

    assert codec.directory_capacity(PAGE_SIZE) == expected


def test_directory_rejects_too_many_entries() -> None:
    entries = tuple(range(codec.directory_capacity(PAGE_SIZE) + 1))

    with pytest.raises(ValueError):
        codec.encode_directory(
            PAGE_SIZE,
            codec.DirectoryPage(
                next_page_id=None,
                bucket_page_ids=entries,
            ),
        )


def test_bucket_roundtrip() -> None:
    bucket = codec.BucketPage(
        local_depth=2,
        overflow_page_id=None,
        entries=(
            (10, RID(1, 2)),
            (20, RID(3, 4)),
        ),
    )

    page = codec.encode_bucket(PAGE_SIZE, bucket)

    assert len(page) == PAGE_SIZE
    assert codec.page_type(page) == codec.PAGE_TYPE_BUCKET
    assert codec.decode_bucket(page) == bucket


def test_bucket_supports_string_and_composite_keys() -> None:
    bucket = codec.BucketPage(
        local_depth=1,
        overflow_page_id=None,
        entries=(
            ("árbol", RID(1, 0)),
            (("department", 42), RID(2, 1)),
        ),
    )

    assert codec.decode_bucket(codec.encode_bucket(PAGE_SIZE, bucket)) == bucket


def test_bucket_overflow_pointer_roundtrip() -> None:
    bucket = codec.BucketPage(
        local_depth=4,
        overflow_page_id=9,
        entries=((7, RID(0, 1)),),
    )

    assert codec.decode_bucket(codec.encode_bucket(PAGE_SIZE, bucket)) == bucket


def test_empty_bucket_roundtrip() -> None:
    bucket = codec.BucketPage(
        local_depth=0,
        overflow_page_id=None,
        entries=(),
    )

    assert codec.decode_bucket(codec.encode_bucket(PAGE_SIZE, bucket)) == bucket


def test_bucket_rejects_payload_larger_than_page() -> None:
    bucket = codec.BucketPage(
        local_depth=0,
        overflow_page_id=None,
        entries=(("x" * PAGE_SIZE, RID(0, 0)),),
    )

    with pytest.raises(ValueError):
        codec.encode_bucket(PAGE_SIZE, bucket)


def test_bucket_rejects_invalid_local_depth() -> None:
    with pytest.raises(ValueError):
        codec.encode_bucket(
            PAGE_SIZE,
            codec.BucketPage(
                local_depth=65,
                overflow_page_id=None,
                entries=(),
            ),
        )


def test_decode_rejects_invalid_magic() -> None:
    page = bytearray(
        codec.encode_metadata(
            PAGE_SIZE,
            global_depth=0,
            first_directory_page_id=1,
            max_global_depth=16,
        )
    )
    page[0:4] = b"BAD!"

    with pytest.raises(ValueError):
        codec.decode_metadata(bytes(page))


def test_decode_bucket_rejects_directory_page() -> None:
    page = codec.encode_directory(
        PAGE_SIZE,
        codec.DirectoryPage(
            next_page_id=None,
            bucket_page_ids=(2,),
        ),
    )

    with pytest.raises(ValueError):
        codec.decode_bucket(page)


def test_page_size_smaller_than_header_is_rejected() -> None:
    with pytest.raises(ValueError):
        codec.encode_metadata(
            codec.HEADER_SIZE - 1,
            global_depth=0,
            first_directory_page_id=1,
            max_global_depth=16,
        )
