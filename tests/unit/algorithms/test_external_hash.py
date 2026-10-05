from pathlib import Path

import pytest

from engine.algorithms.external_hash import (
    external_hash_group_by,
    external_hash_join,
)
from engine.common.record import Record


def _record(key: int, value: str) -> Record:
    return Record(data=f"{key}:{value}".encode())


def _key(record: Record) -> int:
    return int(record.data.split(b":", 1)[0])


def _count(current: int | None, _record: Record) -> int:
    return 1 if current is None else current + 1


def test_group_by_counts_records() -> None:
    records = [
        _record(1, "a"),
        _record(2, "b"),
        _record(1, "c"),
        _record(3, "d"),
        _record(2, "e"),
        _record(1, "f"),
    ]

    result = dict(
        external_hash_group_by(
            records,
            _key,
            _count,
            memory_limit_bytes=128,
        )
    )

    assert result == {
        1: 3,
        2: 2,
        3: 1,
    }


def test_group_by_with_small_memory_uses_external_partitions(
    tmp_path: Path,
) -> None:
    records = [_record(value % 7, str(value)) for value in range(100)]

    result = dict(
        external_hash_group_by(
            records,
            _key,
            _count,
            memory_limit_bytes=24,
            partition_count=4,
            temp_dir=tmp_path,
        )
    )

    assert sum(result.values()) == 100
    assert set(result) == set(range(7))


def test_join_matches_equal_keys() -> None:
    left = [
        _record(1, "left-a"),
        _record(2, "left-b"),
        _record(3, "left-c"),
    ]
    right = [
        _record(2, "right-a"),
        _record(3, "right-b"),
        _record(4, "right-c"),
    ]

    result = list(
        external_hash_join(
            left,
            right,
            _key,
            _key,
            memory_limit_bytes=128,
        )
    )

    assert sorted((left.data, right.data) for left, right in result) == sorted(
        [
            (_record(2, "left-b").data, _record(2, "right-a").data),
            (_record(3, "left-c").data, _record(3, "right-b").data),
        ]
    )


def test_join_preserves_duplicate_cross_product() -> None:
    left = [
        _record(1, "left-a"),
        _record(1, "left-b"),
    ]
    right = [
        _record(1, "right-a"),
        _record(1, "right-b"),
        _record(1, "right-c"),
    ]

    result = list(
        external_hash_join(
            left,
            right,
            _key,
            _key,
            memory_limit_bytes=128,
        )
    )

    assert len(result) == 6

    assert {(left_record.data, right_record.data) for left_record, right_record in result} == {
        (left_record.data, right_record.data) for left_record in left for right_record in right
    }


def test_join_with_small_memory_and_many_records(
    tmp_path: Path,
) -> None:
    left = [_record(value % 10, f"L{value}") for value in range(60)]
    right = [_record(value % 10, f"R{value}") for value in range(40)]

    result = list(
        external_hash_join(
            left,
            right,
            _key,
            _key,
            memory_limit_bytes=32,
            partition_count=4,
            max_partition_depth=2,
            temp_dir=tmp_path,
        )
    )

    assert len(result) == 240

    for left_record, right_record in result:
        assert _key(left_record) == _key(right_record)


def test_join_with_no_matches_returns_empty() -> None:
    result = list(
        external_hash_join(
            [_record(1, "a")],
            [_record(2, "b")],
            _key,
            _key,
            memory_limit_bytes=64,
        )
    )

    assert result == []


def test_empty_group_by_returns_empty() -> None:
    result = list(
        external_hash_group_by(
            [],
            _key,
            _count,
            memory_limit_bytes=64,
        )
    )

    assert result == []


def test_empty_join_returns_empty() -> None:
    result = list(
        external_hash_join(
            [],
            [],
            _key,
            _key,
            memory_limit_bytes=64,
        )
    )

    assert result == []


def test_temporary_files_are_cleaned_after_group_by(
    tmp_path: Path,
) -> None:
    records = [_record(value % 4, str(value)) for value in range(40)]

    list(
        external_hash_group_by(
            records,
            _key,
            _count,
            memory_limit_bytes=16,
            partition_count=4,
            temp_dir=tmp_path,
        )
    )

    assert list(tmp_path.iterdir()) == []


def test_temporary_files_are_cleaned_after_join(
    tmp_path: Path,
) -> None:
    left = [_record(value % 3, str(value)) for value in range(20)]
    right = [_record(value % 3, str(value)) for value in range(20)]

    list(
        external_hash_join(
            left,
            right,
            _key,
            _key,
            memory_limit_bytes=16,
            partition_count=3,
            temp_dir=tmp_path,
        )
    )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("memory_limit", [0, -1])
def test_invalid_memory_limit_is_rejected(memory_limit: int) -> None:
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        list(
            external_hash_group_by(
                [],
                _key,
                _count,
                memory_limit_bytes=memory_limit,
            )
        )


@pytest.mark.parametrize("partition_count", [0, 1])
def test_invalid_partition_count_is_rejected(
    partition_count: int,
) -> None:
    with pytest.raises(ValueError, match="partition_count"):
        list(
            external_hash_join(
                [],
                [],
                _key,
                _key,
                memory_limit_bytes=64,
                partition_count=partition_count,
            )
        )


def test_invalid_partition_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_partition_depth"):
        list(
            external_hash_group_by(
                [],
                _key,
                _count,
                memory_limit_bytes=64,
                max_partition_depth=-1,
            )
        )
