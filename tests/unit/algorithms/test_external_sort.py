from pathlib import Path

import pytest

from engine.algorithms.external_sort import external_sort
from engine.common.record import Record


def _record(value: int) -> Record:
    return Record(data=str(value).encode())


def _value(record: Record) -> int:
    return int(record.data)


def _ascending(left: Record, right: Record) -> int:
    return (_value(left) > _value(right)) - (_value(left) < _value(right))


def _descending(left: Record, right: Record) -> int:
    return (_value(right) > _value(left)) - (_value(right) < _value(left))


def test_empty_input_returns_empty_result() -> None:
    result = list(
        external_sort(
            [],
            _ascending,
            memory_limit_bytes=32,
        )
    )

    assert result == []


def test_sorts_records_in_ascending_order() -> None:
    records = [_record(value) for value in (5, 1, 4, 2, 3)]

    result = external_sort(
        records,
        _ascending,
        memory_limit_bytes=1024,
    )

    assert [_value(record) for record in result] == [1, 2, 3, 4, 5]


def test_custom_comparator_supports_descending_order() -> None:
    records = [_record(value) for value in (5, 1, 4, 2, 3)]

    result = external_sort(
        records,
        _descending,
        memory_limit_bytes=1024,
    )

    assert [_value(record) for record in result] == [5, 4, 3, 2, 1]


def test_multiple_initial_runs_are_merged(tmp_path: Path) -> None:
    records = [_record(value) for value in range(30, -1, -1)]

    result = external_sort(
        records,
        _ascending,
        memory_limit_bytes=12,
        fan_in=4,
        temp_dir=tmp_path,
    )

    assert [_value(record) for record in result] == list(range(31))


def test_multiple_merge_passes_are_supported(tmp_path: Path) -> None:
    records = [_record(value) for value in range(49, -1, -1)]

    result = external_sort(
        records,
        _ascending,
        memory_limit_bytes=8,
        fan_in=2,
        temp_dir=tmp_path,
    )

    assert [_value(record) for record in result] == list(range(50))


def test_duplicate_values_are_preserved() -> None:
    records = [_record(value) for value in (3, 1, 3, 2, 1, 3)]

    result = external_sort(
        records,
        _ascending,
        memory_limit_bytes=10,
        fan_in=2,
    )

    assert [_value(record) for record in result] == [1, 1, 2, 3, 3, 3]


def test_single_record_larger_than_memory_budget_is_still_sorted() -> None:
    record = Record(data=b"12345678901234567890")

    result = list(
        external_sort(
            [record],
            _ascending,
            memory_limit_bytes=4,
        )
    )

    assert result == [record]


def test_temporary_files_are_cleaned_after_consumption(tmp_path: Path) -> None:
    records = [_record(value) for value in range(20, -1, -1)]

    result = list(
        external_sort(
            records,
            _ascending,
            memory_limit_bytes=8,
            fan_in=2,
            temp_dir=tmp_path,
        )
    )

    assert [_value(record) for record in result] == list(range(21))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("memory_limit", [0, -1])
def test_invalid_memory_limit_is_rejected(memory_limit: int) -> None:
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        list(
            external_sort(
                [_record(1)],
                _ascending,
                memory_limit_bytes=memory_limit,
            )
        )


@pytest.mark.parametrize("fan_in", [0, 1])
def test_invalid_fan_in_is_rejected(fan_in: int) -> None:
    with pytest.raises(ValueError, match="fan_in"):
        list(
            external_sort(
                [_record(1)],
                _ascending,
                memory_limit_bytes=32,
                fan_in=fan_in,
            )
        )
