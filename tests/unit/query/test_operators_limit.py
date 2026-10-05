"""Tests for the Limit volcano operator (LIMIT/OFFSET execution)."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.operators import Limit, TableScan
from tests.fakes.fake_storage import FakeFileOrganization

SCHEMA = (ColumnDef("id", ColumnType.INT),)
ROWS = ((0,), (1,), (2,), (3,), (4,))


def limit_rows(limit: int, offset: int = 0) -> list[tuple[object, ...]]:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, SCHEMA)))
    op = Limit(TableScan(fake, SCHEMA), limit, offset)
    op.open()
    try:
        result = []
        while True:
            record = op.next()
            if record is None:
                return result
            result.append(decode_row(record.data, SCHEMA))
    finally:
        op.close()


def test_limit_returns_first_rows() -> None:
    assert limit_rows(3) == [(0,), (1,), (2,)]


def test_limit_zero_returns_nothing() -> None:
    assert limit_rows(0) == []


def test_limit_greater_than_input_returns_all() -> None:
    assert limit_rows(10) == list(ROWS)


def test_limit_with_offset_skips_prefix() -> None:
    assert limit_rows(2, 1) == [(1,), (2,)]


def test_offset_beyond_input_returns_nothing() -> None:
    assert limit_rows(3, 20) == []


def test_limit_negative_raises() -> None:
    fake = FakeFileOrganization()
    with pytest.raises(QueryExecutionError, match="LIMIT"):
        Limit(TableScan(fake, SCHEMA), -1)


def test_offset_negative_raises() -> None:
    fake = FakeFileOrganization()
    with pytest.raises(QueryExecutionError, match="OFFSET"):
        Limit(TableScan(fake, SCHEMA), 3, -1)


def test_limit_explain_reports_quota() -> None:
    fake = FakeFileOrganization()
    op = Limit(TableScan(fake, SCHEMA), 3, 1)
    node = op.explain()
    assert node.op == "Limit"
    assert node.detail == {"limit": 3, "offset": 1}
