"""Tests for the IndexLookup and IndexRangeScan volcano operators."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.query.operators import IndexLookup, IndexRangeScan
from tests.fakes.fake_index import FakeIndex
from tests.fakes.fake_storage import FakeFileOrganization

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("anio", ColumnType.INT),
    ColumnDef("venue", ColumnType.TEXT),
)
ROWS = (
    (1, 2019, "VLDB"),
    (2, 2019, "SIGMOD"),
    (3, 2020, "VLDB"),
    (4, 2021, "SIGMOD"),
)


def build(rows: tuple[tuple[object, ...], ...]) -> tuple[FakeFileOrganization, FakeIndex]:
    fake = FakeFileOrganization()
    index = FakeIndex()
    for row in rows:
        rid = fake.insert(Record(data=encode_row(row, PAPERS)))
        index.insert(row[1], rid)
    return fake, index


def drain(operator) -> list[tuple[object, ...]]:
    operator.open()
    try:
        result = []
        while True:
            record = operator.next()
            if record is None:
                return result
            result.append(decode_row(record.data, operator.schema))
    finally:
        operator.close()


def test_index_lookup_equality() -> None:
    fake, index = build(ROWS)
    lookup = IndexLookup(index, fake.fetch, 2020, PAPERS)
    assert drain(lookup) == [(3, 2020, "VLDB")]


def test_index_lookup_multiple_matches() -> None:
    fake, index = build(ROWS)
    lookup = IndexLookup(index, fake.fetch, 2019, PAPERS)
    assert {row[0] for row in drain(lookup)} == {1, 2}


def test_index_lookup_missing_key() -> None:
    fake, index = build(ROWS)
    lookup = IndexLookup(index, fake.fetch, 2050, PAPERS)
    assert drain(lookup) == []


def test_index_lookup_skips_stale_rid() -> None:
    fake, index = build(ROWS)
    index.insert(2022, RID(page_id=0, slot=99))
    lookup = IndexLookup(index, fake.fetch, 2022, PAPERS)
    assert drain(lookup) == []


def test_index_range_inclusive() -> None:
    fake, index = build(ROWS)
    lookup = IndexRangeScan(index, fake.fetch, 2019, 2020, PAPERS)
    assert {row[0] for row in drain(lookup)} == {1, 2, 3}


def test_index_range_emits_bounded_rows() -> None:
    fake, index = build(ROWS)
    lookup = IndexRangeScan(index, fake.fetch, 2019, 2019, PAPERS)
    assert {row[0] for row in drain(lookup)} == {1, 2}


def test_index_range_unsupported_raises() -> None:
    fake, index = build(ROWS)
    index.close()
    plain = FakeIndex(supports_range=False)
    lookup = IndexRangeScan(plain, fake.fetch, 2019, 2021, PAPERS)
    with pytest.raises(QueryExecutionError):
        lookup.open()
    plain.close()


def test_index_lookup_explain() -> None:
    fake, index = build(ROWS)
    lookup = IndexLookup(index, fake.fetch, 2020, PAPERS)
    drain(lookup)
    plan = lookup.explain()
    assert plan.op == "IndexLookup"
    assert plan.rows == 1
    assert plan.children == []


def test_index_range_explain() -> None:
    fake, index = build(ROWS)
    lookup = IndexRangeScan(index, fake.fetch, 2019, 2020, PAPERS)
    drain(lookup)
    plan = lookup.explain()
    assert plan.op == "IndexRangeScan"
    assert plan.rows == 3
    assert plan.children == []