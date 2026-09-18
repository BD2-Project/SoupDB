"""Tests for the spill mode of the Aggregate volcano operator."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import ColumnRef, FunctionExpr
from engine.query.operators import Aggregate, TableScan
from tests.fakes.fake_storage import FakeFileOrganization

VENUES = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("anio", ColumnType.INT),
    ColumnDef("venue", ColumnType.TEXT),
)
ROWS = (
    (1, 2019, "VLDB"),
    (2, 2020, "VLDB"),
    (3, 2021, "SIGMOD"),
    (4, 2020, "SIGMOD"),
)


def agg_spill(
    group_by: tuple[ColumnRef, ...],
    aggregates: tuple[FunctionExpr, ...],
    memory: int,
    rows: tuple[tuple[object, ...], ...] = ROWS,
) -> list[tuple[object, ...]]:
    fake = FakeFileOrganization()
    for row in rows:
        fake.insert(Record(data=encode_row(row, VENUES)))
    aggregate = Aggregate(
        TableScan(fake, VENUES),
        group_by,
        aggregates,
        memory_limit_bytes=memory,
    )
    aggregate.open()
    try:
        result = []
        while True:
            record = aggregate.next()
            if record is None:
                return result
            result.append(decode_row(record.data, aggregate.schema))
    finally:
        aggregate.close()


def in_memory_agg(group_by, aggregates) -> list[tuple[object, ...]]:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, VENUES)))
    aggregate = Aggregate(TableScan(fake, VENUES), group_by, aggregates)
    aggregate.open()
    try:
        result = []
        while True:
            record = aggregate.next()
            if record is None:
                return result
            result.append(decode_row(record.data, aggregate.schema))
    finally:
        aggregate.close()


def test_agg_spill_group_by_count_and_sum() -> None:
    out = agg_spill(
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", None), FunctionExpr("SUM", ColumnRef("anio"))),
        memory=256,
    )
    assert sorted(out) == sorted([("VLDB", 2, 4039), ("SIGMOD", 2, 4041)])


def test_agg_spill_multi_key_group() -> None:
    out = agg_spill(
        (ColumnRef("venue"), ColumnRef("anio")),
        (FunctionExpr("COUNT", None),),
        memory=128,
    )
    assert sorted(out) == sorted(
        [("VLDB", 2019, 1), ("VLDB", 2020, 1), ("SIGMOD", 2021, 1), ("SIGMOD", 2020, 1)]
    )


def test_agg_spill_avg_and_min() -> None:
    out = agg_spill(
        (ColumnRef("venue"),),
        (FunctionExpr("AVG", ColumnRef("anio")), FunctionExpr("MIN", ColumnRef("anio"))),
        memory=128,
    )
    assert sorted(out) == sorted([("VLDB", 2019.5, 2019), ("SIGMOD", 2020.5, 2020)])


def test_agg_spill_tiny_memory_matches_in_memory() -> None:
    group_by = (ColumnRef("venue"),)
    aggregates = (FunctionExpr("COUNT", None), FunctionExpr("SUM", ColumnRef("anio")))
    assert sorted(agg_spill(group_by, aggregates, memory=1)) == sorted(
        in_memory_agg(group_by, aggregates)
    )


def test_agg_spill_empty_input() -> None:
    out = agg_spill(
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", None),),
        memory=128,
        rows=(),
    )
    assert out == []


def test_agg_spill_distinct_falls_back_to_memory() -> None:
    out = agg_spill(
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", ColumnRef("anio"), distinct=True),),
        memory=128,
    )
    assert out == [("VLDB", 2), ("SIGMOD", 2)]


def test_agg_spill_without_grouping_stays_in_memory() -> None:
    out = agg_spill(
        (),
        (FunctionExpr("COUNT", None), FunctionExpr("AVG", ColumnRef("anio"))),
        memory=128,
    )
    assert out == [(4, 2020.0)]


def test_agg_spill_avg_empty_raises() -> None:
    with pytest.raises(QueryExecutionError):
        agg_spill(
            (),
            (FunctionExpr("AVG", ColumnRef("anio")),),
            memory=128,
            rows=(),
        )


def test_agg_spill_explain_reports_rows() -> None:
    out = agg_spill(
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", None),),
        memory=256,
    )
    assert len(out) == 2
