"""Tests for the Aggregate volcano operator."""

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


def agg_rows(
    group_by: tuple[ColumnRef, ...],
    aggregates: tuple[FunctionExpr, ...],
    rows: tuple[tuple[object, ...], ...] = ROWS,
) -> list[tuple[object, ...]]:
    fake = FakeFileOrganization()
    for row in rows:
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


def test_aggregate_group_by_count_and_sum() -> None:
    out = agg_rows(
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", None), FunctionExpr("SUM", ColumnRef("anio"))),
    )
    assert out == [("VLDB", 2, 4039), ("SIGMOD", 2, 4041)]


def test_aggregate_output_schema() -> None:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, VENUES)))
    aggregate = Aggregate(
        TableScan(fake, VENUES),
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", None), FunctionExpr("SUM", ColumnRef("anio"))),
    )
    assert aggregate.schema[0].name == "venue"
    assert aggregate.schema[0].type_name is ColumnType.TEXT
    assert aggregate.schema[1].name == "count_1"
    assert aggregate.schema[1].type_name is ColumnType.INT
    assert aggregate.schema[2].name == "sum_2"
    assert aggregate.schema[2].type_name is ColumnType.INT


def test_aggregate_count_distinct() -> None:
    rows = (
        (1, 2019, "VLDB"),
        (2, 2020, "VLDB"),
        (3, 2020, "SIGMOD"),
        (4, 2020, "SIGMOD"),
    )
    out = agg_rows(
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", ColumnRef("anio"), distinct=True),),
        rows=rows,
    )
    assert out == [("VLDB", 2), ("SIGMOD", 1)]


def test_aggregate_multi_key_group() -> None:
    out = agg_rows(
        (ColumnRef("venue"), ColumnRef("anio")),
        (FunctionExpr("COUNT", None),),
    )
    assert out == [
        ("VLDB", 2019, 1),
        ("VLDB", 2020, 1),
        ("SIGMOD", 2021, 1),
        ("SIGMOD", 2020, 1),
    ]


def test_aggregate_no_grouping() -> None:
    out = agg_rows(
        (),
        (FunctionExpr("COUNT", None), FunctionExpr("AVG", ColumnRef("anio"))),
    )
    assert out == [(4, 2020.0)]


def test_aggregate_empty_input_count_and_sum() -> None:
    out = agg_rows(
        (),
        (FunctionExpr("COUNT", None), FunctionExpr("SUM", ColumnRef("anio"))),
        rows=(),
    )
    assert out == [(0, 0)]


def test_aggregate_min_over_group() -> None:
    out = agg_rows(
        (ColumnRef("anio"),),
        (FunctionExpr("MIN", ColumnRef("venue")),),
    )
    assert out == [(2019, "VLDB"), (2020, "SIGMOD"), (2021, "SIGMOD")]
    assert agg_output_types(out, (ColumnRef("anio"),))[1] is ColumnType.TEXT


def agg_output_types(
    _rows: list[tuple[object, ...]],
    group_by: tuple[ColumnRef, ...],
) -> tuple:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, VENUES)))
    aggregate = Aggregate(
        TableScan(fake, VENUES),
        group_by,
        (FunctionExpr("MIN", ColumnRef("venue")),),
    )
    return tuple(column.type_name for column in aggregate.schema)


def test_aggregate_explain() -> None:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, VENUES)))
    aggregate = Aggregate(
        TableScan(fake, VENUES),
        (ColumnRef("venue"),),
        (FunctionExpr("COUNT", None),),
    )
    agg_rows((ColumnRef("venue"),), (FunctionExpr("COUNT", None),))
    aggregate.open()
    while aggregate.next() is not None:
        pass
    aggregate.close()
    plan = aggregate.explain()
    assert plan.op == "Aggregate"
    assert plan.rows == 2
    assert len(plan.children) == 1
    assert plan.children[0].op == "TableScan"
