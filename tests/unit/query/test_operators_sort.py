"""Tests for the Sort volcano operator."""

from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import ColumnRef, OrderByItem
from engine.query.operators import Sort, TableScan
from tests.fakes.fake_storage import FakeFileOrganization

PAPERS_SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
    ColumnDef("anio", ColumnType.INT),
)
ROWS = (
    (1, "A", 2021),
    (2, "C", 2020),
    (3, "B", 2021),
    (4, "D", 2020),
)


def sort_rows(order_by: tuple[OrderByItem, ...]) -> list[tuple[object, ...]]:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, PAPERS_SCHEMA)))
    sort = Sort(TableScan(fake, PAPERS_SCHEMA), order_by)
    sort.open()
    try:
        result = []
        while True:
            record = sort.next()
            if record is None:
                return result
            result.append(decode_row(record.data, PAPERS_SCHEMA))
    finally:
        sort.close()


def test_sort_ascending() -> None:
    assert sort_rows((OrderByItem(ColumnRef("anio")),)) == [
        (2, "C", 2020),
        (4, "D", 2020),
        (1, "A", 2021),
        (3, "B", 2021),
    ]


def test_sort_descending() -> None:
    assert sort_rows((OrderByItem(ColumnRef("anio"), ascending=False),)) == [
        (1, "A", 2021),
        (3, "B", 2021),
        (2, "C", 2020),
        (4, "D", 2020),
    ]


def test_sort_multi_key_mixed_direction() -> None:
    order = (
        OrderByItem(ColumnRef("anio"), ascending=False),
        OrderByItem(ColumnRef("titulo")),
    )
    assert sort_rows(order) == [
        (3, "B", 2021),
        (1, "A", 2021),
        (2, "C", 2020),
        (4, "D", 2020),
    ]


def test_sort_empty_input() -> None:
    fake = FakeFileOrganization()
    sort = Sort(TableScan(fake, PAPERS_SCHEMA), (OrderByItem(ColumnRef("anio")),))
    sort.open()
    assert sort.next() is None
    sort.close()


def test_sort_explain() -> None:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, PAPERS_SCHEMA)))
    sort = Sort(
        TableScan(fake, PAPERS_SCHEMA), (OrderByItem(ColumnRef("anio"), ascending=False),)
    )
    sort.open()
    while sort.next() is not None:
        pass
    sort.close()
    plan = sort.explain()
    assert plan.op == "Sort"
    assert plan.rows == 4
    assert len(plan.children) == 1
    assert plan.children[0].op == "TableScan"