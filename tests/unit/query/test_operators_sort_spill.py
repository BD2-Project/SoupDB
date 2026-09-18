"""Tests for the spill mode of the Sort volcano operator."""

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


def sort_spill(order_by: tuple[OrderByItem, ...], memory: int) -> list[tuple[object, ...]]:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, PAPERS_SCHEMA)))
    sort = Sort(
        TableScan(fake, PAPERS_SCHEMA),
        order_by,
        memory_limit_bytes=memory,
    )
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


def test_sort_spill_ascending() -> None:
    assert sort_spill((OrderByItem(ColumnRef("anio")),), memory=256) == [
        (2, "C", 2020),
        (4, "D", 2020),
        (1, "A", 2021),
        (3, "B", 2021),
    ]


def test_sort_spill_descending() -> None:
    assert sort_spill((OrderByItem(ColumnRef("anio"), ascending=False),), memory=256) == [
        (1, "A", 2021),
        (3, "B", 2021),
        (2, "C", 2020),
        (4, "D", 2020),
    ]


def test_sort_spill_multi_key_mixed_direction() -> None:
    order = (
        OrderByItem(ColumnRef("anio"), ascending=False),
        OrderByItem(ColumnRef("titulo")),
    )
    assert sort_spill(order, memory=128) == [
        (1, "A", 2021),
        (3, "B", 2021),
        (2, "C", 2020),
        (4, "D", 2020),
    ]


def test_sort_spill_tiny_memory_matches_in_memory_order() -> None:
    order = (OrderByItem(ColumnRef("anio")),)
    spill_rows = sort_spill(order, memory=1)
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, PAPERS_SCHEMA)))
    sort = Sort(TableScan(fake, PAPERS_SCHEMA), order)
    sort.open()
    in_memory = []
    while True:
        record = sort.next()
        if record is None:
            break
        in_memory.append(decode_row(record.data, PAPERS_SCHEMA))
    sort.close()
    assert spill_rows == in_memory


def test_sort_spill_empty_input() -> None:
    fake = FakeFileOrganization()
    sort = Sort(
        TableScan(fake, PAPERS_SCHEMA),
        (OrderByItem(ColumnRef("anio")),),
        memory_limit_bytes=256,
    )
    sort.open()
    assert sort.next() is None
    sort.close()


def test_sort_spill_explain_reports_rows() -> None:
    fake = FakeFileOrganization()
    for row in ROWS:
        fake.insert(Record(data=encode_row(row, PAPERS_SCHEMA)))
    sort = Sort(
        TableScan(fake, PAPERS_SCHEMA),
        (OrderByItem(ColumnRef("anio")),),
        memory_limit_bytes=256,
    )
    sort.open()
    while sort.next() is not None:
        pass
    sort.close()
    plan = sort.explain()
    assert plan.op == "Sort"
    assert plan.rows == 4