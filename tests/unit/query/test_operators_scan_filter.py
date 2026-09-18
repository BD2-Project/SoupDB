"""Tests for the TableScan and Filter volcano operators."""

import pytest

from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import ColumnRef, CompareExpr, Literal
from engine.query.operators import Filter, TableScan
from tests.fakes.fake_storage import FakeFileOrganization

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
    ColumnDef("anio", ColumnType.INT),
)
ROWS = (
    (1, "A", 2020),
    (2, "B", 2021),
    (3, "C", 2020),
)


def load_fake(rows: tuple[tuple[object, ...], ...]) -> FakeFileOrganization:
    fake = FakeFileOrganization()
    for row in rows:
        fake.insert(Record(data=encode_row(row, PAPERS)))
    return fake


def drain(operator) -> list[tuple[object, ...]]:
    operator.open()
    try:
        rows = []
        while True:
            record = operator.next()
            if record is None:
                return rows
            rows.append(decode_row(record.data, PAPERS))
    finally:
        operator.close()


def test_table_scan_returns_all_rows() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS)
    assert drain(scan) == [*ROWS]


def test_table_scan_empty_table() -> None:
    scan = TableScan(load_fake(()), PAPERS)
    assert drain(scan) == []


def test_table_scan_explain() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS)
    scan.open()
    plan = scan.explain()
    assert plan.op == "TableScan"
    assert plan.rows == 0
    scan.close()


def test_table_scan_rows_metric_after_drain() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS)
    drain(scan)
    plan = scan.explain()
    assert plan.op == "TableScan"
    assert plan.rows == 3
    assert plan.disk_reads == 0
    assert plan.disk_writes == 0


def test_filter_keeps_matching_rows() -> None:
    predicate = CompareExpr(ColumnRef("anio"), ">", Literal(2020))
    scan = TableScan(load_fake(ROWS), PAPERS)
    filter = Filter(scan, predicate, PAPERS)
    assert drain(filter) == [(2, "B", 2021)]


def test_filter_empty_result() -> None:
    predicate = CompareExpr(ColumnRef("anio"), ">", Literal(2100))
    scan = TableScan(load_fake(ROWS), PAPERS)
    filter = Filter(scan, predicate, PAPERS)
    assert drain(filter) == []


def test_filter_explain_chain() -> None:
    predicate = CompareExpr(ColumnRef("anio"), ">", Literal(2020))
    scan = TableScan(load_fake(ROWS), PAPERS)
    filter = Filter(scan, predicate, PAPERS)
    drain(filter)
    plan = filter.explain()
    assert plan.op == "Filter"
    assert plan.rows == 1
    assert len(plan.children) == 1
    assert plan.children[0].op == "TableScan"