"""Integration tests for operator pipelines and disk metrics."""

from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import (
    ColumnRef,
    CompareExpr,
    FunctionExpr,
    Literal,
    OrderByItem,
    SelectColumn,
)
from engine.query.operators import Aggregate, Filter, Project, Sort, TableScan
from engine.storage.disk_manager import DiskManager
from tests.fakes.fake_storage import FakeFileOrganization

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
    ColumnDef("anio", ColumnType.INT),
)
ROWS = (
    (1, "A", 2020),
    (2, "B", 2021),
    (3, "C", 2019),
)


def load_fake(rows: tuple[tuple[object, ...], ...], schema: tuple = PAPERS) -> FakeFileOrganization:
    fake = FakeFileOrganization()
    for row in rows:
        fake.insert(Record(data=encode_row(row, schema)))
    return fake


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


def test_pipeline_scan_filter_sort_project() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS)
    filter = Filter(scan, CompareExpr(ColumnRef("anio"), ">", Literal(2019)), PAPERS)
    sort = Sort(filter, (OrderByItem(ColumnRef("anio"), ascending=False),))
    project = Project(sort, (SelectColumn(ColumnRef("titulo")),))
    assert drain(project) == [("B",), ("A",)]


def test_pipeline_scan_aggregate_sort() -> None:
    venues = FakeFileOrganization()
    for row in ROWS:
        venues.insert(Record(data=encode_row(row, PAPERS)))
    aggregate = Aggregate(
        TableScan(venues, PAPERS),
        (ColumnRef("anio"),),
        (FunctionExpr("COUNT", None),),
    )
    sort = Sort(aggregate, (OrderByItem(ColumnRef("anio"), ascending=False),))
    assert drain(sort) == [(2021, 1), (2020, 1), (2019, 1)]


def test_pipeline_explain_children() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS)
    filter = Filter(scan, CompareExpr(ColumnRef("anio"), ">", Literal(2019)), PAPERS)
    project = Project(filter, (SelectColumn(ColumnRef("titulo")),))
    drain(project)
    plan = project.explain()
    base = plan
    while base.children:
        base = base.children[0]
    assert plan.op == "Project"
    assert plan.children[0].op == "Filter"
    assert base.op == "TableScan"


def test_explain_reports_disk_delta(tmp_path) -> None:
    dm = DiskManager(tmp_path / "db.bin", page_size=4096)
    scan = TableScan(load_fake(ROWS), PAPERS, disk_manager=dm)
    scan.open()
    dm.reads += 3
    dm.writes += 2
    plan = scan.explain()
    assert plan.disk_reads == 3
    assert plan.disk_writes == 2
    scan.close()
