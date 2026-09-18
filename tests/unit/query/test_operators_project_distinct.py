"""Tests for the Project and Distinct volcano operators."""

from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import ColumnRef, CompareExpr, Literal, SelectColumn
from engine.query.operators import Distinct, Project, TableScan
from tests.fakes.fake_storage import FakeFileOrganization

PAPERS_SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
    ColumnDef("anio", ColumnType.INT),
)
ROWS = (
    (1, "A", 2020),
    (2, "B", 2021),
)


def load_fake(rows: tuple[tuple[object, ...], ...]) -> FakeFileOrganization:
    fake = FakeFileOrganization()
    for row in rows:
        fake.insert(Record(data=encode_row(row, PAPERS_SCHEMA)))
    return fake


def paper_rows(operator) -> list[tuple[object, ...]]:
    operator.open()
    try:
        rows = []
        while True:
            record = operator.next()
            if record is None:
                return rows
            rows.append(decode_row(record.data, operator.schema))
    finally:
        operator.close()


def test_project_alias_columns() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS_SCHEMA)
    project = Project(
        scan,
        (SelectColumn(ColumnRef("titulo"), alias="title"),),
    )
    assert paper_rows(project) == [("A",), ("B",)]
    assert project.schema[0].name == "title"
    assert project.schema[0].type_name is ColumnType.TEXT


def test_project_infers_column_types() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS_SCHEMA)
    project = Project(
        scan,
        (SelectColumn(ColumnRef("id")), SelectColumn(ColumnRef("anio"))),
    )
    assert paper_rows(project) == [(1, 2020), (2, 2021)]
    assert project.schema[0].type_name is ColumnType.INT
    assert project.schema[1].type_name is ColumnType.INT


def test_project_default_aliases() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS_SCHEMA)
    project = Project(scan, (SelectColumn(ColumnRef("titulo")),))
    assert [column.name for column in project.schema] == ["column_1"]


def test_project_literal() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS_SCHEMA)
    project = Project(scan, (SelectColumn(Literal("X"), alias="const"),))
    assert paper_rows(project) == [("X",), ("X",)]
    assert project.schema[0].type_name is ColumnType.TEXT


def test_project_boolean_expression() -> None:
    scan = TableScan(load_fake(ROWS), PAPERS_SCHEMA)
    recent = CompareExpr(ColumnRef("anio"), ">", Literal(2020))
    project = Project(scan, (SelectColumn(recent, alias="reciente"),))
    assert paper_rows(project) == [(False,), (True,)]
    assert project.schema[0].type_name is ColumnType.BOOL


def test_distinct_removes_duplicates() -> None:
    rows = ((1, "A", 2020), (1, "A", 2020), (2, "B", 2021), (1, "A", 2020))
    scan = TableScan(load_fake(rows), PAPERS_SCHEMA)
    distinct = Distinct(scan, PAPERS_SCHEMA)
    assert paper_rows(distinct) == [(1, "A", 2020), (2, "B", 2021)]


def test_distinct_all_duplicates() -> None:
    rows = ((1, "A", 2020), (1, "A", 2020))
    scan = TableScan(load_fake(rows), PAPERS_SCHEMA)
    distinct = Distinct(scan, PAPERS_SCHEMA)
    assert paper_rows(distinct) == [(1, "A", 2020)]


def test_distinct_empty() -> None:
    scan = TableScan(load_fake(()), PAPERS_SCHEMA)
    distinct = Distinct(scan, PAPERS_SCHEMA)
    assert paper_rows(distinct) == []


def test_distinct_rows_metric() -> None:
    rows = ((1, "A", 2020), (1, "A", 2020), (2, "B", 2021))
    scan = TableScan(load_fake(rows), PAPERS_SCHEMA)
    distinct = Distinct(scan, PAPERS_SCHEMA)
    paper_rows(distinct)
    plan = distinct.explain()
    assert plan.op == "Distinct"
    assert plan.rows == 2