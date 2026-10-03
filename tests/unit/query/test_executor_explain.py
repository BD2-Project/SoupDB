"""Tests for EXPLAIN / EXPLAIN ANALYZE execution.

EXPLAIN renders the operator tree without executing it (all metrics zero).
EXPLAIN ANALYZE runs the inner statement and reports real per-node metrics in
the ``QUERY PLAN`` text column.
"""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import ExplainStatement
from engine.query.errors import QueryParseError
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog

PAPERS = (
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


def run(sql: str, catalog: FakeCatalog):
    return execute(plan(parse(sql), catalog), catalog)


def make_catalog(rows: tuple[tuple[object, ...], ...] = ROWS) -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("papers", PAPERS)
    for row in rows:
        catalog.insert("papers", row)
    return catalog


def _lines(result) -> list[str]:
    assert [column.name for column in result.columns] == ["QUERY PLAN"]
    return [row[0] for row in result.rows]


def test_explain_returns_query_plan_column() -> None:
    result = run("EXPLAIN SELECT * FROM papers", make_catalog())
    lines = _lines(result)
    assert any(line.startswith("-> TableScan") for line in lines)


def test_explain_shows_nested_operator_tree() -> None:
    result = run(
        "EXPLAIN SELECT venue FROM papers WHERE anio > 2019 ORDER BY venue",
        make_catalog(),
    )
    text = "\n".join(_lines(result))
    assert "-> Sort" in text
    assert "-> Filter" in text
    assert "-> TableScan" in text


def test_explain_without_analyze_reports_zero_rows_and_time() -> None:
    result = run("EXPLAIN SELECT * FROM papers", make_catalog())
    lines = _lines(result)
    assert lines, "expected at least one plan line"
    assert all("rows=0" in line for line in lines)
    assert all("elapsed_ms=0.0" in line for line in lines)


def test_explain_analyze_reports_real_row_counts() -> None:
    result = run(
        "EXPLAIN ANALYZE SELECT * FROM papers WHERE anio = 2020",
        make_catalog(),
    )
    text = "\n".join(_lines(result))
    assert "-> Filter rows=2" in text
    assert "-> TableScan rows=4" in text


def test_explain_analyze_select_does_not_run_twice() -> None:
    catalog = make_catalog()
    run("EXPLAIN ANALYZE SELECT * FROM papers", catalog)
    assert len(run("SELECT * FROM papers", catalog).rows) == 4


def test_explain_insert_describes_statement() -> None:
    result = run("EXPLAIN INSERT INTO papers VALUES (5, 2022, 'VLDB')", make_catalog())
    text = "\n".join(_lines(result))
    assert "Insert" in text
    assert "'table': 'papers'" in text


def test_explain_delete_describes_statement() -> None:
    result = run("EXPLAIN DELETE FROM papers WHERE anio = 2020", make_catalog())
    assert any("Delete" in line for line in _lines(result))


def test_explain_analyze_insert_affects_rows() -> None:
    catalog = make_catalog()
    result = run("EXPLAIN ANALYZE INSERT INTO papers VALUES (5, 2022, 'VLDB')", catalog)
    text = "\n".join(_lines(result))
    assert "rows=1" in text
    assert len(run("SELECT * FROM papers", catalog).rows) == 5


def test_explain_analyze_unknown_table_raises() -> None:
    with pytest.raises(QueryExecutionError):
        run("EXPLAIN ANALYZE SELECT * FROM nope", make_catalog())


def test_explain_requires_inner_statement() -> None:
    with pytest.raises(QueryParseError):
        parse("EXPLAIN")


def test_parse_explain_analyze_flag() -> None:
    stmt = parse("EXPLAIN ANALYZE SELECT * FROM papers")
    assert isinstance(stmt, ExplainStatement)
    assert stmt.analyze is True
