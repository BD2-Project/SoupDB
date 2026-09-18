"""Tests for the public execute_sql API."""

import engine.query as query_module
from engine.common.schema import ColumnDef, ColumnType
from engine.query import ResultSet, execute_sql, parse, plan
from engine.query.ast import SelectStatement
from tests.fakes.fake_catalog import FakeCatalog

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("anio", ColumnType.INT),
    ColumnDef("venue", ColumnType.TEXT),
)


def make_catalog() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("papers", PAPERS)
    for row in ((1, 2019, "VLDB"), (2, 2020, "SIGMOD")):
        catalog.insert("papers", row)
    return catalog


def test_execute_sql_select() -> None:
    result = execute_sql("SELECT * FROM papers", make_catalog())
    assert isinstance(result, ResultSet)
    assert result.rows == ((1, 2019, "VLDB"), (2, 2020, "SIGMOD"))


def test_execute_sql_insert_affected() -> None:
    catalog = make_catalog()
    result = execute_sql("INSERT INTO papers VALUES (3, 2021, 'ICDE')", catalog)
    assert result.affected == 1
    assert len(execute_sql("SELECT * FROM papers", catalog).rows) == 3


def test_execute_sql_delete_affected() -> None:
    catalog = make_catalog()
    result = execute_sql("DELETE FROM papers WHERE anio = 2020", catalog)
    assert result.affected == 1
    assert len(execute_sql("SELECT * FROM papers", catalog).rows) == 1


def test_execute_sql_create_table() -> None:
    catalog = FakeCatalog()
    execute_sql("CREATE TABLE t (a INT, b TEXT)", catalog)
    execute_sql("INSERT INTO t VALUES (1, 'x')", catalog)
    assert execute_sql("SELECT * FROM t", catalog).rows == ((1, "x"),)


def test_parse_returns_statement() -> None:
    assert isinstance(parse("SELECT * FROM papers"), SelectStatement)


def test_plan_returns_plan() -> None:
    assert plan(parse("SELECT * FROM papers"), make_catalog()).root is not None


def test_module_exports_public_api() -> None:
    assert {"execute_sql", "parse", "plan", "ResultSet"} <= set(query_module.__all__)