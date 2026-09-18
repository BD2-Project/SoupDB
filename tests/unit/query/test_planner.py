"""Tests for the query planner."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.parser import parse
from engine.query.planner import Plan, plan
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_index import FakeIndex

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("anio", ColumnType.INT),
    ColumnDef("venue", ColumnType.TEXT),
)


def make_catalog() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("papers", PAPERS)
    return catalog


def plan_select(sql: str, catalog: FakeCatalog) -> Plan:
    return plan(parse(sql), catalog)


def test_plan_select_star_builds_table_scan() -> None:
    result = plan_select("SELECT * FROM papers", make_catalog())
    assert [column.name for column in result.output_schema] == ["id", "anio", "venue"]
    assert result.root.explain().op == "TableScan"


def test_plan_select_eq_uses_index_lookup() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio")
    result = plan_select("SELECT * FROM papers WHERE anio = 2020", catalog)
    tree = result.root.explain()
    assert tree.op == "Filter"
    assert tree.children[0].op == "IndexLookup"


def test_plan_select_between_uses_range_scan() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio")
    result = plan_select("SELECT * FROM papers WHERE anio BETWEEN 2019 AND 2021", catalog)
    tree = result.root.explain()
    assert tree.op == "Filter"
    assert tree.children[0].op == "IndexRangeScan"


def test_plan_select_rejects_range_on_flat_index() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio", FakeIndex(supports_range=False))
    result = plan_select("SELECT * FROM papers WHERE anio BETWEEN 2019 AND 2021", catalog)
    tree = result.root.explain()
    assert tree.op == "Filter"
    assert tree.children[0].op == "TableScan"


def test_plan_select_without_index_uses_scan() -> None:
    result = plan_select("SELECT * FROM papers WHERE anio = 2020", make_catalog())
    tree = result.root.explain()
    assert tree.op == "Filter"
    assert tree.children[0].op == "TableScan"


def test_plan_select_group_by_builds_aggregate() -> None:
    result = plan_select(
        "SELECT venue, COUNT(*) FROM papers GROUP BY venue",
        make_catalog(),
    )
    tree = result.root.explain()
    assert tree.op == "Aggregate"
    assert tree.detail["group_by"] == ["venue"]
    assert tree.detail["aggregates"] == ["COUNT"]


def test_plan_select_projection_and_sort() -> None:
    result = plan_select(
        "SELECT venue AS v FROM papers ORDER BY anio DESC",
        make_catalog(),
    )
    tree = result.root.explain()
    assert tree.op == "Project"
    assert tree.children[0].op == "Sort"
    assert result.output_schema[0].name == "v"


def test_plan_select_distinct_sorted() -> None:
    result = plan_select("SELECT DISTINCT venue FROM papers ORDER BY venue", make_catalog())
    tree = result.root.explain()
    assert tree.op == "Distinct"
    assert tree.children[0].op == "Project"
    assert tree.children[0].children[0].op == "Sort"


def test_plan_star_with_group_by_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan_select("SELECT * FROM papers GROUP BY venue", make_catalog())


def test_plan_select_unknown_table_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan_select("SELECT * FROM nope", make_catalog())


def test_plan_insert_keeps_statement() -> None:
    result = plan(parse("INSERT INTO papers VALUES (1, 2020, 'VLDB')"), make_catalog())
    assert result.statement.table == "papers"
    assert result.root is None


def test_plan_insert_partial_columns_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("INSERT INTO papers (id) VALUES (1)"), make_catalog())


def test_plan_insert_value_count_mismatch_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("INSERT INTO papers VALUES (1, 2020)"), make_catalog())


def test_plan_create_duplicate_column_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("CREATE TABLE t (a INT, a INT)"), make_catalog())


def test_plan_delete_unknown_table_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("DELETE FROM nope WHERE id = 1"), make_catalog())