"""Tests for the query planner."""

import pathlib

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.parser import parse
from engine.query.planner import Plan, plan
from engine.storage.disk_manager import DiskManager
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
    assert tree.op == "Project"
    assert tree.children[0].op == "Aggregate"
    aggregate = tree.children[0]
    assert aggregate.detail["group_by"] == ["ColumnRef(name='venue')"]
    assert aggregate.detail["aggregates"] == ["COUNT"]


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


def test_plan_select_disk_manager_propagated(tmp_path: pathlib.Path) -> None:
    catalog = make_catalog()
    catalog.disk_manager = DiskManager(tmp_path / "db.bin", page_size=4096)
    tree = plan_select("SELECT anio FROM papers WHERE anio = 2020", catalog).root.explain()
    assert tree.op == "Project"
    assert tree.disk_reads == 0
    assert tree.disk_writes == 0


def test_plan_select_unknown_column_in_where_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan_select("SELECT * FROM papers WHERE missing = 1", make_catalog())


def test_plan_select_unknown_column_in_order_by_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan_select("SELECT * FROM papers ORDER BY missing", make_catalog())


def test_plan_select_order_by_aggregate_resolved() -> None:
    result = plan_select(
        "SELECT venue, COUNT(*) FROM papers GROUP BY venue ORDER BY COUNT(*)",
        make_catalog(),
    )
    tree = result.root.explain()
    assert tree.op == "Project"
    sort = tree.children[0]
    assert sort.op == "Sort"
    assert sort.children[0].op == "Aggregate"
    assert any("count_1" in key for key in sort.detail["keys"])


def test_plan_insert_keeps_statement() -> None:
    result = plan(parse("INSERT INTO papers VALUES (1, 2020, 'VLDB')"), make_catalog())
    assert result.statement.table == "papers"
    assert result.root is None


def test_plan_insert_partial_columns_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("INSERT INTO papers (id) VALUES (1)"), make_catalog())


def test_plan_insert_duplicate_columns_raises() -> None:
    sql = "INSERT INTO papers (id, id, anio, venue) VALUES (1, 2, 2020, 'VLDB')"
    with pytest.raises(QueryExecutionError):
        plan(parse(sql), make_catalog())


def test_plan_delete_unknown_column_in_where_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("DELETE FROM papers WHERE missing = 1"), make_catalog())


def test_plan_delete_with_where_builds() -> None:
    result = plan(parse("DELETE FROM papers WHERE anio = 2020"), make_catalog())
    assert result.statement is not None


def test_plan_insert_value_count_mismatch_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("INSERT INTO papers VALUES (1, 2020)"), make_catalog())


def test_plan_create_duplicate_column_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("CREATE TABLE t (a INT, a INT)"), make_catalog())


def test_plan_create_table_unknown_engine_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("CREATE TABLE t (a INT) ENGINE MEMORY"), make_catalog())


def test_plan_create_table_valid_engine_builds() -> None:
    result = plan(parse("CREATE TABLE t (a INT) ENGINE HEAP"), make_catalog())
    assert result.statement is not None


def test_plan_create_index_builds() -> None:
    result = plan(parse("CREATE INDEX idx_anio ON papers (anio) TYPE BTREE"), make_catalog())
    assert result.statement is not None


def test_plan_create_index_unknown_table_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("CREATE INDEX idx ON nope (a)"), make_catalog())


def test_plan_create_index_unknown_column_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("CREATE INDEX idx ON papers (missing)"), make_catalog())


def test_plan_create_index_unknown_type_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("CREATE INDEX idx ON papers (anio) TYPE BM25"), make_catalog())


def test_plan_drop_table_builds() -> None:
    result = plan(parse("DROP TABLE papers"), make_catalog())
    assert result.statement is not None


def test_plan_drop_table_unknown_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("DROP TABLE nope"), make_catalog())


def test_plan_drop_table_sys_table_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("DROP TABLE SysTables"), make_catalog())


def test_plan_drop_index_builds() -> None:
    catalog = make_catalog()
    catalog.add_index("papers", "anio", index_name="idx_anio")
    result = plan(parse("DROP INDEX idx_anio"), catalog)
    assert result.statement is not None


def test_plan_drop_index_unknown_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("DROP INDEX nope"), make_catalog())


def test_plan_delete_unknown_table_raises() -> None:
    with pytest.raises(QueryExecutionError):
        plan(parse("DELETE FROM nope WHERE id = 1"), make_catalog())
