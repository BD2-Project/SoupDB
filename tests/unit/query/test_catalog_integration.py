"""End-to-end SQL tests over the persistent catalog."""

from pathlib import Path

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.query import execute_sql


def make_catalog(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path, page_size=256, buffer_capacity=4)


def test_create_insert_select_roundtrip(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 'RAG', 2020)", catalog)
    execute_sql("INSERT INTO papers VALUES (2, 'Vector', 2021)", catalog)
    result = execute_sql("SELECT titulo FROM papers WHERE anio = 2021", catalog)
    assert result.rows == (("Vector",),)


def test_ddl_persists_across_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 'RAG', 2020)", catalog)
    catalog.close()

    reopened = make_catalog(tmp_path)
    result = execute_sql("SELECT anio FROM papers WHERE id = 1", reopened)
    assert result.rows == ((2020,),)
    reopened.close()


def test_create_table_with_sequential_engine(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE logs (id INT, msg TEXT) ENGINE SEQUENTIAL", catalog)
    assert catalog.strategy("logs") == "SEQUENTIAL"
    execute_sql("INSERT INTO logs VALUES (1, 'a')", catalog)
    assert execute_sql("SELECT msg FROM logs WHERE id = 1", catalog).rows == (("a",),)


def test_create_index_then_query_uses_it(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 'RAG', 2020)", catalog)
    execute_sql("CREATE INDEX idx_anio ON papers (anio) TYPE BTREE", catalog)
    assert "idx_anio" in catalog.indexes("papers")
    result = execute_sql("SELECT titulo FROM papers WHERE anio = 2020", catalog)
    assert result.rows == (("RAG",),)


def test_index_persists_across_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 2020)", catalog)
    execute_sql("CREATE INDEX idx_anio ON papers (anio)", catalog)
    catalog.close()

    reopened = make_catalog(tmp_path)
    assert "idx_anio" in reopened.indexes("papers")
    result = execute_sql("SELECT id FROM papers WHERE anio = 2020", reopened)
    assert result.rows == ((1,),)
    reopened.close()


def test_insert_maintains_index_on_catalog(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("CREATE INDEX idx_anio ON papers (anio)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 2020)", catalog)
    execute_sql("INSERT INTO papers VALUES (2, 2021)", catalog)
    assert execute_sql("SELECT id FROM papers WHERE anio = 2021", catalog).rows == ((2,),)


def test_delete_maintains_index_on_catalog(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("CREATE INDEX idx_anio ON papers (anio)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 2020)", catalog)
    execute_sql("INSERT INTO papers VALUES (2, 2021)", catalog)
    execute_sql("DELETE FROM papers WHERE anio = 2021", catalog)
    assert execute_sql("SELECT id FROM papers WHERE anio = 2021", catalog).rows == ()


def test_sys_tables_queryable(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT)", catalog)
    result = execute_sql("SELECT table_name FROM SysTables ORDER BY table_name", catalog)
    names = [row[0] for row in result.rows]
    assert "papers" in names
    assert "SysTables" in names


def test_sys_columns_queryable(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT)", catalog)
    result = execute_sql(
        "SELECT column_name FROM SysColumns WHERE table_name = 'papers' ORDER BY position",
        catalog,
    )
    assert result.rows == (("id",), ("titulo",))


def test_invalid_engine_rejected(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryExecutionError):
        execute_sql("CREATE TABLE t (a INT) ENGINE MEMORY", catalog)


def test_duplicate_index_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("CREATE INDEX idx_anio ON papers (anio)", catalog)
    with pytest.raises(QueryExecutionError):
        execute_sql("CREATE INDEX idx_anio ON papers (anio)", catalog)


def test_drop_table_end_to_end(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 'RAG')", catalog)
    execute_sql("DROP TABLE papers", catalog)
    with pytest.raises(QueryExecutionError):
        execute_sql("SELECT * FROM papers", catalog)
    assert "papers" not in catalog.tables()


def test_drop_table_persists_after_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT)", catalog)
    execute_sql("DROP TABLE papers", catalog)
    catalog.close()
    reopened = make_catalog(tmp_path)
    assert "papers" not in reopened.tables()
    reopened.close()


def test_drop_table_removes_from_sys_columns(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, titulo TEXT)", catalog)
    execute_sql("DROP TABLE papers", catalog)
    result = execute_sql("SELECT column_name FROM SysColumns WHERE table_name = 'papers'", catalog)
    assert result.rows == ()


def test_drop_index_end_to_end(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 2020)", catalog)
    execute_sql("CREATE INDEX idx_anio ON papers (anio)", catalog)
    execute_sql("DROP INDEX idx_anio", catalog)
    assert catalog.indexes("papers") == {}
    assert execute_sql("SELECT id FROM papers WHERE anio = 2020", catalog).rows == ((1,),)


def test_drop_index_persists_after_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("CREATE INDEX idx_anio ON papers (anio)", catalog)
    execute_sql("DROP INDEX idx_anio", catalog)
    catalog.close()
    reopened = make_catalog(tmp_path)
    assert reopened.indexes("papers") == {}
    reopened.close()


def test_drop_table_then_recreate_and_use(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("DROP TABLE papers", catalog)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (2, 2021)", catalog)
    assert execute_sql("SELECT id FROM papers WHERE anio = 2021", catalog).rows == ((2,),)


def test_two_indexes_on_same_column_end_to_end(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 2020)", catalog)
    execute_sql("CREATE INDEX idx_bt ON papers (anio) TYPE BTREE", catalog)
    execute_sql("CREATE INDEX idx_hash ON papers (anio) TYPE HASH", catalog)
    assert set(catalog.indexes("papers")) == {"idx_bt", "idx_hash"}
    assert execute_sql("SELECT id FROM papers WHERE anio = 2020", catalog).rows == ((1,),)
    assert execute_sql("SELECT id FROM papers WHERE anio BETWEEN 2020 AND 2021", catalog).rows == (
        (1,),
    )


def test_two_indexes_on_same_column_persist(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("CREATE INDEX idx_bt ON papers (anio) TYPE BTREE", catalog)
    execute_sql("CREATE INDEX idx_hash ON papers (anio) TYPE HASH", catalog)
    catalog.close()
    reopened = make_catalog(tmp_path)
    assert set(reopened.indexes("papers")) == {"idx_bt", "idx_hash"}
    assert execute_sql("SELECT id FROM papers WHERE anio = 2020", reopened).rows == ()
    reopened.close()


def test_drop_one_index_keeps_the_other(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE papers (id INT, anio INT)", catalog)
    execute_sql("INSERT INTO papers VALUES (1, 2020)", catalog)
    execute_sql("CREATE INDEX idx_bt ON papers (anio) TYPE BTREE", catalog)
    execute_sql("CREATE INDEX idx_hash ON papers (anio) TYPE HASH", catalog)
    execute_sql("DROP INDEX idx_hash", catalog)
    assert set(catalog.indexes("papers")) == {"idx_bt"}
    assert execute_sql("SELECT id FROM papers WHERE anio = 2020", catalog).rows == ((1,),)
    assert not (tmp_path / "ix_idx_hash.db").exists()
