"""Tests for the persistent catalog backed by system tables."""

from pathlib import Path

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.common.record import Record, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.storage.sequential_file import SequentialFile

PAPERS = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
    ColumnDef("anio", ColumnType.INT),
)


def make_catalog(tmp_path: Path, **kwargs) -> Catalog:
    return Catalog(tmp_path, page_size=128, buffer_capacity=4, **kwargs)


def test_sys_tables_are_registered(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    assert catalog.tables() == ["SysTables", "SysColumns", "SysIndexes"]


def test_sys_tables_reopen_persists(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.close()
    assert make_catalog(tmp_path).tables() == ["SysTables", "SysColumns", "SysIndexes"]


def test_create_table_adds_schema(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    assert catalog.schema("papers") == PAPERS


def test_create_table_persists_across_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    catalog.close()
    assert make_catalog(tmp_path).schema("papers") == PAPERS


def test_sys_tables_registered_after_user_table_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    catalog.close()
    reopened = make_catalog(tmp_path)
    assert "papers" in reopened.tables()
    assert "SysTables" in reopened.tables()


def test_create_table_duplicate_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    with pytest.raises(QueryExecutionError):
        catalog.create_table("papers", PAPERS)


def test_create_table_reserved_name_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryExecutionError):
        catalog.create_table("SysTables", PAPERS)


def test_create_table_unknown_engine_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryExecutionError):
        catalog.create_table("papers", PAPERS, engine="MEMORY")


def test_unknown_table_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryExecutionError):
        catalog.schema("nope")


def test_heap_engine_default(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    assert catalog.strategy("papers") == "HEAP"
    from engine.storage.heap_file import HeapFile

    assert isinstance(catalog.file_org("papers"), HeapFile)


def test_sequential_engine(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS, engine="SEQUENTIAL")
    assert catalog.strategy("papers") == "SEQUENTIAL"
    assert isinstance(catalog.file_org("papers"), SequentialFile)


def test_insert_and_scan_roundtrip(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    row = (1, "RAG sobre papers", 2020)
    rid = catalog.file_org("papers").insert(Record(data=encode_row(row, PAPERS)))
    (stored_rid, record) = next(catalog.file_org("papers").scan())
    assert stored_rid == rid
    assert record.data == encode_row(row, PAPERS)


def test_create_index_btree_and_search(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    catalog.file_org("papers").insert(Record(data=encode_row((1, "titulo", 2020), PAPERS)))
    catalog.create_index("idx_anio", "papers", "anio", index_type="BTREE")
    assert catalog.indexes("papers")["anio"].search(2020)


def test_create_index_backfills_existing_rows(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    catalog.file_org("papers").insert(Record(data=encode_row((1, "a", 2020), PAPERS)))
    catalog.create_index("idx_anio", "papers", "anio", index_type="BTREE")
    assert catalog.indexes("papers")["anio"].search(2020)


def test_create_index_persists_across_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    catalog.create_index("idx_anio", "papers", "anio", index_type="BTREE")
    catalog.close()
    reopened = make_catalog(tmp_path)
    assert "anio" in reopened.indexes("papers")


def test_create_index_unknown_table_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryExecutionError):
        catalog.create_index("idx", "nope", "anio")


def test_create_index_unknown_column_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    with pytest.raises(QueryExecutionError):
        catalog.create_index("idx", "papers", "missing")


def test_create_index_duplicate_name_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    catalog.create_index("idx_anio", "papers", "anio")
    with pytest.raises(QueryExecutionError):
        catalog.create_index("idx_anio", "papers", "anio")


def test_create_index_unknown_type_raises(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    catalog.create_table("papers", PAPERS)
    with pytest.raises(QueryExecutionError):
        catalog.create_index("idx", "papers", "anio", index_type="BM25")
