"""Tests for the SQL parser: CREATE TABLE statement."""

import pytest

from engine.query.ast import ColumnDef, ColumnType, CreateTableStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_create_table_basic_types() -> None:
    stmt = parse("CREATE TABLE papers (id INT, titulo TEXT, anio INT)")
    assert isinstance(stmt, CreateTableStatement)
    assert stmt.table == "papers"
    assert stmt.columns == (
        ColumnDef("id", ColumnType.INT),
        ColumnDef("titulo", ColumnType.TEXT),
        ColumnDef("anio", ColumnType.INT),
    )


def test_create_table_varchar_with_length() -> None:
    stmt = parse("CREATE TABLE autores (nombre VARCHAR(60))")
    assert stmt.columns == (ColumnDef("nombre", ColumnType.VARCHAR, length=60),)


def test_create_table_float_and_bool() -> None:
    stmt = parse("CREATE TABLE papers (score FLOAT, activo BOOL)")
    assert stmt.columns == (
        ColumnDef("score", ColumnType.FLOAT),
        ColumnDef("activo", ColumnType.BOOL),
    )


def test_create_table_type_case_insensitive() -> None:
    stmt = parse("create table papers (id int, titulo text)")
    assert stmt.columns == (
        ColumnDef("id", ColumnType.INT),
        ColumnDef("titulo", ColumnType.TEXT),
    )


def test_create_table_missing_columns_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("CREATE TABLE papers")


def test_create_table_unknown_type_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("CREATE TABLE papers (id MYTYPE)")


def test_create_table_varchar_without_length_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("CREATE TABLE papers (titulo VARCHAR)")


def test_create_table_default_engine_is_heap() -> None:
    stmt = parse("CREATE TABLE papers (id INT)")
    assert isinstance(stmt, CreateTableStatement)
    assert stmt.engine == "HEAP"


def test_create_table_with_engine_clause() -> None:
    stmt = parse("CREATE TABLE papers (id INT) ENGINE HEAP")
    assert isinstance(stmt, CreateTableStatement)
    assert stmt.engine == "HEAP"


def test_create_table_with_sequential_engine() -> None:
    stmt = parse("CREATE TABLE papers (id INT) ENGINE SEQUENTIAL")
    assert isinstance(stmt, CreateTableStatement)
    assert stmt.engine == "SEQUENTIAL"


def test_create_table_engine_case_insensitive() -> None:
    stmt = parse("create table papers (id int) engine sequential")
    assert isinstance(stmt, CreateTableStatement)
    assert stmt.engine == "SEQUENTIAL"


def test_create_table_engine_raises_when_duplicate() -> None:
    with pytest.raises(QueryParseError):
        parse("CREATE TABLE papers (id INT) ENGINE HEAP ENGINE SEQUENTIAL")
