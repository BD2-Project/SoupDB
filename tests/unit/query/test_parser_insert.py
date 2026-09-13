"""Tests for the SQL parser: INSERT statement."""

import pytest

from engine.query.ast import InsertStatement, Literal
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_insert_with_columns_single_row() -> None:
    stmt = parse("INSERT INTO papers (id, titulo) VALUES (1, 'RAG')")
    assert isinstance(stmt, InsertStatement)
    assert stmt.table == "papers"
    assert stmt.columns == ("id", "titulo")
    assert stmt.values == ((Literal(1), Literal("RAG")),)


def test_insert_without_columns() -> None:
    stmt = parse("INSERT INTO papers VALUES (1, 'RAG')")
    assert isinstance(stmt, InsertStatement)
    assert stmt.columns == ()
    assert stmt.values == ((Literal(1), Literal("RAG")),)


def test_insert_multiple_rows() -> None:
    stmt = parse("INSERT INTO papers (id) VALUES (1), (2), (3)")
    assert stmt.values == ((Literal(1),), (Literal(2),), (Literal(3),))


def test_insert_with_numbers_and_booleans() -> None:
    stmt = parse("INSERT INTO papers (anio, activo) VALUES (2020, 1)")
    assert stmt.values == ((Literal(2020), Literal(1)),)


def test_insert_case_insensitive() -> None:
    stmt = parse("insert into papers values ('x')")
    assert isinstance(stmt, InsertStatement)
    assert stmt.values == ((Literal("x"),),)


def test_insert_missing_into_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("INSERT papers VALUES (1)")


def test_insert_missing_values_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("INSERT INTO papers (id)")
