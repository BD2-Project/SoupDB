"""Tests for the SQL parser: NULL literal in expressions."""

from engine.query.ast import (
    ColumnRef,
    CompareExpr,
    InsertStatement,
    Literal,
    SelectColumn,
    SelectStatement,
    UpdateStatement,
)
from engine.query.parser import parse


def test_select_null_literal_projection() -> None:
    stmt = parse("SELECT NULL FROM papers")
    assert isinstance(stmt, SelectStatement)
    assert stmt.columns == (SelectColumn(Literal(None)),)


def test_null_literal_in_comparison() -> None:
    stmt = parse("SELECT * FROM papers WHERE titulo = NULL")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where == CompareExpr(ColumnRef("titulo"), "=", Literal(None))


def test_insert_null_value() -> None:
    stmt = parse("INSERT INTO papers (titulo) VALUES (NULL)")
    assert isinstance(stmt, InsertStatement)
    assert stmt.values == ((Literal(None),),)


def test_update_set_null() -> None:
    stmt = parse("UPDATE papers SET titulo = NULL WHERE id = 1")
    assert isinstance(stmt, UpdateStatement)
    assert stmt.assignments == (("titulo", Literal(None)),)
    assert stmt.where is not None


def test_null_in_value_rows() -> None:
    stmt = parse("INSERT INTO papers (id, titulo) VALUES (1, NULL), (2, 'x')")
    assert stmt.values == ((Literal(1), Literal(None)), (Literal(2), Literal("x")))
