"""Tests for the SQL parser: IS NULL / IS NOT NULL predicate."""

import pytest

from engine.query.ast import ColumnRef, IsNullExpr, LogicalExpr, SelectColumn, SelectStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_where_is_null() -> None:
    stmt = parse("SELECT * FROM papers WHERE titulo IS NULL")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where == IsNullExpr(ColumnRef("titulo"), negated=False)


def test_where_is_not_null() -> None:
    stmt = parse("SELECT * FROM papers WHERE titulo IS NOT NULL")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where == IsNullExpr(ColumnRef("titulo"), negated=True)


def test_is_null_in_projection() -> None:
    stmt = parse("SELECT titulo IS NULL FROM papers")
    assert isinstance(stmt, SelectStatement)
    assert stmt.columns == (SelectColumn(IsNullExpr(ColumnRef("titulo"))),)


def test_is_null_combined_with_logical_operators() -> None:
    stmt = parse("SELECT * FROM papers WHERE anio IS NULL OR anio > 2000")
    assert isinstance(stmt.where, LogicalExpr)
    assert stmt.where.op == "OR"
    assert stmt.where.left == IsNullExpr(ColumnRef("anio"))


def test_is_null_works_against_arithmetic_operand() -> None:
    stmt = parse("SELECT * FROM papers WHERE (anio + 1) IS NULL")
    from engine.query.ast import BinaryExpr, Literal

    assert stmt.where == IsNullExpr(BinaryExpr(ColumnRef("anio"), "+", Literal(1)), negated=False)


def test_is_without_null_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers WHERE anio IS mango")
