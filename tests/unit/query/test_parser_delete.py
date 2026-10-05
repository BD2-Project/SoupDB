"""Tests for the SQL parser: DELETE statement."""

import pytest

from engine.query.ast import ColumnRef, CompareExpr, DeleteStatement, Literal
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_delete_minimal() -> None:
    stmt = parse("DELETE FROM papers")
    assert isinstance(stmt, DeleteStatement)
    assert stmt.table == "papers"
    assert stmt.where is None


def test_delete_with_where() -> None:
    stmt = parse("DELETE FROM papers WHERE id = 42")
    assert isinstance(stmt, DeleteStatement)
    assert stmt.where == CompareExpr(ColumnRef("id"), "=", Literal(42))


def test_delete_case_insensitive() -> None:
    stmt = parse("delete from papers where anio < 2010")
    assert isinstance(stmt, DeleteStatement)
    assert stmt.where == CompareExpr(ColumnRef("anio"), "<", Literal(2010))


def test_delete_missing_from_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("DELETE papers")


def test_delete_missing_table_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("DELETE FROM")


def test_delete_with_order_by_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("DELETE FROM papers ORDER BY anio")
