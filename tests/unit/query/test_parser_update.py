"""Tests for the SQL parser: UPDATE statement."""

import pytest

from engine.query.ast import ColumnRef, CompareExpr, Literal, UpdateStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_update_single_assignment() -> None:
    stmt = parse("UPDATE papers SET anio = 2021")
    assert isinstance(stmt, UpdateStatement)
    assert stmt.table == "papers"
    assert stmt.assignments == (("anio", Literal(2021)),)
    assert stmt.where is None


def test_update_multiple_assignments() -> None:
    stmt = parse("UPDATE papers SET anio = 2021, titulo = 'RAG'")
    assert stmt.assignments == (
        ("anio", Literal(2021)),
        ("titulo", Literal("RAG")),
    )


def test_update_with_where() -> None:
    stmt = parse("UPDATE papers SET anio = 2021 WHERE id = 1")
    assert stmt.where == CompareExpr(ColumnRef("id"), "=", Literal(1))


def test_update_case_insensitive_keywords() -> None:
    stmt = parse("update papers set anio = 2021")
    assert isinstance(stmt, UpdateStatement)
    assert stmt.assignments == (("anio", Literal(2021)),)


def test_update_missing_set_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("UPDATE papers anio = 1")


def test_update_missing_equals_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("UPDATE papers SET anio 2021")


def test_update_missing_assignment_value_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("UPDATE papers SET anio =")


def test_update_trailing_garbage_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("UPDATE papers SET anio = 2021 extra")
