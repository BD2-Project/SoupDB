"""Tests for the SQL parser: JOIN clauses in SELECT."""

import pytest

from engine.query.ast import ColumnRef, CompareExpr, JoinClause, Literal, LogicalExpr, SelectStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_select_single_join() -> None:
    stmt = parse("SELECT * FROM papers JOIN authors ON author = paper")
    assert isinstance(stmt, SelectStatement)
    assert stmt.joins == (
        JoinClause(
            "authors",
            CompareExpr(ColumnRef("author"), "=", ColumnRef("paper")),
        ),
    )


def test_select_inner_join_keyword_is_optional() -> None:
    stmt = parse("SELECT * FROM papers INNER JOIN authors ON author = paper")
    assert stmt.joins == (
        JoinClause(
            "authors",
            CompareExpr(ColumnRef("author"), "=", ColumnRef("paper")),
        ),
    )


def test_select_multiple_chained_joins() -> None:
    stmt = parse(
        "SELECT * FROM papers "
        "JOIN authors ON author = paper "
        "JOIN venues ON venues_id = author"
    )
    assert stmt.joins == (
        JoinClause("authors", CompareExpr(ColumnRef("author"), "=", ColumnRef("paper"))),
        JoinClause("venues", CompareExpr(ColumnRef("venues_id"), "=", ColumnRef("author"))),
    )


def test_select_join_on_boolean_expression() -> None:
    stmt = parse("SELECT * FROM papers JOIN authors ON author = paper AND anio = 2020")
    assert stmt.joins == (
        JoinClause(
            "authors",
            LogicalExpr(
                CompareExpr(ColumnRef("author"), "=", ColumnRef("paper")),
                "AND",
                CompareExpr(ColumnRef("anio"), "=", Literal(2020)),
            ),
        ),
    )


def test_select_join_with_where() -> None:
    stmt = parse("SELECT * FROM papers JOIN authors ON author = paper WHERE anio > 2010")
    assert isinstance(stmt, SelectStatement)
    assert len(stmt.joins) == 1
    assert stmt.where is not None


def test_select_join_without_on_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers JOIN authors")


def test_select_inner_without_join_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers INNER authors ON author = paper")