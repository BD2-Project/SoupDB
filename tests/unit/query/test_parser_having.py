"""Tests for the SQL parser: HAVING clause."""

import pytest

from engine.query.ast import (
    ColumnRef,
    CompareExpr,
    FunctionExpr,
    HavingClause,
    Literal,
    LogicalExpr,
    SelectStatement,
)
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_select_having_after_group_by() -> None:
    stmt = parse("SELECT venue, COUNT(*) FROM papers GROUP BY venue HAVING COUNT(*) > 1")
    assert isinstance(stmt, SelectStatement)
    assert stmt.group_by == (ColumnRef("venue"),)
    assert stmt.having == HavingClause(
        CompareExpr(FunctionExpr("COUNT", arg=None), ">", Literal(1))
    )


def test_select_having_after_where_without_group_by() -> None:
    stmt = parse("SELECT venue FROM papers WHERE anio > 2010 HAVING COUNT(*) > 1")
    assert stmt.where is not None
    assert stmt.having == HavingClause(
        CompareExpr(FunctionExpr("COUNT", arg=None), ">", Literal(1))
    )


def test_select_having_complex_boolean_expression() -> None:
    stmt = parse("SELECT * FROM papers HAVING SUM(anio) > 100 AND COUNT(*) < 50")
    assert isinstance(stmt.having, HavingClause)
    assert isinstance(stmt.having.expr, LogicalExpr)


def test_select_having_not_set_by_default() -> None:
    stmt = parse("SELECT * FROM papers")
    assert stmt.having is None


def test_select_having_before_group_by_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers HAVING COUNT(*) > 1 GROUP BY venue")
