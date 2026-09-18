"""Tests for the SQL parser: LIMIT and OFFSET clauses."""

import pytest

from engine.query.ast import LimitClause, Literal, SelectStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_select_limit_only() -> None:
    stmt = parse("SELECT * FROM papers LIMIT 10")
    assert isinstance(stmt, SelectStatement)
    assert stmt.limit == LimitClause(limit=Literal(10), offset=None)


def test_select_limit_with_offset() -> None:
    stmt = parse("SELECT * FROM papers LIMIT 10 OFFSET 5")
    assert stmt.limit == LimitClause(limit=Literal(10), offset=Literal(5))


def test_select_limit_after_order_by() -> None:
    stmt = parse("SELECT * FROM papers ORDER BY anio LIMIT 3")
    assert stmt.limit == LimitClause(limit=Literal(3), offset=None)


def test_select_limit_after_where_and_group_by() -> None:
    stmt = parse(
        "SELECT venue, COUNT(*) FROM papers WHERE anio > 2010 GROUP BY venue LIMIT 5"
    )
    assert stmt.limit == LimitClause(limit=Literal(5), offset=None)


def test_select_no_limit_by_default() -> None:
    stmt = parse("SELECT * FROM papers")
    assert stmt.limit is None


def test_select_limit_zero_is_allowed() -> None:
    stmt = parse("SELECT * FROM papers LIMIT 0")
    assert stmt.limit == LimitClause(limit=Literal(0), offset=None)


def test_select_limit_string_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers LIMIT '10'")


def test_select_limit_negative_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers LIMIT -1")


def test_select_limit_decimal_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers LIMIT 1.5")


def test_select_offset_negative_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers LIMIT 5 OFFSET -1")