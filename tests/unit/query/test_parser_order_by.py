"""Tests for the SQL parser: ORDER BY clause."""

import pytest

from engine.query.ast import ColumnRef, OrderByItem, SelectStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_order_by_ascending_default() -> None:
    stmt = parse("SELECT * FROM papers ORDER BY anio")
    assert isinstance(stmt, SelectStatement)
    assert stmt.order_by == (OrderByItem(ColumnRef("anio"), ascending=True),)


def test_order_by_descending() -> None:
    stmt = parse("SELECT * FROM papers ORDER BY anio DESC")
    assert stmt.order_by == (OrderByItem(ColumnRef("anio"), ascending=False),)


def test_order_by_ascending_explicit() -> None:
    stmt = parse("SELECT * FROM papers ORDER BY anio ASC")
    assert stmt.order_by == (OrderByItem(ColumnRef("anio"), ascending=True),)


def test_order_by_multiple_keys() -> None:
    stmt = parse("SELECT * FROM papers ORDER BY venue, anio DESC")
    assert stmt.order_by == (
        OrderByItem(ColumnRef("venue"), ascending=True),
        OrderByItem(ColumnRef("anio"), ascending=False),
    )


def test_order_by_case_insensitive_keywords() -> None:
    stmt = parse("select * from papers order by anio desc")
    assert stmt.order_by == (OrderByItem(ColumnRef("anio"), ascending=False),)


def test_order_by_appears_after_where() -> None:
    stmt = parse("SELECT * FROM papers WHERE anio > 2010 ORDER BY titulo")
    assert stmt.where is not None
    assert stmt.order_by == (OrderByItem(ColumnRef("titulo"), ascending=True),)


def test_order_by_without_column_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers ORDER BY")
