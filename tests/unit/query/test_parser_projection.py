"""Tests for the SQL parser: projections with alias and DISTINCT."""

import pytest

from engine.query.ast import ColumnRef, Literal, SelectColumn, SelectStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_select_column_list() -> None:
    stmt = parse("SELECT id, titulo, anio FROM papers")
    assert isinstance(stmt, SelectStatement)
    assert stmt.columns == (
        SelectColumn(ColumnRef("id")),
        SelectColumn(ColumnRef("titulo")),
        SelectColumn(ColumnRef("anio")),
    )


def test_select_alias_with_as_keyword() -> None:
    stmt = parse("SELECT titulo AS t FROM papers")
    assert stmt.columns == (SelectColumn(ColumnRef("titulo"), alias="t"),)


def test_select_alias_without_as_keyword() -> None:
    stmt = parse("SELECT titulo t FROM papers")
    assert stmt.columns == (SelectColumn(ColumnRef("titulo"), alias="t"),)


def test_select_alias_is_not_consumed_as_column() -> None:
    stmt = parse("SELECT a AS x, b AS y FROM papers")
    assert stmt.columns == (
        SelectColumn(ColumnRef("a"), alias="x"),
        SelectColumn(ColumnRef("b"), alias="y"),
    )


def test_select_mix_star_and_column_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT *, titulo FROM papers")


def test_select_distinct_flag() -> None:
    stmt = parse("SELECT DISTINCT venue FROM papers")
    assert stmt.distinct is True
    assert stmt.columns == (SelectColumn(ColumnRef("venue")),)


def test_select_distinct_star() -> None:
    stmt = parse("SELECT DISTINCT * FROM papers")
    assert stmt.distinct is True
    assert stmt.columns == ()


def test_select_literal_projection() -> None:
    stmt = parse("SELECT 1 FROM papers")
    assert stmt.columns == (SelectColumn(Literal(1)),)


def test_select_decimal_literal_projection() -> None:
    stmt = parse("SELECT 1.25 FROM papers")
    assert stmt.columns == (SelectColumn(Literal(1.25)),)


def test_select_boolean_literals() -> None:
    stmt = parse("SELECT TRUE, FALSE FROM papers")
    assert stmt.columns == (
        SelectColumn(Literal(True)),
        SelectColumn(Literal(False)),
    )
