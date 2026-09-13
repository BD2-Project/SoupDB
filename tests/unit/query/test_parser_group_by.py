"""Tests for the SQL parser: GROUP BY clause and aggregate functions."""

import pytest

from engine.query.ast import (
    ColumnRef,
    FunctionExpr,
    SelectColumn,
    SelectStatement,
)
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_group_by_single_column() -> None:
    stmt = parse("SELECT * FROM papers GROUP BY venue")
    assert isinstance(stmt, SelectStatement)
    assert stmt.group_by == (ColumnRef("venue"),)


def test_group_by_multiple_columns() -> None:
    stmt = parse("SELECT * FROM papers GROUP BY venue, anio")
    assert stmt.group_by == (ColumnRef("venue"), ColumnRef("anio"))


def test_group_by_after_where() -> None:
    stmt = parse("SELECT * FROM papers WHERE anio > 2010 GROUP BY venue")
    assert stmt.where is not None
    assert stmt.group_by == (ColumnRef("venue"),)


def test_count_star_projection() -> None:
    stmt = parse("SELECT COUNT(*) FROM papers")
    assert stmt.columns == (SelectColumn(FunctionExpr("COUNT", arg=None)),)


def test_count_column_projection() -> None:
    stmt = parse("SELECT COUNT(id) FROM papers")
    assert stmt.columns == (SelectColumn(FunctionExpr("COUNT", ColumnRef("id"))),)


def test_count_distinct_projection() -> None:
    stmt = parse("SELECT COUNT(DISTINCT venue) FROM papers")
    assert stmt.columns == (SelectColumn(FunctionExpr("COUNT", ColumnRef("venue"), distinct=True)),)


def test_sum_min_max_avg_projections() -> None:
    funcs = ["SUM", "MIN", "MAX", "AVG"]
    for name in funcs:
        stmt = parse(f"SELECT {name}(anio) FROM papers")
        assert stmt.columns == (SelectColumn(FunctionExpr(name, ColumnRef("anio"))),)


def test_group_by_with_aggregate_query() -> None:
    stmt = parse("SELECT venue, COUNT(*) FROM papers GROUP BY venue")
    assert stmt.group_by == (ColumnRef("venue"),)
    assert stmt.columns == (
        SelectColumn(ColumnRef("venue")),
        SelectColumn(FunctionExpr("COUNT", arg=None)),
    )


def test_aggregate_without_parens_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT COUNT id FROM papers")


def test_group_by_without_column_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers GROUP BY")
