"""Tests for the SQL parser: POINT(x, y) and distance(a, b) expressions."""

import pytest

from engine.query.ast import (
    ColumnRef,
    ColumnType,
    CompareExpr,
    CreateTableStatement,
    DistanceExpr,
    InsertStatement,
    Literal,
    OrderByItem,
    PointExpr,
    SelectStatement,
)
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_parse_where_distance_point() -> None:
    stmt = parse("SELECT * FROM t WHERE distance(ubicacion, POINT(1, 2)) < 5.0")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where == CompareExpr(
        DistanceExpr(
            ColumnRef("ubicacion"),
            PointExpr(Literal(1), Literal(2)),
        ),
        "<",
        Literal(5.0),
    )


def test_parse_order_by_distance_desc() -> None:
    stmt = parse("SELECT nombre FROM t ORDER BY distance(ubicacion, POINT(0, 0)) DESC")
    assert isinstance(stmt, SelectStatement)
    assert stmt.order_by == (
        OrderByItem(
            DistanceExpr(ColumnRef("ubicacion"), PointExpr(Literal(0), Literal(0))),
            ascending=False,
        ),
    )


def test_parse_project_distance_alias() -> None:
    sql = (
        "SELECT distance(ubicacion, POINT(0, 0)) AS d FROM t "
        "ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 3"
    )
    stmt = parse(sql)
    assert isinstance(stmt, SelectStatement)
    assert stmt.limit is not None
    assert stmt.limit.limit == Literal(3)


def test_parse_distance_with_decimal_coordinates() -> None:
    stmt = parse("SELECT * FROM t WHERE distance(ubicacion, POINT(1.5, -2.25)) <= 10.0")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where == CompareExpr(
        DistanceExpr(
            ColumnRef("ubicacion"),
            PointExpr(Literal(1.5), Literal(-2.25)),
        ),
        "<=",
        Literal(10.0),
    )


def test_parse_point_in_insert_values() -> None:
    stmt = parse("INSERT INTO t (nombre, ubicacion) VALUES ('A', POINT(3, 4))")
    assert isinstance(stmt, InsertStatement)
    assert stmt.values == ((Literal("A"), PointExpr(Literal(3), Literal(4))),)


def test_parse_create_table_with_point_column() -> None:
    stmt = parse("CREATE TABLE lugares (nombre TEXT, ubicacion POINT)")
    assert isinstance(stmt, CreateTableStatement)
    assert stmt.columns[1].name == "ubicacion"
    assert stmt.columns[1].type_name is ColumnType.POINT


def test_parse_point_without_parenthesis_is_not_function() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM t WHERE POINT = 5")
