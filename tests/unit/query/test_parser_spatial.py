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


# --- distance(a, b, 'metrica') -------------------------------------------


def test_parse_distance_without_metric_defaults_to_euclidean() -> None:
    stmt = parse("SELECT * FROM t WHERE distance(ubicacion, POINT(1, 2)) < 5.0")
    assert isinstance(stmt, SelectStatement)
    assert isinstance(stmt.where, CompareExpr)
    assert stmt.where.left == DistanceExpr(
        ColumnRef("ubicacion"), PointExpr(Literal(1), Literal(2))
    )
    assert stmt.where.left.metric == "euclidean"


def test_parse_distance_with_haversine_metric() -> None:
    stmt = parse("SELECT * FROM t WHERE distance(ubicacion, POINT(1, 2), 'haversine') < 100.0")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where == CompareExpr(
        DistanceExpr(ColumnRef("ubicacion"), PointExpr(Literal(1), Literal(2)), "haversine"),
        "<",
        Literal(100.0),
    )


def test_parse_distance_metric_is_normalized_to_lowercase() -> None:
    for literal in ("'haversine'", "'Haversine'", "'  HAVersine  '"):
        stmt = parse(f"SELECT nombre FROM t ORDER BY distance(p, POINT(0, 0), {literal}) DESC")
        assert isinstance(stmt, SelectStatement)
        assert stmt.order_by[0].expr.metric == "haversine"
        assert stmt.order_by[0].ascending is False


def test_parse_distance_metric_in_projection_and_where() -> None:
    sql = (
        "SELECT distance(ubicacion, POINT(0, 0), 'haversine') AS d FROM t "
        "WHERE distance(ubicacion, POINT(0, 0), 'haversine') < 10.0 LIMIT 5"
    )
    stmt = parse(sql)
    assert isinstance(stmt, SelectStatement)
    assert isinstance(stmt.columns[0].expr, DistanceExpr)
    assert stmt.columns[0].expr.metric == "haversine"
    assert isinstance(stmt.where, CompareExpr)
    assert stmt.where.left.metric == "haversine"
    assert stmt.limit is not None and stmt.limit.limit == Literal(5)


def test_parse_distance_with_unknown_metric_fails() -> None:
    with pytest.raises(QueryParseError, match="unknown distance metric"):
        parse("SELECT * FROM t WHERE distance(ubicacion, POINT(1, 2), 'manhattan') < 5.0")


def test_parse_distance_unknown_metric_error_mentions_position_and_names() -> None:
    with pytest.raises(QueryParseError) as excinfo:
        parse("SELECT * FROM t WHERE distance(ubicacion, POINT(1, 2), 'manhattan') < 5.0")
    message = str(excinfo.value)
    assert "manhattan" in message
    assert "euclidean" in message and "haversine" in message
    assert "position" in message


@pytest.mark.parametrize("argument", ["haversine", "1", "metrico", "POINT(1, 2)", "NULL"])
def test_parse_distance_metric_must_be_a_string_literal(argument: str) -> None:
    with pytest.raises(QueryParseError, match="metric must be a string literal"):
        parse(f"SELECT * FROM t WHERE distance(ubicacion, POINT(1, 2), {argument}) < 5.0")


def test_parse_distance_with_too_many_arguments_fails() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM t WHERE distance(ubicacion, POINT(1, 2), 'haversine', 3) < 5.0")
