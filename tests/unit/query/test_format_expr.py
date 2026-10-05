"""El detalle del plan se lee como SQL, no como el `repr` del AST."""

import pytest

from engine.query.ast import (
    BetweenExpr,
    ColumnRef,
    CompareExpr,
    DistanceExpr,
    FunctionExpr,
    InExpr,
    IsNullExpr,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    PointExpr,
)
from engine.query.format_expr import format_expr
from engine.query.parser import parse


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (Literal(14), "14"),
        (Literal(None), "NULL"),
        (Literal(True), "TRUE"),
        (Literal("Pérez"), "'Pérez'"),
        # Una comilla dentro del texto se duplica, como en SQL.
        (Literal("O'Hara"), "'O''Hara'"),
        (ColumnRef("nota"), "nota"),
        (CompareExpr(ColumnRef("nota"), ">=", Literal(14)), "nota >= 14"),
        (IsNullExpr(ColumnRef("nota")), "nota IS NULL"),
        (IsNullExpr(ColumnRef("nota"), negated=True), "nota IS NOT NULL"),
        (NotExpr(ColumnRef("activo")), "NOT activo"),
        (BetweenExpr(ColumnRef("nota"), Literal(10), Literal(20)), "nota BETWEEN 10 AND 20"),
        (InExpr(ColumnRef("id"), (Literal(1), Literal(2))), "id IN (1, 2)"),
        (LikeExpr(ColumnRef("nombre"), Literal("P%")), "nombre LIKE 'P%'"),
        (FunctionExpr("COUNT", None), "COUNT(*)"),
        (FunctionExpr("COUNT", ColumnRef("id"), distinct=True), "COUNT(DISTINCT id)"),
        # El literal se muestra como se escribe: latitud y después longitud.
        (PointExpr(Literal(-77.04), Literal(-12.05)), "POINT(-12.05, -77.04)"),
    ],
)
def test_renders_as_sql(expr, expected: str) -> None:
    assert format_expr(expr) == expected


def test_logical_operands_are_parenthesised() -> None:
    # Sin paréntesis, `a AND (b OR c)` y `(a AND b) OR c` se verían igual.
    inner = LogicalExpr(ColumnRef("b"), "OR", ColumnRef("c"))
    assert format_expr(LogicalExpr(ColumnRef("a"), "AND", inner)) == "(a AND (b OR c))"


def test_distance_keeps_the_metric() -> None:
    expr = DistanceExpr(
        ColumnRef("ubicacion"), PointExpr(Literal(-77.0), Literal(-12.0)), "haversine"
    )
    assert format_expr(expr) == "distancia(ubicacion, POINT(-12.0, -77.0), 'haversine')"


def test_plan_detail_of_a_parsed_query() -> None:
    statement = parse("SELECT * FROM t WHERE nota >= 14 AND nombre LIKE 'P%'")
    assert format_expr(statement.where) == "(nota >= 14 AND nombre LIKE 'P%')"
