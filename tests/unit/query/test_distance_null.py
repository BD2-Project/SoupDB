import pytest

from engine.common.errors import QueryExecutionError
from engine.query.ast import CompareExpr, DistanceExpr, Literal, PointExpr
from engine.query.evaluator import evaluate


def test_distance_returns_none_when_left_is_null() -> None:
    expr = DistanceExpr(Literal(None), PointExpr(Literal(0.0), Literal(0.0)))
    assert evaluate(expr, (), ()) is None


def test_distance_returns_none_when_right_is_null() -> None:
    expr = DistanceExpr(PointExpr(Literal(1.0), Literal(2.0)), Literal(None))
    assert evaluate(expr, (), ()) is None


def test_distance_returns_none_when_both_null() -> None:
    expr = DistanceExpr(Literal(None), Literal(None))
    assert evaluate(expr, (), ()) is None


def test_compare_with_null_distance_is_false() -> None:
    expr = CompareExpr(
        DistanceExpr(Literal(None), PointExpr(Literal(0.0), Literal(0.0))),
        ">=",
        Literal(0.0),
    )
    assert evaluate(expr, (), ()) is False


def test_distance_raises_for_non_null_non_point_left() -> None:
    expr = DistanceExpr(Literal(5), PointExpr(Literal(0.0), Literal(0.0)))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, (), ())


def test_distance_raises_for_non_null_non_point_right() -> None:
    expr = DistanceExpr(PointExpr(Literal(0.0), Literal(0.0)), Literal("x"))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, (), ())


def test_compare_with_null_literal_is_false() -> None:
    assert evaluate(CompareExpr(Literal(None), "=", Literal(5)), (), ()) is False
    assert evaluate(CompareExpr(Literal(5), ">", Literal(None)), (), ()) is False
