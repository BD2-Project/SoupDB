"""Tests for the SQL expression evaluator."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.query.ast import (
    BetweenExpr,
    ColumnDef,
    ColumnRef,
    ColumnType,
    CompareExpr,
    FunctionExpr,
    InExpr,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
)
from engine.query.evaluator import evaluate, evaluate_aggregate

SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("titulo", ColumnType.TEXT),
    ColumnDef("anio", ColumnType.INT),
    ColumnDef("score", ColumnType.FLOAT),
    ColumnDef("activo", ColumnType.BOOL),
)
ROW = (7, "RAG sobre papers", 2020, 0.9, True)

NOTAS_SCHEMA = (ColumnDef("nota", ColumnType.INT),)
NOTAS = [(5,), (7,), (5,), (3,)]
WITH_NULL = [(5,), (7,), (None,), (3,)]  # type: ignore[list-item]
FLOATS = [(1.5,), (2.5,)]
MIXED_NUM = [(1,), (2.5,)]
NAMES_SCHEMA = (ColumnDef("name", ColumnType.TEXT),)
NAMES = [("apple",), ("banana",), ("apple",)]


def test_evaluate_literal() -> None:
    assert evaluate(Literal(42), ROW, SCHEMA) == 42
    assert evaluate(Literal(3.5), ROW, SCHEMA) == 3.5
    assert evaluate(Literal("texto"), ROW, SCHEMA) == "texto"
    assert evaluate(Literal(True), ROW, SCHEMA) is True


def test_evaluate_column_ref() -> None:
    assert evaluate(ColumnRef("anio"), ROW, SCHEMA) == 2020
    assert evaluate(ColumnRef("activo"), ROW, SCHEMA) is True


def test_evaluate_unknown_column_raises() -> None:
    with pytest.raises(QueryExecutionError):
        evaluate(ColumnRef("missing"), ROW, SCHEMA)


def test_evaluate_compare_equality() -> None:
    expr = CompareExpr(ColumnRef("anio"), "=", Literal(2020))
    assert evaluate(expr, ROW, SCHEMA) is True
    expr = CompareExpr(ColumnRef("anio"), "=", Literal(2019))
    assert evaluate(expr, ROW, SCHEMA) is False


def test_evaluate_compare_operators() -> None:
    assert evaluate(CompareExpr(Literal(1), "<>", Literal(2)), ROW, SCHEMA) is True
    assert evaluate(CompareExpr(Literal(2), "<", Literal(1)), ROW, SCHEMA) is False
    assert evaluate(CompareExpr(Literal(1), "<=", Literal(1)), ROW, SCHEMA) is True
    assert evaluate(CompareExpr(Literal(3), ">", Literal(2)), ROW, SCHEMA) is True
    assert evaluate(CompareExpr(Literal(2), ">=", Literal(3)), ROW, SCHEMA) is False


def test_evaluate_compare_floats() -> None:
    assert evaluate(CompareExpr(Literal(0.9), "=", Literal(0.9)), ROW, SCHEMA) is True


def test_evaluate_compare_strings() -> None:
    assert evaluate(CompareExpr(Literal("b"), "<", Literal("c")), ROW, SCHEMA) is True
    assert evaluate(CompareExpr(Literal("b"), "=", Literal("a")), ROW, SCHEMA) is False


def test_evaluate_compare_ints_and_floats_raises() -> None:
    expr = CompareExpr(Literal(1), "=", Literal(1.0))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, ROW, SCHEMA)


def test_evaluate_compare_int_and_str_raises() -> None:
    expr = CompareExpr(Literal(1), "=", Literal("1"))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, ROW, SCHEMA)


def test_evaluate_compare_bool_and_int_raises() -> None:
    expr = CompareExpr(Literal(True), "=", Literal(1))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, ROW, SCHEMA)


def test_evaluate_logical_and() -> None:
    expr = LogicalExpr(Literal(True), "AND", Literal(True))
    assert evaluate(expr, ROW, SCHEMA) is True
    expr = LogicalExpr(Literal(True), "AND", Literal(False))
    assert evaluate(expr, ROW, SCHEMA) is False


def test_evaluate_logical_or() -> None:
    expr = LogicalExpr(Literal(False), "OR", Literal(False))
    assert evaluate(expr, ROW, SCHEMA) is False
    expr = LogicalExpr(Literal(False), "OR", Literal(True))
    assert evaluate(expr, ROW, SCHEMA) is True


def test_evaluate_logical_non_bool_raises() -> None:
    expr = LogicalExpr(Literal(1), "AND", Literal(True))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, ROW, SCHEMA)


def test_evaluate_not() -> None:
    assert evaluate(NotExpr(Literal(True)), ROW, SCHEMA) is False
    assert evaluate(NotExpr(Literal(False)), ROW, SCHEMA) is True


def test_evaluate_not_non_bool_raises() -> None:
    with pytest.raises(QueryExecutionError):
        evaluate(NotExpr(Literal(1)), ROW, SCHEMA)


def test_evaluate_between_inclusive() -> None:
    expr = BetweenExpr(Literal(5), Literal(1), Literal(10))
    assert evaluate(expr, ROW, SCHEMA) is True
    expr = BetweenExpr(Literal(1), Literal(1), Literal(10))
    assert evaluate(expr, ROW, SCHEMA) is True
    expr = BetweenExpr(Literal(11), Literal(1), Literal(10))
    assert evaluate(expr, ROW, SCHEMA) is False


def test_evaluate_between_strings() -> None:
    expr = BetweenExpr(Literal("m"), Literal("a"), Literal("z"))
    assert evaluate(expr, ROW, SCHEMA) is True


def test_evaluate_between_mixed_types_raises() -> None:
    expr = BetweenExpr(Literal(5), Literal(1.0), Literal(10))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, ROW, SCHEMA)


def test_evaluate_in() -> None:
    expr = InExpr(Literal(2), (Literal(1), Literal(2), Literal(3)))
    assert evaluate(expr, ROW, SCHEMA) is True
    expr = InExpr(Literal(9), (Literal(1), Literal(2), Literal(3)))
    assert evaluate(expr, ROW, SCHEMA) is False


def test_evaluate_in_strings() -> None:
    expr = InExpr(Literal("x"), (Literal("a"), Literal("b")))
    assert evaluate(expr, ROW, SCHEMA) is False


def test_evaluate_in_mixed_types_raises() -> None:
    expr = InExpr(Literal(2), (Literal("a"), Literal(2)))
    with pytest.raises(QueryExecutionError):
        evaluate(expr, ROW, SCHEMA)


def test_evaluate_like_exact() -> None:
    expr = LikeExpr(Literal("RAG"), Literal("RAG"))
    assert evaluate(expr, ROW, SCHEMA) is True
    expr = LikeExpr(Literal("RAG"), Literal("rag"))
    assert evaluate(expr, ROW, SCHEMA) is False


def test_evaluate_like_percent_wildcard() -> None:
    assert evaluate(LikeExpr(Literal("RAG papers"), Literal("RAG%")), ROW, SCHEMA) is True
    assert evaluate(LikeExpr(Literal("RAG papers"), Literal("%papers")), ROW, SCHEMA) is True
    assert evaluate(LikeExpr(Literal("RAG papers"), Literal("%sobre%")), ROW, SCHEMA) is False


def test_evaluate_like_underscore_wildcard() -> None:
    assert evaluate(LikeExpr(Literal("RAG"), Literal("R_G")), ROW, SCHEMA) is True
    assert evaluate(LikeExpr(Literal("RA"), Literal("R_G")), ROW, SCHEMA) is False


def test_evaluate_like_non_string_raises() -> None:
    with pytest.raises(QueryExecutionError):
        evaluate(LikeExpr(Literal(5), Literal("5")), ROW, SCHEMA)
    with pytest.raises(QueryExecutionError):
        evaluate(LikeExpr(Literal("5"), Literal(5)), ROW, SCHEMA)


def test_evaluate_aggregate_in_scalar_context_raises() -> None:
    expr = FunctionExpr("COUNT", arg=None)
    with pytest.raises(QueryExecutionError):
        evaluate(expr, ROW, SCHEMA)


def test_aggregate_count_star() -> None:
    expr = FunctionExpr("COUNT", arg=None)
    assert evaluate_aggregate(expr, NOTAS, NOTAS_SCHEMA) == 4


def test_aggregate_count_column() -> None:
    expr = FunctionExpr("COUNT", ColumnRef("nota"))
    assert evaluate_aggregate(expr, WITH_NULL, NOTAS_SCHEMA) == 3


def test_aggregate_count_distinct() -> None:
    expr = FunctionExpr("COUNT", ColumnRef("nota"), distinct=True)
    assert evaluate_aggregate(expr, NOTAS, NOTAS_SCHEMA) == 3


def test_aggregate_count_empty_group() -> None:
    expr = FunctionExpr("COUNT", arg=None)
    assert evaluate_aggregate(expr, [], NOTAS_SCHEMA) == 0


def test_aggregate_sum() -> None:
    expr = FunctionExpr("SUM", ColumnRef("nota"))
    assert evaluate_aggregate(expr, NOTAS, NOTAS_SCHEMA) == 20


def test_aggregate_sum_distinct() -> None:
    expr = FunctionExpr("SUM", ColumnRef("nota"), distinct=True)
    assert evaluate_aggregate(expr, NOTAS, NOTAS_SCHEMA) == 15


def test_aggregate_sum_floats() -> None:
    expr = FunctionExpr("SUM", ColumnRef("nota"))
    assert evaluate_aggregate(expr, FLOATS, NOTAS_SCHEMA) == 4.0


def test_aggregate_sum_mixed_numeric() -> None:
    expr = FunctionExpr("SUM", ColumnRef("nota"))
    assert evaluate_aggregate(expr, MIXED_NUM, NOTAS_SCHEMA) == 3.5


def test_aggregate_sum_empty_group() -> None:
    expr = FunctionExpr("SUM", ColumnRef("nota"))
    assert evaluate_aggregate(expr, [], NOTAS_SCHEMA) == 0


def test_aggregate_sum_over_strings_raises() -> None:
    expr = FunctionExpr("SUM", ColumnRef("name"))
    with pytest.raises(QueryExecutionError):
        evaluate_aggregate(expr, NAMES, NAMES_SCHEMA)


def test_aggregate_avg() -> None:
    expr = FunctionExpr("AVG", ColumnRef("nota"))
    assert evaluate_aggregate(expr, NOTAS, NOTAS_SCHEMA) == 5.0


def test_aggregate_avg_empty_group_raises() -> None:
    expr = FunctionExpr("AVG", ColumnRef("nota"))
    with pytest.raises(QueryExecutionError):
        evaluate_aggregate(expr, [], NOTAS_SCHEMA)


def test_aggregate_avg_over_strings_raises() -> None:
    expr = FunctionExpr("AVG", ColumnRef("name"))
    with pytest.raises(QueryExecutionError):
        evaluate_aggregate(expr, NAMES, NAMES_SCHEMA)


def test_aggregate_min_max() -> None:
    min_expr = FunctionExpr("MIN", ColumnRef("nota"))
    max_expr = FunctionExpr("MAX", ColumnRef("nota"))
    assert evaluate_aggregate(min_expr, NOTAS, NOTAS_SCHEMA) == 3
    assert evaluate_aggregate(max_expr, NOTAS, NOTAS_SCHEMA) == 7


def test_aggregate_min_max_strings() -> None:
    min_expr = FunctionExpr("MIN", ColumnRef("name"))
    max_expr = FunctionExpr("MAX", ColumnRef("name"))
    assert evaluate_aggregate(min_expr, NAMES, NAMES_SCHEMA) == "apple"
    assert evaluate_aggregate(max_expr, NAMES, NAMES_SCHEMA) == "banana"


def test_aggregate_min_max_empty_group_raises() -> None:
    for name in ("MIN", "MAX"):
        expr = FunctionExpr(name, ColumnRef("nota"))
        with pytest.raises(QueryExecutionError):
            evaluate_aggregate(expr, [], NOTAS_SCHEMA)


def test_aggregate_min_max_mixed_types_raises() -> None:
    rows = [(1,), ("2",)]  # type: ignore[list-item]
    for name in ("MIN", "MAX"):
        expr = FunctionExpr(name, ColumnRef("nota"))
        with pytest.raises(QueryExecutionError):
            evaluate_aggregate(expr, rows, NOTAS_SCHEMA)


def test_aggregate_unknown_function_raises() -> None:
    expr = FunctionExpr("MEDIAN", ColumnRef("nota"))
    with pytest.raises(QueryExecutionError):
        evaluate_aggregate(expr, NOTAS, NOTAS_SCHEMA)
