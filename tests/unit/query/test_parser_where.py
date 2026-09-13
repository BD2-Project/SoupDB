"""Tests for the SQL parser: WHERE clause expressions."""

import pytest

from engine.query.ast import (
    BetweenExpr,
    ColumnRef,
    CompareExpr,
    InExpr,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    SelectStatement,
)
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def parse_where(sql: str):
    stmt = parse(f"SELECT * FROM papers WHERE {sql}")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where is not None
    return stmt.where


def test_where_comparison_equality() -> None:
    expr = parse_where("id = 42")
    assert expr == CompareExpr(ColumnRef("id"), "=", Literal(42))


def test_where_comparison_number_and_string() -> None:
    expr = parse_where("titulo = 'RAG'")
    assert expr == CompareExpr(ColumnRef("titulo"), "=", Literal("RAG"))


def test_where_all_comparison_operators() -> None:
    ops = ["=", "<>", "!=", "<", "<=", ">", ">="]
    for op in ops:
        expr = parse_where(f"anio {op} 2020")
        assert expr == CompareExpr(ColumnRef("anio"), op, Literal(2020))


def test_where_and_precedence() -> None:
    expr = parse_where("a = 1 AND b = 2")
    assert expr == LogicalExpr(
        CompareExpr(ColumnRef("a"), "=", Literal(1)),
        "AND",
        CompareExpr(ColumnRef("b"), "=", Literal(2)),
    )


def test_where_or_precedence() -> None:
    expr = parse_where("a = 1 OR b = 2")
    assert isinstance(expr, LogicalExpr)
    assert expr.op == "OR"


def test_where_ands_chain_left_associative() -> None:
    expr = parse_where("a = 1 AND b = 2 AND c = 3")
    assert isinstance(expr, LogicalExpr)
    assert isinstance(expr.left, LogicalExpr)
    assert expr.left.left == CompareExpr(ColumnRef("a"), "=", Literal(1))


def test_where_or_binds_looser_than_and() -> None:
    expr = parse_where("a = 1 OR b = 2 AND c = 3")
    assert isinstance(expr, LogicalExpr)
    assert expr.op == "OR"
    assert isinstance(expr.right, LogicalExpr)
    assert expr.right.op == "AND"


def test_where_not() -> None:
    expr = parse_where("NOT a = 1")
    assert isinstance(expr, NotExpr)
    assert expr.operand == CompareExpr(ColumnRef("a"), "=", Literal(1))


def test_where_not_and_precedence() -> None:
    expr = parse_where("NOT a = 1 AND b = 2")
    assert isinstance(expr, LogicalExpr)
    assert expr.op == "AND"
    assert isinstance(expr.left, NotExpr)


def test_where_between_inclusive() -> None:
    expr = parse_where("anio BETWEEN 2015 AND 2020")
    assert isinstance(expr, BetweenExpr)
    assert expr.value == ColumnRef("anio")
    assert expr.lo == Literal(2015)
    assert expr.hi == Literal(2020)


def test_where_in_list() -> None:
    expr = parse_where("id IN (1, 2, 3)")
    assert isinstance(expr, InExpr)
    assert expr.value == ColumnRef("id")
    assert expr.items == (Literal(1), Literal(2), Literal(3))


def test_where_in_with_strings() -> None:
    expr = parse_where("venue IN ('SIGMOD', 'VLDB')")
    assert isinstance(expr, InExpr)
    assert expr.items == (Literal("SIGMOD"), Literal("VLDB"))


def test_where_like() -> None:
    expr = parse_where("titulo LIKE '%rag%'")
    assert isinstance(expr, LikeExpr)
    assert expr.value == ColumnRef("titulo")
    assert expr.pattern == Literal("%rag%")


def test_where_comparison_missing_rhs_raises() -> None:
    with pytest.raises(QueryParseError):
        parse_where("a =")


def test_where_bare_identifier_raises() -> None:
    with pytest.raises(QueryParseError):
        parse_where("a")


def test_where_literal_only_raises() -> None:
    with pytest.raises(QueryParseError):
        parse_where("1 ==")


def test_where_missing_close_paren() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers WHERE (a = 1")
