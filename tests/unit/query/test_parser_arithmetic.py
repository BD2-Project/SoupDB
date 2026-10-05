"""Tests for the SQL parser: binary arithmetic expressions.

Precedence (high to low) inside a predicate operand: primary >
``* / %`` > ``+ -`` > comparison. A leading minus glued to a number is a
literal, while ``-`` between two operands is a binary operator.
"""

import pytest

from engine.query.ast import (
    BinaryExpr,
    ColumnRef,
    CompareExpr,
    Literal,
    NotExpr,
    SelectStatement,
)
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def parse_expr(sql: str):
    stmt = parse(f"SELECT {sql} FROM papers")
    assert isinstance(stmt, SelectStatement)
    return stmt.columns[0].expr


def test_select_binary_subtraction() -> None:
    expr = parse_expr("id - 1")
    assert expr == BinaryExpr(ColumnRef("id"), "-", Literal(1))


def test_select_leading_minus_is_literal_not_binary() -> None:
    expr = parse_expr("-1")
    assert expr == Literal(-1)


def test_select_binary_addition() -> None:
    expr = parse_expr("id + 1")
    assert expr == BinaryExpr(ColumnRef("id"), "+", Literal(1))


def test_select_binary_multiplication() -> None:
    expr = parse_expr("anio * 2")
    assert expr == BinaryExpr(ColumnRef("anio"), "*", Literal(2))


def test_select_binary_division() -> None:
    expr = parse_expr("anio / 2")
    assert expr == BinaryExpr(ColumnRef("anio"), "/", Literal(2))


def test_select_binary_modulo() -> None:
    expr = parse_expr("anio % 2")
    assert expr == BinaryExpr(ColumnRef("anio"), "%", Literal(2))


def test_multiplication_binds_tighter_than_addition() -> None:
    expr = parse_expr("1 + 2 * 3")
    assert expr == BinaryExpr(Literal(1), "+", BinaryExpr(Literal(2), "*", Literal(3)))


def test_add_subtraction_are_left_associative() -> None:
    expr = parse_expr("id - 1 + 2")
    assert expr == BinaryExpr(BinaryExpr(ColumnRef("id"), "-", Literal(1)), "+", Literal(2))


def test_division_binds_tighter_than_subtraction() -> None:
    expr = parse_expr("id - 6 / 3")
    assert expr == BinaryExpr(ColumnRef("id"), "-", BinaryExpr(Literal(6), "/", Literal(3)))


def test_parenthesized_arithmetic() -> None:
    expr = parse_expr("(1 + 2) * 3")
    assert expr == BinaryExpr(BinaryExpr(Literal(1), "+", Literal(2)), "*", Literal(3))


def test_comparison_binds_looser_than_arithmetic() -> None:
    stmt = parse("SELECT * FROM papers WHERE anio + 1 = 2020")
    assert isinstance(stmt, SelectStatement)
    assert stmt.where == CompareExpr(
        BinaryExpr(ColumnRef("anio"), "+", Literal(1)), "=", Literal(2020)
    )


def test_arithmetic_inside_not_operand() -> None:
    stmt = parse("SELECT * FROM papers WHERE NOT anio = 2020 + 1")
    assert isinstance(stmt, SelectStatement)
    assert isinstance(stmt.where, NotExpr)
    assert stmt.where.operand == CompareExpr(
        ColumnRef("anio"), "=", BinaryExpr(Literal(2020), "+", Literal(1))
    )


def test_arithmetic_in_update_assignment() -> None:
    from engine.query.ast import UpdateStatement

    stmt = parse("UPDATE papers SET anio = anio + 1")
    assert isinstance(stmt, UpdateStatement)
    assert stmt.assignments == (("anio", BinaryExpr(ColumnRef("anio"), "+", Literal(1))),)


def test_select_star_still_maps_to_empty_columns() -> None:
    stmt = parse("SELECT * FROM papers")
    assert isinstance(stmt, SelectStatement)
    assert stmt.columns == ()


def test_select_two_columns_with_star_between_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT id, * FROM papers")
