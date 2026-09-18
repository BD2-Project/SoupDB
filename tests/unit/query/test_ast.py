"""Tests for the SQL abstract syntax tree nodes."""

from engine.query.ast import (
    BetweenExpr,
    ColumnDef,
    ColumnRef,
    ColumnType,
    CompareExpr,
    CreateTableStatement,
    DeleteStatement,
    FunctionExpr,
    InExpr,
    InsertStatement,
    LikeExpr,
    Literal,
    LogicalExpr,
    NotExpr,
    OrderByItem,
    SelectColumn,
    SelectStatement,
)


def test_compare_expression_structure() -> None:
    expr = CompareExpr(ColumnRef("anio"), "=", Literal(2020))
    assert expr.left == ColumnRef("anio")
    assert expr.op == "="
    assert expr.right == Literal(2020)


def test_logical_not_and_between_tree() -> None:
    between = BetweenExpr(ColumnRef("anio"), Literal(2015), Literal(2020))
    expr = LogicalExpr(between, "AND", NotExpr(ColumnRef("retired")))
    assert isinstance(expr.left, BetweenExpr)
    assert isinstance(expr.right, NotExpr)
    assert isinstance(expr.right.operand, ColumnRef)


def test_like_and_in_expressions() -> None:
    like = LikeExpr(ColumnRef("titulo"), Literal("%rag%"))
    in_expr = InExpr(ColumnRef("venue"), (Literal("SIGMOD"), Literal("VLDB")))
    assert like.pattern.value == "%rag%"
    assert in_expr.value == ColumnRef("venue")
    assert len(in_expr.items) == 2


def test_function_expression() -> None:
    fn = FunctionExpr("COUNT", ColumnRef("id"), distinct=False)
    assert fn.name == "COUNT"
    assert fn.arg == ColumnRef("id")
    assert fn.distinct is False


def test_select_statement_full_shape() -> None:
    stmt = SelectStatement(
        columns=(SelectColumn(ColumnRef("titulo")),),
        table="papers",
        where=CompareExpr(ColumnRef("anio"), ">=", Literal(2018)),
        group_by=(ColumnRef("venue"),),
        order_by=(OrderByItem(ColumnRef("anio"), ascending=False),),
        distinct=True,
    )
    assert stmt.table == "papers"
    assert stmt.columns[0].expr == ColumnRef("titulo")
    assert stmt.group_by == (ColumnRef("venue"),)
    assert stmt.order_by[0].ascending is False
    assert stmt.distinct is True


def test_minimal_select_statement() -> None:
    stmt = SelectStatement(columns=(), table="papers")
    assert stmt.where is None
    assert stmt.group_by == ()
    assert stmt.order_by == ()
    assert stmt.distinct is False


def test_insert_statement_shape() -> None:
    stmt = InsertStatement(
        table="papers",
        columns=("id", "titulo"),
        values=((Literal(1), Literal("RAG")),),
    )
    assert stmt.columns == ("id", "titulo")
    assert stmt.values[0][1] == Literal("RAG")


def test_delete_statement_shape() -> None:
    stmt = DeleteStatement(
        table="papers",
        where=CompareExpr(ColumnRef("anio"), "<", Literal(2010)),
    )
    assert stmt.table == "papers"
    assert stmt.where is not None


def test_create_table_statement_shape() -> None:
    stmt = CreateTableStatement(
        table="papers",
        columns=(
            ColumnDef("id", ColumnType.INT),
            ColumnDef("titulo", ColumnType.TEXT),
            ColumnDef("nombre", ColumnType.VARCHAR, length=60),
        ),
    )
    assert stmt.columns[0].type_name == ColumnType.INT
    assert stmt.columns[2].length == 60
