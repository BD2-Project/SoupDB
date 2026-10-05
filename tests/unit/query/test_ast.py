"""Tests for the SQL abstract syntax tree nodes."""

from engine.query.ast import (
    BetweenExpr,
    BinaryExpr,
    ColumnDef,
    ColumnRef,
    ColumnType,
    CompareExpr,
    CreateTableStatement,
    DeleteStatement,
    DistanceExpr,
    ExplainStatement,
    FunctionExpr,
    HavingClause,
    InExpr,
    InsertStatement,
    IsNullExpr,
    JoinClause,
    LikeExpr,
    LimitClause,
    Literal,
    LogicalExpr,
    NotExpr,
    OrderByItem,
    PointExpr,
    SelectColumn,
    SelectStatement,
    UpdateStatement,
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


def test_point_expression_structure() -> None:
    point = PointExpr(Literal(1), Literal(2.5))
    assert point.x == Literal(1)
    assert point.y == Literal(2.5)
    assert point == PointExpr(Literal(1), Literal(2.5))
    assert point != PointExpr(Literal(1), Literal(3))


def test_distance_expression_structure() -> None:
    point = PointExpr(Literal(3), Literal(4))
    dist = DistanceExpr(ColumnRef("ubicacion"), point)
    assert dist.left == ColumnRef("ubicacion")
    assert dist.right == point
    assert dist == DistanceExpr(ColumnRef("ubicacion"), point)
    assert dist != DistanceExpr(point, ColumnRef("ubicacion"))


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


def test_binary_expression_structure() -> None:
    expr = BinaryExpr(ColumnRef("a"), "+", Literal(1))
    assert expr.left == ColumnRef("a")
    assert expr.op == "+"
    assert expr.right == Literal(1)
    assert expr == BinaryExpr(ColumnRef("a"), "+", Literal(1))
    assert expr != BinaryExpr(ColumnRef("a"), "-", Literal(1))


def test_is_null_expression() -> None:
    null = IsNullExpr(ColumnRef("titulo"))
    not_null = IsNullExpr(ColumnRef("titulo"), negated=True)
    assert null.value == ColumnRef("titulo")
    assert null.negated is False
    assert not_null.negated is True
    assert null == IsNullExpr(ColumnRef("titulo"))
    assert null != not_null


def test_having_clause_shape() -> None:
    having = HavingClause(
        CompareExpr(FunctionExpr("COUNT", ColumnRef("id")), ">", Literal(10)),
    )
    assert having.expr.op == ">"
    assert having == HavingClause(having.expr)


def test_limit_clause_shape() -> None:
    limit_only = LimitClause(limit=Literal(10))
    limit_offset = LimitClause(limit=Literal(10), offset=Literal(20))
    assert limit_only.limit == Literal(10)
    assert limit_only.offset is None
    assert limit_offset.offset == Literal(20)
    assert limit_only == LimitClause(limit=Literal(10))
    assert limit_only != limit_offset


def test_join_clause_shape() -> None:
    on_expr = CompareExpr(ColumnRef("p.author_id"), "=", ColumnRef("a.id"))
    join = JoinClause("authors", on_expr)
    assert join.table == "authors"
    assert isinstance(join.on, CompareExpr)
    assert join == JoinClause("authors", on_expr)


def test_select_statement_with_having_limit_joins() -> None:
    stmt = SelectStatement(
        columns=(SelectColumn(ColumnRef("venue")),),
        table="papers",
        group_by=(ColumnRef("venue"),),
        having=HavingClause(
            CompareExpr(FunctionExpr("COUNT", ColumnRef("id")), ">", Literal(10)),
        ),
        limit=LimitClause(limit=Literal(5)),
        joins=(
            JoinClause(
                "venues",
                CompareExpr(ColumnRef("p.venue_id"), "=", ColumnRef("v.id")),
            ),
        ),
    )
    assert stmt.having is not None
    assert stmt.having.expr.op == ">"
    assert stmt.limit is not None
    assert stmt.limit.limit == Literal(5)
    assert stmt.limit.offset is None
    assert stmt.joins[0].table == "venues"
    assert stmt == SelectStatement(
        columns=(SelectColumn(ColumnRef("venue")),),
        table="papers",
        group_by=(ColumnRef("venue"),),
        having=HavingClause(
            CompareExpr(FunctionExpr("COUNT", ColumnRef("id")), ">", Literal(10)),
        ),
        limit=LimitClause(limit=Literal(5)),
        joins=(
            JoinClause(
                "venues",
                CompareExpr(ColumnRef("p.venue_id"), "=", ColumnRef("v.id")),
            ),
        ),
    )


def test_select_statement_new_fields_default() -> None:
    stmt = SelectStatement(columns=(), table="papers")
    assert stmt.having is None
    assert stmt.limit is None
    assert stmt.joins == ()


def test_update_statement_shape() -> None:
    stmt = UpdateStatement(
        table="papers",
        assignments=(("anio", Literal(2021)), ("titulo", Literal("New"))),
        where=CompareExpr(ColumnRef("id"), "=", Literal(1)),
    )
    assert stmt.table == "papers"
    assert stmt.assignments[0] == ("anio", Literal(2021))
    assert stmt.assignments[1] == ("titulo", Literal("New"))
    assert stmt.where is not None
    assert stmt == UpdateStatement(
        table="papers",
        assignments=(("anio", Literal(2021)), ("titulo", Literal("New"))),
        where=CompareExpr(ColumnRef("id"), "=", Literal(1)),
    )


def test_explain_statement_shape() -> None:
    inner = SelectStatement(
        columns=(SelectColumn(ColumnRef("titulo")),),
        table="papers",
    )
    stmt = ExplainStatement(sql="SELECT titulo FROM papers", statement=inner)
    assert stmt.sql == "SELECT titulo FROM papers"
    assert stmt.statement == inner
    assert stmt == ExplainStatement(
        sql="SELECT titulo FROM papers",
        statement=inner,
    )


def test_explain_statement_without_parsed() -> None:
    stmt = ExplainStatement(sql="SELECT 1")
    assert stmt.statement is None
