"""C05: ranked SQL must fail closed until the ranked planner is integrated."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import (
    BoundScoreExpr,
    ColumnRef,
    CompareExpr,
    Literal,
    MatchExpr,
    OrderByItem,
    ScoreExpr,
    SearchOptions,
    SelectColumn,
    SelectStatement,
    StarExpr,
)
from engine.query.format_expr import format_expr
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM documentos WHERE MATCH(contenido, 'índice') USING TF_IDF LIMIT 10",
        "SELECT *, SCORE() AS relevancia FROM documentos "
        "WHERE MATCH(contenido, 'búsqueda') USING BM25 "
        "ORDER BY relevancia DESC",
        "SELECT id FROM documentos WHERE MATCH(contenido, 'consulta')",
        "SELECT id FROM documentos USING BM25",
        "SELECT SCORE() FROM documentos",
        "SELECT id FROM documentos ORDER BY SCORE() DESC",
        "SELECT id FROM documentos WHERE SCORE() > 0.5",
        "SELECT id FROM documentos WHERE NOT MATCH(contenido, 'texto')",
        "SELECT id FROM documentos WHERE id = 1 OR MATCH(contenido, 'texto')",
    ],
)
def test_ranked_sql_is_rejected_before_catalog_access(sql: str) -> None:
    """A ranked query cannot accidentally fall through to the scalar planner."""
    statement = parse(sql)

    with pytest.raises(QueryExecutionError, match="ranked SQL search requires the ranked planner"):
        plan(statement, object())


def test_explain_does_not_bypass_ranked_guard() -> None:
    statement = parse(
        "EXPLAIN SELECT id FROM documentos WHERE MATCH(contenido, 'texto') USING BM25"
    )

    with pytest.raises(QueryExecutionError, match="ranked SQL search requires the ranked planner"):
        plan(statement, object())


@pytest.mark.parametrize(
    "statement",
    [
        SelectStatement(
            columns=(SelectColumn(BoundScoreExpr(position=2, kind="text")),),
            table="documentos",
        ),
        SelectStatement(
            columns=(SelectColumn(ColumnRef("id")),),
            table="documentos",
            order_by=(OrderByItem(BoundScoreExpr(position=2, kind="text")),),
        ),
        SelectStatement(
            columns=(SelectColumn(StarExpr()), SelectColumn(ScoreExpr())),
            table="documentos",
        ),
        SelectStatement(
            columns=(SelectColumn(ColumnRef("id")),),
            table="documentos",
            where=CompareExpr(ScoreExpr(), ">", Literal(0.5)),
        ),
        SelectStatement(
            columns=(SelectColumn(ColumnRef("id")),),
            table="documentos",
            search=SearchOptions(strategy="BM25"),
        ),
    ],
)
def test_manual_ast_cannot_bypass_ranked_guard(statement: SelectStatement) -> None:
    with pytest.raises(QueryExecutionError, match="ranked SQL search requires the ranked planner"):
        plan(statement, object())


def test_normal_selects_still_use_existing_planner() -> None:
    catalog = FakeCatalog()
    catalog.create_table(
        "documentos",
        (ColumnDef("id", ColumnType.INT), ColumnDef("contenido", ColumnType.TEXT)),
    )
    for sql in (
        "SELECT * FROM documentos",
        "SELECT id FROM documentos WHERE id > 0",
        "SELECT id FROM documentos ORDER BY id DESC LIMIT 5",
    ):
        result = plan(parse(sql), catalog)
        assert result.root is not None
        assert result.output_schema is not None


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (StarExpr(), "*"),
        (ScoreExpr(), "SCORE()"),
        (BoundScoreExpr(2, "text"), "SCORE()"),
        (ScoreExpr(kind="vector"), "SIMILARITY_SCORE()"),
        (BoundScoreExpr(2, "vector"), "SIMILARITY_SCORE()"),
        (
            MatchExpr(ColumnRef("contenido"), Literal("índice invertido")),
            "MATCH(contenido, 'índice invertido')",
        ),
    ],
)
def test_format_ranked_expressions_without_private_slot(expression: object, expected: str) -> None:
    assert format_expr(expression) == expected
    assert "ranked_score" not in format_expr(expression)
