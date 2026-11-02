"""Parser-only tests for Part 3 ranked full-text SQL syntax.

Execution and index binding are intentionally tested in later commits.
"""

import pytest

from engine.query.ast import (
    ColumnRef,
    LimitClause,
    Literal,
    LogicalExpr,
    MatchExpr,
    OrderByItem,
    ScoreExpr,
    SearchOptions,
    SelectColumn,
    SelectStatement,
    StarExpr,
)
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_official_tfidf_example() -> None:
    statement = parse(
        "SELECT * FROM documentos "
        "WHERE MATCH(contenido, 'base de datos vectorial') "
        "USING TF_IDF LIMIT 10;"
    )

    assert isinstance(statement, SelectStatement)
    assert statement.table == "documentos"
    assert statement.columns == ()  # Plain SELECT * remains backward compatible.
    assert statement.where == MatchExpr(ColumnRef("contenido"), Literal("base de datos vectorial"))
    assert statement.search == SearchOptions(strategy="TF_IDF")
    assert statement.limit == LimitClause(Literal(10))


def test_official_bm25_example_with_score_alias() -> None:
    statement = parse(
        "SELECT *, SCORE() AS relevancia FROM articulos "
        "WHERE MATCH(texto, 'machine learning') "
        "USING BM25 ORDER BY relevancia DESC;"
    )

    assert statement.columns == (
        SelectColumn(StarExpr()),
        SelectColumn(ScoreExpr(kind="text"), alias="relevancia"),
    )
    assert statement.where == MatchExpr(ColumnRef("texto"), Literal("machine learning"))
    assert statement.search == SearchOptions(strategy="BM25")
    assert statement.order_by == (OrderByItem(ColumnRef("relevancia"), ascending=False),)
    assert statement.limit is None  # No implicit top-k is introduced by the parser.


def test_match_can_be_combined_with_another_boolean_predicate() -> None:
    statement = parse(
        "SELECT id FROM documentos WHERE MATCH(contenido, 'vector') AND publicado = TRUE USING BM25"
    )

    assert isinstance(statement.where, LogicalExpr)
    assert statement.where.op == "AND"
    assert statement.where.left == MatchExpr(ColumnRef("contenido"), Literal("vector"))
    # Planner will decide whether this combination can execute safely.


def test_text_search_preserves_unicode_and_escaped_quotes() -> None:
    statement = parse(
        "SELECT * FROM documentos WHERE match(contenido, 'O''Reilly y minería') USING bm25"
    )

    assert statement.where == MatchExpr(ColumnRef("contenido"), Literal("O'Reilly y minería"))
    assert statement.search == SearchOptions(strategy="BM25")


@pytest.mark.parametrize("strategy", ["TF_IDF", "tf_idf", "Tf_Idf", "BM25", "bm25"])
def test_using_strategy_is_case_insensitive(strategy: str) -> None:
    statement = parse(f"SELECT * FROM documentos WHERE MATCH(contenido, 'query') USING {strategy}")
    assert statement.search == SearchOptions(strategy=strategy.upper())


def test_limit_offset_is_preserved_for_text_search() -> None:
    statement = parse(
        "SELECT SCORE() AS relevancia FROM documentos "
        "WHERE MATCH(contenido, 'index') USING BM25 LIMIT 5 OFFSET 2"
    )

    assert statement.columns == (SelectColumn(ScoreExpr(), alias="relevancia"),)
    assert statement.limit == LimitClause(Literal(5), Literal(2))


def test_using_without_match_is_only_syntax_at_this_stage() -> None:
    statement = parse("SELECT id FROM documentos USING BM25")
    assert statement.search == SearchOptions(strategy="BM25")
    # The planner must reject this semantically; the parser has no catalog access.


def test_no_using_retains_select_defaults() -> None:
    statement = parse("SELECT * FROM documentos")
    assert statement.search is None
    assert statement.columns == ()


def test_search_words_remain_ordinary_column_identifiers() -> None:
    statement = parse("SELECT match, score, using, metric FROM documentos")
    assert statement.columns == tuple(
        SelectColumn(ColumnRef(name)) for name in ("match", "score", "using", "metric")
    )
    assert statement.search is None


def test_score_can_be_selected_without_star() -> None:
    statement = parse("SELECT id, SCORE() relevancia FROM documentos")
    assert statement.columns == (
        SelectColumn(ColumnRef("id")),
        SelectColumn(ScoreExpr(), alias="relevancia"),
    )


@pytest.mark.parametrize(
    "expression",
    [
        "MATCH('not_a_column', 'q')",
        "MATCH(contenido, id)",
        "MATCH(contenido, 42)",
        "MATCH(contenido, NULL)",
        "MATCH(contenido)",
        "MATCH(contenido, 'q', 'extra')",
        "MATCH()",
    ],
)
def test_match_rejects_invalid_arguments(expression: str) -> None:
    with pytest.raises(QueryParseError):
        parse(f"SELECT * FROM documentos WHERE {expression} USING BM25")


@pytest.mark.parametrize("expression", ["SCORE('x')", "SCORE(1)", "SCORE(*)"])
def test_score_requires_zero_arguments(expression: str) -> None:
    with pytest.raises(QueryParseError):
        parse(f"SELECT {expression} FROM documentos")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT *, id FROM documentos",
        "SELECT *, score FROM documentos",
        "SELECT *, SCORE(), id FROM documentos",
    ],
)
def test_mixed_star_rejects_unsupported_projections(sql: str) -> None:
    with pytest.raises(QueryParseError):
        parse(sql)


@pytest.mark.parametrize(
    "clause",
    ["USING IVF", "USING HNSW", "USING NO_EXISTE", "USING", "USING BM25 WITH METRIC=cosine"],
)
def test_text_parser_rejects_unsupported_using_clause(clause: str) -> None:
    with pytest.raises(QueryParseError):
        parse(f"SELECT * FROM documentos WHERE MATCH(contenido, 'q') {clause}")


def test_match_column_without_call_is_not_a_boolean_expression() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM documentos WHERE match")
