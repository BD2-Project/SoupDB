"""C11: multimedia SQL grammar is contextual and side-effect free."""

import pytest

from engine.query.ast import Literal, ScoreExpr, SimilarToExpr
from engine.query.errors import QueryParseError
from engine.query.format_expr import format_expr
from engine.query.parser import parse

IMAGE = (
    "SELECT * FROM imagenes WHERE SIMILAR_TO('foto_consulta.jpg', k=10) "
    "USING HNSW WITH METRIC=cosine"
)
AUDIO = (
    "SELECT nombre, artista, SIMILARITY_SCORE() AS score FROM canciones "
    "WHERE SIMILAR_TO('audio_query.mp3', k=5) "
    "USING IVF WITH METRIC=euclidean ORDER BY score DESC"
)


def test_parse_image_official_example():
    stmt = parse(IMAGE)
    assert stmt.columns == ()
    assert stmt.search.strategy == "HNSW"
    assert stmt.search.metric == "cosine"
    assert stmt.where == SimilarToExpr(Literal("foto_consulta.jpg"), Literal(10))


def test_parse_audio_official_example():
    stmt = parse(AUDIO)
    assert stmt.columns[2].expr == ScoreExpr(kind="vector")
    assert stmt.columns[2].alias == "score"
    assert stmt.search.strategy == "IVF"
    assert stmt.search.metric == "euclidean"
    assert stmt.where.k.value == 5
    assert stmt.order_by[0].ascending is False


def test_select_star_mixed_with_similarity_score():
    stmt = parse(
        "SELECT *, SIMILARITY_SCORE() AS relevance FROM imagenes "
        "WHERE SIMILAR_TO('x', k=2) USING HNSW WITH METRIC=cosine"
    )
    assert len(stmt.columns) == 2
    assert stmt.columns[-1].expr.kind == "vector"


def test_format_similar_to_uses_sql_syntax():
    assert format_expr(SimilarToExpr(Literal("O'Hara.jpg"), Literal(2))) == (
        "SIMILAR_TO('O''Hara.jpg', k=2)"
    )


def test_zero_k_is_explicit():
    assert parse(IMAGE.replace("k=10", "k=0")).where.k.value == 0


@pytest.mark.parametrize("metric", ["COSINE", "Euclidean", "INNER_PRODUCT"])
def test_case_insensitive_metric(metric):
    statement = parse(IMAGE.replace("cosine", metric))
    assert statement.search.metric == metric.lower()


@pytest.mark.parametrize(
    "sql",
    [
        IMAGE.replace("k=10", "10"),
        IMAGE.replace("k=10", "n=10"),
        IMAGE.replace("k=10", "k=-1"),
        IMAGE.replace("k=10", "k=1.5"),
        IMAGE.replace("k=10", "k=TRUE"),
        IMAGE.replace("k=10", "k='1'"),
        IMAGE.replace("k=10", "k=2, x=1"),
        IMAGE.replace("k=10", "k=1, k=2"),
        IMAGE.replace("'foto_consulta.jpg'", "123"),
        IMAGE.replace("'foto_consulta.jpg'", "id"),
        IMAGE.replace(" WITH METRIC=cosine", ""),
        IMAGE.replace("cosine", "haversine"),
        IMAGE.replace("METRIC=cosine", "METRIC cosine"),
        IMAGE.replace("METRIC=cosine", "FOO=cosine"),
        IMAGE.replace("USING HNSW", "USING BTREE"),
        IMAGE.replace("USING HNSW", "USING BM25"),
        IMAGE.replace("SIMILAR_TO('foto_consulta.jpg', k=10)", "SIMILAR_TO()"),
        IMAGE.replace("SIMILAR_TO('foto_consulta.jpg', k=10)", "SIMILAR_TO('x')"),
        AUDIO.replace("SIMILARITY_SCORE()", "SIMILARITY_SCORE(1)"),
    ],
)
def test_reject_invalid_multimedia_syntax(sql):
    with pytest.raises(QueryParseError):
        parse(sql)


def test_ordinary_scalar_select_still_parses():
    stmt = parse("SELECT id FROM imagenes WHERE id = 4 LIMIT 2")
    assert stmt.search is None
    assert stmt.columns[0].expr.name == "id"
