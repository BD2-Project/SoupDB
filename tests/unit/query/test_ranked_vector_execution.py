"""C12-C14: IVF/HNSW SQL execution and capability-safe behavior."""

import json
from dataclasses import replace
from math import inf, nan

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_query_vectorizer import FakeQueryVectorizer
from tests.fakes.fake_ranked_search import ScriptedRankedIndex


@pytest.fixture
def image_case():
    catalog = FakeCatalog()
    catalog.create_table(
        "imagenes",
        (ColumnDef("id", ColumnType.INT), ColumnDef("archivo", ColumnType.TEXT)),
    )
    for row in ((1, "uno.jpg"), (2, "dos.jpg"), (3, "tres.jpg")):
        catalog.insert("imagenes", row)
    rids = [rid for rid, _ in catalog.file_org("imagenes").scan()]
    vectorizer = FakeQueryVectorizer()
    vectorizer.register("consulta.jpg", "image-v1", (0.1, 0.2))
    catalog.query_vectorizer = vectorizer
    index = ScriptedRankedIndex("hnsw")
    index.set_results(
        (0.1, 0.2),
        [(rids[1], 0.9), (rids[0], 0.8), (rids[2], 0.7)],
        metric="cosine",
    )
    binding = RankedBinding(
        "idx_imagenes",
        "imagenes",
        "archivo",
        "HNSW",
        index,
        metric="cosine",
        dimension=2,
        profile_id="image-v1",
        item_count=3,
    )
    catalog.add_ranked_index(binding)
    return catalog, vectorizer, index, binding


@pytest.fixture
def audio_case():
    catalog = FakeCatalog()
    catalog.create_table(
        "canciones",
        (ColumnDef("nombre", ColumnType.TEXT), ColumnDef("artista", ColumnType.TEXT)),
    )
    catalog.insert("canciones", ("primera", "ana"))
    catalog.insert("canciones", ("segunda", "beto"))
    rids = [rid for rid, _ in catalog.file_org("canciones").scan()]
    vectorizer = FakeQueryVectorizer()
    vectorizer.register("consulta.mp3", "audio-v1", [2, 4])
    catalog.query_vectorizer = vectorizer
    index = ScriptedRankedIndex("ivf")
    index.set_results((2.0, 4.0), [(rids[1], 0.95), (rids[0], 0.3)], metric="euclidean")
    binding = RankedBinding(
        "idx_audio",
        "canciones",
        "nombre",
        "IVF",
        index,
        metric="euclidean",
        dimension=2,
        profile_id="audio-v1",
        item_count=2,
    )
    catalog.add_ranked_index(binding)
    return catalog, vectorizer, index, binding


def run(catalog, sql):
    return execute(plan(parse(sql), catalog), catalog)


def image_sql(suffix=""):
    return (
        "SELECT * FROM imagenes WHERE SIMILAR_TO('consulta.jpg', k=2) "
        "USING HNSW WITH METRIC=cosine" + suffix
    )


def test_image_top_k_returns_provider_order(image_case):
    cat, vectorizer, index, _ = image_case
    result = run(cat, image_sql())
    assert result.rows == ((2, "dos.jpg"), (1, "uno.jpg"))
    assert tuple(col.name for col in result.columns) == ("id", "archivo")
    assert index.calls == [((0.1, 0.2), 2, {"metric": "cosine"})]
    assert vectorizer.calls == [("consulta.jpg", "image-v1")]
    assert index.closed is False


def test_audio_score_is_original_provider_score(audio_case):
    cat, vectorizer, index, _ = audio_case
    sql = (
        "SELECT nombre, artista, SIMILARITY_SCORE() AS score FROM canciones "
        "WHERE SIMILAR_TO('consulta.mp3', k=2) "
        "USING IVF WITH METRIC=euclidean ORDER BY score DESC"
    )
    result = run(cat, sql)
    assert result.rows == (("segunda", "beto", 0.95), ("primera", "ana", 0.3))
    assert result.columns[-1].type_name is ColumnType.FLOAT
    assert index.calls == [((2.0, 4.0), 2, {"metric": "euclidean"})]
    assert vectorizer.calls == [("consulta.mp3", "audio-v1")]


def test_mixed_star_score_only_exposes_requested_score(image_case):
    cat, _, _, _ = image_case
    sql = image_sql().replace("SELECT *", "SELECT *, SIMILARITY_SCORE() AS score")
    assert run(cat, sql).rows == ((2, "dos.jpg", 0.9), (1, "uno.jpg", 0.8))


def test_explain_does_not_vectorize_or_search(image_case):
    cat, vectorizer, index, _ = image_case
    result = run(cat, "EXPLAIN (FORMAT JSON) " + image_sql())
    plan_json = json.loads(result.rows[0][0])

    def nodes(node):
        yield node
        for child in node["children"]:
            yield from nodes(child)

    leaf = next(item for item in nodes(plan_json) if item["op"] == "HNSWSearch")
    assert leaf["detail"]["metric"] == "cosine"
    assert leaf["detail"]["profile_id"] == "image-v1"
    assert leaf["detail"]["dimension"] == 2
    assert leaf["detail"]["requested_k"] == 2
    assert leaf["detail"]["options"] == {"metric": "cosine"}
    assert vectorizer.calls == []
    assert index.calls == []


def test_explain_analyze_executes_once(image_case):
    cat, vectorizer, index, _ = image_case
    result = run(cat, "EXPLAIN ANALYZE (FORMAT JSON) " + image_sql())
    assert "HNSWSearch" in result.rows[0][0]
    assert len(vectorizer.calls) == 1
    assert len(index.calls) == 1


def test_vector_k_zero_does_not_call_provider(image_case):
    cat, vectorizer, index, _ = image_case
    result = run(cat, image_sql().replace("k=2", "k=0"))
    assert result.rows == ()
    assert vectorizer.calls == []
    assert index.calls == []


def test_limit_does_not_change_defined_k(image_case):
    cat, _, index, _ = image_case
    result = run(cat, image_sql().replace("k=2", "k=3") + " LIMIT 1 OFFSET 1")
    assert result.rows == ((1, "uno.jpg"),)
    assert index.calls[0][1] == 3


def test_order_by_regular_column_is_over_selected_k(image_case):
    cat, _, index, _ = image_case
    result = run(cat, image_sql() + " ORDER BY id ASC")
    assert result.rows == ((1, "uno.jpg"), (2, "dos.jpg"))
    assert index.calls[0][1] == 2


def test_no_index_fails_before_vectorization(image_case):
    cat, vectorizer, index, binding = image_case
    cat.drop_index(binding.name)
    with pytest.raises(QueryExecutionError, match="no HNSW index"):
        run(cat, image_sql())
    assert vectorizer.calls == []
    assert index.calls == []


def test_ambiguous_indexes_fail_closed(image_case):
    cat, vectorizer, index, binding = image_case
    cat.add_ranked_index(replace(binding, name="idx_imagenes_2"))
    with pytest.raises(QueryExecutionError, match="ambiguous vector indexes"):
        run(cat, image_sql())
    assert vectorizer.calls == []
    assert index.calls == []


def test_metric_incompatible_fails_closed(image_case):
    cat, vectorizer, _, _ = image_case
    with pytest.raises(QueryExecutionError, match="no HNSW index"):
        run(cat, image_sql().replace("METRIC=cosine", "METRIC=euclidean"))
    assert vectorizer.calls == []


def test_missing_vectorizer_fails_before_index(image_case):
    cat, _, index, _ = image_case
    del cat.query_vectorizer
    with pytest.raises(QueryExecutionError, match="vectorizer"):
        run(cat, image_sql())
    assert index.calls == []


@pytest.mark.parametrize(
    "vector",
    [[], [0.1], [0.1, 0.2, 0.3], [True, 0.2], [nan, 0.2], [inf, 0.2], "0.1,0.2"],
)
def test_invalid_vectors_never_reach_index(image_case, vector):
    cat, vectorizer, index, _ = image_case
    vectorizer.register("consulta.jpg", "image-v1", vector)
    with pytest.raises(QueryExecutionError, match="vectorize"):
        run(cat, image_sql())
    assert index.calls == []


def test_unexpected_vectorizer_failure_is_wrapped(image_case):
    cat, vectorizer, index, _ = image_case
    vectorizer.vectors.clear()
    with pytest.raises(QueryExecutionError, match="failed to vectorize"):
        run(cat, image_sql())
    assert index.calls == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM imagenes WHERE id = 1 AND SIMILAR_TO('consulta.jpg', k=2) "
        "USING HNSW WITH METRIC=cosine",
        "SELECT * FROM imagenes WHERE SIMILAR_TO('consulta.jpg', k=2) OR id = 1 "
        "USING HNSW WITH METRIC=cosine",
        "SELECT * FROM imagenes WHERE NOT SIMILAR_TO('consulta.jpg', k=2) "
        "USING HNSW WITH METRIC=cosine",
        "SELECT DISTINCT * FROM imagenes WHERE SIMILAR_TO('consulta.jpg', k=2) "
        "USING HNSW WITH METRIC=cosine",
        "SELECT COUNT(*) FROM imagenes WHERE SIMILAR_TO('consulta.jpg', k=2) "
        "USING HNSW WITH METRIC=cosine",
        "SELECT * FROM imagenes WHERE SIMILAR_TO('consulta.jpg', k=2) "
        "USING HNSW WITH METRIC=cosine GROUP BY id",
        "SELECT * FROM imagenes WHERE SIMILAR_TO('consulta.jpg', k=2) "
        "USING HNSW WITH METRIC=cosine HAVING id = 1",
    ],
)
def test_unsupported_filtering_and_grouping_fail_closed(image_case, sql):
    cat, vectorizer, index, _ = image_case
    with pytest.raises(QueryExecutionError):
        run(cat, sql)
    assert vectorizer.calls == []
    assert index.calls == []


def test_nested_similarity_score_rejected(image_case):
    cat, vectorizer, index, _ = image_case
    with pytest.raises(QueryExecutionError):
        run(cat, image_sql().replace("SELECT *", "SELECT SIMILARITY_SCORE() + 1"))
    assert vectorizer.calls == []
    assert index.calls == []


def test_vector_score_cannot_be_used_for_text_query(image_case):
    cat, _, index, _ = image_case
    with pytest.raises(QueryExecutionError):
        run(cat, image_sql().replace("SELECT *", "SELECT SCORE()"))
    assert index.calls == []
