"""C07/C08: MATCH top-k, SCORE binding, alias resolution and fail-closed SQL."""

from dataclasses import replace

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_ranked_search import ScriptedRankedIndex

SCHEMA = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("contenido", ColumnType.TEXT),
)


@pytest.fixture
def data():
    catalog = FakeCatalog()
    catalog.create_table("documentos", SCHEMA)
    for row in ((1, "primero"), (2, "segundo"), (3, "tercero")):
        catalog.insert("documentos", row)
    rids = [rid for rid, _ in catalog.file_org("documentos").scan()]
    index = ScriptedRankedIndex("text")
    index.set_results("vector", [(rids[1], 0.9), (rids[0], 0.8), (rids[2], 0.1)], method="bm25")
    index.set_results("vector", [(rids[2], 2.5), (rids[0], 1.5)], method="tf_idf")
    binding = RankedBinding(
        "idx_texto",
        "documentos",
        "contenido",
        "INVERTED",
        index,
        item_count=3,
        exact=True,
        exhaustive=True,
    )
    catalog.add_ranked_index(binding)
    return catalog, index, binding


def run(sql, catalog):
    return execute(plan(parse(sql), catalog), catalog)


def test_simple_match_limit_keeps_provider_order(data):
    catalog, index, _ = data
    result = run(
        "SELECT * FROM documentos WHERE MATCH(contenido, 'vector') USING BM25 LIMIT 2",
        catalog,
    )
    assert result.rows == ((2, "segundo"), (1, "primero"))
    assert [col.name for col in result.columns] == ["id", "contenido"]
    assert index.calls == [("vector", 2, {"method": "bm25"})]


def test_tfidf_forwards_method(data):
    catalog, index, _ = data
    result = run(
        "SELECT id FROM documentos WHERE MATCH(contenido, 'vector') USING TF_IDF LIMIT 1",
        catalog,
    )
    assert result.rows == ((3,),)
    assert index.calls == [("vector", 1, {"method": "tf_idf"})]


def test_score_projection_and_desc_alias(data):
    catalog, index, _ = data
    result = run(
        "SELECT *, SCORE() AS relevancia FROM documentos "
        "WHERE MATCH(contenido, 'vector') USING BM25 "
        "ORDER BY relevancia DESC LIMIT 2",
        catalog,
    )
    assert [col.name for col in result.columns] == ["id", "contenido", "relevancia"]
    assert result.rows == ((2, "segundo", 0.9), (1, "primero", 0.8))
    assert index.calls == [("vector", 2, {"method": "bm25"})]


def test_score_direct_order(data):
    catalog, _, _ = data
    rows = run(
        "SELECT SCORE() AS score FROM documentos WHERE MATCH(contenido, 'vector') "
        "USING BM25 ORDER BY SCORE() DESC LIMIT 2",
        catalog,
    ).rows
    assert rows == ((0.9,), (0.8,))


def test_limit_offset_requests_required_candidates(data):
    catalog, index, _ = data
    result = run(
        "SELECT id FROM documentos WHERE MATCH(contenido, 'vector') USING BM25 LIMIT 1 OFFSET 1",
        catalog,
    )
    assert result.rows == ((1,),)
    assert index.calls == [("vector", 2, {"method": "bm25"})]


def test_limit_zero_never_calls_index(data):
    catalog, index, _ = data
    assert (
        run(
            "SELECT * FROM documentos WHERE MATCH(contenido, 'vector') USING BM25 LIMIT 0",
            catalog,
        ).rows
        == ()
    )
    assert index.calls == []


def test_no_index_rejected(data):
    catalog, _, binding = data
    catalog.drop_index(binding.name)
    with pytest.raises(QueryExecutionError, match="no INVERTED index"):
        run(
            "SELECT * FROM documentos WHERE MATCH(contenido, 'vector') USING BM25 LIMIT 2",
            catalog,
        )


def test_ambiguous_indexes_rejected(data):
    catalog, _, binding = data
    catalog.add_ranked_index(replace(binding, name="idx_texto_2"))
    with pytest.raises(QueryExecutionError, match="ambiguous"):
        run(
            "SELECT * FROM documentos WHERE MATCH(contenido, 'vector') USING BM25 LIMIT 2",
            catalog,
        )


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM documentos USING BM25 LIMIT 2",
        "SELECT SCORE() FROM documentos",
        "SELECT id FROM documentos WHERE MATCH(contenido, 'vector') LIMIT 2",
        "SELECT * FROM documentos WHERE NOT MATCH(contenido, 'vector') USING BM25 LIMIT 2",
    ],
)
def test_unsupported_search_is_explicit(data, sql):
    catalog, index, _ = data
    with pytest.raises(QueryExecutionError):
        run(sql, catalog)
    assert index.calls == []
