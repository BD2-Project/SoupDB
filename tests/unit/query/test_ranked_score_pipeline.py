"""C09 ranked SQL correctness against a scripted full-universe oracle."""

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

SCHEMA = (ColumnDef("id", ColumnType.INT), ColumnDef("texto", ColumnType.TEXT))
DATA = ((1, "uno"), (2, "dos"), (3, "tres"), (4, "dos"), (5, "cinco"))


@pytest.fixture
def context():
    cat = FakeCatalog()
    cat.create_table("articulos", SCHEMA)
    for row in DATA:
        cat.insert("articulos", row)
    rids = [rid for rid, _ in cat.file_org("articulos").scan()]
    index = ScriptedRankedIndex("text")
    index.set_results(
        "machine learning",
        [
            (rids[0], 0.99),
            (rids[1], 0.94),
            (rids[2], 0.85),
            (rids[3], 0.71),
            (rids[4], 0.65),
        ],
        method="bm25",
    )
    binding = RankedBinding(
        "idx_articulos",
        "articulos",
        "texto",
        "INVERTED",
        index,
        item_count=5,
        exact=True,
        exhaustive=True,
    )
    cat.add_ranked_index(binding)
    return cat, index, binding


def run(context, sql):
    return execute(plan(parse(sql), context[0]), context[0])


def test_official_bm25_without_limit_has_no_hidden_k(context):
    result = run(
        context,
        "SELECT *, SCORE() AS relevancia FROM articulos "
        "WHERE MATCH(texto, 'machine learning') USING BM25 "
        "ORDER BY relevancia DESC",
    )
    assert len(result.rows) == 5
    assert result.rows[0] == (1, "uno", 0.99)
    assert result.rows[-1] == (5, "cinco", 0.65)
    assert context[1].calls == [("machine learning", 5, {"method": "bm25"})]


def test_residual_filter_fills_limit_after_dropping_top_hits(context):
    result = run(
        context,
        "SELECT id FROM articulos WHERE MATCH(texto, 'machine learning') "
        "AND id >= 3 USING BM25 LIMIT 2",
    )
    assert result.rows == ((3,), (4,))
    assert context[1].calls == [("machine learning", 5, {"method": "bm25"})]


def test_reverse_conjunction_order_works(context):
    result = run(
        context,
        "SELECT id FROM articulos WHERE id > 3 AND MATCH(texto, 'machine learning') "
        "USING BM25 LIMIT 1",
    )
    assert result.rows == ((4,),)


def test_distinct_is_applied_before_limit(context):
    result = run(
        context,
        "SELECT DISTINCT texto FROM articulos WHERE MATCH(texto, 'machine learning') "
        "USING BM25 LIMIT 3",
    )
    assert result.rows == (("uno",), ("dos",), ("tres",))
    assert context[1].calls == [("machine learning", 5, {"method": "bm25"})]


def test_order_by_non_score_requires_full_universe(context):
    result = run(
        context,
        "SELECT id FROM articulos WHERE MATCH(texto, 'machine learning') "
        "USING BM25 ORDER BY id DESC LIMIT 2",
    )
    assert result.rows == ((5,), (4,))
    assert context[1].calls == [("machine learning", 5, {"method": "bm25"})]


def test_order_by_score_ascending_requires_full_universe(context):
    result = run(
        context,
        "SELECT id, SCORE() AS relevancia FROM articulos "
        "WHERE MATCH(texto, 'machine learning') USING BM25 "
        "ORDER BY relevancia ASC LIMIT 2",
    )
    assert result.rows == ((5, 0.65), (4, 0.71))
    assert context[1].calls == [("machine learning", 5, {"method": "bm25"})]


def test_limit_offset_after_residual_filter(context):
    result = run(
        context,
        "SELECT id FROM articulos WHERE MATCH(texto, 'machine learning') "
        "AND id >= 2 USING BM25 LIMIT 2 OFFSET 1",
    )
    assert result.rows == ((3,), (4,))
    assert context[1].calls == [("machine learning", 5, {"method": "bm25"})]


def test_no_limit_and_no_score_keeps_all_matches(context):
    result = run(
        context, "SELECT * FROM articulos WHERE MATCH(texto, 'machine learning') USING BM25"
    )
    assert result.rows == DATA


@pytest.mark.parametrize(
    "changes",
    [
        {"exact": False},
        {"exhaustive": False},
        {"item_count": None},
    ],
)
def test_missing_capability_fails_before_provider_call(context, changes):
    cat, index, binding = context
    cat.drop_index(binding.name)
    cat.add_ranked_index(replace(binding, **changes))
    with pytest.raises(QueryExecutionError, match="exact exhaustive"):
        run(context, "SELECT id FROM articulos WHERE MATCH(texto, 'machine learning') USING BM25")
    assert index.calls == []


def test_or_match_is_explicitly_unsupported(context):
    with pytest.raises(QueryExecutionError, match="exactly one MATCH"):
        run(
            context,
            "SELECT id FROM articulos WHERE MATCH(texto, 'machine learning') "
            "OR id = 3 USING BM25 LIMIT 2",
        )
    assert context[1].calls == []


def test_unknown_residual_column_rejected_before_provider_call(context):
    with pytest.raises(QueryExecutionError, match="unknown column"):
        run(
            context,
            "SELECT id FROM articulos WHERE MATCH(texto, 'machine learning') "
            "AND desconocida = 3 USING BM25 LIMIT 2",
        )
    assert context[1].calls == []


def test_existing_scalar_sql_is_unaffected(context):
    result = run(context, "SELECT id FROM articulos WHERE id >= 4")
    assert result.rows == ((4,), (5,))
    assert context[1].calls == []
