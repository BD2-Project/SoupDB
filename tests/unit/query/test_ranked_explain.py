"""C10 EXPLAIN JSON and ANALYZE metadata for ranked SQL (no timing snapshots)."""

import json

from engine.common.ranked import RankedBinding
from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_ranked_search import ScriptedRankedIndex


def _case():
    cat = FakeCatalog()
    cat.create_table(
        "documentos", (ColumnDef("id", ColumnType.INT), ColumnDef("texto", ColumnType.TEXT))
    )
    cat.insert("documentos", (1, "documento"))
    rid = next(iter(cat.file_org("documentos").scan()))[0]
    index = ScriptedRankedIndex("text")
    index.set_results("q", [(rid, 1.5)], method="bm25")
    cat.add_ranked_index(
        RankedBinding(
            "idx_documentos",
            "documentos",
            "texto",
            "INVERTED",
            index,
            item_count=1,
            exact=True,
            exhaustive=True,
        )
    )
    return cat, index


def _query(cat, sql):
    return execute(plan(parse(sql), cat), cat)


def _nodes(node):
    yield node
    for child in node["children"]:
        yield from _nodes(child)


def test_explain_json_does_not_call_provider():
    cat, index = _case()
    result = _query(
        cat,
        "EXPLAIN (FORMAT JSON) SELECT id, SCORE() AS relevancia FROM documentos "
        "WHERE MATCH(texto, 'q') USING BM25 LIMIT 1",
    )
    plan_json = json.loads(result.rows[0][0])
    assert set(plan_json) == {
        "op",
        "detail",
        "rows",
        "elapsed_ms",
        "disk_reads",
        "disk_writes",
        "children",
    }
    ranked = next(node for node in _nodes(plan_json) if node["op"] == "InvertedIndexScan")
    assert ranked["detail"]["index"] == "idx_documentos"
    assert ranked["detail"]["requested_k"] == 1
    assert ranked["detail"]["candidate_policy"] == "bounded_top_k"
    assert ranked["detail"]["options"] == {"method": "bm25"}
    assert ranked["rows"] == 0
    assert index.calls == []
    assert "ranked_score" not in result.rows[0][0]


def test_explain_analyze_reports_real_ranked_hits():
    cat, index = _case()
    result = _query(
        cat,
        "EXPLAIN ANALYZE (FORMAT JSON) SELECT *, SCORE() AS relevancia "
        "FROM documentos WHERE MATCH(texto, 'q') USING BM25 LIMIT 1",
    )
    plan_json = json.loads(result.rows[0][0])
    ranked = next(node for node in _nodes(plan_json) if node["op"] == "InvertedIndexScan")
    assert ranked["rows"] == 1
    assert ranked["detail"]["provider_hits"] == 1
    assert ranked["detail"]["rows_fetched"] == 1
    assert ranked["detail"]["exhaustive"] is True
    assert index.calls == [("q", 1, {"method": "bm25"})]


def test_explain_text_mentions_ranked_leaf():
    cat, index = _case()
    result = _query(
        cat, "EXPLAIN SELECT * FROM documentos WHERE MATCH(texto, 'q') USING BM25 LIMIT 1"
    )
    assert "InvertedIndexScan" in "\n".join(row[0] for row in result.rows)
    assert index.calls == []
