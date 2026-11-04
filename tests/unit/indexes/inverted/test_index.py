import math

import pytest

from engine.common.rid import RID
from engine.indexes.inverted.index import InvertedIndex
from engine.indexes.inverted.tokenizer import Tokenizer

ITEMS = [("alpha alpha beta", RID(7, 0)), ("alpha gamma", RID(1, 3)), ("delta", RID(2, 0))]


@pytest.fixture
def index(tmp_path):
    index = InvertedIndex.build(tmp_path / "index", ITEMS, memory_limit_bytes=1,
                                tokenizer=Tokenizer(stemming=False), fan_in=2)
    yield index
    index.close()


def test_cosine_uses_entire_document_vector(index):
    i, j = math.log(1.5), math.log(3)
    expected = (2 * i * i + j * j) / (math.sqrt(4 * i * i + j * j) * math.hypot(i, j))
    hits = index.top_k("alpha beta", 10)
    assert [rid for rid, _ in hits] == [RID(7, 0), RID(1, 3), RID(2, 0)]
    assert hits[0][1] == pytest.approx(expected)
    assert hits[1][1] == pytest.approx(i * i / (i * i + j * j))
    assert hits[2][1] == 0.0
    assert index.top_k("alpha beta missing", 10) == hits


def test_bm25_matches_numeric_example_and_query_term_frequency(index):
    hits = index.top_k("alpha", 10, method="bm25")
    assert hits[0][0] == RID(7, 0)
    assert hits[0][1] == pytest.approx(math.log(1.6) * 4.4 / 3.65)
    assert hits[1][1] == pytest.approx(math.log(1.6))
    twice = index.top_k("alpha alpha", 10, method="bm25")
    assert [score for _, score in twice] == pytest.approx([2 * score for _, score in hits])


@pytest.mark.parametrize("query", ["", "the and el", "unseenword"])
def test_match_empty_stopwords_and_unknown_differ_from_full_contract(index, query):
    assert index.match_top_k(query, 10) == []
    assert index.top_k(query, 10) == [(RID(1, 3), 0.0), (RID(2, 0), 0.0), (RID(7, 0), 0.0)]


def test_zero_score_match_is_retained(tmp_path):
    index = InvertedIndex.build(tmp_path / "common", [("common", RID(3, 0)),
                                                     ("common", RID(1, 0))])
    try:
        assert index.match_top_k("common", 10) == [(RID(1, 0), 0.0), (RID(3, 0), 0.0)]
        assert index.match_top_k("common", 10, method="bm25")[0][1] > 0.0
    finally:
        index.close()


@pytest.mark.parametrize("method", ["tf_idf", "bm25"])
def test_reload_reconstructs_full_scores_and_match_membership(tmp_path, method):
    path = tmp_path / "index"
    index = InvertedIndex.build(path, ITEMS)
    before = index.top_k("alpha beta", 10, method=method)
    matching = index.match_top_k("alpha", 10, method=method)
    index.close()
    index = InvertedIndex.load(path)
    try:
        assert index.top_k("alpha beta", 10, method=method) == before
        assert index.match_top_k("alpha", 10, method=method) == matching
    finally:
        index.close()


@pytest.mark.parametrize("options", [{"bogus": 1}, {"method": "wrong"},
                                    {"matches_only": 1}, {"b": .2},
                                    {"method": "bm25", "k1": 0},
                                    {"method": "bm25", "b": math.nan}])
def test_invalid_search_options(index, options):
    with pytest.raises(ValueError):
        index.top_k("alpha", 3, **options)


@pytest.mark.parametrize("k", [-1, True, 1.2])
def test_invalid_k(index, k):
    with pytest.raises(ValueError):
        index.top_k("alpha", k)


def test_k_zero_query_type_empty_index_and_closed_state(tmp_path, index):
    assert index.top_k("alpha", 0) == []
    with pytest.raises(TypeError):
        index.top_k(4, 3)
    empty = InvertedIndex.build(tmp_path / "empty", [])
    assert empty.top_k("alpha", 3) == []
    empty.close()
    with pytest.raises(ValueError, match="closed"):
        empty.top_k("alpha", 1)
