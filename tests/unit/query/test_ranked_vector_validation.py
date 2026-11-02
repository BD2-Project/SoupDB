"""C13 validation of vector-ranked SQL contracts."""

from math import inf, nan

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.schema import ColumnDef, ColumnType
from engine.query.vector_operators import HNSWSearch, IVFSearch
from tests.fakes.fake_query_vectorizer import FakeQueryVectorizer
from tests.fakes.fake_ranked_search import ScriptedRankedIndex
from tests.fakes.fake_storage import FakeFileOrganization

SCHEMA = (ColumnDef("reference", ColumnType.TEXT),)


def binding(algorithm="HNSW", metric="cosine"):
    return RankedBinding(
        "idx",
        "tabla",
        "reference",
        algorithm,
        ScriptedRankedIndex(algorithm.lower()),
        metric=metric,
        dimension=2,
        profile_id="profile-1",
    )


@pytest.mark.parametrize("vector", [(1, 2), [0.2, -0.5], (0.0, 0.0)])
def test_valid_query_vectors_are_converted_to_float(vector):
    search = binding()
    vectorizer = FakeQueryVectorizer()
    vectorizer.register("x", "profile-1", vector)
    search.index.set_results(tuple(float(v) for v in vector), [], metric="cosine")
    op = HNSWSearch(search, FakeFileOrganization(), "x", SCHEMA, 1, vectorizer)
    op.open()
    try:
        assert op.next() is None
        assert search.index.calls == [(tuple(float(v) for v in vector), 1, {"metric": "cosine"})]
    finally:
        op.close()


def test_no_secondary_score_normalization():
    from engine.common.record import Record, decode_row, encode_row

    store = FakeFileOrganization()
    rid = store.insert(Record(encode_row(("x",), SCHEMA)))
    search = binding()
    vectorizer = FakeQueryVectorizer()
    vectorizer.register("x", "profile-1", (0.0, 1.0))
    search.index.set_results((0.0, 1.0), [(rid, -0.8)], metric="cosine")
    op = HNSWSearch(search, store, "x", SCHEMA, 1, vectorizer)
    op.open()
    try:
        record = op.next()
        assert decode_row(record.data, op.schema) == ("x", -0.8)
    finally:
        op.close()


@pytest.mark.parametrize("score", [nan, inf, -inf, 1, True, None])
def test_invalid_provider_scores_are_rejected(score):
    from engine.common.record import Record, encode_row

    store = FakeFileOrganization()
    rid = store.insert(Record(encode_row(("x",), SCHEMA)))
    search = binding()
    vectorizer = FakeQueryVectorizer()
    vectorizer.register("x", "profile-1", (0.0, 1.0))
    search.index.set_results((0.0, 1.0), [(rid, score)], metric="cosine")
    op = HNSWSearch(search, store, "x", SCHEMA, 1, vectorizer)
    with pytest.raises(QueryExecutionError, match="finite FLOAT"):
        op.open()
    assert search.index.closed is False


def test_wrong_index_family_is_rejected():
    vectorizer = FakeQueryVectorizer()
    with pytest.raises(QueryExecutionError, match="IVF binding"):
        IVFSearch(binding("HNSW"), FakeFileOrganization(), "x", SCHEMA, 1, vectorizer)
