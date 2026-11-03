from types import SimpleNamespace

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked_catalog import RankedMetadata
from engine.indexes.ranked_factory import DelegatingRankedFactory


class Delegate:
    def __init__(self, resolver=None):
        self.query_vectorizer = resolver
        self.calls = []

    def build(self, *args):
        self.calls.append(("build", args))
        return "built"

    def load(self, *args):
        self.calls.append(("load", args))
        return "loaded"

    def drop(self, *args):
        self.calls.append(("drop", args))


def test_delegates_all_lifecycle_calls_to_algorithm_provider():
    text, vector = Delegate(), Delegate(object())
    delegates = {"INVERTED": text, "IVF": vector, "HNSW": vector}
    factory = DelegatingRankedFactory(delegates)
    delegates.clear()
    meta = RankedMetadata("ix", "table", "text", "INVERTED")
    file_org, schema = object(), ()
    assert factory.build(meta, file_org, schema) == "built"
    assert factory.load(meta, file_org, schema) == "loaded"
    factory.drop(meta)
    assert [call[0] for call in text.calls] == ["build", "load", "drop"]
    assert vector.calls == []
    assert factory.query_vectorizer is vector.query_vectorizer
    vector_meta = RankedMetadata("iv", "table", "text", "IVF")
    assert factory.load(vector_meta, file_org, schema) == "loaded"
    assert vector.calls == [("load", (vector_meta, file_org, schema))]


def test_missing_algorithm_fails_clearly_and_optional_drop_is_noop():
    factory = DelegatingRankedFactory({"INVERTED": SimpleNamespace()})
    meta = RankedMetadata("ix", "table", "text", "INVERTED")
    factory.drop(meta)
    missing = RankedMetadata("iv", "table", "text", "IVF")
    with pytest.raises(QueryExecutionError, match="no ranked provider"):
        factory.load(missing, None, ())
    with pytest.raises(ValueError, match="unsupported"):
        DelegatingRankedFactory({"UNKNOWN": Delegate()})


def test_distinct_vector_resolvers_require_explicit_shared_resolver():
    factories = {"IVF": Delegate(object()), "HNSW": Delegate(object())}
    with pytest.raises(ValueError, match="shared query_vectorizer"):
        DelegatingRankedFactory(factories)
    shared = object()
    assert DelegatingRankedFactory(factories, query_vectorizer=shared).query_vectorizer is shared
    assert DelegatingRankedFactory({"INVERTED": Delegate(object())}).query_vectorizer is None
