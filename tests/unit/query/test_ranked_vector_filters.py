"""C14 guards: no filter-over-top-k semantic drift."""

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
def catalog():
    cat = FakeCatalog()
    cat.create_table(
        "imagenes", (ColumnDef("id", ColumnType.INT), ColumnDef("path", ColumnType.TEXT))
    )
    cat.insert("imagenes", (1, "x"))
    fake = ScriptedRankedIndex("hnsw")
    rid = next(iter(cat.file_org("imagenes").scan()))[0]
    fake.set_results((1.0, 0.0), [(rid, 0.95)], metric="cosine")
    cat.add_ranked_index(
        RankedBinding(
            "idx",
            "imagenes",
            "path",
            "HNSW",
            fake,
            metric="cosine",
            dimension=2,
            profile_id="p1",
            exact=False,
            exhaustive=False,
        )
    )
    vec = FakeQueryVectorizer()
    vec.register("q", "p1", (1.0, 0.0))
    cat.query_vectorizer = vec
    return cat, fake, vec


def run(cat, sql):
    return execute(plan(parse(sql), cat), cat)


@pytest.mark.parametrize(
    "predicate",
    [
        "id = 1 AND SIMILAR_TO('q', k=1)",
        "SIMILAR_TO('q', k=1) AND id = 1",
        "id = 1 OR SIMILAR_TO('q', k=1)",
        "NOT SIMILAR_TO('q', k=1)",
        "SIMILAR_TO('q', k=1) OR SIMILAR_TO('q', k=1)",
    ],
)
def test_filtered_ann_is_not_accepted_without_guarantees(catalog, predicate):
    cat, fake, vec = catalog
    with pytest.raises(QueryExecutionError):
        run(cat, f"SELECT * FROM imagenes WHERE {predicate} USING HNSW WITH METRIC=cosine")
    assert fake.calls == []
    assert vec.calls == []


def test_limit_cannot_increase_k(catalog):
    cat, fake, _ = catalog
    result = run(
        cat,
        "SELECT * FROM imagenes WHERE SIMILAR_TO('q', k=1) USING HNSW WITH METRIC=cosine LIMIT 20",
    )
    assert result.rows == ((1, "x"),)
    assert fake.calls[0][1] == 1
