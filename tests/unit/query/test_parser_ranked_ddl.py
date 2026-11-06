"""C16: specialized index creation syntax and legacy compatibility."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.ast import CreateIndexStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog


@pytest.mark.parametrize("kind", ["IVF", "HNSW"])
def test_vector_ddl_metadata(kind):
    stmt = parse(
        f"CREATE INDEX idx ON assets (path) USING {kind} "
        "WITH METRIC=cosine, DIMENSION=128, PROFILE='image-v1'"
    )
    assert stmt == CreateIndexStatement(
        "idx",
        "assets",
        "path",
        kind,
        "cosine",
        128,
        "image-v1",
        (("metric", "cosine"), ("dimension", 128), ("profile", "image-v1")),
    )


def test_option_order_is_irrelevant():
    stmt = parse(
        "CREATE INDEX idx ON assets (path) USING IVF "
        "WITH PROFILE='audio-v1', DIMENSION=256, METRIC=euclidean"
    )
    assert (stmt.profile_id, stmt.dimension, stmt.metric) == ("audio-v1", 256, "euclidean")


def test_inverted_ddl():
    stmt = parse("CREATE INDEX idx ON docs (texto) USING INVERTED")
    assert stmt.index_type == "INVERTED"
    assert (stmt.metric, stmt.dimension, stmt.profile_id) == (None, None, None)


def test_legacy_ddl_is_unchanged():
    for sql, expected in [
        ("CREATE INDEX idx ON docs (texto)", "BTREE"),
        ("CREATE INDEX idx ON docs (texto) TYPE BTREE", "BTREE"),
        ("CREATE INDEX idx ON docs (texto) TYPE HASH", "HASH"),
        ("CREATE INDEX idx ON docs (texto) TYPE RTREE", "RTREE"),
    ]:
        stmt = parse(sql)
        assert stmt.index_type == expected
        assert (stmt.metric, stmt.dimension, stmt.profile_id) == (None, None, None)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE INDEX idx ON assets (path) USING HNSW",
        "CREATE INDEX idx ON assets (path) USING IVF WITH METRIC=cosine",
        "CREATE INDEX idx ON assets (path) USING IVF WITH METRIC=cosine, DIMENSION=3",
        (
            "CREATE INDEX idx ON assets (path) USING HNSW "
            "WITH METRIC=unknown, DIMENSION=3, PROFILE='x'"
        ),
        "CREATE INDEX idx ON assets (path) USING HNSW WITH METRIC=cosine, DIMENSION=0, PROFILE='x'",
        (
            "CREATE INDEX idx ON assets (path) USING HNSW "
            "WITH METRIC=cosine, DIMENSION=2.5, PROFILE='x'"
        ),
        "CREATE INDEX idx ON assets (path) USING HNSW WITH METRIC=cosine, DIMENSION=3, PROFILE=''",
        (
            "CREATE INDEX idx ON assets (path) USING HNSW "
            "WITH METRIC=cosine, METRIC=cosine, DIMENSION=3, PROFILE='x'"
        ),
        "CREATE INDEX idx ON docs (texto) USING INVERTED WITH METRIC=cosine",
        "CREATE INDEX idx ON docs (texto) TYPE HASH USING INVERTED",
    ],
)
def test_invalid_ranked_ddl_rejected(sql):
    with pytest.raises(QueryParseError):
        parse(sql)


def test_unknown_plugin_is_rejected_at_plan_time():
    catalog = FakeCatalog()
    catalog.create_table("docs", (ColumnDef("texto", ColumnType.TEXT),))
    stmt = parse("CREATE INDEX idx ON docs (texto) USING WHATEVER")
    with pytest.raises(QueryExecutionError, match="unsupported index type"):
        plan(stmt, catalog)


def test_ddl_has_explicit_guard_before_factories():
    catalog = FakeCatalog()
    catalog.create_table("docs", (ColumnDef("texto", ColumnType.TEXT),))
    stmt = parse("CREATE INDEX idx ON docs (texto) USING INVERTED")
    with pytest.raises(QueryExecutionError, match="factory"):
        plan(stmt, catalog)
