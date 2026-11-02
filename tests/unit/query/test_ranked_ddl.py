"""C18: ranked factory-backed DDL, reopen and safe write rejection."""

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_ranked_factory import FakeRankedFactory


def run(c, sql):
    return execute(plan(parse(sql), c), c)


def test_build_reopen_drop_and_write_guard(tmp_path):
    factory = FakeRankedFactory()
    c = Catalog(tmp_path, page_size=256, buffer_capacity=4, ranked_factory=factory)
    c.create_table("docs", (ColumnDef("id", ColumnType.INT), ColumnDef("texto", ColumnType.TEXT)))
    run(c, "CREATE INDEX idx_texto ON docs (texto) USING INVERTED")
    assert factory.built == ["idx_texto"]
    assert "idx_texto" in c.ranked_indexes_for("docs")
    with pytest.raises(QueryExecutionError, match="ranked DML is blocked"):
        run(c, "INSERT INTO docs VALUES (1, 'foo')")
    with pytest.raises(QueryExecutionError, match="ranked DML is blocked"):
        run(c, "DELETE FROM docs")
    assert list(c.file_org("docs").scan()) == []
    c.close()

    reopened = Catalog(tmp_path, page_size=256, buffer_capacity=4, ranked_factory=factory)
    assert factory.loaded == ["idx_texto"]
    assert "idx_texto" in reopened.ranked_indexes_for("docs")
    run(reopened, "DROP INDEX idx_texto")
    assert reopened.ranked_metadata_for("docs") == {}
    run(reopened, "INSERT INTO docs VALUES (1, 'foo')")
    reopened.close()


def test_without_factory_does_not_create_index(tmp_path):
    c = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    c.create_table("docs", (ColumnDef("texto", ColumnType.TEXT),))
    with pytest.raises(QueryExecutionError, match="factory"):
        run(c, "CREATE INDEX idx ON docs (texto) USING INVERTED")
    assert c.index_location("idx") is None
    c.close()


def test_vector_metadata_is_preserved(tmp_path):
    factory = FakeRankedFactory()
    c = Catalog(tmp_path, page_size=256, buffer_capacity=4, ranked_factory=factory)
    c.create_table("assets", (ColumnDef("ruta", ColumnType.TEXT),))
    run(
        c,
        "CREATE INDEX idx_vec ON assets (ruta) USING HNSW "
        "WITH METRIC=cosine, DIMENSION=8, PROFILE='image-v1'",
    )
    meta = c.ranked_metadata_for("assets")["idx_vec"]
    assert (meta.algorithm, meta.metric, meta.dimension, meta.profile_id) == (
        "HNSW",
        "cosine",
        8,
        "image-v1",
    )
    c.close()


def test_explicit_transaction_cannot_create_ranked_ddl(tmp_path):
    from engine.common.errors import TransactionError
    from engine.transactions.session import TransactionalSession

    factory = FakeRankedFactory()
    c = Catalog(tmp_path, page_size=256, buffer_capacity=4, ranked_factory=factory)
    c.create_table("docs", (ColumnDef("texto", ColumnType.TEXT),))
    session = TransactionalSession(c)
    try:
        session.begin()
        with pytest.raises(TransactionError, match="not reversible"):
            session.execute("CREATE INDEX ix ON docs (texto) USING INVERTED")
        assert c.index_location("ix") is None
    finally:
        if session.current_transaction() is not None:
            session.rollback()
        session.close()
        c.close()


def test_autocommit_session_creates_ranked_index(tmp_path):
    from engine.transactions.session import TransactionalSession

    factory = FakeRankedFactory()
    c = Catalog(tmp_path, page_size=256, buffer_capacity=4, ranked_factory=factory)
    c.create_table("docs", (ColumnDef("texto", ColumnType.TEXT),))
    session = TransactionalSession(c)
    try:
        session.execute("CREATE INDEX ix ON docs (texto) USING INVERTED")
        assert c.index_location("ix") == ("docs", "texto")
        session.execute("DROP INDEX ix")
        assert c.index_location("ix") is None
    finally:
        session.close()
        c.close()
