"""Real SQL, heap RIDs, persisted SPIMI and MATCH candidate semantics."""

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.indexes.inverted.provider import OnDiskInvertedFactory
from engine.indexes.ranked_factory import DelegatingRankedFactory
from engine.query import execute_sql

pytestmark = pytest.mark.integration


@pytest.fixture
def database(tmp_path):
    physical = OnDiskInvertedFactory(tmp_path / "physical", memory_limit_bytes=128, fan_in=2)
    factory = DelegatingRankedFactory({"INVERTED": physical})
    path = tmp_path / "sql"
    catalog = Catalog(path, ranked_factory=factory)
    execute_sql("CREATE TABLE docs (id INT, contenido TEXT)", catalog)
    for sql in (
        "INSERT INTO docs VALUES (1, 'alpha alpha common')",
        "INSERT INTO docs VALUES (2, 'beta common')",
        "INSERT INTO docs VALUES (3, '')",
        "INSERT INTO docs VALUES (4, NULL)",
    ):
        execute_sql(sql, catalog)
    execute_sql("CREATE INDEX idx_text ON docs (contenido) USING INVERTED", catalog)
    yield path, factory, catalog
    catalog.close()


@pytest.mark.parametrize("method", ["TF_IDF", "BM25"])
def test_sql_match_filters_nonmatches_but_top_k_preserves_contract(database, method):
    _, _, catalog = database
    binding = catalog.ranked_indexes_for("docs")["idx_text"]
    assert binding.item_count == 3  # NULL omitted; empty string retained.
    assert binding.exact and binding.exhaustive
    hits = binding.index.top_k("alpha", 100, method=method.lower())
    assert len(hits) == 3
    result = execute_sql(
        f"SELECT id, SCORE() AS score FROM docs WHERE MATCH(contenido, 'alpha') "
        f"USING {method} LIMIT 100",
        catalog,
    )
    assert len(result.rows) == 1 and result.rows[0][0] == 1
    assert type(result.rows[0][1]) is float
    assert result.rows[0][1] > 0.0


@pytest.mark.parametrize("query", ["", "the and el", "unknownword"])
@pytest.mark.parametrize("method", ["TF_IDF", "BM25"])
def test_empty_stopword_and_unknown_match_return_no_rows(database, query, method):
    _, _, catalog = database
    for limit in ("", " LIMIT 2"):
        result = execute_sql(
            f"SELECT id FROM docs WHERE MATCH(contenido, '{query}') USING {method}{limit}",
            catalog,
        )
        assert result.rows == ()


def test_exhaustive_sql_filters_distinct_reorders_and_offsets(database):
    _, _, catalog = database
    query = "SELECT id FROM docs WHERE MATCH(contenido, 'common') USING BM25"
    assert execute_sql(query + " LIMIT 1 OFFSET 1", catalog).rows == ((1,),)
    assert execute_sql(query + " ORDER BY id ASC", catalog).rows == ((1,), (2,))
    assert execute_sql(
        "SELECT DISTINCT id FROM docs WHERE MATCH(contenido, 'common') AND id > 1 "
        "USING BM25 ORDER BY id ASC LIMIT 1",
        catalog,
    ).rows == ((2,),)


def test_tfidf_zero_score_still_matches(tmp_path):
    factory = OnDiskInvertedFactory(tmp_path / "physical")
    catalog = Catalog(tmp_path / "sql", ranked_factory=factory)
    try:
        execute_sql("CREATE TABLE docs (id INT, contenido TEXT)", catalog)
        execute_sql("INSERT INTO docs VALUES (1, 'common')", catalog)
        execute_sql("INSERT INTO docs VALUES (2, 'common')", catalog)
        execute_sql("CREATE INDEX idx ON docs (contenido) USING INVERTED", catalog)
        assert execute_sql(
            "SELECT id, SCORE() FROM docs WHERE MATCH(contenido, 'common') "
            "USING TF_IDF ORDER BY id ASC",
            catalog,
        ).rows == ((1, 0.0), (2, 0.0))
    finally:
        catalog.close()


def test_sql_reopen_both_methods_drop_and_rebuild(database):
    path, factory, catalog = database
    queries = [
        f"SELECT id, SCORE() FROM docs WHERE MATCH(contenido, 'common') "
        f"USING {method} ORDER BY SCORE() DESC"
        for method in ("TF_IDF", "BM25")
    ]
    before = [execute_sql(query, catalog).rows for query in queries]
    with pytest.raises(QueryExecutionError, match="blocked"):
        execute_sql("INSERT INTO docs VALUES (5, 'common')", catalog)
    catalog.close()
    reopened = Catalog(path, ranked_factory=factory)
    try:
        assert [execute_sql(query, reopened).rows for query in queries] == before
        execute_sql("DROP INDEX idx_text", reopened)
        assert reopened.ranked_indexes_for("docs") == {}
        execute_sql("CREATE INDEX idx_text ON docs (contenido) USING INVERTED", reopened)
        assert [execute_sql(query, reopened).rows for query in queries] == before
    finally:
        reopened.close()
