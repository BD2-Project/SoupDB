import pytest
from soupdb_inverted.plugin import InvertedPlugin

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.indexes.ranked_factory import DelegatingRankedFactory
from engine.plugins import PluginRegistry
from engine.query import execute_sql
from tests.fakes.fake_ranked_index import FakeRankedIndex, text_scorer


def _inverted_factory(tmp_path):
    registry = PluginRegistry()
    registry.register(InvertedPlugin({"storage_root": str(tmp_path / "plugins")}))
    return DelegatingRankedFactory.from_registry(registry)


def test_system_tables_are_readable_but_not_writable(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        rows = execute_sql("SELECT type_name FROM SysTypes", catalog).rows
        assert ("INT",) in rows and ("POINT",) in rows
        tables = execute_sql("SELECT table_name FROM SysTables", catalog).rows
        assert ("SysPluginIndexes",) in tables
        with pytest.raises(QueryExecutionError, match="read-only"):
            execute_sql("INSERT INTO SysTypes VALUES ('X', '', 0, 0)", catalog)
        with pytest.raises(QueryExecutionError, match="read-only"):
            execute_sql("DELETE FROM SysTables", catalog)
        with pytest.raises(QueryExecutionError, match="read-only"):
            execute_sql("CREATE INDEX bad ON SysTypes (type_name) USING INVERTED", catalog)
    finally:
        catalog.close()


def test_plugin_config_is_persisted_and_reloaded_without_factory(tmp_path):
    db_path = tmp_path / "db"
    catalog = Catalog(db_path, ranked_factory=_inverted_factory(tmp_path))
    execute_sql("CREATE TABLE docs (id INT, contenido TEXT)", catalog)
    execute_sql("INSERT INTO docs VALUES (1, 'alpha beta')", catalog)
    execute_sql("INSERT INTO docs VALUES (2, 'gamma delta')", catalog)
    execute_sql("CREATE INDEX idx_docs ON docs (contenido) USING INVERTED", catalog)

    plugins = execute_sql("SELECT plugin_name FROM SysPlugins", catalog).rows
    assert ("INVERTED",) in plugins
    stored = execute_sql("SELECT options_json FROM SysPluginIndexes", catalog).rows
    assert stored
    catalog.close()

    # No factory injected: the catalog discovers the installed plugin and
    # applies the configuration persisted in SysPlugins.
    reopened = Catalog(db_path)
    try:
        result = execute_sql(
            "SELECT id, SCORE() FROM docs WHERE MATCH(contenido, 'alpha') USING TF_IDF LIMIT 5",
            reopened,
        )
        assert result.rows[0][0] == 1
    finally:
        reopened.close()


def test_metadata_without_stored_plugin_config_stays_metadata_only(tmp_path):
    db_path = tmp_path / "db"
    catalog = Catalog(db_path)
    execute_sql("CREATE TABLE docs (id INT, texto TEXT)", catalog)
    catalog.register_ranked_index(
        RankedBinding(
            name="idx",
            table="docs",
            column="texto",
            algorithm="INVERTED",
            index=FakeRankedIndex(text_scorer),
            item_count=0,
            exact=True,
            exhaustive=True,
        )
    )
    catalog.close()

    reopened = Catalog(db_path)
    try:
        assert reopened.ranked_indexes_for("docs") == {}
        assert reopened.ranked_metadata_for("docs")["idx"].algorithm == "INVERTED"
    finally:
        reopened.close()
