import json

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.query import execute_sql
from engine.transactions.session import TransactionalSession


def _seed_docs(catalog: Catalog, storage) -> None:
    execute_sql(f"CREATE EXTENSION inverted WITH (storage_root = '{storage.as_posix()}')", catalog)
    execute_sql("CREATE TABLE docs (id INT, contenido TEXT)", catalog)
    execute_sql("INSERT INTO docs VALUES (1, 'alpha beta')", catalog)
    execute_sql("INSERT INTO docs VALUES (2, 'gamma delta')", catalog)
    execute_sql("CREATE INDEX idx ON docs (contenido) USING INVERTED", catalog)


def _search(catalog: Catalog):
    return execute_sql(
        "SELECT id, SCORE() FROM docs WHERE SEARCH(contenido, 'alpha') USING INVERTED LIMIT 5",
        catalog,
    ).rows


def test_create_extension_activates_plugin_on_fresh_database(tmp_path):
    db_path, storage = tmp_path / "db", tmp_path / "plugins"
    catalog = Catalog(db_path)
    _seed_docs(catalog, storage)
    assert execute_sql("SELECT plugin_name FROM SysPlugins", catalog).rows == (("INVERTED",),)
    assert _search(catalog)[0][0] == 1
    catalog.close()

    # The engine reopens without an injected factory: configuration is persisted.
    reopened = Catalog(db_path)
    try:
        assert _search(reopened)[0][0] == 1
    finally:
        reopened.close()


def test_create_extension_is_idempotent(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        first = f"CREATE EXTENSION inverted WITH (storage_root = '{(tmp_path / 'a').as_posix()}')"
        second = f"CREATE EXTENSION inverted WITH (storage_root = '{(tmp_path / 'b').as_posix()}')"
        execute_sql(first, catalog)
        execute_sql(second, catalog)
        rows = execute_sql("SELECT plugin_name, enabled FROM SysPlugins", catalog).rows
        assert rows == (("INVERTED", True),)
        assert (tmp_path / "b").as_posix() in execute_sql(
            "SELECT config_json FROM SysPlugins", catalog
        ).rows[0][0]
    finally:
        catalog.close()


def test_create_extension_rejects_uninstalled_plugin(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        with pytest.raises(QueryExecutionError, match="not installed"):
            execute_sql("CREATE EXTENSION no_existe", catalog)
    finally:
        catalog.close()


def test_drop_extension_restricts_dependent_indexes(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        _seed_docs(catalog, tmp_path / "plugins")
        with pytest.raises(QueryExecutionError, match="indexes depend"):
            execute_sql("DROP EXTENSION inverted", catalog)
        execute_sql("DROP INDEX idx", catalog)
        execute_sql("DROP EXTENSION inverted", catalog)
        assert execute_sql("SELECT plugin_name FROM SysPlugins", catalog).rows == ()
        with pytest.raises(QueryExecutionError, match="not installed"):
            execute_sql("CREATE INDEX idx2 ON docs (contenido) USING INVERTED", catalog)
    finally:
        catalog.close()


def test_create_extension_through_transactional_session(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        session = TransactionalSession(catalog)
        session.execute(
            f"CREATE EXTENSION inverted WITH (storage_root = '{(tmp_path / 'p').as_posix()}')"
        )
        session.execute("CREATE TABLE docs (id INT, contenido TEXT)")
        session.execute("INSERT INTO docs VALUES (1, 'alpha')")
        session.execute("CREATE INDEX idx ON docs (contenido) USING INVERTED")
        assert _search(catalog)[0][0] == 1
    finally:
        catalog.close()


def test_create_extension_ivf_with_custom_storage_root(tmp_path):
    db_path = tmp_path / "db"
    catalog = Catalog(db_path)
    try:
        execute_sql(
            f"CREATE EXTENSION ivf WITH (storage_root = '{(tmp_path / 'ivf').as_posix()}')",
            catalog,
        )
        assert execute_sql("SELECT plugin_name FROM SysPlugins", catalog).rows == (("IVF",),)
    finally:
        catalog.close()

    # Reopen without factory: only IVF is configured; HNSW stays discovered-only.
    reopened = Catalog(db_path)
    try:
        assert execute_sql("SELECT plugin_name FROM SysPlugins", reopened).rows == (("IVF",),)
    finally:
        reopened.close()


def test_both_vector_plugins_with_same_config_share_the_vectorizer(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        execute_sql("CREATE EXTENSION ivf", catalog)
        execute_sql("CREATE EXTENSION hnsw", catalog)
        rows = execute_sql("SELECT plugin_name FROM SysPlugins", catalog).rows
        assert set(rows) == {("HNSW",), ("IVF",)}
    finally:
        catalog.close()


def test_incompatible_vector_configs_fail_cleanly_and_do_not_persist(tmp_path):
    catalog = Catalog(tmp_path / "db")
    try:
        execute_sql(
            f"CREATE EXTENSION ivf WITH (storage_root = '{(tmp_path / 'a').as_posix()}')",
            catalog,
        )
        with pytest.raises(QueryExecutionError, match="cannot activate plugin 'HNSW'"):
            execute_sql(
                f"CREATE EXTENSION hnsw WITH (storage_root = '{(tmp_path / 'b').as_posix()}')",
                catalog,
            )
        assert execute_sql("SELECT plugin_name FROM SysPlugins", catalog).rows == (("IVF",),)
    finally:
        catalog.close()


def test_profiles_json_is_persisted_and_reloaded(tmp_path):
    db_path = tmp_path / "db"
    profiles = json.dumps(
        {
            "img-v1": {
                "kind": "image",
                "root": "/datasets/imagenes",
                "codebook": "/codebooks/img.npz",
            }
        }
    )
    catalog = Catalog(db_path)
    try:
        execute_sql(
            "CREATE EXTENSION ivf WITH "
            f"(storage_root = '{(tmp_path / 'ivf').as_posix()}', profiles = '{profiles}')",
            catalog,
        )
        config_json = execute_sql("SELECT config_json FROM SysPlugins", catalog).rows[0][0]
        assert "img-v1" in config_json
    finally:
        catalog.close()

    reopened = Catalog(db_path)
    try:
        assert reopened.plugin_capabilities("IVF") is not None
        plugin = reopened._plugin_registry.plugin("IVF")
        assert "img-v1" in plugin._profiles()
    finally:
        reopened.close()
