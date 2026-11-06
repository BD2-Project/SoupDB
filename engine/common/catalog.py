"""Metadata for tables and indexes.

The :class:`Catalog` is the persistent, duck-typed contract consumed by the
planner and executor. It stores its own metadata in three system tables - each
a regular heap file managed through :class:`engine.storage.file_manager.FileManager`
- so the catalog bootstraps itself on first open and reloads on later opens:

- ``SysTables``: one row per table (including the system tables themselves)
  with its name, physical file id and storage strategy (HEAP or SEQUENTIAL).
- ``SysColumns``: one row per column: owner table, name, SQL type and the
  declared length (character count for VARCHAR, 0 otherwise). The byte size of
  a value is computed per record by the codec.
- ``SysIndexes``: one row per index: name, owner table, indexed column and
  structure type (BTREE, HASH or RTREE).

``RTREE`` is the spatial structure (:mod:`engine.indexes.rtree`): it indexes a
POINT column and only answers box (range) queries, so it is registered apart
from the scalar structures and the catalog validates the column type at
``create_index`` time.
"""

from collections.abc import Callable
from dataclasses import dataclass
from json import dumps as json_dumps
from json import loads as json_loads
from pathlib import Path

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding, validate_ranked_binding
from engine.common.ranked_catalog import (
    RANKED_METADATA_FILE,
    RANKED_SCHEMA,
    RankedMetadata,
    check_ranked_metadata,
)
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.base import Index
from engine.indexes.bplus_tree import BPlusTree
from engine.indexes.extendible_hash import ExtendibleHash
from engine.indexes.ranked_factory import DelegatingRankedFactory, RankedFactory
from engine.indexes.rtree import RTree
from engine.plugins.errors import PluginError, PluginNotFoundError
from engine.plugins.registry import PluginRegistry
from engine.storage.base import FileOrganization
from engine.storage.file_manager import FileManager
from engine.storage.heap_file import HeapFile
from engine.storage.io_counters import AggregateIOCounters
from engine.storage.sequential_file import SequentialFile

SYS_TABLES_SCHEMA = (
    ColumnDef("table_name", ColumnType.TEXT),
    ColumnDef("file_id", ColumnType.TEXT),
    ColumnDef("strategy", ColumnType.TEXT),
    ColumnDef("position", ColumnType.INT),
)

SYS_COLUMNS_SCHEMA = (
    ColumnDef("table_name", ColumnType.TEXT),
    ColumnDef("position", ColumnType.INT),
    ColumnDef("column_name", ColumnType.TEXT),
    ColumnDef("type_name", ColumnType.TEXT),
    ColumnDef("length", ColumnType.INT),
)

SYS_INDEXES_SCHEMA = (
    ColumnDef("index_name", ColumnType.TEXT),
    ColumnDef("table_name", ColumnType.TEXT),
    ColumnDef("column_name", ColumnType.TEXT),
    ColumnDef("index_type", ColumnType.TEXT),
)

SYS_PLUGINS_SCHEMA = (
    ColumnDef("plugin_name", ColumnType.TEXT),
    ColumnDef("version", ColumnType.TEXT),
    ColumnDef("api_version", ColumnType.INT),
    ColumnDef("enabled", ColumnType.BOOL),
    ColumnDef("config_json", ColumnType.TEXT),
)

SYS_TYPES_SCHEMA = (
    ColumnDef("type_name", ColumnType.TEXT),
    ColumnDef("plugin_name", ColumnType.TEXT),
    ColumnDef("encoding", ColumnType.INT),
    ColumnDef("size", ColumnType.INT),
)

SYS_PLUGIN_INDEXES_SCHEMA = (
    ColumnDef("index_name", ColumnType.TEXT),
    ColumnDef("table_name", ColumnType.TEXT),
    ColumnDef("column_name", ColumnType.TEXT),
    ColumnDef("index_type", ColumnType.TEXT),
    ColumnDef("plugin_name", ColumnType.TEXT),
    ColumnDef("options_json", ColumnType.TEXT),
    ColumnDef("capabilities_json", ColumnType.TEXT),
    ColumnDef("item_count", ColumnType.INT),
    ColumnDef("artifact_root", ColumnType.TEXT),
)

_SYS_TABLES = (
    ("SysTables", SYS_TABLES_SCHEMA),
    ("SysColumns", SYS_COLUMNS_SCHEMA),
    ("SysIndexes", SYS_INDEXES_SCHEMA),
    ("SysPlugins", SYS_PLUGINS_SCHEMA),
    ("SysTypes", SYS_TYPES_SCHEMA),
    ("SysPluginIndexes", SYS_PLUGIN_INDEXES_SCHEMA),
)

#: (type_name, plugin_name, encoding, size). Native types declare plugin_name="".
_NATIVE_TYPES = (
    ("INT", "", 0, 4),
    ("FLOAT", "", 0, 8),
    ("VARCHAR", "", 0, 0),
    ("TEXT", "", 0, 0),
    ("BOOL", "", 0, 1),
    ("POINT", "", 0, 16),
)

_RESERVED_PREFIX = "Sys"
_SUPPORTED_ENGINES = ("HEAP", "SEQUENTIAL")
#: Scalar structures: keys are values of a single column.
_SCALAR_INDEX_TYPES = ("BTREE", "HASH")
#: Spatial structure: the key is the POINT value of the indexed column.
_SPATIAL_INDEX_TYPE = "RTREE"
_SUPPORTED_INDEX_TYPES = (*_SCALAR_INDEX_TYPES, _SPATIAL_INDEX_TYPE)


@dataclass
class _IndexEntry:
    index_name: str
    column: str
    index_type: str
    index: Index


@dataclass
class _Table:
    name: str
    schema: tuple[ColumnDef, ...]
    file_id: str
    strategy: str
    file: FileOrganization
    indexes: dict[str, _IndexEntry]


def _length_or_none(length: int) -> int | None:
    return None if length == 0 else length


def _key_fn_for(columns: tuple[ColumnDef, ...]):
    """Order key for SEQUENTIAL files: value of the first column."""

    def key_fn(record: Record) -> object:
        return decode_row(record.data, columns)[0]

    return key_fn


class Catalog:
    """Persistent catalog backed by user-facing system tables."""

    def __init__(
        self,
        db_path: Path,
        page_size: int = 4096,
        buffer_capacity: int = 16,
        ranked_factory: RankedFactory | None = None,
    ) -> None:
        self._ranked_factory = ranked_factory
        self._files = FileManager(db_path, page_size=page_size, buffer_capacity=buffer_capacity)
        self._tables: dict[str, _Table] = {}
        self._ranked_meta: dict[str, RankedMetadata] = {}
        self._ranked_ready: dict[str, RankedBinding] = {}
        self._plugin_registry: PluginRegistry | None = None
        self._seed_sys_metadata()
        self._load()
        self._ensure_sys_tables()
        self._seed_types()
        self._load_plugins()
        self._migrate_ranked_metadata()
        for row in self._read_sys_rows("SysPluginIndexes", SYS_PLUGIN_INDEXES_SCHEMA):
            meta = _meta_from_row(row)
            check_ranked_metadata(meta, self.schema(meta.table))
            if meta.name in self._ranked_meta:
                raise QueryExecutionError(f"duplicate ranked metadata {meta.name!r}")
            self._ranked_meta[meta.name] = meta
            if self._ranked_factory is not None:
                binding = self._ranked_factory.load(
                    meta, self.file_org(meta.table), self.schema(meta.table)
                )
                validate_ranked_binding(binding, self.schema(meta.table))
                if binding.name != meta.name or binding.algorithm != meta.algorithm:
                    raise QueryExecutionError("ranked factory loaded inconsistent metadata")
                self._ranked_ready[meta.name] = binding
        self._io = AggregateIOCounters(lambda: self._files.disk_managers)

    @property
    def disk_manager(self) -> AggregateIOCounters:
        """Contadores de E/S sumados sobre todos los archivos abiertos.

        El planner los consulta para calcular, por operador, cuántas páginas se
        tocaron. Sin esto el plan reporta cero en todos los nodos, que es
        justamente la métrica que el curso evalúa.
        """
        return self._io

    def close(self) -> None:
        """Flush backing files and release loaded ranked index instances."""
        try:
            for binding in self._ranked_ready.values():
                binding.index.close()
        finally:
            self._files.close()

    # --- Duck-typed catalog contract ------------------------------------

    def schema(self, name: str) -> tuple[ColumnDef, ...]:
        return self._table(name).schema

    def file_org(self, name: str) -> FileOrganization:
        return self._table(name).file

    def strategy(self, name: str) -> str:
        return self._table(name).strategy

    def indexes(self, name: str) -> dict[str, Index]:
        """Physical indexes on a table keyed by index name."""
        table = self._table(name)
        return {index_name: entry.index for index_name, entry in table.indexes.items()}

    def indexes_for(self, name: str, column: str) -> dict[str, Index]:
        """Physical indexes on a table that cover ``column``, keyed by name."""
        table = self._table(name)
        return {
            index_name: entry.index
            for index_name, entry in table.indexes.items()
            if entry.column == column
        }

    @property
    def query_vectorizer(self):
        """Optional SQL multimedia resolver attached to the configured factory."""
        return getattr(self._ranked_factory, "query_vectorizer", None)

    @property
    def ranked_factory(self) -> RankedFactory | None:
        return self._ranked_factory

    def plugin_capabilities(self, name: str):
        """Declared capabilities of a registered plugin, or None if unknown."""
        registry = self._plugin_registry or getattr(self._ranked_factory, "registry", None)
        if registry is None:
            return None
        try:
            return registry.plugin(name).capabilities()
        except PluginNotFoundError:
            return None

    def validate_plugin_options(self, name: str, options: dict[str, object]) -> dict[str, object]:
        """Validate plugin-specific DDL options through the plugin contract."""
        registry = self._plugin_registry or getattr(self._ranked_factory, "registry", None)
        if registry is None:
            return dict(options)
        try:
            plugin = registry.plugin(name)
        except PluginNotFoundError as exc:
            raise QueryExecutionError(f"plugin {name!r} is not installed") from exc
        validator = getattr(plugin, "validate_options", None)
        return validator(options) if callable(validator) else dict(options)

    def create_ranked_index(self, metadata: RankedMetadata) -> None:
        """Build first, persist metadata after provider succeeds."""
        if self.is_system_table(metadata.table):
            raise QueryExecutionError(f"system table {metadata.table!r} is read-only")
        schema = self.schema(metadata.table)
        check_ranked_metadata(metadata, schema)
        if metadata.name in self._index_names():
            raise QueryExecutionError(f"index {metadata.name!r} already exists")
        if self._ranked_factory is None:
            raise QueryExecutionError("ranked CREATE INDEX requires a configured index factory")
        binding = self._ranked_factory.build(metadata, self.file_org(metadata.table), schema)
        try:
            if (
                binding.name,
                binding.table,
                binding.column,
                binding.algorithm,
                binding.metric,
                binding.dimension,
                binding.profile_id,
            ) != (
                metadata.name,
                metadata.table,
                metadata.column,
                metadata.algorithm,
                metadata.metric,
                metadata.dimension,
                metadata.profile_id,
            ):
                raise QueryExecutionError("ranked factory returned inconsistent index metadata")
            self.register_ranked_index(binding)
        except BaseException:
            binding.index.close()
            raise

    def ranked_metadata_for(
        self, name: str, column: str | None = None
    ) -> dict[str, RankedMetadata]:
        """All ranked registry entries, including providers unavailable after reopen."""
        self._table(name)
        return {
            key: meta
            for key, meta in self._ranked_meta.items()
            if meta.table == name and (column is None or meta.column == column)
        }

    def ranked_indexes_for(self, name: str, column: str | None = None) -> dict[str, RankedBinding]:
        """Only bindings with a valid live provider (not metadata-only entries)."""
        self._table(name)
        return {
            key: binding
            for key, binding in self._ranked_ready.items()
            if binding.table == name and (column is None or binding.column == column)
        }

    def register_ranked_index(self, binding: RankedBinding) -> None:
        """Publish and persist an already-built ranked index atomically at registry level."""
        schema = self.schema(binding.table)
        validate_ranked_binding(binding, schema)
        if binding.name in self._index_names():
            raise QueryExecutionError(f"index {binding.name!r} already exists")
        meta = RankedMetadata(
            binding.name,
            binding.table,
            binding.column,
            binding.algorithm,
            binding.metric,
            binding.dimension,
            binding.profile_id,
            binding.item_count,
            binding.exact,
            binding.exhaustive,
            binding.mutable,
        )
        check_ranked_metadata(meta, schema)
        self._upsert_sys_row(
            "SysPluginIndexes",
            SYS_PLUGIN_INDEXES_SCHEMA,
            _ranked_row(meta),
            key=lambda row: row[0] == meta.name,
        )
        self._persist_plugin_config(meta.algorithm)
        self._ranked_meta[meta.name] = meta
        self._ranked_ready[meta.name] = binding

    def tables(self) -> list[str]:
        return list(self._tables)

    def is_system_table(self, name: str) -> bool:
        """Whether ``name`` is a read-only system catalog table."""
        return name.startswith(_RESERVED_PREFIX)

    def types(self) -> tuple[tuple[object, ...], ...]:
        """Rows of ``SysTypes``: type name, plugin name, wire encoding and size."""
        return tuple(self._read_sys_rows("SysTypes", SYS_TYPES_SCHEMA))

    # --- DDL -------------------------------------------------------------

    def create_table(
        self,
        name: str,
        columns: tuple[ColumnDef, ...],
        engine: str = "HEAP",
    ) -> None:
        if name in self._tables:
            raise QueryExecutionError(f"table {name!r} already exists")
        if name.startswith(_RESERVED_PREFIX):
            raise QueryExecutionError(
                f"table name {name!r} is reserved for system tables ({_RESERVED_PREFIX}*)"
            )
        if engine not in _SUPPORTED_ENGINES:
            raise QueryExecutionError(
                f"unsupported engine {engine!r}; use one of {', '.join(_SUPPORTED_ENGINES)}"
            )
        _check_columns(columns)

        file_id = f"t_{name}"
        file = self._open_file(file_id, columns, engine)
        position = len(self._tables)
        self._tables[name] = _Table(
            name=name,
            schema=columns,
            file_id=file_id,
            strategy=engine,
            file=file,
            indexes={},
        )
        self._insert_sys_row("SysTables", SYS_TABLES_SCHEMA, (name, file_id, engine, position))
        for column_position, column in enumerate(columns):
            self._insert_sys_row(
                "SysColumns",
                SYS_COLUMNS_SCHEMA,
                (name, column_position, column.name, column.type_name.value, column.length or 0),
            )

    def create_index(
        self,
        index_name: str,
        table_name: str,
        column: str,
        index_type: str = "BTREE",
    ) -> None:
        if index_name in self._index_names():
            raise QueryExecutionError(f"index {index_name!r} already exists")
        if self.is_system_table(table_name):
            raise QueryExecutionError(f"system table {table_name!r} is read-only")
        table = self._table(table_name)
        if column not in {c.name for c in table.schema}:
            raise QueryExecutionError(f"unknown column {column!r} in table {table_name!r}")
        if index_type not in _SUPPORTED_INDEX_TYPES:
            raise QueryExecutionError(
                f"unsupported index type {index_type!r}; use {', '.join(_SUPPORTED_INDEX_TYPES)}"
            )
        _check_indexable_column(table, table_name, column, index_type)

        index = self._open_index(index_name, index_type)
        self._fill_index(table, index, column, index_type)
        table.indexes[index_name] = _IndexEntry(
            index_name=index_name,
            column=column,
            index_type=index_type,
            index=index,
        )
        self._insert_sys_row(
            "SysIndexes",
            SYS_INDEXES_SCHEMA,
            (index_name, table_name, column, index_type),
        )

    def index_location(self, index_name: str) -> tuple[str, str] | None:
        """Owning ``(table, column)`` of an index by name, or None if missing."""
        ranked = self._ranked_meta.get(index_name)
        if ranked is not None:
            return ranked.table, ranked.column
        for table in self._tables.values():
            entry = table.indexes.get(index_name)
            if entry is not None:
                return (table.name, entry.column)
        return None

    def drop_table(self, name: str) -> None:
        """Remove a table, its sys rows, its indexes and its backing file."""
        if name.startswith(_RESERVED_PREFIX):
            raise QueryExecutionError(
                f"table name {name!r} is reserved for system tables ({_RESERVED_PREFIX}*)"
            )
        table = self._table(name)
        for meta in list(self.ranked_metadata_for(name).values()):
            self._forget_ranked(meta.name)
        for entry in list(table.indexes.values()):
            self._files.drop(f"ix_{entry.index_name}")
            self._delete_sys_rows(
                "SysIndexes",
                SYS_INDEXES_SCHEMA,
                lambda row, index_name=entry.index_name: row[0] == index_name,
            )
        self._delete_sys_rows("SysTables", SYS_TABLES_SCHEMA, lambda row: row[0] == name)
        self._delete_sys_rows("SysColumns", SYS_COLUMNS_SCHEMA, lambda row: row[0] == name)
        del self._tables[name]
        self._files.drop(f"t_{name}")

    def _forget_ranked(self, index_name: str) -> None:
        meta = self._ranked_meta[index_name]
        binding = self._ranked_ready.pop(index_name, None)
        self._delete_sys_rows(
            "SysPluginIndexes",
            SYS_PLUGIN_INDEXES_SCHEMA,
            lambda row: row[0] == index_name,
        )
        self._ranked_meta.pop(index_name, None)
        if binding is not None:
            binding.index.close()
        drop = getattr(self._ranked_factory, "drop", None)
        if callable(drop):
            drop(meta)

    def drop_index(self, index_name: str) -> None:
        """Remove an index, its sys row and its backing file."""
        if index_name in self._ranked_meta:
            self._forget_ranked(index_name)
            return
        for table in self._tables.values():
            entry = table.indexes.get(index_name)
            if entry is None:
                continue
            del table.indexes[index_name]
            self._delete_sys_rows(
                "SysIndexes", SYS_INDEXES_SCHEMA, lambda row: row[0] == index_name
            )
            self._files.drop(f"ix_{index_name}")
            return
        raise QueryExecutionError(f"unknown index {index_name!r}")

    # --- Internal bootstrap and reload -----------------------------------

    def _table(self, name: str) -> _Table:
        table = self._tables.get(name)
        if table is None:
            raise QueryExecutionError(f"unknown table {name!r}")
        return table

    def _seed_sys_metadata(self) -> None:
        """Write the system-table rows the first time this database is opened."""
        tables_file = self._open_file("SysTables", SYS_TABLES_SCHEMA, "HEAP")
        if any(True for _ in tables_file.scan()):
            return
        for position, (sys_name, sys_schema) in enumerate(_SYS_TABLES):
            self._insert_sys_row(
                "SysTables",
                SYS_TABLES_SCHEMA,
                (sys_name, sys_name, "HEAP", position),
            )
            for column_position, column in enumerate(sys_schema):
                self._insert_sys_row(
                    "SysColumns",
                    SYS_COLUMNS_SCHEMA,
                    (
                        sys_name,
                        column_position,
                        column.name,
                        column.type_name.value,
                        column.length or 0,
                    ),
                )

    def _ensure_sys_tables(self) -> None:
        """Register system tables introduced after this database was created."""
        for sys_name, sys_schema in _SYS_TABLES:
            if sys_name in self._tables:
                continue
            self._insert_sys_row(
                "SysTables",
                SYS_TABLES_SCHEMA,
                (sys_name, sys_name, "HEAP", len(self._tables)),
            )
            for column_position, column in enumerate(sys_schema):
                self._insert_sys_row(
                    "SysColumns",
                    SYS_COLUMNS_SCHEMA,
                    (
                        sys_name,
                        column_position,
                        column.name,
                        column.type_name.value,
                        column.length or 0,
                    ),
                )
            self._tables[sys_name] = _Table(
                name=sys_name,
                schema=sys_schema,
                file_id=sys_name,
                strategy="HEAP",
                file=self._open_file(sys_name, sys_schema, "HEAP"),
                indexes={},
            )

    def _seed_types(self) -> None:
        """Register the native column types in ``SysTypes`` on first open."""
        if self._read_sys_rows("SysTypes", SYS_TYPES_SCHEMA):
            return
        for row in _NATIVE_TYPES:
            self._insert_sys_row("SysTypes", SYS_TYPES_SCHEMA, row)

    def _load_plugins(self) -> None:
        """Discover installed plugins and apply configurations stored in SysPlugins.

        When a factory is injected explicitly (tests, embeds), it wins. Otherwise
        the catalog rebuilds the factory from the installed entry points and the
        persisted configuration, so reopening a database no longer requires the
        caller to re-inject every plugin parameter by hand.
        """
        if self._ranked_factory is not None:
            self._plugin_registry = getattr(self._ranked_factory, "registry", None)
            return
        registry = PluginRegistry()
        try:
            registry.discover()
        except PluginError:
            self._plugin_registry = registry
            return
        stored = self._read_sys_rows("SysPlugins", SYS_PLUGINS_SCHEMA)
        if not stored:
            # No plugin was persisted for this database: keep metadata-only
            # entries instead of trying to reopen artifacts that were never
            # registered with a plugin configuration.
            self._plugin_registry = registry
            return
        for row in stored:
            plugin_name, _version, _api_version, enabled, config_json = row
            if not enabled or plugin_name not in registry:
                continue
            try:
                config = json_loads(config_json or "{}")
            except ValueError as exc:
                raise QueryExecutionError(
                    f"invalid stored configuration for plugin {plugin_name!r}: {exc}"
                ) from exc
            if not isinstance(config, dict):
                raise QueryExecutionError(
                    f"stored configuration for plugin {plugin_name!r} must be an object"
                )
            registry.register(registry.plugin(plugin_name), config, replace=True, source="catalog")
        self._plugin_registry = registry
        if registry.available():
            self._ranked_factory = DelegatingRankedFactory.from_registry(registry)

    def _migrate_ranked_metadata(self) -> None:
        """One-version migration from the legacy ``SysRankedIndexes`` layout."""
        if self._read_sys_rows("SysPluginIndexes", SYS_PLUGIN_INDEXES_SCHEMA):
            return
        for row in self._read_sys_rows(RANKED_METADATA_FILE, RANKED_SCHEMA):
            self._insert_sys_row(
                "SysPluginIndexes",
                SYS_PLUGIN_INDEXES_SCHEMA,
                _ranked_row(RankedMetadata.from_row(row)),
            )

    def _persist_plugin_config(self, algorithm: str) -> None:
        """Store the plugin configuration needed to reload this index later."""
        registry = getattr(self._ranked_factory, "registry", None)
        if registry is None:
            return
        try:
            registration = registry.get(algorithm)
        except PluginNotFoundError:
            return
        plugin = registration.plugin
        row = (
            str(plugin.name),
            str(getattr(plugin, "version", "0.0.0")),
            int(getattr(plugin, "api_version", 1)),
            True,
            json_dumps(dict(registration.config), sort_keys=True, default=str),
        )
        self._upsert_sys_row(
            "SysPlugins", SYS_PLUGINS_SCHEMA, row, key=lambda item: item[0] == row[0]
        )

    def _load(self) -> None:
        t_rows = self._read_sys_rows("SysTables", SYS_TABLES_SCHEMA)
        c_rows = self._read_sys_rows("SysColumns", SYS_COLUMNS_SCHEMA)
        i_rows = self._read_sys_rows("SysIndexes", SYS_INDEXES_SCHEMA)

        for table_name, file_id, strategy, _position in sorted(t_rows, key=lambda row: row[3]):
            columns = tuple(
                ColumnDef(
                    column_name,
                    ColumnType(type_name),
                    _length_or_none(length),
                )
                for _t_name, _col_pos, column_name, type_name, length in sorted(
                    (row for row in c_rows if row[0] == table_name),
                    key=lambda row: row[1],
                )
            )
            self._tables[table_name] = _Table(
                name=table_name,
                schema=columns,
                file_id=file_id,
                strategy=strategy,
                file=self._open_file(file_id, columns, strategy),
                indexes={},
            )

        for index_name, table_name, column, index_type in i_rows:
            table = self._tables.get(table_name)
            if table is None:
                continue
            table.indexes[index_name] = _IndexEntry(
                index_name=index_name,
                column=column,
                index_type=index_type,
                index=self._build_index(table, index_name, column, index_type),
            )

    def _index_names(self) -> set[str]:
        return set(self._ranked_meta) | {
            name for table in self._tables.values() for name in table.indexes
        }

    def _open_file(
        self,
        file_id: str,
        columns: tuple[ColumnDef, ...],
        strategy: str,
    ) -> FileOrganization:
        disk_manager, buffer_manager = self._files.storage(file_id)
        if strategy == "HEAP":
            return HeapFile(disk_manager, buffer_manager)
        if strategy == "SEQUENTIAL":
            return SequentialFile(disk_manager, buffer_manager, _key_fn_for(columns))
        raise QueryExecutionError(f"unsupported engine {strategy!r}")

    def _build_index(
        self,
        table: _Table,
        index_name: str,
        column: str,
        index_type: str,
    ) -> Index:
        """Open ``index_name`` with the entries of ``column`` already in ``table``.

        BTREE and HASH persist their own pages: the file they were filled into is
        all they need to come back. The spatial index has no page-backed form in
        its public API (``RTree`` only serializes to a filesystem path with
        ``save``/``load``/``open``, and the paginated variant lives in a private
        module), so the catalog treats it as a *derived* structure and rebuilds
        it from the table every time the database is opened. Rebuilding is what
        guarantees a reopened index answers exactly like the table it indexes,
        instead of coming back empty; the cost is one sequential pass per
        spatial index, paid once at open time.
        """
        index = self._open_index(index_name, index_type)
        if index_type == _SPATIAL_INDEX_TYPE:
            self._fill_index(table, index, column, index_type)
        return index

    def _fill_index(
        self,
        table: _Table,
        index: Index,
        column: str,
        index_type: str,
    ) -> None:
        """Insert every row of ``table`` into ``index``, keyed by ``column``."""
        position = _column_position(table.schema, column)
        for rid, record in table.file.scan():
            value = decode_row(record.data, table.schema)[position]
            if value is None and index_type == _SPATIAL_INDEX_TYPE:
                # A NULL coordinate is not a location: there is nothing to index,
                # and a row that cannot be located cannot answer a radius query.
                continue
            index.insert(value, rid)

    def _open_index(self, index_name: str, index_type: str) -> Index:
        if index_type == _SPATIAL_INDEX_TYPE:
            return RTree()
        disk_manager, buffer_manager = self._files.storage(f"ix_{index_name}")
        if index_type == "BTREE":
            return BPlusTree(disk_manager, buffer_manager)
        if index_type == "HASH":
            return ExtendibleHash(disk_manager, buffer_manager)
        raise QueryExecutionError(f"unsupported index type {index_type!r}")

    def _insert_sys_row(
        self, name: str, schema: tuple[ColumnDef, ...], row: tuple[object, ...]
    ) -> None:
        disk_manager, buffer_manager = self._files.storage(name)
        file = HeapFile(disk_manager, buffer_manager)
        file.insert(Record(data=encode_row(row, schema)))

    def _upsert_sys_row(
        self,
        name: str,
        schema: tuple[ColumnDef, ...],
        row: tuple[object, ...],
        *,
        key: Callable[[tuple[object, ...]], bool],
    ) -> None:
        """Replace the first matching system row, or append a new one."""
        self._delete_sys_rows(name, schema, key)
        self._insert_sys_row(name, schema, row)

    def _read_sys_rows(self, name: str, schema: tuple[ColumnDef, ...]) -> list[tuple[object, ...]]:
        disk_manager, buffer_manager = self._files.storage(name)
        file = HeapFile(disk_manager, buffer_manager)
        return [decode_row(record.data, schema) for _rid, record in file.scan()]

    def _delete_sys_rows(
        self,
        name: str,
        schema: tuple[ColumnDef, ...],
        predicate: Callable[[tuple[object, ...]], bool],
    ) -> None:
        disk_manager, buffer_manager = self._files.storage(name)
        file = HeapFile(disk_manager, buffer_manager)
        awaiting = [
            rid for rid, record in file.scan() if predicate(decode_row(record.data, schema))
        ]
        for rid in awaiting:
            file.remove(rid)


def _ranked_row(meta: RankedMetadata) -> tuple[object, ...]:
    """Map ``RankedMetadata`` to a ``SysPluginIndexes`` row."""
    options: dict[str, object] = {}
    if meta.metric is not None:
        options["metric"] = meta.metric
    if meta.dimension is not None:
        options["dimension"] = meta.dimension
    if meta.profile_id is not None:
        options["profile_id"] = meta.profile_id
    capabilities = {
        "exact": bool(meta.exact),
        "exhaustive": bool(meta.exhaustive),
        "mutable": bool(meta.mutable),
    }
    return (
        meta.name,
        meta.table,
        meta.column,
        meta.algorithm,
        meta.algorithm,
        json_dumps(options, sort_keys=True),
        json_dumps(capabilities, sort_keys=True),
        meta.item_count,
        "",
    )


def _meta_from_row(row: tuple[object, ...]) -> RankedMetadata:
    """Rebuild ``RankedMetadata`` from a ``SysPluginIndexes`` row."""
    (
        name,
        table,
        column,
        index_type,
        plugin_name,
        options_json,
        capabilities_json,
        item_count,
        _artifact_root,
    ) = row
    try:
        options = json_loads(options_json) if options_json else {}
        capabilities = json_loads(capabilities_json) if capabilities_json else {}
    except ValueError as exc:
        raise QueryExecutionError(f"invalid plugin index metadata JSON: {exc}") from exc
    if not isinstance(options, dict) or not isinstance(capabilities, dict):
        raise QueryExecutionError("plugin index metadata JSON must contain objects")
    return RankedMetadata(
        name,
        table,
        column,
        plugin_name or index_type,
        options.get("metric"),
        options.get("dimension"),
        options.get("profile_id"),
        item_count,
        bool(capabilities.get("exact", False)),
        bool(capabilities.get("exhaustive", False)),
        bool(capabilities.get("mutable", False)),
    )


def _check_columns(columns: tuple[ColumnDef, ...]) -> None:
    seen: set[str] = set()
    for column in columns:
        if column.name in seen:
            raise QueryExecutionError(f"duplicate column {column.name!r}")
        seen.add(column.name)


def _check_indexable_column(
    table: _Table,
    table_name: str,
    column: str,
    index_type: str,
) -> None:
    """Reject a spatial index on a column that cannot hold a location."""
    if index_type != _SPATIAL_INDEX_TYPE:
        return
    definition = next(item for item in table.schema if item.name == column)
    if definition.type_name is not ColumnType.POINT:
        raise QueryExecutionError(
            f"a spatial index requires a POINT column; column {column!r} of table "
            f"{table_name!r} is {definition.type_name.value}"
        )


def _column_position(columns: tuple[ColumnDef, ...], name: str) -> int:
    for index, column in enumerate(columns):
        if column.name == name:
            return index
    raise QueryExecutionError(f"unknown column {name!r}")
