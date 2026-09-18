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
  structure type (BTREE or HASH).
"""

from dataclasses import dataclass
from pathlib import Path

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.base import Index
from engine.indexes.bplus_tree import BPlusTree
from engine.indexes.extendible_hash import ExtendibleHash
from engine.storage.base import FileOrganization
from engine.storage.file_manager import FileManager
from engine.storage.heap_file import HeapFile
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

_SYS_TABLES = (
    ("SysTables", SYS_TABLES_SCHEMA),
    ("SysColumns", SYS_COLUMNS_SCHEMA),
    ("SysIndexes", SYS_INDEXES_SCHEMA),
)

_RESERVED_PREFIX = "Sys"
_SUPPORTED_ENGINES = ("HEAP", "SEQUENTIAL")
_SUPPORTED_INDEX_TYPES = ("BTREE", "HASH")


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
    ) -> None:
        self._files = FileManager(db_path, page_size=page_size, buffer_capacity=buffer_capacity)
        self._tables: dict[str, _Table] = {}
        self._seed_sys_metadata()
        self._load()

    def close(self) -> None:
        """Flush all backing files."""
        self._files.close()

    # --- Duck-typed catalog contract ------------------------------------

    def schema(self, name: str) -> tuple[ColumnDef, ...]:
        return self._table(name).schema

    def file_org(self, name: str) -> FileOrganization:
        return self._table(name).file

    def strategy(self, name: str) -> str:
        return self._table(name).strategy

    def indexes(self, name: str) -> dict[str, Index]:
        table = self._table(name)
        return {column: entry.index for column, entry in table.indexes.items()}

    def tables(self) -> list[str]:
        return list(self._tables)

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
        for table in self._tables.values():
            if any(entry.index_name == index_name for entry in table.indexes.values()):
                raise QueryExecutionError(f"index {index_name!r} already exists")
        table = self._table(table_name)
        if column not in {c.name for c in table.schema}:
            raise QueryExecutionError(f"unknown column {column!r} in table {table_name!r}")
        if index_type not in _SUPPORTED_INDEX_TYPES:
            raise QueryExecutionError(f"unsupported index type {index_type!r}; use BTREE or HASH")

        index = self._open_index(index_name, index_type)
        position = _column_position(table.schema, column)
        for _rid, record in table.file.scan():
            row = decode_row(record.data, table.schema)
            index.insert(row[position], _rid)
        table.indexes[column] = _IndexEntry(
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
            table.indexes[column] = _IndexEntry(
                index_name=index_name,
                column=column,
                index_type=index_type,
                index=self._open_index(index_name, index_type),
            )

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

    def _open_index(self, index_name: str, index_type: str) -> Index:
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

    def _read_sys_rows(self, name: str, schema: tuple[ColumnDef, ...]) -> list[tuple[object, ...]]:
        disk_manager, buffer_manager = self._files.storage(name)
        file = HeapFile(disk_manager, buffer_manager)
        return [decode_row(record.data, schema) for _rid, record in file.scan()]


def _check_columns(columns: tuple[ColumnDef, ...]) -> None:
    seen: set[str] = set()
    for column in columns:
        if column.name in seen:
            raise QueryExecutionError(f"duplicate column {column.name!r}")
        seen.add(column.name)


def _column_position(columns: tuple[ColumnDef, ...], name: str) -> int:
    for index, column in enumerate(columns):
        if column.name == name:
            return index
    raise QueryExecutionError(f"unknown column {name!r}")
