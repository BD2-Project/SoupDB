"""In-memory catalog oracle for planner and executor tests.

Mimics the duck-typed interface the planner docs: ``schema``, ``file_org``,
``indexes`` and ``create_table`` plus extra test helpers (``insert``,
``add_index``).
"""

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.indexes.base import Index
from engine.query.evaluator import Schema
from tests.fakes.fake_index import FakeIndex
from tests.fakes.fake_storage import FakeFileOrganization


def _column_index(schema: Schema, name: str) -> int:
    for position, column in enumerate(schema):
        if column.name == name:
            return position
    raise QueryExecutionError(f"unknown column {name!r}")


class _Table:
    def __init__(self, schema: Schema, engine: str = "HEAP") -> None:
        self.schema = schema
        self.engine = engine
        self.file_org = FakeFileOrganization()
        self.indexes: dict[str, Index] = {}


class FakeCatalog:
    """In-memory catalog backing tables, file organizations and indexes."""

    def __init__(self) -> None:
        self._tables: dict[str, _Table] = {}

    def create_table(self, name: str, columns: Schema, engine: str = "HEAP") -> None:
        if name in self._tables:
            raise QueryExecutionError(f"table {name} already exists")
        self._tables[name] = _Table(columns, engine)

    def strategy(self, name: str) -> str:
        table = self._tables.get(name)
        if table is None:
            raise QueryExecutionError(f"unknown table {name!r}")
        return table.engine

    def schema(self, name: str) -> Schema:
        table = self._tables.get(name)
        if table is None:
            raise QueryExecutionError(f"unknown table {name!r}")
        return table.schema

    def file_org(self, name: str) -> FakeFileOrganization:
        table = self._tables.get(name)
        if table is None:
            raise QueryExecutionError(f"unknown table {name!r}")
        return table.file_org

    def indexes(self, name: str) -> dict[str, Index]:
        table = self._tables.get(name)
        if table is None:
            raise QueryExecutionError(f"unknown table {name!r}")
        return dict(table.indexes)

    def add_index(self, name: str, column: str, index: FakeIndex | None = None) -> FakeIndex:
        table = self._tables[name]
        index = index or FakeIndex()
        table.indexes[column] = index
        position = _column_index(table.schema, column)
        for rid, record in table.file_org.scan():
            row = decode_row(record.data, table.schema)
            index.insert(row[position], rid)
        return index

    def insert(self, name: str, row: tuple[object, ...]) -> None:
        schema = self.schema(name)
        table = self._tables[name]
        rid = table.file_org.insert(Record(data=encode_row(row, schema)))
        for position, column in enumerate(schema):
            index = table.indexes.get(column.name)
            if index is not None:
                index.insert(row[position], rid)
