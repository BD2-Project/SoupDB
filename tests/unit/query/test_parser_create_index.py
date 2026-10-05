"""Tests for the SQL parser: CREATE INDEX statement."""

import pytest

from engine.query.ast import CreateIndexStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_create_index_btree() -> None:
    stmt = parse("CREATE INDEX idx_venue ON papers (venue) TYPE BTREE")
    assert isinstance(stmt, CreateIndexStatement)
    assert stmt.index_name == "idx_venue"
    assert stmt.table == "papers"
    assert stmt.column == "venue"
    assert stmt.index_type == "BTREE"


def test_create_index_hash() -> None:
    stmt = parse("CREATE INDEX idx_venue ON papers (venue) TYPE HASH")
    assert isinstance(stmt, CreateIndexStatement)
    assert stmt.index_type == "HASH"


def test_create_index_default_type() -> None:
    stmt = parse("CREATE INDEX idx_venue ON papers (venue)")
    assert isinstance(stmt, CreateIndexStatement)
    assert stmt.index_type == "BTREE"


def test_create_index_case_insensitive() -> None:
    stmt = parse("create index idx on papers (venue) type hash")
    assert isinstance(stmt, CreateIndexStatement)
    assert stmt.index_type == "HASH"


def test_create_index_requires_table() -> None:
    with pytest.raises(QueryParseError):
        parse("CREATE INDEX idx ON (venue)")


def test_create_index_missing_column_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("CREATE INDEX idx ON papers ()")
