"""Tests for the SQL parser: DROP TABLE and DROP INDEX statements."""

import pytest

from engine.query.ast import DropIndexStatement, DropTableStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_drop_table_builds() -> None:
    stmt = parse("DROP TABLE papers")
    assert isinstance(stmt, DropTableStatement)
    assert stmt.table == "papers"


def test_drop_table_case_insensitive() -> None:
    stmt = parse("drop table papers")
    assert isinstance(stmt, DropTableStatement)
    assert stmt.table == "papers"


def test_drop_index_builds() -> None:
    stmt = parse("DROP INDEX idx_anio")
    assert isinstance(stmt, DropIndexStatement)
    assert stmt.index_name == "idx_anio"


def test_drop_requires_object_type() -> None:
    with pytest.raises(QueryParseError):
        parse("DROP papers")
    with pytest.raises(QueryParseError):
        parse("DROP VIEW papers")


def test_drop_requires_name() -> None:
    with pytest.raises(QueryParseError):
        parse("DROP TABLE")


def test_drop_trailing_tokens_raise() -> None:
    with pytest.raises(QueryParseError):
        parse("DROP TABLE papers extra")
