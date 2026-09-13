"""Tests for the SQL parser: SELECT * FROM table."""

import pytest

from engine.query.ast import ColumnRef, SelectStatement
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_parse_select_star_from_table() -> None:
    stmt = parse("SELECT * FROM papers")
    assert isinstance(stmt, SelectStatement)
    assert stmt.table == "papers"
    assert stmt.columns == ()
    assert stmt.where is None


def test_parse_select_star_is_case_insensitive() -> None:
    stmt = parse("select * from papers")
    assert isinstance(stmt, SelectStatement)
    assert stmt.table == "papers"


def test_parse_single_column_select() -> None:
    stmt = parse("SELECT * FROM papers")
    assert isinstance(stmt, SelectStatement)


def test_parse_rejects_missing_from() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * papers")


def test_parse_rejects_unknown_keyword() -> None:
    with pytest.raises(QueryParseError):
        parse("UPSERT * FROM papers")


def test_parse_rejects_incomplete_statement() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM")


def test_parse_rejects_trailing_garbage() -> None:
    with pytest.raises(QueryParseError):
        parse("SELECT * FROM papers extra")


def test_parse_select_stores_column_refs() -> None:
    stmt = parse("SELECT titulo FROM papers")
    assert isinstance(stmt, SelectStatement)
    assert stmt.columns[0].expr == ColumnRef("titulo")
