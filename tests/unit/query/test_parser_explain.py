"""Tests for the SQL parser: EXPLAIN statement.

EXPLAIN keeps the raw inner SQL text plus its parsed AST. The parser stores the
original SQL string, extracts ``sql`` by slicing from the offset of the first
inner token, and then parses the remaining tokens in place via ``parse_statement``
(no re-tokenization, so trailing-token bookkeeping stays correct).
"""

import pytest

from engine.query.ast import (
    DeleteStatement,
    ExplainStatement,
    InsertStatement,
    Literal,
    SelectStatement,
    UpdateStatement,
)
from engine.query.errors import QueryParseError
from engine.query.parser import parse


def test_explain_select() -> None:
    stmt = parse("EXPLAIN SELECT * FROM papers")
    assert isinstance(stmt, ExplainStatement)
    assert stmt.sql == "SELECT * FROM papers"
    assert isinstance(stmt.statement, SelectStatement)
    assert stmt.statement.table == "papers"
    assert stmt.statement.columns == ()


def test_explain_select_with_where() -> None:
    stmt = parse("EXPLAIN SELECT titulo FROM papers WHERE anio > 2010")
    assert isinstance(stmt, ExplainStatement)
    assert stmt.sql == "SELECT titulo FROM papers WHERE anio > 2010"
    inner = stmt.statement
    assert isinstance(inner, SelectStatement)
    assert inner.where is not None


def test_explain_insert() -> None:
    stmt = parse("EXPLAIN INSERT INTO papers (id) VALUES (1)")
    assert isinstance(stmt, ExplainStatement)
    assert stmt.sql == "INSERT INTO papers (id) VALUES (1)"
    assert isinstance(stmt.statement, InsertStatement)
    assert stmt.statement.values == ((Literal(1),),)


def test_explain_delete() -> None:
    stmt = parse("EXPLAIN DELETE FROM papers WHERE id = 1")
    assert isinstance(stmt, ExplainStatement)
    assert stmt.sql == "DELETE FROM papers WHERE id = 1"
    assert isinstance(stmt.statement, DeleteStatement)


def test_explain_update() -> None:
    stmt = parse("EXPLAIN UPDATE papers SET anio = 2021 WHERE id = 1")
    assert isinstance(stmt, ExplainStatement)
    assert stmt.sql == "UPDATE papers SET anio = 2021 WHERE id = 1"
    assert isinstance(stmt.statement, UpdateStatement)
    assert stmt.statement.assignments == (("anio", Literal(2021)),)


def test_explain_case_insensitive() -> None:
    stmt = parse("explain select * from papers")
    assert isinstance(stmt, ExplainStatement)
    assert stmt.sql == "select * from papers"
    assert isinstance(stmt.statement, SelectStatement)


def test_explain_with_trailing_semicolon() -> None:
    stmt = parse("EXPLAIN SELECT * FROM papers;")
    assert isinstance(stmt, ExplainStatement)
    assert isinstance(stmt.statement, SelectStatement)


def test_explain_unsupported_inner_statement_raises() -> None:
    with pytest.raises(QueryParseError):
        parse("EXPLAIN UPSERT INTO papers VALUES (1)")