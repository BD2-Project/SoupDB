"""Tests for the transaction control syntax: BEGIN / END [TRANSACTION]."""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query import execute_sql
from engine.query.ast import BeginTransactionStatement, EndTransactionStatement
from engine.query.errors import QueryParseError
from engine.query.executor import execute
from engine.query.lexer import tokenize
from engine.query.parser import parse
from engine.query.planner import plan
from engine.query.tokens import TokenKind
from tests.fakes.fake_catalog import FakeCatalog

PAPERS = (ColumnDef("id", ColumnType.INT), ColumnDef("anio", ColumnType.INT))


def make_catalog() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("papers", PAPERS)
    return catalog


def test_begin_transaction_parses() -> None:
    assert parse("BEGIN TRANSACTION") == BeginTransactionStatement()


def test_begin_short_form_parses() -> None:
    assert parse("BEGIN") == BeginTransactionStatement()


def test_end_transaction_parses() -> None:
    assert parse("END TRANSACTION") == EndTransactionStatement()


def test_end_short_form_parses() -> None:
    assert parse("END") == EndTransactionStatement()


def test_transaction_control_is_case_insensitive_and_tolerates_semicolon() -> None:
    assert parse("begin transaction;") == BeginTransactionStatement()
    assert parse("End Transaction;") == EndTransactionStatement()


def test_begin_and_end_are_different_nodes() -> None:
    assert not isinstance(parse("BEGIN"), EndTransactionStatement)
    assert not isinstance(parse("END"), BeginTransactionStatement)


def test_begin_transaction_tokens() -> None:
    tokens = [tok for tok in tokenize("BEGIN TRANSACTION") if tok.kind is not TokenKind.EOF]
    assert [(tok.kind, tok.value) for tok in tokens] == [
        (TokenKind.KEYWORD, "BEGIN"),
        (TokenKind.KEYWORD, "TRANSACTION"),
    ]


def test_end_transaction_tokens() -> None:
    tokens = [tok for tok in tokenize("END TRANSACTION") if tok.kind is not TokenKind.EOF]
    assert [(tok.kind, tok.value) for tok in tokens] == [
        (TokenKind.KEYWORD, "END"),
        (TokenKind.KEYWORD, "TRANSACTION"),
    ]


@pytest.mark.parametrize(
    "sql",
    [
        "BEGIN TRANSACTION TRANSACTION",
        "END TRANSACTION TRANSACTION",
        "BEGIN TRANSACTION SELECT * FROM papers",
        "END TRANSACTION SELECT * FROM papers",
    ],
)
def test_trailing_tokens_after_transaction_control_raise(sql: str) -> None:
    with pytest.raises(QueryParseError):
        parse(sql)


@pytest.mark.parametrize("sql", ["BEGIN TRANSACTION extra", "END TRANSACTION extra"])
def test_trailing_garbage_after_transaction_control_raises(sql: str) -> None:
    with pytest.raises(QueryParseError):
        parse(sql)


def test_plan_begin_transaction_has_no_operator_tree() -> None:
    catalog = make_catalog()
    result = plan(parse("BEGIN TRANSACTION"), catalog)
    assert isinstance(result.statement, BeginTransactionStatement)
    assert result.root is None
    assert result.output_schema is None


def test_plan_end_transaction_has_no_operator_tree() -> None:
    catalog = make_catalog()
    result = plan(parse("END TRANSACTION"), catalog)
    assert isinstance(result.statement, EndTransactionStatement)
    assert result.root is None
    assert result.output_schema is None


def test_plan_transaction_control_does_not_touch_catalog() -> None:
    class ExplodingCatalog:
        def __getattr__(self, name: str):
            raise AssertionError(f"the planner must not call {name}")

    assert plan(parse("BEGIN"), ExplodingCatalog()).root is None
    assert plan(parse("END"), ExplodingCatalog()).root is None


def test_execute_begin_transaction_raises_actionable_error() -> None:
    catalog = make_catalog()
    with pytest.raises(QueryExecutionError) as excinfo:
        execute(plan(parse("BEGIN TRANSACTION"), catalog), catalog)
    message = str(excinfo.value)
    assert "BEGIN TRANSACTION" in message
    assert "TransactionalSession" in message


def test_execute_end_transaction_raises_actionable_error() -> None:
    catalog = make_catalog()
    with pytest.raises(QueryExecutionError) as excinfo:
        execute(plan(parse("END TRANSACTION"), catalog), catalog)
    message = str(excinfo.value)
    assert "END TRANSACTION" in message
    assert "TransactionalSession" in message


def test_execute_sql_begin_transaction_without_session_raises() -> None:
    with pytest.raises(QueryExecutionError) as excinfo:
        execute_sql("BEGIN TRANSACTION", make_catalog())
    assert "TransactionalSession" in str(excinfo.value)


def test_execute_sql_end_transaction_without_session_raises() -> None:
    with pytest.raises(QueryExecutionError) as excinfo:
        execute_sql("END TRANSACTION", make_catalog())
    assert "TransactionalSession" in str(excinfo.value)


def test_execute_sql_transaction_control_has_no_silent_side_effect() -> None:
    catalog = make_catalog()
    for sql in ("BEGIN TRANSACTION", "END TRANSACTION"):
        with pytest.raises(QueryExecutionError):
            execute_sql(sql, catalog)
    assert catalog.schema("papers") == PAPERS
