"""Tests for the session interception of SQL transaction control."""

import pytest

from engine.common.errors import QueryExecutionError, QueryParseError, TransactionError
from engine.common.schema import ColumnDef, ColumnType
from engine.query import execute_sql
from engine.transactions.session import TransactionalSession
from engine.transactions.transaction_manager import TransactionManager
from tests.fakes.fake_catalog import FakeCatalog

ACCOUNTS = (ColumnDef("id", ColumnType.INT), ColumnDef("balance", ColumnType.INT))


def make_catalog() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("accounts", ACCOUNTS)
    catalog.insert("accounts", (1, 100))
    catalog.insert("accounts", (2, 200))
    return catalog


def rows_of(session: TransactionalSession) -> tuple[tuple[object, ...], ...]:
    return session.execute("SELECT * FROM accounts").rows


def test_sql_begin_transaction_opens_thread_transaction() -> None:
    session = TransactionalSession(make_catalog())
    try:
        assert session.current_transaction() is None
        result = session.execute("BEGIN TRANSACTION")
        assert result.columns == ()
        assert result.affected == 0
        assert session.current_transaction() is not None
    finally:
        session.close()


def test_sql_begin_insert_end_commits_and_persists() -> None:
    session = TransactionalSession(make_catalog())
    try:
        session.execute("BEGIN TRANSACTION")
        session.execute("INSERT INTO accounts VALUES (3, 300)")
        assert session.execute("END TRANSACTION").affected == 0
        assert (3, 300) in rows_of(session)
        assert session.current_transaction() is None
    finally:
        session.close()


def test_sql_group_keeps_statements_in_one_transaction() -> None:
    session = TransactionalSession(make_catalog())
    try:
        session.execute("BEGIN TRANSACTION")
        tx_id = session.current_transaction().tx_id
        session.execute("INSERT INTO accounts VALUES (3, 300)")
        session.execute("INSERT INTO accounts VALUES (4, 400)")
        session.execute("DELETE FROM accounts WHERE id = 1")
        assert session.current_transaction().tx_id == tx_id
        assert len(session.current_transaction().journal) == 3
        session.execute("END TRANSACTION")
        assert rows_of(session) == ((2, 200), (3, 300), (4, 400))
    finally:
        session.close()


def test_sql_group_is_invisible_to_other_sessions_until_end() -> None:
    catalog = make_catalog()
    manager = TransactionManager()
    writer = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=30)
    reader = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=30)
    try:
        writer.execute("BEGIN TRANSACTION")
        writer.execute("INSERT INTO accounts VALUES (3, 300)")
        with pytest.raises(TransactionError):
            reader.execute("SELECT * FROM accounts")
        writer.execute("END TRANSACTION")
        assert (3, 300) in reader.execute("SELECT * FROM accounts").rows
    finally:
        writer.close()
        reader.close()


def test_sql_short_begin_and_end_forms() -> None:
    session = TransactionalSession(make_catalog())
    try:
        session.execute("BEGIN")
        session.execute("INSERT INTO accounts VALUES (3, 300)")
        session.execute("END")
        assert (3, 300) in rows_of(session)
        assert session.current_transaction() is None
    finally:
        session.close()


def test_sql_lowercase_and_semicolon_forms() -> None:
    session = TransactionalSession(make_catalog())
    try:
        session.execute("begin transaction;")
        session.execute("insert into accounts values (3, 300)")
        session.execute("end transaction;")
        assert (3, 300) in rows_of(session)
        assert session.current_transaction() is None
    finally:
        session.close()


def test_api_rollback_after_sql_begin_restores_previous_state() -> None:
    session = TransactionalSession(make_catalog())
    try:
        before = rows_of(session)
        session.execute("BEGIN TRANSACTION")
        session.execute("INSERT INTO accounts VALUES (3, 300)")
        session.execute("DELETE FROM accounts WHERE id = 1")
        session.rollback()
        restored = set(rows_of(session))
        assert restored == set(before)
        assert (3, 300) not in restored
        assert (1, 100) in restored
        assert session.current_transaction() is None
    finally:
        session.close()


def test_sql_end_after_rollback_reports_no_active_transaction() -> None:
    session = TransactionalSession(make_catalog())
    try:
        session.execute("BEGIN TRANSACTION")
        session.execute("INSERT INTO accounts VALUES (3, 300)")
        session.rollback()
        with pytest.raises(TransactionError) as excinfo:
            session.execute("END TRANSACTION")
        assert "no active transaction" in str(excinfo.value)
    finally:
        session.close()


def test_nested_sql_begin_raises() -> None:
    session = TransactionalSession(make_catalog())
    try:
        session.execute("BEGIN TRANSACTION")
        with pytest.raises(TransactionError) as excinfo:
            session.execute("BEGIN TRANSACTION")
        assert "a transaction is already active on this thread" in str(excinfo.value)
        assert session.current_transaction() is not None
    finally:
        session.close()


def test_sql_end_without_active_transaction_raises() -> None:
    session = TransactionalSession(make_catalog())
    try:
        with pytest.raises(TransactionError) as excinfo:
            session.execute("END TRANSACTION")
        assert "no active transaction" in str(excinfo.value)
    finally:
        session.close()


def test_sql_begin_inside_api_transaction_raises() -> None:
    session = TransactionalSession(make_catalog())
    try:
        session.begin()
        with pytest.raises(TransactionError):
            session.execute("BEGIN")
        session.rollback()
    finally:
        session.close()


def test_malformed_transaction_control_does_not_open_a_transaction() -> None:
    session = TransactionalSession(make_catalog())
    try:
        with pytest.raises(QueryParseError):
            session.execute("BEGIN TRANSACTION extra")
        assert session.current_transaction() is None
    finally:
        session.close()


def test_execute_sql_begin_transaction_without_session_raises() -> None:
    catalog = make_catalog()
    with pytest.raises(QueryExecutionError) as excinfo:
        execute_sql("BEGIN TRANSACTION", catalog)
    assert "TransactionalSession" in str(excinfo.value)


def test_execute_sql_end_transaction_without_session_raises() -> None:
    catalog = make_catalog()
    with pytest.raises(QueryExecutionError) as excinfo:
        execute_sql("END TRANSACTION", catalog)
    assert "TransactionalSession" in str(excinfo.value)
