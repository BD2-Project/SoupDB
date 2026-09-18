"""Tests for the transactional session."""

import pytest

from engine.common.errors import LockNotGranted, QueryParseError, TransactionError
from engine.common.schema import ColumnDef, ColumnType
from engine.transactions.base import LockMode
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


def test_begin_sets_current_transaction() -> None:
    session = TransactionalSession(make_catalog())
    tx = session.begin()
    assert session.current_transaction() is tx
    session.rollback()
    session.close()


def test_select_acquires_shared_lock() -> None:
    session = TransactionalSession(make_catalog())
    tx = session.begin()
    session.execute("SELECT * FROM accounts")
    locks = session.transaction_manager.lock_manager.locks_held(tx.tx_id)
    assert ("table", "accounts") in locks
    session.rollback()
    session.close()


def test_insert_acquires_exclusive_lock() -> None:
    session = TransactionalSession(make_catalog())
    tx = session.begin()
    session.execute("INSERT INTO accounts VALUES (3, 300)")
    locks = session.transaction_manager.lock_manager.locks_held(tx.tx_id)
    assert ("table", "accounts") in locks
    session.rollback()
    session.close()


def test_commit_releases_all_locks() -> None:
    session = TransactionalSession(make_catalog())
    tx = session.begin()
    session.execute("INSERT INTO accounts VALUES (3, 300)")
    session.commit()
    assert tx.state.value == "committed"
    assert session.transaction_manager.lock_manager.locks_held(tx.tx_id) == frozenset()
    session.close()


def test_autocommit_persists_single_statement() -> None:
    session = TransactionalSession(make_catalog())
    session.execute("INSERT INTO accounts VALUES (3, 300)")
    rows = session.execute("SELECT * FROM accounts")
    assert (3, 300) in rows.rows
    session.close()


def test_exclusive_write_blocks_concurrent_reader() -> None:
    catalog = make_catalog()
    manager = TransactionManager()
    s1 = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=30)
    s2 = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=30)
    t1 = s1.begin()
    s1.execute("INSERT INTO accounts VALUES (9, 999)")
    s2.begin()
    with pytest.raises(LockNotGranted):
        s2.execute("SELECT * FROM accounts")
    t1.rollback()
    s1.close()
    s2.close()


def test_rollback_undoes_insert() -> None:
    catalog = make_catalog()
    session = TransactionalSession(catalog)
    session.begin()
    session.execute("INSERT INTO accounts VALUES (3, 300)")
    session.rollback()
    rows = session.execute("SELECT * FROM accounts")
    assert (3, 300) not in rows.rows
    session.close()


def test_rollback_undoes_delete() -> None:
    catalog = make_catalog()
    session = TransactionalSession(catalog)
    session.begin()
    session.execute("DELETE FROM accounts WHERE id = 1")
    session.rollback()
    rows = session.execute("SELECT * FROM accounts")
    assert (1, 100) in rows.rows
    session.close()


def test_transaction_context_commits_on_success() -> None:
    session = TransactionalSession(make_catalog())
    with session.transaction():
        session.execute("INSERT INTO accounts VALUES (3, 300)")
    rows = session.execute("SELECT * FROM accounts")
    assert (3, 300) in rows.rows
    session.close()


def test_transaction_context_rolls_back_on_error() -> None:
    session = TransactionalSession(make_catalog())
    with pytest.raises(QueryParseError):
        with session.transaction():
            session.execute("INSERT INTO accounts VALUES (3, 300)")
            session.execute("BOGUS SQL")
    rows = session.execute("SELECT * FROM accounts")
    assert (3, 300) not in rows.rows
    session.close()


def test_requires_active_transaction_for_locking() -> None:
    session = TransactionalSession(make_catalog())
    with pytest.raises(TransactionError):
        session._lock(("table", "accounts"), LockMode.EXCLUSIVE)
    session.close()
