"""Tests for the transaction manager lifecycle."""

import pytest

from engine.common.errors import LockNotGranted, TransactionError
from engine.transactions.base import LockMode, TransactionState
from engine.transactions.transaction_manager import TransactionManager


def test_begin_creates_active_transaction() -> None:
    manager = TransactionManager()
    tx = manager.begin()
    assert tx.state is TransactionState.ACTIVE
    assert tx.tx_id >= 1
    manager.close()


def test_commit_releases_locks_and_marks_committed() -> None:
    manager = TransactionManager()
    tx = manager.begin()
    manager.lock_manager.acquire(tx.tx_id, ("record", "r1"), LockMode.EXCLUSIVE)
    tx.commit()
    assert tx.state is TransactionState.COMMITTED
    assert not manager.lock_manager.is_locked(("record", "r1"))
    manager.close()


def test_rollback_releases_locks_and_marks_aborted() -> None:
    manager = TransactionManager()
    tx = manager.begin()
    manager.lock_manager.acquire(tx.tx_id, ("record", "r1"), LockMode.EXCLUSIVE)
    tx.rollback()
    assert tx.state is TransactionState.ABORTED
    assert not manager.lock_manager.is_locked(("record", "r1"))
    manager.close()


def test_commit_on_inactive_raises() -> None:
    manager = TransactionManager()
    tx = manager.begin()
    tx.commit()
    with pytest.raises(TransactionError):
        tx.commit()
    manager.close()


def test_rollback_after_commit_raises() -> None:
    manager = TransactionManager()
    tx = manager.begin()
    tx.commit()
    with pytest.raises(TransactionError):
        tx.rollback()
    manager.close()


def test_commit_twice_raises() -> None:
    manager = TransactionManager()
    tx = manager.begin()
    tx.commit()
    with pytest.raises(TransactionError):
        tx.commit()
    manager.close()


def test_locks_isolated_between_transactions() -> None:
    manager = TransactionManager()
    t1 = manager.begin()
    t2 = manager.begin()
    manager.lock_manager.acquire(t1.tx_id, ("record", "r1"), LockMode.EXCLUSIVE)
    with pytest.raises(LockNotGranted):
        manager.lock_manager.acquire(t2.tx_id, ("record", "r1"), LockMode.EXCLUSIVE, timeout_ms=20)
    t1.commit()
    manager.lock_manager.acquire(t2.tx_id, ("record", "r1"), LockMode.EXCLUSIVE)
    t2.commit()
    manager.close()


def test_active_transactions_are_tracked() -> None:
    manager = TransactionManager()
    tx = manager.begin()
    assert manager.active_transactions() == (tx,)
    tx.commit()
    assert manager.active_transactions() == ()
    manager.close()
