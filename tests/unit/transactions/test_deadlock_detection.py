"""Tests for deadlock detection (wait-for graph) and timeout rollback."""

import threading

import pytest

from engine.common.errors import DeadlockDetected, LockNotGranted, TransactionError
from engine.transactions.base import LockMode
from engine.transactions.session import TransactionalSession
from engine.transactions.strategies.strict_2pl import StrictTwoPhaseLocking
from engine.transactions.transaction_manager import TransactionManager
from tests.unit.transactions.test_session import make_catalog


def test_timeout_raises_lock_not_granted() -> None:
    strategy = StrictTwoPhaseLocking()
    strategy.acquire(1, ("record", "r1"), LockMode.EXCLUSIVE)
    with pytest.raises(LockNotGranted):
        strategy.acquire(2, ("record", "r1"), LockMode.EXCLUSIVE, timeout_ms=20)
    strategy.close()


def test_wait_for_cycle_is_detected() -> None:
    strategy = StrictTwoPhaseLocking()
    strategy.acquire(1, ("record", "a"), LockMode.EXCLUSIVE)
    strategy.acquire(2, ("record", "b"), LockMode.EXCLUSIVE)
    strategy._waiting[1] = ("record", "b")
    strategy._waiting[2] = ("record", "a")
    assert strategy._deadlock_detected(1)
    assert strategy._deadlock_detected(2)
    strategy.close()


def test_thread_deadlock_aborts_a_victim() -> None:
    strategy = StrictTwoPhaseLocking()
    strategy.acquire(1, ("record", "a"), LockMode.EXCLUSIVE)
    strategy.acquire(2, ("record", "b"), LockMode.EXCLUSIVE)
    errors: list[tuple[int, BaseException]] = []

    def tx1() -> None:
        try:
            strategy.acquire(1, ("record", "b"), LockMode.EXCLUSIVE, timeout_ms=500)
        except BaseException as exc:  # noqa: BLE001
            errors.append((1, exc))

    def tx2() -> None:
        try:
            strategy.acquire(2, ("record", "a"), LockMode.EXCLUSIVE, timeout_ms=500)
        except BaseException as exc:  # noqa: BLE001
            errors.append((2, exc))

    thread1 = threading.Thread(target=tx1)
    thread2 = threading.Thread(target=tx2)
    thread1.start()
    thread2.start()
    thread1.join(1)
    thread2.join(1)
    assert not thread1.is_alive() and not thread2.is_alive()
    assert any(isinstance(exc, DeadlockDetected) for _, exc in errors)
    strategy.close()


def test_session_rolls_back_waiting_transaction_on_timeout() -> None:
    catalog = make_catalog()
    manager = TransactionManager()
    s1 = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    s2 = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    t1 = s1.begin()
    s1.execute("INSERT INTO accounts VALUES (9, 999)")
    t2 = s2.begin()
    with pytest.raises(TransactionError):
        s2.execute("SELECT * FROM accounts")
    assert t2.state.value == "aborted"
    assert manager.lock_manager.locks_held(t2.tx_id) == frozenset()
    t1.rollback()
    s2.begin()
    rows = s2.execute("SELECT * FROM accounts")
    assert (9, 999) not in rows.rows
    s2.rollback()
    s1.close()
    s2.close()
