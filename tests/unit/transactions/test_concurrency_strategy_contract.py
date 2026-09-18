"""Conformance suite for every ConcurrencyStrategy implementation.

The suite parametrizes over all implementations; the pessimistic strict 2PL
strategy joins the params when it exists.
"""

import pytest

from engine.transactions.base import ConcurrencyStrategy, LockMode, TransactionState
from tests.fakes.fake_concurrency import FakeConcurrencyStrategy


@pytest.fixture(params=[FakeConcurrencyStrategy])
def strategy(request):
    impl = request.param()
    yield impl
    impl.close()


def test_lock_modes_have_values() -> None:
    assert LockMode.SHARED.value == "S"
    assert LockMode.UPDATE.value == "U"
    assert LockMode.EXCLUSIVE.value == "X"


def test_transaction_states() -> None:
    assert TransactionState.ACTIVE.value == "active"
    assert TransactionState.PARTIALLY_COMMITTED.value == "partially_committed"
    assert TransactionState.COMMITTED.value == "committed"
    assert TransactionState.FAILED.value == "failed"
    assert TransactionState.ABORTED.value == "aborted"


def test_strategy_is_abstract() -> None:
    with pytest.raises(TypeError):
        ConcurrencyStrategy()  # type: ignore[abstract]


def test_acquire_and_release(strategy) -> None:
    resource = ("record", "r1")
    strategy.acquire(1, resource, LockMode.SHARED)
    assert resource in strategy.locks_held(1)
    assert strategy.is_locked(resource)
    strategy.release(1, resource)
    assert resource not in strategy.locks_held(1)
    assert not strategy.is_locked(resource)


def test_release_all(strategy) -> None:
    strategy.acquire(1, ("record", "r1"), LockMode.SHARED)
    strategy.acquire(1, ("table", "t1"), LockMode.EXCLUSIVE)
    strategy.release_all(1)
    assert strategy.locks_held(1) == frozenset()
    assert not strategy.is_locked(("record", "r1"))
    assert not strategy.is_locked(("table", "t1"))


def test_locks_are_isolated_per_transaction(strategy) -> None:
    strategy.acquire(1, ("record", "r1"), LockMode.SHARED)
    strategy.acquire(2, ("record", "r2"), LockMode.SHARED)
    assert ("record", "r1") in strategy.locks_held(1)
    assert ("record", "r1") not in strategy.locks_held(2)
    assert ("record", "r2") in strategy.locks_held(2)
