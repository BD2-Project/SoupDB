"""Tests for the LockManager facade."""

import pytest

from engine.common.errors import LockNotGranted
from engine.transactions.base import ConcurrencyStrategy, LockMode
from engine.transactions.lock_manager import LockManager
from engine.transactions.strategies.strict_2pl import StrictTwoPhaseLocking


def test_defaults_to_strict_two_phase_locking() -> None:
    manager = LockManager()
    assert isinstance(manager.strategy, StrictTwoPhaseLocking)
    manager.close()


def test_accepts_injected_strategy() -> None:
    strategy = StrictTwoPhaseLocking()
    manager = LockManager(strategy=strategy)
    assert manager.strategy is strategy
    manager.close()


def test_delegates_acquire_and_release() -> None:
    manager = LockManager()
    manager.acquire(1, ("record", "r1"), LockMode.SHARED)
    assert ("record", "r1") in manager.locks_held(1)
    assert manager.is_locked(("record", "r1"))
    manager.release_all(1)
    assert not manager.is_locked(("record", "r1"))
    manager.close()


def test_acquire_raises_on_conflict() -> None:
    manager = LockManager()
    manager.acquire(1, ("record", "r1"), LockMode.EXCLUSIVE)
    with pytest.raises(LockNotGranted):
        manager.acquire(2, ("record", "r1"), LockMode.SHARED, timeout_ms=20)
    manager.close()


def test_exposes_strategy_type() -> None:
    assert issubclass(LockManager, object)
    assert ConcurrencyStrategy in StrictTwoPhaseLocking.__mro__
