"""Tests for the pessimistic strict 2PL strategy."""

import pytest

from engine.common.errors import LockNotGranted
from engine.transactions.base import LockMode
from engine.transactions.strategies.strict_2pl import StrictTwoPhaseLocking


@pytest.fixture
def s2pl():
    strategy = StrictTwoPhaseLocking()
    yield strategy
    strategy.close()


def test_shared_locks_are_compatible(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.SHARED)
    s2pl.acquire(2, ("record", "r1"), LockMode.SHARED)
    assert ("record", "r1") in s2pl.locks_held(1)
    assert ("record", "r1") in s2pl.locks_held(2)


def test_shared_blocks_exclusive(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.SHARED)
    with pytest.raises(LockNotGranted):
        s2pl.acquire(2, ("record", "r1"), LockMode.EXCLUSIVE, timeout_ms=20)


def test_exclusive_blocks_shared(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.EXCLUSIVE)
    with pytest.raises(LockNotGranted):
        s2pl.acquire(2, ("record", "r1"), LockMode.SHARED, timeout_ms=20)


def test_exclusive_blocks_exclusive(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.EXCLUSIVE)
    with pytest.raises(LockNotGranted):
        s2pl.acquire(2, ("record", "r1"), LockMode.EXCLUSIVE, timeout_ms=20)


def test_update_allows_shared_readers(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.UPDATE)
    s2pl.acquire(2, ("record", "r1"), LockMode.SHARED)
    with pytest.raises(LockNotGranted):
        s2pl.acquire(3, ("record", "r1"), LockMode.EXCLUSIVE, timeout_ms=20)
    with pytest.raises(LockNotGranted):
        s2pl.acquire(3, ("record", "r1"), LockMode.UPDATE, timeout_ms=20)


def test_upgrade_shared_to_exclusive(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.SHARED)
    s2pl.acquire(1, ("record", "r1"), LockMode.EXCLUSIVE)
    assert s2pl._grants[("record", "r1")][1][0] == LockMode.EXCLUSIVE


def test_upgrade_waits_for_other_reader(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.SHARED)
    s2pl.acquire(2, ("record", "r1"), LockMode.SHARED)
    with pytest.raises(LockNotGranted):
        s2pl.acquire(1, ("record", "r1"), LockMode.EXCLUSIVE, timeout_ms=20)


def test_release_all_after_commit(s2pl) -> None:
    s2pl.acquire(1, ("table", "t1"), LockMode.EXCLUSIVE)
    s2pl.release_all(1)
    s2pl.acquire(2, ("table", "t1"), LockMode.EXCLUSIVE)
    assert ("table", "t1") in s2pl.locks_held(2)


def test_lock_counts_same_tx(s2pl) -> None:
    s2pl.acquire(1, ("record", "r1"), LockMode.SHARED)
    s2pl.acquire(1, ("record", "r1"), LockMode.SHARED)
    s2pl.release(1, ("record", "r1"))
    assert ("record", "r1") in s2pl.locks_held(1)
    s2pl.release(1, ("record", "r1"))
    assert ("record", "r1") not in s2pl.locks_held(1)
