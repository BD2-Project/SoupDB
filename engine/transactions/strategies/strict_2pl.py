"""Pessimistic strict two-phase locking (strict 2PL) strategy.

Locks (S/U/X) are granted following the classic compatibility matrix and are
held until ``release_all`` (commit or rollback), which guarantees isolation
and serializable schedules. Deadlock is handled by lock timeout.
"""

import threading
import time

from engine.common.errors import LockNotGranted
from engine.transactions.base import ConcurrencyStrategy, LockMode, Resource

_LOCK_PRIORITY = {
    LockMode.SHARED: 0,
    LockMode.UPDATE: 1,
    LockMode.EXCLUSIVE: 2,
}

# Compatibility matrix: can a lock in `mode` be granted when another
# transaction holds a lock in `held`?
_COMPAT = {
    LockMode.SHARED: {
        LockMode.SHARED: True,
        LockMode.UPDATE: True,
        LockMode.EXCLUSIVE: False,
    },
    LockMode.UPDATE: {
        LockMode.SHARED: True,
        LockMode.UPDATE: False,
        LockMode.EXCLUSIVE: False,
    },
    LockMode.EXCLUSIVE: {
        LockMode.SHARED: False,
        LockMode.UPDATE: False,
        LockMode.EXCLUSIVE: False,
    },
}

_POLL_INTERVAL_S = 0.001


def _stronger(held: LockMode, requested: LockMode) -> LockMode:
    """Return the stronger mode for a transaction re-acquiring a resource."""
    if _LOCK_PRIORITY[requested] > _LOCK_PRIORITY[held]:
        return requested
    return held


class StrictTwoPhaseLocking(ConcurrencyStrategy):
    """Strict 2PL strategy with table/record granularity."""

    def __init__(self) -> None:
        # resource -> {tx_id: (mode, count)}
        self._grants: dict[Resource, dict[int, tuple[LockMode, int]]] = {}
        self._mutex = threading.RLock()

    def _try_grant(self, tx_id: int, resource: Resource, mode: LockMode) -> bool:
        grants = self._grants.get(resource)
        if not grants:
            return True
        if tx_id in grants:
            desired = _stronger(grants[tx_id][0], mode)
            for other, (other_mode, _) in grants.items():
                if other != tx_id and not _COMPAT[other_mode][desired]:
                    return False
            return True
        return all(_COMPAT[other_mode][mode] for other_mode, _ in grants.values())

    def acquire(
        self,
        tx_id: int,
        resource: Resource,
        mode: LockMode,
        timeout_ms: int | None = None,
    ) -> None:
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000
        while True:
            with self._mutex:
                if self._try_grant(tx_id, resource, mode):
                    grants = self._grants.setdefault(resource, {})
                    if tx_id in grants:
                        held, count = grants[tx_id]
                        grants[tx_id] = (_stronger(held, mode), count + 1)
                    else:
                        grants[tx_id] = (mode, 1)
                    return
            if deadline is not None and time.monotonic() >= deadline:
                raise LockNotGranted(
                    f"lock not granted on {resource} for tx {tx_id} within timeout"
                )
            time.sleep(_POLL_INTERVAL_S)

    def release(self, tx_id: int, resource: Resource) -> None:
        with self._mutex:
            grants = self._grants.get(resource)
            if not grants or tx_id not in grants:
                return
            mode, count = grants[tx_id]
            if count > 1:
                grants[tx_id] = (mode, count - 1)
            else:
                del grants[tx_id]
                if not grants:
                    del self._grants[resource]

    def release_all(self, tx_id: int) -> None:
        with self._mutex:
            for resource in list(self._grants):
                grants = self._grants[resource]
                if tx_id in grants:
                    del grants[tx_id]
                if not grants:
                    del self._grants[resource]

    def locks_held(self, tx_id: int) -> frozenset[Resource]:
        with self._mutex:
            return frozenset(
                resource for resource, grants in self._grants.items() if tx_id in grants
            )

    def is_locked(self, resource: Resource) -> bool:
        with self._mutex:
            return resource in self._grants and bool(self._grants[resource])

    def close(self) -> None:
        with self._mutex:
            self._grants.clear()
