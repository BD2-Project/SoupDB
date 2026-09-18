"""In-memory concurrency strategy: oracle and unblocking fake."""

from engine.transactions.base import ConcurrencyStrategy, LockMode, Resource


class FakeConcurrencyStrategy(ConcurrencyStrategy):
    """In-memory no-op strategy used for contract tests."""

    def __init__(self) -> None:
        self._locks: dict[int, set[Resource]] = {}

    def acquire(self, tx_id: int, resource: Resource, mode: LockMode) -> None:
        self._locks.setdefault(tx_id, set()).add(resource)

    def release(self, tx_id: int, resource: Resource) -> None:
        self._locks.get(tx_id, set()).discard(resource)

    def release_all(self, tx_id: int) -> None:
        self._locks.pop(tx_id, None)

    def locks_held(self, tx_id: int) -> frozenset[Resource]:
        return frozenset(self._locks.get(tx_id, ()))

    def is_locked(self, resource: Resource) -> bool:
        return any(resource in held for held in self._locks.values())

    def close(self) -> None:
        self._locks.clear()
