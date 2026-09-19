"""Lock manager facade.

Delegates to an injectable :class:`ConcurrencyStrategy`, so strategies can be
swapped without touching callers.
"""

from engine.transactions.base import ConcurrencyStrategy, LockMode, Resource
from engine.transactions.strategies.strict_2pl import StrictTwoPhaseLocking


class LockManager:
    """Facade over a concurrency strategy."""

    def __init__(self, strategy: ConcurrencyStrategy | None = None) -> None:
        self._strategy = strategy if strategy is not None else StrictTwoPhaseLocking()

    @property
    def strategy(self) -> ConcurrencyStrategy:
        """The active concurrency strategy."""
        return self._strategy

    def acquire(
        self,
        tx_id: int,
        resource: Resource,
        mode: LockMode,
        timeout_ms: int | None = None,
    ) -> None:
        self._strategy.acquire(tx_id, resource, mode, timeout_ms=timeout_ms)

    def release(self, tx_id: int, resource: Resource) -> None:
        self._strategy.release(tx_id, resource)

    def release_all(self, tx_id: int) -> None:
        self._strategy.release_all(tx_id)

    def locks_held(self, tx_id: int) -> frozenset[Resource]:
        return self._strategy.locks_held(tx_id)

    def is_locked(self, resource: Resource) -> bool:
        return self._strategy.is_locked(resource)

    def close(self) -> None:
        self._strategy.close()
