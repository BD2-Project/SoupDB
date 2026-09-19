"""Frozen contract for concurrency control.

Concurrency strategies are swappable: the rest of the engine depends on
:class:`ConcurrencyStrategy`, never on a concrete implementation. The
pessimistic strict 2PL strategy is one implementation; optimistic control,
MVCC or timestamp ordering can be added later without touching callers.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

#: A lockable resource. Granularity is encoded in the key: a table lock uses
#: ``("table", name)`` and a record lock uses ``("record", RID)``.
Resource = Any


class LockMode(Enum):
    """Lock modes for pessimistic control."""

    SHARED = "S"
    UPDATE = "U"
    EXCLUSIVE = "X"


class TransactionState(Enum):
    """Lifecycle states of a transaction."""

    ACTIVE = "active"
    PARTIALLY_COMMITTED = "partially_committed"
    COMMITTED = "committed"
    FAILED = "failed"
    ABORTED = "aborted"


class ConcurrencyStrategy(ABC):
    """Base interface for concurrency control strategies."""

    @abstractmethod
    def acquire(
        self,
        tx_id: int,
        resource: Resource,
        mode: LockMode,
        timeout_ms: int | None = None,
    ) -> None:
        """Acquire a lock, blocking until granted or raising LockNotGranted on timeout."""

    @abstractmethod
    def release(self, tx_id: int, resource: Resource) -> None:
        """Release one lock instance of a resource held by a transaction."""

    @abstractmethod
    def release_all(self, tx_id: int) -> None:
        """Release every lock held by a transaction (commit or rollback)."""

    @abstractmethod
    def locks_held(self, tx_id: int) -> frozenset[Resource]:
        """Return the resources currently locked by a transaction."""

    @abstractmethod
    def is_locked(self, resource: Resource) -> bool:
        """Whether a resource has at least one granted lock."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the strategy."""
