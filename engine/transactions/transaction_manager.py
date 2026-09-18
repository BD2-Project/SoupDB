"""Transaction lifecycle management (ACID states)."""

from collections.abc import Callable
from threading import Lock

from engine.common.errors import TransactionError
from engine.transactions.base import TransactionState
from engine.transactions.lock_manager import LockManager


class Transaction:
    """A single transaction context with a lifecycle state."""

    def __init__(self, tx_id: int, manager: "TransactionManager") -> None:
        self.tx_id = tx_id
        self._manager = manager
        self.state = TransactionState.ACTIVE
        self.journal: list[object] = []
        self.undo_callback: Callable[[list[object]], None] | None = None

    def commit(self) -> None:
        self._manager.commit(self)

    def rollback(self) -> None:
        if self.undo_callback is not None:
            self.undo_callback(self.journal)
        self._manager.rollback(self)


class TransactionManager:
    """Manages transaction lifecycle and integrates the lock manager."""

    def __init__(self, lock_manager: LockManager | None = None) -> None:
        self._lock_manager = lock_manager or LockManager()
        self._transactions: dict[int, Transaction] = {}
        self._next_id = 1
        self._mutex = Lock()

    @property
    def lock_manager(self) -> LockManager:
        """The lock manager used by every transaction."""
        return self._lock_manager

    def begin(self) -> Transaction:
        with self._mutex:
            tx = Transaction(self._next_id, self)
            self._next_id += 1
            self._transactions[tx.tx_id] = tx
        return tx

    def commit(self, tx: Transaction) -> None:
        self._finish(tx, TransactionState.COMMITTED)

    def rollback(self, tx: Transaction) -> None:
        self._finish(tx, TransactionState.ABORTED)

    def _finish(self, tx: Transaction, final: TransactionState) -> None:
        if tx.state is not TransactionState.ACTIVE:
            raise TransactionError(f"transaction {tx.tx_id} is not active")
        tx.state = (
            TransactionState.PARTIALLY_COMMITTED
            if final is TransactionState.COMMITTED
            else TransactionState.FAILED
        )
        self._lock_manager.release_all(tx.tx_id)
        tx.state = final
        with self._mutex:
            self._transactions.pop(tx.tx_id, None)

    def active_transactions(self) -> tuple[Transaction, ...]:
        with self._mutex:
            return tuple(self._transactions.values())

    def close(self) -> None:
        self._lock_manager.close()
