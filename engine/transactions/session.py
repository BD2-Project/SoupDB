"""Transactional session: executes statements inside transactions with locking.

The session wraps a catalog so that every access path goes through strict 2PL
locks. Each thread holds its own active transaction (thread-local); statements
run inside ``BEGIN/COMMIT`` explicitly or are auto-committed as a single
statement transaction when no transaction is active.
"""

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from engine.common.errors import (
    DeadlockDetected,
    LockNotGranted,
    TransactionError,
)
from engine.common.record import Record
from engine.common.rid import RID
from engine.query import ResultSet
from engine.query.executor import execute as execute_plan
from engine.query.parser import parse
from engine.query.planner import plan as build_plan
from engine.storage.base import FileOrganization
from engine.transactions.base import LockMode
from engine.transactions.transaction_manager import Transaction, TransactionManager

_DEFAULT_LOCK_TIMEOUT_MS = int(os.getenv("LOCK_TIMEOUT_MS", "3000"))


@dataclass
class _UndoInsert:
    """Undo an insert by removing the produced RID."""

    table: str
    rid: RID


@dataclass
class _UndoRemove:
    """Undo a remove by re-inserting the removed record."""

    table: str
    rid: RID
    record: Record


class TransactionalSession:
    """Runs SQL statements with transparent strict 2PL locking."""

    def __init__(
        self,
        catalog: Any,
        transaction_manager: TransactionManager | None = None,
        lock_timeout_ms: int | None = None,
    ) -> None:
        self._catalog = catalog
        self._tm = transaction_manager or TransactionManager()
        self._lock_timeout_ms = (
            lock_timeout_ms if lock_timeout_ms is not None else _DEFAULT_LOCK_TIMEOUT_MS
        )
        self._local = threading.local()

    @property
    def catalog(self) -> Any:
        return self._catalog

    @property
    def transaction_manager(self) -> TransactionManager:
        return self._tm

    def begin(self) -> Transaction:
        if getattr(self._local, "tx", None) is not None:
            raise TransactionError("a transaction is already active on this thread")
        tx = self._tm.begin()
        tx.undo_callback = self._apply_undo
        self._local.tx = tx
        return tx

    def commit(self) -> None:
        tx = self._current()
        self._tm.commit(tx)
        self._local.tx = None

    def rollback(self) -> None:
        tx = self._current()
        tx.rollback()
        self._local.tx = None

    def current_transaction(self) -> Transaction | None:
        return getattr(self._local, "tx", None)

    def execute(self, sql: str) -> ResultSet:
        return self.execute_statement(parse(sql))

    def execute_statement(self, statement: Any) -> ResultSet:
        if self.current_transaction() is None:
            self.begin()
            try:
                return self._run(statement)
            except BaseException:
                if self.current_transaction() is not None:
                    self.rollback()
                raise
        return self._run(statement)

    @contextmanager
    def transaction(self):
        tx = self.begin()
        try:
            yield tx
        except BaseException:
            self.rollback()
            raise
        else:
            self.commit()

    def _run(self, statement: Any) -> ResultSet:
        locking_catalog = _LockingCatalog(self._catalog, self)
        try:
            return execute_plan(build_plan(statement, locking_catalog), locking_catalog)
        except (LockNotGranted, DeadlockDetected) as exc:
            tx = self.current_transaction()
            if tx is not None:
                self.rollback()
            raise TransactionError(
                f"deadlock or lock timeout in transaction {tx.tx_id if tx else '?'}: {exc}"
            ) from exc

    # --- Locking helpers used by the catalog proxies ---------------------

    def _lock(self, resource: tuple[object, object], mode: LockMode) -> None:
        tx = self._current()
        self._tm.lock_manager.acquire(tx.tx_id, resource, mode, timeout_ms=self._lock_timeout_ms)

    def _journal_append(self, entry: object) -> None:
        self._current().journal.append(entry)

    def _current(self) -> Transaction:
        tx = getattr(self._local, "tx", None)
        if tx is None:
            raise TransactionError("no active transaction on this thread")
        return tx

    def _apply_undo(self, journal: list[object]) -> None:
        for entry in reversed(journal):
            if isinstance(entry, _UndoInsert):
                self._catalog.file_org(entry.table).remove(entry.rid)
            elif isinstance(entry, _UndoRemove):
                self._catalog.file_org(entry.table).insert(entry.record)

    def close(self) -> None:
        self._tm.close()


class _LockingCatalog:
    """Catalog proxy that returns locking file organizations per access."""

    def __init__(self, catalog: Any, session: TransactionalSession) -> None:
        self._catalog = catalog
        self._session = session

    def schema(self, name: str):
        return self._catalog.schema(name)

    def strategy(self, name: str) -> str:
        return self._catalog.strategy(name)

    def indexes(self, name: str):
        return self._catalog.indexes(name)

    def indexes_for(self, name: str, column: str):
        return self._catalog.indexes_for(name, column)

    def index_location(self, name: str):
        return self._catalog.index_location(name)

    def tables(self):
        return self._catalog.tables()

    @property
    def disk_manager(self):
        return getattr(self._catalog, "disk_manager", None)

    def file_org(self, name: str) -> FileOrganization:
        return _LockingFileOrganization(name, self._catalog.file_org(name), self._session)

    def create_table(self, name: str, columns: tuple[object, ...], engine: str = "HEAP") -> None:
        self._session._lock(("table", name), LockMode.EXCLUSIVE)
        self._catalog.create_table(name, columns, engine)

    def create_index(
        self,
        index_name: str,
        table: str,
        column: str,
        index_type: str = "BTREE",
    ) -> None:
        self._session._lock(("table", table), LockMode.EXCLUSIVE)
        self._catalog.create_index(index_name, table, column, index_type)

    def drop_table(self, name: str) -> None:
        self._session._lock(("table", name), LockMode.EXCLUSIVE)
        self._catalog.drop_table(name)

    def drop_index(self, index_name: str) -> None:
        location = self._catalog.index_location(index_name)
        if location is not None:
            self._session._lock(("table", location[0]), LockMode.EXCLUSIVE)
        self._catalog.drop_index(index_name)


class _LockingFileOrganization(FileOrganization):
    """File organization proxy that acquires locks before touching storage."""

    def __init__(
        self,
        table: str,
        inner: FileOrganization,
        session: TransactionalSession,
    ) -> None:
        self._table = table
        self._inner = inner
        self._session = session

    def insert(self, record: Record) -> RID:
        self._session._lock(("table", self._table), LockMode.EXCLUSIVE)
        rid = self._inner.insert(record)
        self._session._journal_append(_UndoInsert(self._table, rid))
        return rid

    def fetch(self, rid: RID) -> Record | None:
        self._session._lock(("record", rid), LockMode.SHARED)
        return self._inner.fetch(rid)

    def remove(self, rid: RID) -> bool:
        self._session._lock(("record", rid), LockMode.EXCLUSIVE)
        record = self._inner.fetch(rid)
        if record is None:
            return False
        self._inner.remove(rid)
        self._session._journal_append(_UndoRemove(self._table, rid, record))
        return True

    def scan(self) -> Any:
        self._session._lock(("table", self._table), LockMode.SHARED)
        return self._inner.scan()
