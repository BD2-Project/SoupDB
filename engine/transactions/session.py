"""Transactional session: executes statements inside transactions with locking.

The session wraps a catalog so that every access path goes through strict 2PL
locks. Each thread holds its own active transaction (thread-local); statements
run inside ``BEGIN/COMMIT`` explicitly or are auto-committed as a single
statement transaction when no transaction is active. Boundaries can also be
driven from SQL text (``BEGIN [TRANSACTION]`` / ``END [TRANSACTION]``), which
this session intercepts before the auto-commit path.
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
from engine.common.record import Record, decode_row
from engine.common.rid import RID
from engine.query import ResultSet
from engine.query.ast import BeginTransactionStatement, EndTransactionStatement
from engine.query.executor import _indexes_for_column, _skip_index_key
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
        if isinstance(statement, BeginTransactionStatement):
            return self._execute_begin_transaction()
        if isinstance(statement, EndTransactionStatement):
            return self._execute_end_transaction()
        if self.current_transaction() is None:
            self.begin()
            try:
                result = self._run(statement)
            except BaseException:
                if self.current_transaction() is not None:
                    self.rollback()
                raise
            else:
                self.commit()
            return result
        return self._run(statement)

    def _execute_begin_transaction(self) -> ResultSet:
        """``BEGIN [TRANSACTION]`` opens the thread transaction.

        ``begin()`` rejects a nested BEGIN with "a transaction is already active
        on this thread"; the statement never reaches the executor, so the
        auto-commit path cannot fire and close a transaction the caller believes
        is still open.
        """
        self.begin()
        return ResultSet(columns=(), affected=0)

    def _execute_end_transaction(self) -> ResultSet:
        """``END [TRANSACTION]`` commits the active thread transaction.

        ``commit()`` goes through ``_current()``, so ending without an active
        transaction raises "no active transaction on this thread" instead of
        silently doing nothing.
        """
        self.commit()
        return ResultSet(columns=(), affected=0)

    @contextmanager
    def transaction(self):
        tx = self.begin()
        try:
            yield tx
        except BaseException:
            if self.current_transaction() is not None:
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
        """Undo the journal, storage and indexes together.

        Both directions touch the indexes because the journal entry alone no longer
        describes them: undoing a DELETE re-inserts the row under a NEW RID, and
        undoing an INSERT has to take back the entry the INSERT added.

        The order is storage first and indexes second, and there is no compensating
        step if an index insert fails halfway. That is the same fail-stop contract
        the executor's own INSERT and DELETE already have, and the undo is no more
        fragile than the statement that wrote the entry it is undoing. What the undo
        must never do is leave a live row unindexed while reporting success, which is
        why the index update is not swallowed by a ``try``.
        """
        for entry in reversed(journal):
            if isinstance(entry, _UndoInsert):
                file_org = self._catalog.file_org(entry.table)
                record = file_org.fetch(entry.rid)
                if record is not None:
                    self._unindex_row(entry.table, record, entry.rid)
                file_org.remove(entry.rid)
            elif isinstance(entry, _UndoRemove):
                rid = self._catalog.file_org(entry.table).insert(entry.record)
                self._index_row(entry.table, entry.record, rid)

    def _index_row(self, table: str, record: Record, rid: RID) -> None:
        """Index a row that undoing a DELETE has just put back.

        Undoing a DELETE gives the row a NEW RID, so the entry the DELETE removed
        has to be added again under that new one. Leaving it out makes the row live
        in the table and absent from every index, and an index that does not
        describe its table answers wrongly and silently: a rollback would leave
        every indexed lookup - spatial or scalar - reading a table it no longer
        matches.
        """
        row = self._row_values(table, record)
        for position, column in enumerate(self._catalog.schema(table)):
            for index in _indexes_for_column(self._catalog, table, column.name).values():
                if not _skip_index_key(index, row[position]):
                    index.insert(row[position], rid)

    def _unindex_row(self, table: str, record: Record, rid: RID) -> None:
        """Drop from the indexes the row that undoing an INSERT is about to remove."""
        row = self._row_values(table, record)
        for position, column in enumerate(self._catalog.schema(table)):
            for index in _indexes_for_column(self._catalog, table, column.name).values():
                if not _skip_index_key(index, row[position]):
                    index.remove(row[position], rid)

    def _row_values(self, table: str, record: Record) -> tuple[object, ...]:
        return decode_row(record.data, self._catalog.schema(table))

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

    def lock_shared(self) -> None:
        self._session._lock(("table", self._table), LockMode.SHARED)

