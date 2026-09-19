"""Demostración de concurrencia con hilos (Avance 1).

Corre transferencias simultáneas sobre la tabla ``accounts`` comparando el
acceso directo sin control de concurrencia (race condition / actualizaciones
perdidas) contra el acceso transaccional con strict 2PL (resultado
serializable, con reintento tras deadlocks).
"""

import threading
import time
from dataclasses import dataclass
from typing import Any

from engine.common.errors import TransactionError
from engine.query import execute_sql
from engine.transactions.session import TransactionalSession
from engine.transactions.transaction_manager import TransactionManager


@dataclass(frozen=True)
class DemoResult:
    """Resultado de una simulación de transferencias."""

    final: int
    expected: int
    transfers: int
    deadlocks: int


def _balance(catalog: Any, account_id: int) -> int:
    rows = execute_sql(f"SELECT balance FROM accounts WHERE id = {account_id}", catalog)
    return rows.rows[0][0]


def _set_balance(catalog: Any, account_id: int, balance: int) -> None:
    execute_sql(f"DELETE FROM accounts WHERE id = {account_id}", catalog)
    execute_sql(f"INSERT INTO accounts VALUES ({account_id}, {balance})", catalog)


def race_result(
    catalog: Any,
    account_id: int = 0,
    initial: int = 100,
    threads: int = 2,
    rounds: int = 25,
    delta: int = 10,
    gap: float = 0.0005,
) -> DemoResult:
    """Read-modify-write sin locks: sufre actualizaciones perdidas.

    ``gap`` ensancha la ventana entre la lectura y la escritura para que la
    intercalación de hilos sea observable de forma determinista.
    """
    barrier = threading.Barrier(threads)

    def worker() -> None:
        barrier.wait()
        for _ in range(rounds):
            balance = _balance(catalog, account_id)
            time.sleep(gap)
            _set_balance(catalog, account_id, balance + delta)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    final = _balance(catalog, account_id)
    expected = initial + threads * rounds * delta
    return DemoResult(final=final, expected=expected, transfers=threads * rounds, deadlocks=0)


def transactional_result(
    catalog: Any,
    account_id: int = 0,
    initial: int = 100,
    threads: int = 2,
    rounds: int = 25,
    delta: int = 10,
    lock_timeout_ms: int = 200,
) -> DemoResult:
    """Transferencias transaccionales con strict 2PL: serializables."""
    manager = TransactionManager()
    barrier = threading.Barrier(threads)
    deadlocks = 0
    guard = threading.Lock()

    def worker() -> None:
        nonlocal deadlocks
        session = TransactionalSession(
            catalog, transaction_manager=manager, lock_timeout_ms=lock_timeout_ms
        )
        barrier.wait()
        for _ in range(rounds):
            while True:
                try:
                    with session.transaction():
                        rows = session.execute(
                            f"SELECT balance FROM accounts WHERE id = {account_id}"
                        )
                        balance = rows.rows[0][0]
                        session.execute(f"DELETE FROM accounts WHERE id = {account_id}")
                        session.execute(
                            f"INSERT INTO accounts VALUES ({account_id}, {balance + delta})"
                        )
                    break
                except TransactionError:
                    with guard:
                        deadlocks += 1
        session.close()

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    final = _balance(catalog, account_id)
    expected = initial + threads * rounds * delta
    return DemoResult(
        final=final,
        expected=expected,
        transfers=threads * rounds,
        deadlocks=deadlocks,
    )
