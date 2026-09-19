"""Demostración de concurrencia con hilos (Avance 1).

Ejecuta transferencias simultáneas sobre la tabla ``accounts``:

1. Sin control de concurrencia: se ven actualizaciones perdidas (race condition).
2. Con transacciones strict 2PL: resultado serializable, con deadlocks
   detectados y resueltos automáticamente.

Uso: uv run python scripts/demo_concurrency.py
"""

import tempfile
from pathlib import Path

from engine.common.catalog import Catalog
from engine.common.schema import ColumnDef, ColumnType
from engine.query import execute_sql
from engine.transactions import demo

ACCOUNTS = (ColumnDef("id", ColumnType.INT), ColumnDef("balance", ColumnType.INT))
INITIAL = 100
THREADS = 2
ROUNDS = 25
DELTA = 10


def _fresh_catalog(root: Path) -> Catalog:
    catalog = Catalog(root)
    catalog.create_table("accounts", ACCOUNTS, "HEAP")
    execute_sql("INSERT INTO accounts VALUES (0, 100)", catalog)
    return catalog


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="soupdb_demo_") as tmp:
        race_root = Path(tmp) / "race"
        race_root.mkdir()
        race_catalog = _fresh_catalog(race_root)
        race = demo.race_result(
            race_catalog, initial=INITIAL, threads=THREADS, rounds=ROUNDS, delta=DELTA
        )
        race_catalog.close()

        tx_root = Path(tmp) / "tx"
        tx_root.mkdir()
        tx_catalog = _fresh_catalog(tx_root)
        transactional = demo.transactional_result(
            tx_catalog, initial=INITIAL, threads=THREADS, rounds=ROUNDS, delta=DELTA
        )
        tx_catalog.close()

    lost = race.expected - race.final
    print("== SoupDB - demostracion de concurrencia (Avance 1) ==")
    print()
    print("1) Sin control de concurrencia (race condition)")
    print(
        f"   transferencias: {race.transfers} | saldo final: {race.final} | "
        f"esperado: {race.expected}"
    )
    print(f"   => se perdieron {lost} incrementos (actualizaciones perdidas)")
    print()
    print("2) Con transacciones strict 2PL")
    print(
        f"   transferencias: {transactional.transfers} | saldo final: "
        f"{transactional.final} | esperado: {transactional.expected}"
    )
    print(f"   deadlocks detectados y resueltos: {transactional.deadlocks}")
    print("   => resultado serializable, sin perdidas")
    print()

    if transactional.final != transactional.expected:
        print("ERROR: el resultado transaccional no es serializable")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
