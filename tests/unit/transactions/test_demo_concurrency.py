"""Tests for the concurrency demo simulation."""

from engine.common.schema import ColumnDef, ColumnType
from engine.transactions import demo
from tests.fakes.fake_catalog import FakeCatalog

ACCOUNTS = (ColumnDef("id", ColumnType.INT), ColumnDef("balance", ColumnType.INT))


def make_accounts(initial: int = 100) -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("accounts", ACCOUNTS)
    catalog.insert("accounts", (0, initial))
    return catalog


def test_transactional_transfers_are_serializable() -> None:
    result = demo.transactional_result(make_accounts(), threads=2, rounds=15, delta=10)
    assert result.final == result.expected


def test_transactional_never_loses_updates() -> None:
    result = demo.transactional_result(make_accounts(), threads=3, rounds=10, delta=5)
    assert result.final == 100 + 3 * 10 * 5


def test_race_result_loses_updates() -> None:
    result = demo.race_result(make_accounts(), threads=2, rounds=25, delta=10)
    assert result.final < result.expected
