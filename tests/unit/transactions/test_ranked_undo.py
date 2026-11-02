"""End-to-end reversible ranked DML via strict-2PL transaction sessions."""

from dataclasses import replace

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.record import decode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from engine.transactions.session import TransactionalSession
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_ranked_maintenance import FakeReversibleRankedIndex

SCHEMA = (ColumnDef("id", ColumnType.INT), ColumnDef("texto", ColumnType.TEXT))


def make_session(*, existing: bool = True):
    catalog = FakeCatalog()
    catalog.create_table("docs", SCHEMA)
    if existing:
        catalog.insert("docs", (1, "original"))
    idx = FakeReversibleRankedIndex()
    for rid, record in catalog.file_org("docs").scan():
        idx.insert_document(rid, decode_row(record.data, SCHEMA)[1])
    binding = RankedBinding(
        "idx_docs",
        "docs",
        "texto",
        "INVERTED",
        idx,
        mutable=True,
        reversible=True,
    )
    catalog.add_ranked_index(binding)
    return catalog, TransactionalSession(catalog), idx


def current(catalog, index) -> tuple[tuple[object, ...], ...]:
    storage = {
        rid: decode_row(record.data, SCHEMA) for rid, record in catalog.file_org("docs").scan()
    }
    assert index.documents == {rid: row[1] for rid, row in storage.items() if row[1] is not None}
    return tuple(sorted(storage.values()))


def test_insert_autocommit_indexes_new_record() -> None:
    catalog, session, index = make_session()
    try:
        assert session.execute("INSERT INTO docs VALUES (2, 'nuevo')").affected == 1
        assert current(catalog, index) == ((1, "original"), (2, "nuevo"))
    finally:
        session.close()


def test_explicit_insert_rollback_restores_both() -> None:
    catalog, session, index = make_session()
    try:
        session.begin()
        session.execute("INSERT INTO docs VALUES (2, 'nuevo')")
        assert len(current(catalog, index)) == 2
        session.rollback()
        assert current(catalog, index) == ((1, "original"),)
    finally:
        session.close()


def test_delete_autocommit_updates_ranked_index() -> None:
    catalog, session, index = make_session()
    try:
        assert session.execute("DELETE FROM docs WHERE id = 1").affected == 1
        assert current(catalog, index) == ()
    finally:
        session.close()


def test_delete_rollback_reindexes_new_rid() -> None:
    catalog, session, index = make_session()
    try:
        session.begin()
        session.execute("DELETE FROM docs WHERE id = 1")
        assert current(catalog, index) == ()
        session.rollback()
        assert current(catalog, index) == ((1, "original"),)
    finally:
        session.close()


def test_insert_then_delete_in_same_tx_does_not_resurrect_row() -> None:
    catalog, session, index = make_session(existing=False)
    try:
        session.begin()
        session.execute("INSERT INTO docs VALUES (9, 'temporal')")
        session.execute("DELETE FROM docs WHERE id = 9")
        session.rollback()
        assert current(catalog, index) == ()
    finally:
        session.close()


@pytest.mark.parametrize("operation", ["insert", "remove"])
def test_failed_ranked_mutation_auto_rolls_back(operation: str) -> None:
    catalog, session, index = make_session()
    index.fail_next = operation
    try:
        sql = (
            "INSERT INTO docs VALUES (2, 'nuevo')"
            if operation == "insert"
            else "DELETE FROM docs WHERE id = 1"
        )
        with pytest.raises(RuntimeError, match="simulated"):
            session.execute(sql)
        assert current(catalog, index) == ((1, "original"),)
    finally:
        session.close()


def test_plain_executor_is_rejected_before_storage_changes() -> None:
    catalog, _, index = make_session()
    with pytest.raises(QueryExecutionError, match="TransactionalSession"):
        execute(plan(parse("INSERT INTO docs VALUES (2, 'nuevo')"), catalog), catalog)
    assert current(catalog, index) == ((1, "original"),)


def test_nonreversible_provider_blocks_before_storage_mutation() -> None:
    catalog, session, index = make_session()
    previous = catalog.ranked_indexes_for("docs")["idx_docs"]
    catalog.drop_index("idx_docs")
    catalog.add_ranked_index(replace(previous, reversible=False))
    try:
        with pytest.raises(QueryExecutionError, match="no reversible provider"):
            session.execute("INSERT INTO docs VALUES (2, 'nuevo')")
        assert current(catalog, index) == ((1, "original"),)
    finally:
        session.close()


def test_two_reversible_indexes_stay_consistent_during_rollback() -> None:
    catalog, session, first = make_session()
    second = FakeReversibleRankedIndex()
    for rid, record in catalog.file_org("docs").scan():
        second.insert_document(rid, decode_row(record.data, SCHEMA)[1])
    catalog.add_ranked_index(
        RankedBinding(
            "idx_docs_2",
            "docs",
            "texto",
            "INVERTED",
            second,
            mutable=True,
            reversible=True,
        )
    )
    try:
        session.begin()
        session.execute("INSERT INTO docs VALUES (2, 'nuevo')")
        session.rollback()
        assert current(catalog, first) == ((1, "original"),)
        current(catalog, second)
    finally:
        session.close()


def test_partial_multi_index_insert_is_repaired_by_auto_rollback() -> None:
    catalog, session, first = make_session()
    second = FakeReversibleRankedIndex()
    for rid, record in catalog.file_org("docs").scan():
        second.insert_document(rid, decode_row(record.data, SCHEMA)[1])
    catalog.add_ranked_index(
        RankedBinding(
            "idx_docs_2",
            "docs",
            "texto",
            "INVERTED",
            second,
            mutable=True,
            reversible=True,
        )
    )
    second.fail_next = "insert"
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            session.execute("INSERT INTO docs VALUES (2, 'nuevo')")
        assert current(catalog, first) == ((1, "original"),)
        current(catalog, second)
    finally:
        session.close()


def test_empty_or_null_text_does_not_create_stale_ranked_entry() -> None:
    catalog, session, index = make_session(existing=False)
    try:
        session.execute("INSERT INTO docs VALUES (1, NULL)")
        assert current(catalog, index) == ((1, None),)
        session.execute("DELETE FROM docs WHERE id = 1")
        assert current(catalog, index) == ()
    finally:
        session.close()
