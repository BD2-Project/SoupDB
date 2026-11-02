"""Ranked writes must be explicitly reversible, not merely mutable."""

from dataclasses import replace

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.query.index_maintenance import maintain_ranked, ranked_targets
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_ranked_maintenance import FakeReversibleRankedIndex
from tests.fakes.fake_ranked_search import ScriptedRankedIndex

SCHEMA = (ColumnDef("id", ColumnType.INT), ColumnDef("texto", ColumnType.TEXT))


def make_catalog() -> tuple[FakeCatalog, RankedBinding, FakeReversibleRankedIndex]:
    catalog = FakeCatalog()
    catalog.create_table("docs", SCHEMA)
    index = FakeReversibleRankedIndex()
    binding = RankedBinding(
        "idx_docs",
        "docs",
        "texto",
        "INVERTED",
        index,
        mutable=True,
        reversible=True,
    )
    catalog.add_ranked_index(binding)
    return catalog, binding, index


def test_accepts_explicitly_reversible_binding() -> None:
    catalog, binding, _ = make_catalog()
    assert ranked_targets(catalog, "docs") == (binding,)


def test_insert_remove_idempotent() -> None:
    catalog, _, index = make_catalog()
    targets = ranked_targets(catalog, "docs")
    rid = RID(0, 5)
    maintain_ranked(targets, SCHEMA, (5, "texto"), rid, insert=True)
    maintain_ranked(targets, SCHEMA, (5, "texto"), rid, insert=True)
    assert index.documents == {rid: "texto"}
    maintain_ranked(targets, SCHEMA, (5, "texto"), rid, insert=False)
    maintain_ranked(targets, SCHEMA, (5, "texto"), rid, insert=False)
    assert not index.documents


def test_null_value_is_not_indexed() -> None:
    catalog, _, index = make_catalog()
    maintain_ranked(ranked_targets(catalog, "docs"), SCHEMA, (1, None), RID(0, 1), insert=True)
    assert index.documents == {}


@pytest.mark.parametrize("attribute", ["mutable", "reversible"])
def test_missing_capability_is_rejected(attribute: str) -> None:
    catalog, binding, index = make_catalog()
    catalog.drop_index(binding.name)
    catalog.add_ranked_index(replace(binding, **{attribute: False}))
    with pytest.raises(QueryExecutionError, match="no reversible provider"):
        ranked_targets(catalog, "docs")
    assert index.documents == {}


def test_readonly_provider_cannot_fake_capability() -> None:
    catalog, binding, _ = make_catalog()
    catalog.drop_index(binding.name)
    catalog.add_ranked_index(replace(binding, index=ScriptedRankedIndex("text")))
    with pytest.raises(QueryExecutionError, match="no reversible provider"):
        ranked_targets(catalog, "docs")


def test_declared_but_nonatomic_provider_is_rejected() -> None:
    catalog, binding, index = make_catalog()
    index.atomic_mutations = False
    with pytest.raises(QueryExecutionError, match="no reversible provider"):
        ranked_targets(catalog, "docs")


def test_fake_provider_failure_is_not_suppressed() -> None:
    catalog, _, index = make_catalog()
    index.fail_next = "insert"
    with pytest.raises(RuntimeError, match="simulated"):
        maintain_ranked(ranked_targets(catalog, "docs"), SCHEMA, (2, "x"), RID(0, 2), insert=True)
    assert index.documents == {}
