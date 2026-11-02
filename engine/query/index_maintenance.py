"""Reversible ranked-index maintenance, independent of table storage.

Opt-in providers must support atomic, idempotent single-record mutations. The
SQL layer does not infer mutability from a top_k method or a nonempty index.
"""

from typing import Protocol, runtime_checkable

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.rid import RID
from engine.query.evaluator import Schema


@runtime_checkable
class ReversibleRankedIndex(Protocol):
    """A provider with atomic, idempotent mutations and inverse operations."""

    atomic_mutations: bool

    def insert_document(self, rid: RID, value: object) -> None:
        """Add or replace one document, atomically, by RID."""

    def remove_document(self, rid: RID, value: object) -> None:
        """Delete a document by RID; deleting a missing RID is a no-op."""


def ranked_targets(catalog: object, table: str) -> tuple[RankedBinding, ...]:
    """Return verified ranked bindings; reject unavailable/unsafe providers.

    Calling this before touching storage is mandatory. ``None`` / missing
    registries imply an ordinary table without ranked indexes.
    """
    metadata_fn = getattr(catalog, "ranked_metadata_for", None)
    ready_fn = getattr(catalog, "ranked_indexes_for", None)
    metadata = metadata_fn(table) if callable(metadata_fn) else {}
    ready = ready_fn(table) if callable(ready_fn) else {}

    if metadata and set(metadata) != set(ready):
        raise QueryExecutionError("ranked DML is blocked: a ranked provider is unavailable")
    for name, binding in ready.items():
        index = binding.index
        if not (
            binding.mutable is True
            and binding.reversible is True
            and isinstance(index, ReversibleRankedIndex)
            and getattr(index, "atomic_mutations", None) is True
        ):
            raise QueryExecutionError(
                f"ranked DML is blocked: index {name!r} has no reversible provider"
            )
    return tuple(ready.values())


def maintain_ranked(
    targets: tuple[RankedBinding, ...],
    schema: Schema,
    row: tuple[object, ...],
    rid: RID,
    *,
    insert: bool,
) -> None:
    """Apply a row mutation to prevalidated targets.

    Provider methods are required to be idempotent so transaction undo can
    repair a partially completed multi-index operation. The caller controls
    table storage and must invoke rollback on any write failure.
    """
    positions = {column.name: i for i, column in enumerate(schema)}
    for binding in targets:
        try:
            position = positions[binding.column]
        except KeyError as exc:
            raise QueryExecutionError(
                f"ranked column {binding.column!r} is missing from the table"
            ) from exc
        value = row[position]
        if value is None:
            continue
        index = binding.index
        if insert:
            index.insert_document(rid, value)
        else:
            index.remove_document(rid, value)
