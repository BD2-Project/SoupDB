"""Provider boundary for ranked CREATE INDEX / database reopen.

The SQL subsystem does not implement indexing algorithms here. Both creation
and reloading must be supplied by an explicitly configured provider.
"""

from collections.abc import Mapping
from typing import Protocol

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.ranked_catalog import RankedMetadata
from engine.common.schema import ColumnDef
from engine.storage.base import FileOrganization


class RankedFactory(Protocol):
    def build(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        """Build an index over all existing table rows or fail without publishing."""

    def load(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        """Reopen an existing provider from durable data, or raise."""


class DelegatingRankedFactory:
    """Route each index family to an explicitly configured provider.

    IVF/HNSW delegates normally share one query_vectorizer instance. Pass an
    explicit shared resolver when combining delegates with different resolvers.
    No multimedia implementation is imported by this composition boundary.
    """

    def __init__(
        self, factories: Mapping[str, RankedFactory], *, query_vectorizer: object | None = None
    ) -> None:
        if factories.keys() - {"INVERTED", "IVF", "HNSW"}:
            raise ValueError("unsupported ranked factory algorithm")
        self._factories = dict(factories)
        if query_vectorizer is None:
            resolvers = {
                id(resolver): resolver
                for algorithm, factory in self._factories.items()
                if algorithm in {"IVF", "HNSW"}
                and (resolver := getattr(factory, "query_vectorizer", None)) is not None
            }
            if len(resolvers) > 1:
                raise ValueError("vector delegates require an explicit shared query_vectorizer")
            query_vectorizer = next(iter(resolvers.values()), None)
        self.query_vectorizer = query_vectorizer

    def _factory(self, metadata: RankedMetadata) -> RankedFactory:
        try:
            return self._factories[metadata.algorithm]
        except KeyError as exc:
            raise QueryExecutionError(
                f"no ranked provider configured for {metadata.algorithm!r}"
            ) from exc

    def build(
        self, metadata: RankedMetadata, file_org: FileOrganization, schema: tuple[ColumnDef, ...]
    ) -> RankedBinding:
        return self._factory(metadata).build(metadata, file_org, schema)

    def load(
        self, metadata: RankedMetadata, file_org: FileOrganization, schema: tuple[ColumnDef, ...]
    ) -> RankedBinding:
        return self._factory(metadata).load(metadata, file_org, schema)

    def drop(self, metadata: RankedMetadata) -> None:
        drop = getattr(self._factory(metadata), "drop", None)
        if callable(drop):
            drop(metadata)
