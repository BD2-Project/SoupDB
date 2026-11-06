"""Provider boundary for ranked CREATE INDEX / database reopen.

The SQL subsystem does not implement indexing algorithms here. Both creation
and reloading must be supplied by an explicitly configured plugin registry or
by the legacy algorithm->factory mapping kept for compatibility.
"""

from collections.abc import Mapping
from typing import Protocol

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.ranked_catalog import RankedMetadata
from engine.common.schema import ColumnDef
from engine.plugins.errors import PluginNotFoundError
from engine.plugins.registry import PluginRegistry
from engine.storage.base import FileOrganization

_LEGACY_ALGORITHMS = frozenset({"INVERTED", "IVF", "HNSW"})


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
    """Route each ranked index family to its registered plugin.

    Two construction paths are supported:

    - ``DelegatingRankedFactory(mapping)``: legacy algorithm->factory mapping,
      restricted to ``INVERTED``/``IVF``/``HNSW``.
    - ``DelegatingRankedFactory.from_registry(registry)``: plugin registry,
      which accepts any installed plugin name (used by the extensible SQL).
    """

    def __init__(
        self, factories: Mapping[str, RankedFactory], *, query_vectorizer: object | None = None
    ) -> None:
        if factories.keys() - _LEGACY_ALGORITHMS:
            raise ValueError("unsupported ranked factory algorithm")
        self._registry = PluginRegistry()
        for algorithm, factory in factories.items():
            self._registry.register_factory(algorithm, factory, source="legacy-mapping")
        self.query_vectorizer = _resolve_query_vectorizer(factories, query_vectorizer)

    @classmethod
    def from_registry(
        cls, registry: PluginRegistry, *, query_vectorizer: object | None = None
    ) -> "DelegatingRankedFactory":
        instance = cls.__new__(cls)
        instance._registry = registry
        instance.query_vectorizer = _resolve_registry_vectorizer(registry, query_vectorizer)
        return instance

    @property
    def registry(self) -> PluginRegistry:
        return self._registry

    def _registration(self, metadata: RankedMetadata):
        try:
            return self._registry.get(metadata.algorithm)
        except PluginNotFoundError as exc:
            raise QueryExecutionError(
                f"no ranked provider configured for {metadata.algorithm!r}"
            ) from exc

    def build(
        self, metadata: RankedMetadata, file_org: FileOrganization, schema: tuple[ColumnDef, ...]
    ) -> RankedBinding:
        return self._registration(metadata).plugin.build(metadata, file_org, schema)

    def load(
        self, metadata: RankedMetadata, file_org: FileOrganization, schema: tuple[ColumnDef, ...]
    ) -> RankedBinding:
        return self._registration(metadata).plugin.load(metadata, file_org, schema)

    def drop(self, metadata: RankedMetadata) -> None:
        self._registration(metadata).plugin.drop(metadata)


def _factory_of(plugin: object) -> object:
    getter = getattr(plugin, "factory", None)
    return getter() if callable(getter) else plugin


def _resolve_query_vectorizer(
    factories: Mapping[str, object],
    explicit: object | None,
) -> object | None:
    if explicit is not None:
        return explicit
    resolvers = {
        id(resolver): resolver
        for algorithm, factory in factories.items()
        if algorithm in {"IVF", "HNSW"}
        and (resolver := getattr(factory, "query_vectorizer", None)) is not None
    }
    if len(resolvers) > 1:
        raise ValueError("vector delegates require an explicit shared query_vectorizer")
    return next(iter(resolvers.values()), None)


def _resolve_registry_vectorizer(
    registry: PluginRegistry, explicit: object | None
) -> object | None:
    if explicit is not None:
        return explicit
    resolvers = {}
    for name in registry.available():
        resolver = getattr(_factory_of(registry.plugin(name)), "query_vectorizer", None)
        if resolver is not None:
            resolvers[id(resolver)] = resolver
    if len(resolvers) > 1:
        raise ValueError("vector delegates require an explicit shared query_vectorizer")
    return next(iter(resolvers.values()), None)
