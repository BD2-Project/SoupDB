"""Protocols every SoupDB index/search plugin must satisfy.

The engine core never imports a concrete plugin implementation. It discovers
plugins through :class:`engine.plugins.registry.PluginRegistry` and talks to
them only through these protocols.
"""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from engine.common.ranked import RankedBinding
from engine.common.ranked_catalog import RankedMetadata
from engine.common.rid import RID
from engine.common.schema import ColumnDef
from engine.plugins.capabilities import PluginCapabilities
from engine.storage.base import FileOrganization

#: Contract version understood by this core. A plugin declaring another
#: version is rejected at registration time.
API_VERSION = 1


@runtime_checkable
class IndexPlugin(Protocol):
    """A buildable, reloadable index/search provider."""

    name: str
    version: str
    api_version: int

    def capabilities(self) -> PluginCapabilities:
        """Declare search/DML guarantees. Never inferred from methods."""

    def default_config(self) -> dict[str, Any]:
        """Configuration persisted in ``SysPlugins`` to reload the plugin."""

    def configure(self, config: Mapping[str, Any]) -> None:
        """Apply a configuration loaded from ``SysPlugins`` or the caller."""

    def validate_config(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Validate/normalize ``CREATE EXTENSION ... WITH (...)`` options."""

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        """Validate/normalize ``CREATE INDEX ... WITH (...)`` options."""

    def build(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        """Build an index over all existing rows or fail without publishing."""

    def load(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        """Reopen durable index data, or raise."""

    def drop(self, metadata: RankedMetadata) -> None:
        """Remove durable index artifacts. A missing artifact is not an error."""

    def search(
        self,
        binding: RankedBinding,
        query: Any,
        k: int,
        options: Mapping[str, Any],
    ) -> list[tuple[RID, float]]:
        """Answer a ranked search. Scores are always 'higher is better'."""


@runtime_checkable
class MutableIndexPlugin(IndexPlugin, Protocol):
    """Plugin whose durable index is maintained by DML.

    Mutations must be atomic and idempotent per RID: removing a missing RID is
    a no-op. Solvency of a rollback is the engine's responsibility through the
    transaction undo journal.
    """

    def insert_document(self, binding: RankedBinding, rid: RID, value: object) -> None:
        """Add or replace one document by RID."""

    def remove_document(self, binding: RankedBinding, rid: RID, value: object) -> None:
        """Remove one document by RID."""
