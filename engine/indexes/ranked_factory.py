"""Provider boundary for ranked CREATE INDEX / database reopen.

The SQL subsystem does not implement indexing algorithms here. Both creation
and reloading must be supplied by an explicitly configured provider.
"""

from typing import Protocol

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
