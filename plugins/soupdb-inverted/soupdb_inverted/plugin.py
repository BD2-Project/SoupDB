"""Entry-point plugin for the persistent inverted (full-text) index."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.ranked_catalog import RankedMetadata
from engine.common.schema import ColumnDef
from engine.plugins.base import API_VERSION
from engine.plugins.capabilities import PluginCapabilities
from engine.storage.base import FileOrganization
from soupdb_inverted.provider import OnDiskInvertedFactory

DEFAULT_STORAGE_ROOT = os.environ.get("SOUPDB_PLUGIN_STORAGE", ".soupdb-plugins")
DEFAULT_MEMORY_LIMIT = 8 * 1024 * 1024
DEFAULT_PAGE_SIZE = 4096
DEFAULT_FAN_IN = 8


class InvertedPlugin:
    """Full-text plugin backed by SPIMI postings and TF-IDF/BM25 scoring."""

    name = "INVERTED"
    version = "0.1.0"
    api_version = API_VERSION

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            exact=True,
            exhaustive=True,
            mutable=False,
            reversible=False,
            operators=frozenset({"search"}),
            index_options=frozenset({"method"}),
        )

    def default_config(self) -> dict[str, Any]:
        return {
            "storage_root": str(self._config.get("storage_root", DEFAULT_STORAGE_ROOT)),
            "memory_limit_bytes": int(self._config.get("memory_limit_bytes", DEFAULT_MEMORY_LIMIT)),
            "page_size": int(self._config.get("page_size", DEFAULT_PAGE_SIZE)),
            "fan_in": int(self._config.get("fan_in", DEFAULT_FAN_IN)),
        }

    def configure(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(options) - {"method"}
        if unknown:
            raise QueryExecutionError(f"unsupported INVERTED options: {sorted(unknown)}")
        method = options.get("method")
        if method is None:
            return {}
        if not isinstance(method, str) or method.lower() not in {"tf_idf", "bm25"}:
            raise QueryExecutionError("INVERTED method must be 'tf_idf' or 'bm25'")
        return {"method": method.lower()}

    def _factory(self) -> OnDiskInvertedFactory:
        config = self.default_config()
        return OnDiskInvertedFactory(
            Path(config["storage_root"]),
            memory_limit_bytes=config["memory_limit_bytes"],
            page_size=config["page_size"],
            fan_in=config["fan_in"],
        )

    def build(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        return self._factory().build(metadata, file_org, schema)

    def load(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        return self._factory().load(metadata, file_org, schema)

    def drop(self, metadata: RankedMetadata) -> None:
        self._factory().drop(metadata)

    def search(
        self,
        binding: RankedBinding,
        query: Any,
        k: int,
        options: Mapping[str, Any],
    ) -> list[tuple[Any, float]]:
        return binding.index.top_k(query, k, **dict(options))
