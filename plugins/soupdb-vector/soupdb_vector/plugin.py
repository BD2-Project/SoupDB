"""Entry-point plugins for the persistent vector indices (IVF and HNSW)."""

from __future__ import annotations

import json
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
from soupdb_vector.indexes.sql_factory import OnDiskVectorFactory
from soupdb_vector.multimedia.sql_profiles import MultimediaProfile

DEFAULT_STORAGE_ROOT = os.environ.get("SOUPDB_PLUGIN_STORAGE", ".soupdb-plugins")
DEFAULT_PAGE_SIZE = 4096
DEFAULT_NLIST = 4
DEFAULT_BUFFER_CAPACITY = 16

#: Both IVF and HNSW must resolve query vectors through the *same* vectorizer
#: instance when their stored configuration is identical. The cache makes that
#: true even though each plugin class owns its own factory reference.
_FACTORY_CACHE: dict[str, OnDiskVectorFactory] = {}


class _VectorPluginBase:
    """Shared configuration/behavior for IVF and HNSW plugins."""

    name = "IVF"
    version = "0.1.0"
    api_version = API_VERSION

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})

    @property
    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def capabilities(self) -> PluginCapabilities:
        return PluginCapabilities(
            exact=False,
            exhaustive=False,
            approximate=True,
            mutable=True,
            reversible=True,
            query_kind="vector",
            operators=frozenset({"search"}),
            index_options=frozenset({"metric", "dimension", "profile"}),
        )

    def default_config(self) -> dict[str, Any]:
        return {
            "storage_root": str(self._config.get("storage_root", DEFAULT_STORAGE_ROOT)),
            "page_size": int(self._config.get("page_size", DEFAULT_PAGE_SIZE)),
            "nlist": int(self._config.get("nlist", DEFAULT_NLIST)),
            "buffer_capacity": int(self._config.get("buffer_capacity", DEFAULT_BUFFER_CAPACITY)),
            "profiles": dict(self._config.get("profiles", {})),
        }

    def configure(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        unknown = set(options) - {"metric", "dimension", "profile"}
        if unknown:
            raise QueryExecutionError(f"unsupported {self.name} options: {sorted(unknown)}")
        return dict(options)

    def _profiles(self) -> dict[str, MultimediaProfile]:
        raw = self.default_config()["profiles"]
        profiles: dict[str, MultimediaProfile] = {}
        for profile_id, profile in raw.items():
            if not isinstance(profile, Mapping):
                raise QueryExecutionError(f"profile {profile_id!r} must be a mapping")
            profiles[profile_id] = MultimediaProfile(
                profile_id=profile_id,
                kind=profile.get("kind", "image"),
                root=Path(profile["root"]),
                codebook=Path(profile["codebook"]),
                cache=Path(profile["cache"]) if profile.get("cache") else None,
            )
        return profiles

    def _factory(self) -> OnDiskVectorFactory:
        config = self.default_config()
        key = json.dumps(config, sort_keys=True, default=str)
        factory = _FACTORY_CACHE.get(key)
        if factory is None:
            factory = OnDiskVectorFactory(
                Path(config["storage_root"]),
                self._profiles(),
                page_size=config["page_size"],
                nlist=config["nlist"],
                buffer_capacity=config["buffer_capacity"],
            )
            _FACTORY_CACHE[key] = factory
        return factory

    @property
    def query_vectorizer(self):
        """Shared resolver used by the SQL planner for SIMILAR_TO references."""
        return self._factory().query_vectorizer

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


class IVFPlugin(_VectorPluginBase):
    name = "IVF"


class HNSWPlugin(_VectorPluginBase):
    name = "HNSW"
