"""In-memory registry and entry-point discovery for engine plugins."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

from engine.common.ranked import RankedBinding
from engine.common.ranked_catalog import RankedMetadata
from engine.common.schema import ColumnDef
from engine.plugins.base import API_VERSION, IndexPlugin
from engine.plugins.capabilities import PluginCapabilities
from engine.plugins.errors import (
    PluginConfigError,
    PluginError,
    PluginNotFoundError,
)
from engine.storage.base import FileOrganization

_LOGGER = logging.getLogger(__name__)

DEFAULT_ENTRY_POINT_GROUP = "soupdb.plugins"


def _normalize(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise PluginConfigError("plugin name must be a non-empty string")
    return name.strip().upper()


@dataclass(frozen=True, slots=True)
class PluginRegistration:
    """A plugin plus the configuration persisted for reloads."""

    plugin: IndexPlugin
    config: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    source: str = "manual"


class FactoryPluginAdapter:
    """Adapt a legacy ``RankedFactory`` object to the plugin protocol.

    Keeps the previous factory boundary working while the repository migrates
    to plugins. Capabilities are read from the factory when declared and
    otherwise default to conservative (non-mutable) values.
    """

    def __init__(
        self,
        name: str,
        factory: object,
        *,
        version: str = "0.0.0",
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = _normalize(name)
        self.version = version
        self.api_version = API_VERSION
        self._factory = factory
        self._config = dict(config or {})

    def factory(self) -> object:
        return self._factory

    def capabilities(self) -> PluginCapabilities:
        declared = getattr(self._factory, "capabilities", None)
        if isinstance(declared, PluginCapabilities):
            return declared
        if callable(declared):
            return declared()
        return PluginCapabilities(
            mutable=getattr(self._factory, "mutable", False) is True,
            reversible=getattr(self._factory, "reversible", False) is True,
        )

    def default_config(self) -> dict[str, Any]:
        return dict(self._config)

    def validate_options(self, options: Mapping[str, Any]) -> dict[str, Any]:
        return dict(options)

    def build(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        return self._factory.build(metadata, file_org, schema)

    def load(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        return self._factory.load(metadata, file_org, schema)

    def drop(self, metadata: RankedMetadata) -> None:
        drop = getattr(self._factory, "drop", None)
        if callable(drop):
            drop(metadata)

    def search(
        self,
        binding: RankedBinding,
        query: Any,
        k: int,
        options: Mapping[str, Any],
    ) -> list[tuple[Any, float]]:
        return binding.index.top_k(query, k, **dict(options))


class PluginRegistry:
    """Registry of plugins available to this engine process.

    Registration is explicit for tests and embeds; :meth:`discover` adds
    plugins installed through ``soupdb.plugins`` entry points. Persistence of
    plugin configuration across database reopen is handled by the catalog in a
    later phase, not here.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, PluginRegistration] = {}

    def register(
        self,
        plugin: IndexPlugin,
        config: Mapping[str, Any] | None = None,
        *,
        replace: bool = False,
        source: str = "manual",
    ) -> PluginRegistration:
        name = _normalize(getattr(plugin, "name", None))
        if getattr(plugin, "api_version", None) != API_VERSION:
            raise PluginConfigError(
                f"plugin {name!r} declares api_version "
                f"{getattr(plugin, 'api_version', None)!r}; core supports {API_VERSION}"
            )
        if name in self._registrations and not replace:
            raise PluginError(f"plugin {name!r} is already registered")
        registration = PluginRegistration(
            plugin=plugin,
            config=dict(config or plugin.default_config()),
            enabled=True,
            source=source,
        )
        self._registrations[name] = registration
        return registration

    def register_factory(
        self,
        name: str,
        factory: object,
        *,
        version: str = "0.0.0",
        config: Mapping[str, Any] | None = None,
        replace: bool = False,
        source: str = "factory",
    ) -> PluginRegistration:
        adapter = FactoryPluginAdapter(name, factory, version=version, config=config)
        return self.register(adapter, config, replace=replace, source=source)

    def unregister(self, name: str) -> None:
        normalized = _normalize(name)
        if normalized not in self._registrations:
            raise PluginNotFoundError(f"plugin {normalized!r} is not registered")
        del self._registrations[normalized]

    def get(self, name: str) -> PluginRegistration:
        normalized = _normalize(name)
        try:
            return self._registrations[normalized]
        except KeyError:
            raise PluginNotFoundError(f"plugin {normalized!r} is not installed") from None

    def plugin(self, name: str) -> IndexPlugin:
        return self.get(name).plugin

    def config_for(self, name: str) -> Mapping[str, Any]:
        return dict(self.get(name).config)

    def available(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def enabled(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, reg in self._registrations.items() if reg.enabled))

    def __contains__(self, name: object) -> bool:
        try:
            return _normalize(name) in self._registrations
        except PluginConfigError:
            return False

    def __len__(self) -> int:
        return len(self._registrations)

    def discover(self, group: str = DEFAULT_ENTRY_POINT_GROUP) -> tuple[str, ...]:
        """Load plugins declared through package entry points.

        A discovered entry point may resolve to a plugin instance, a plugin
        class, or a zero-argument factory returning a plugin instance.
        """
        discovered: list[str] = []
        for entry_point in entry_points(group=group):
            try:
                loaded = entry_point.load()
                plugin = _instantiate(loaded)
                self.register(plugin, source=f"entry-point:{entry_point.name}")
                discovered.append(_normalize(plugin.name))
            except PluginError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise PluginError(
                    f"cannot load plugin entry point {entry_point.name!r}: {exc}"
                ) from exc
        _LOGGER.debug("discovered plugins: %s", discovered)
        return tuple(discovered)

    def build(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        return self.plugin(metadata.algorithm).build(metadata, file_org, schema)

    def load(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        return self.plugin(metadata.algorithm).load(metadata, file_org, schema)

    def drop(self, metadata: RankedMetadata) -> None:
        self.plugin(metadata.algorithm).drop(metadata)


def _instantiate(candidate: object) -> IndexPlugin:
    if isinstance(candidate, type):
        candidate = candidate()
    if callable(candidate) and not hasattr(candidate, "build"):
        candidate = candidate()
    if not hasattr(candidate, "build") or not hasattr(candidate, "load"):
        raise PluginConfigError("plugin entry point did not return an index plugin")
    if not hasattr(candidate, "name") or not hasattr(candidate, "api_version"):
        raise PluginConfigError("plugin must declare name and api_version")
    return candidate  # type: ignore[return-value]
