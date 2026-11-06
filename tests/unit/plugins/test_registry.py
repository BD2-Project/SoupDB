from types import SimpleNamespace

import pytest

from engine.common.ranked_catalog import RankedMetadata
from engine.indexes.ranked_factory import DelegatingRankedFactory
from engine.plugins import (
    PluginCapabilities,
    PluginConfigError,
    PluginError,
    PluginNotFoundError,
    PluginRegistry,
)


class FakePlugin:
    name = "DUMMY"
    version = "1.2.3"
    api_version = 1

    def __init__(self):
        self.calls = []

    def capabilities(self):
        return PluginCapabilities(exact=True, exhaustive=True)

    def default_config(self):
        return {"root": "/tmp/dummy"}

    def validate_options(self, options):
        return dict(options)

    def build(self, metadata, file_org, schema):
        self.calls.append(("build", metadata.name))
        return "built"

    def load(self, metadata, file_org, schema):
        self.calls.append(("load", metadata.name))
        return "loaded"

    def drop(self, metadata):
        self.calls.append(("drop", metadata.name))

    def search(self, binding, query, k, options):
        return []


class FakeIndex:
    def __init__(self):
        self.calls = []

    def top_k(self, query, k, **options):
        self.calls.append((query, k, options))
        return []


def _metadata(algorithm="DUMMY"):
    return RankedMetadata("ix", "t", "c", algorithm)


def test_register_get_and_available():
    registry = PluginRegistry()
    plugin = FakePlugin()
    registration = registry.register(plugin)
    assert registration.config == {"root": "/tmp/dummy"}
    assert registry.plugin("dummy") is plugin
    assert registry.available() == ("DUMMY",)
    assert "DUMMY" in registry and "other" not in registry
    assert len(registry) == 1


def test_duplicate_and_unknown_plugins_fail_clearly():
    registry = PluginRegistry()
    registry.register(FakePlugin())
    with pytest.raises(PluginError):
        registry.register(FakePlugin())
    registry.register(FakePlugin(), replace=True)
    with pytest.raises(PluginNotFoundError):
        registry.get("missing")
    with pytest.raises(PluginNotFoundError):
        registry.unregister("missing")


def test_api_version_is_enforced():
    registry = PluginRegistry()
    plugin = FakePlugin()
    plugin.api_version = 99
    with pytest.raises(PluginConfigError, match="api_version"):
        registry.register(plugin)


def test_factory_adapter_delegates_lifecycle_and_search():
    class Factory:
        query_vectorizer = None

        def __init__(self):
            self.calls = []

        def build(self, metadata, file_org, schema):
            self.calls.append("build")
            return SimpleNamespace(index=FakeIndex())

        def load(self, metadata, file_org, schema):
            self.calls.append("load")
            return SimpleNamespace(index=FakeIndex())

        def drop(self, metadata):
            self.calls.append("drop")

    registry = PluginRegistry()
    factory = Factory()
    registry.register_factory("dummy", factory)
    binding = registry.build(_metadata(), object(), ())
    assert factory.calls == ["build"]
    registry.load(_metadata(), object(), ())
    registry.drop(_metadata())
    assert factory.calls == ["build", "load", "drop"]
    plugin = registry.plugin("DUMMY")
    assert plugin.search(binding, "query", 5, {"method": "bm25"}) == []
    assert binding.index.calls == [("query", 5, {"method": "bm25"})]


def test_delegating_factory_from_registry_accepts_new_plugin_names():
    registry = PluginRegistry()
    registry.register(FakePlugin())
    factory = DelegatingRankedFactory.from_registry(registry)
    assert factory.build(_metadata(), object(), ()) == "built"
    assert factory.load(_metadata(), object(), ()) == "loaded"
    factory.drop(_metadata())
    with pytest.raises(Exception, match="no ranked provider"):
        factory.load(_metadata("MISSING"), object(), ())


def test_discovery_uses_entry_points(monkeypatch):
    class EntryPoint:
        name = "dummy"

        def load(self):
            return FakePlugin()

    monkeypatch.setattr(
        "engine.plugins.registry.entry_points", lambda group: [EntryPoint()]
    )
    registry = PluginRegistry()
    assert registry.discover() == ("DUMMY",)
    assert registry.available() == ("DUMMY",)
