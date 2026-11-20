"""Plugin contracts and registry for SoupDB extensions."""

from engine.plugins.base import IndexPlugin, MutableIndexPlugin
from engine.plugins.capabilities import PluginCapabilities
from engine.plugins.errors import (
    PluginConfigError,
    PluginError,
    PluginNotFoundError,
)
from engine.plugins.registry import PluginRegistration, PluginRegistry

__all__ = [
    "IndexPlugin",
    "MutableIndexPlugin",
    "PluginCapabilities",
    "PluginConfigError",
    "PluginError",
    "PluginNotFoundError",
    "PluginRegistration",
    "PluginRegistry",
]
