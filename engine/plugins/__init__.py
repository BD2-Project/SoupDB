"""Plugin contracts and registry for SoupDB extensions.

Native engine features (base column types, storage, B+ tree/hash/R-tree) do not
live here. This package is the boundary for installed extensions such as the
full-text and vector search plugins.
"""

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
