"""Errors raised by the plugin boundary."""

from engine.common.errors import SoupDBError


class PluginError(SoupDBError):
    """Base class for plugin loading, configuration and registration errors."""


class PluginNotFoundError(PluginError):
    """A required plugin is not installed or not registered."""


class PluginConfigError(PluginError):
    """A plugin configuration or CREATE INDEX option is invalid."""
