"""Explicit capability declarations for plugins."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PluginCapabilities:
    """Capabilities a plugin declares to the engine.

    ``operators`` names the query operators the plugin can serve (for example
    ``{"search"}``). ``index_options`` lists the option names accepted by
    ``CREATE INDEX ... WITH (...)`` and by the search clause.
    """

    exact: bool = False
    exhaustive: bool = False
    approximate: bool = False
    mutable: bool = False
    reversible: bool = False
    query_kind: str = "text"
    operators: frozenset[str] = field(default_factory=lambda: frozenset({"search"}))
    index_options: frozenset[str] = field(default_factory=frozenset)
