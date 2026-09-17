"""Query processing own exceptions.

``QueryParseError`` is defined in :mod:`engine.common.errors` (canonical path
per ``docs/contratos.md``) and re-exported here so the query domain can import
it without a second source of truth.
"""

from engine.common.errors import QueryParseError

__all__ = ["QueryParseError"]
