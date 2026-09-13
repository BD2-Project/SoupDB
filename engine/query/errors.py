"""Query processing own exceptions.

These subscribe :class:`engine.common.errors.SoupDBError` so the engine keeps
its "no bare Exception" rule while keeping the query domain decoupled from
``common/`` (frozen zone). Can be moved to ``common/errors.py`` by team
agreement.
"""

from engine.common.errors import SoupDBError


class QueryParseError(SoupDBError):
    """Invalid SQL syntax found while tokenizing or parsing."""
