"""Column types and variable-length type handling.

Canonical location for the column type enum and the column definition used by
the engine. ``engine.query.ast`` re-exports both to keep the query namespace.
"""

from dataclasses import dataclass
from enum import Enum


class ColumnType(Enum):
    """Supported column types in a table definition."""

    INT = "INT"
    FLOAT = "FLOAT"
    VARCHAR = "VARCHAR"
    TEXT = "TEXT"
    BOOL = "BOOL"


@dataclass(frozen=True)
class ColumnDef:
    """One column definition inside a CREATE TABLE statement."""

    name: str
    type_name: ColumnType
    length: int | None = None
