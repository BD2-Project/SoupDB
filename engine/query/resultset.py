"""Internal result shape produced by the query executor.

Temporary in-memory shape: columns plus typed rows for ``SELECT`` and an
affected-row counter for ``INSERT``/``DELETE``/``CREATE TABLE``. The wire
format towards ``rsoup``/the frontend is agreed with the Transactions domain
and only affects the serialization of this value.
"""

from dataclasses import dataclass

from engine.query.evaluator import Schema


@dataclass(frozen=True)
class ResultSet:
    """Columns and rows of a query result plus affected-row count."""

    columns: Schema
    rows: tuple[tuple[object, ...], ...] = ()
    affected: int = 0
