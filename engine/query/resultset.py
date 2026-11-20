"""Internal result shape produced by the query executor."""

from dataclasses import dataclass

from engine.query.evaluator import Schema


@dataclass(frozen=True)
class ResultSet:
    """Columns and rows of a query result plus affected-row count."""

    columns: Schema
    rows: tuple[tuple[object, ...], ...] = ()
    affected: int = 0
