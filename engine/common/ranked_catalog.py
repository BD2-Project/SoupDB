"""Persistent ranked metadata independent of the legacy SysIndexes layout.

Metadata are *not* search indexes. A registry entry is only searchable once an
explicit factory has reloaded a compatible RankedIndex provider.
"""

from dataclasses import dataclass

from engine.common.errors import QueryExecutionError
from engine.common.schema import ColumnDef, ColumnType

RANKED_METADATA_FILE = "SysRankedIndexes"
RANKED_SCHEMA = (
    ColumnDef("index_name", ColumnType.TEXT),
    ColumnDef("table_name", ColumnType.TEXT),
    ColumnDef("column_name", ColumnType.TEXT),
    ColumnDef("algorithm", ColumnType.TEXT),
    ColumnDef("metric", ColumnType.TEXT),
    ColumnDef("dimension", ColumnType.INT),
    ColumnDef("profile_id", ColumnType.TEXT),
    ColumnDef("item_count", ColumnType.INT),
    ColumnDef("exact", ColumnType.BOOL),
    ColumnDef("exhaustive", ColumnType.BOOL),
    ColumnDef("mutable", ColumnType.BOOL),
)


@dataclass(frozen=True, slots=True)
class RankedMetadata:
    name: str
    table: str
    column: str
    algorithm: str
    metric: str | None = None
    dimension: int | None = None
    profile_id: str | None = None
    item_count: int | None = None
    exact: bool = False
    exhaustive: bool = False
    mutable: bool = False

    def to_row(self) -> tuple[object, ...]:
        return (
            self.name,
            self.table,
            self.column,
            self.algorithm,
            self.metric,
            self.dimension,
            self.profile_id,
            self.item_count,
            self.exact,
            self.exhaustive,
            self.mutable,
        )

    @classmethod
    def from_row(cls, row: tuple[object, ...]) -> "RankedMetadata":
        return cls(*row)


def check_ranked_metadata(meta: RankedMetadata, schema: tuple[ColumnDef, ...]) -> None:
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (meta.name, meta.table, meta.column)
    ):
        raise QueryExecutionError("ranked index name, table and column are required")
    if meta.algorithm not in ("INVERTED", "IVF", "HNSW"):
        raise QueryExecutionError(f"unsupported ranked algorithm {meta.algorithm!r}")
    column = next((value for value in schema if value.name == meta.column), None)
    if column is None or column.type_name not in (ColumnType.TEXT, ColumnType.VARCHAR):
        raise QueryExecutionError("ranked index requires a TEXT or VARCHAR column")
    if meta.algorithm == "INVERTED":
        if any(value is not None for value in (meta.metric, meta.dimension, meta.profile_id)):
            raise QueryExecutionError("INVERTED cannot use vector options")
    elif (
        meta.metric not in ("euclidean", "cosine", "inner_product")
        or type(meta.dimension) is not int
        or meta.dimension <= 0
        or not isinstance(meta.profile_id, str)
        or not meta.profile_id.strip()
    ):
        raise QueryExecutionError("vector indexes require metric, dimension and profile")
    if meta.item_count is not None and (type(meta.item_count) is not int or meta.item_count < 0):
        raise QueryExecutionError("item_count must be a non-negative integer")
    for attr in ("exact", "exhaustive", "mutable"):
        if type(getattr(meta, attr)) is not bool:
            raise QueryExecutionError(f"{attr} must be boolean")
