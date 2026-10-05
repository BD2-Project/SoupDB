"""Spatial index over the persistent catalog: registration, maintenance, reopen.

The in-memory tests already pin the operator and the planner; these exercise the
real :class:`engine.common.catalog.Catalog` in a temporary directory, where the
index has to survive ``close``/reopen and coexist with the physical files.
"""

from pathlib import Path

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError
from engine.indexes.rtree import RTree
from engine.query import execute_sql

LUGARES = "CREATE TABLE lugares (nombre TEXT, ubicacion POINT)"
RADIUS = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 4.0"
INDEX = "CREATE INDEX idx_ubic ON lugares (ubicacion) TYPE RTREE"

INSERT = "INSERT INTO lugares VALUES ('{name}', POINT({x}, {y}))"
ROWS = (
    ("A", 0.0, 0.0),
    ("B", 3.0, 4.0),
    ("C", 6.0, 8.0),
    ("D", -1.0, 0.0),
    ("E", 0.0, 4.0),
    ("F", 3.0, 3.0),
)


def make_catalog(tmp_path: Path) -> Catalog:
    return Catalog(tmp_path, page_size=256, buffer_capacity=4)


def seed(catalog: Catalog, rows=ROWS) -> None:
    execute_sql(LUGARES, catalog)
    for name, x, y in rows:
        execute_sql(INSERT.format(name=name, x=x, y=y), catalog)


def names(catalog: Catalog, sql: str = RADIUS) -> list[str]:
    return sorted(row[0] for row in execute_sql(sql, catalog).rows)


def index_files(tmp_path: Path) -> list[str]:
    return sorted(path.name for path in tmp_path.iterdir() if path.name.startswith("ix_"))


# --- registro -------------------------------------------------------------


def test_create_spatial_index_registers_the_rtree(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    index = catalog.indexes("lugares")["idx_ubic"]
    assert isinstance(index, RTree)
    rows = execute_sql(
        "SELECT index_type FROM SysIndexes WHERE index_name = 'idx_ubic'", catalog
    ).rows
    assert rows == (("RTREE",),)
    catalog.close()


def test_indexes_for_returns_the_spatial_index(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    assert set(catalog.indexes_for("lugares", "ubicacion")) == {"idx_ubic"}
    catalog.close()


def test_spatial_index_leaves_no_physical_file(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    # El R-Tree vive en memoria y se reconstruye al abrir: no crea un archivo
    # ix_*.db como las estructuras paginadas.
    assert index_files(tmp_path) == []
    catalog.close()
    assert index_files(tmp_path) == []


def test_explain_reports_the_spatial_index(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    lines = execute_sql(f"EXPLAIN {RADIUS}", catalog).rows
    text = "\n".join(line[0] for line in lines)
    assert "SpatialIndexScan" in text
    assert "'index': 'idx_ubic'" in text
    assert "'center': (0.0, 0.0)" in text
    assert "'radius': 4.0" in text
    assert "'metric': 'euclidean'" in text
    catalog.close()


# --- resultados -----------------------------------------------------------


def test_index_and_full_scan_return_the_same_rows(tmp_path: Path) -> None:
    indexed = make_catalog(tmp_path / "con")
    seed(indexed)
    execute_sql(INDEX, indexed)
    scanned = make_catalog(tmp_path / "sin")
    seed(scanned)
    assert names(indexed) == names(scanned) == ["A", "D"]
    indexed.close()
    scanned.close()


def test_point_inside_the_box_but_outside_the_circle_is_filtered(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    # F (3, 3) cae en la caja [-4, 4]² pero a distancia 4.24.
    assert "F" not in names(catalog)
    catalog.close()


# --- mantenimiento --------------------------------------------------------


def test_insert_after_create_index_is_found_by_radius(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    execute_sql(INSERT.format(name="N", x=1.0, y=0.0), catalog)
    assert "N" in names(catalog)
    catalog.close()


def test_delete_removes_the_row_from_the_spatial_index(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    assert "A" in names(catalog)
    execute_sql("DELETE FROM lugares WHERE nombre = 'A'", catalog)
    assert "A" not in names(catalog)
    assert names(catalog) == ["D"]
    catalog.close()


def test_insert_null_point_with_index_does_not_crash(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    # Un POINT NULL no es una ubicación: se guarda en la tabla y se omite del
    # índice, sin romper el INSERT.
    execute_sql("INSERT INTO lugares VALUES ('sin punto', NULL)", catalog)
    assert names(catalog) == ["A", "D"]
    catalog.close()


def test_delete_null_point_with_index_does_not_crash(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    execute_sql("INSERT INTO lugares VALUES ('sin punto', NULL)", catalog)
    execute_sql("DELETE FROM lugares WHERE nombre = 'sin punto'", catalog)
    assert names(catalog) == ["A", "D"]
    catalog.close()


# --- reapertura -----------------------------------------------------------


def test_reopen_rebuilds_the_index_from_the_table(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    assert names(catalog) == ["A", "D"]
    catalog.close()

    reopened = make_catalog(tmp_path)
    assert "idx_ubic" in reopened.indexes("lugares")
    # La consulta con índice tras reabrir debe dar lo mismo que la tabla: un
    # índice vacío devolvería menos filas.
    assert names(reopened) == ["A", "D"]
    lines = execute_sql(f"EXPLAIN {RADIUS}", reopened).rows
    assert any("SpatialIndexScan" in line[0] for line in lines)
    reopened.close()


def test_reopen_after_insert_keeps_the_new_point_indexed(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    execute_sql(INSERT.format(name="N", x=1.0, y=0.0), catalog)
    catalog.close()

    reopened = make_catalog(tmp_path)
    assert "N" in names(reopened)
    reopened.close()


def test_reopen_after_delete_does_not_resurrect_the_row(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    execute_sql("DELETE FROM lugares WHERE nombre = 'A'", catalog)
    catalog.close()

    reopened = make_catalog(tmp_path)
    assert names(reopened) == ["D"]
    reopened.close()


# --- DROP -----------------------------------------------------------------


def test_drop_index_leaves_the_table_and_scan_working(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    execute_sql("DROP INDEX idx_ubic", catalog)
    assert catalog.indexes("lugares") == {}
    assert index_files(tmp_path) == []
    assert names(catalog) == ["A", "D"]
    lines = execute_sql(f"EXPLAIN {RADIUS}", catalog).rows
    assert not any("SpatialIndexScan" in line[0] for line in lines)
    catalog.close()


def test_drop_index_survives_reopen(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    execute_sql("DROP INDEX idx_ubic", catalog)
    catalog.close()

    reopened = make_catalog(tmp_path)
    assert reopened.indexes("lugares") == {}
    assert names(reopened) == ["A", "D"]
    reopened.close()


def test_drop_table_cleans_the_spatial_index(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    seed(catalog)
    execute_sql(INDEX, catalog)
    execute_sql("DROP TABLE lugares", catalog)
    assert catalog.tables().count("lugares") == 0
    assert execute_sql("SELECT index_name FROM SysIndexes", catalog).rows == ()
    assert index_files(tmp_path) == []
    catalog.close()

    reopened = make_catalog(tmp_path)
    assert "lugares" not in reopened.tables()
    reopened.close()


# --- errores --------------------------------------------------------------


def test_spatial_index_on_non_point_column_fails_via_sql(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql(LUGARES, catalog)
    with pytest.raises(QueryExecutionError, match="requires a POINT column"):
        execute_sql("CREATE INDEX idx_nombre ON lugares (nombre) TYPE RTREE", catalog)
    catalog.close()


def test_spatial_index_on_non_point_column_fails_via_catalog(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql(LUGARES, catalog)
    with pytest.raises(QueryExecutionError, match="requires a POINT column"):
        catalog.create_index("idx_nombre", "lugares", "nombre", index_type="RTREE")
    catalog.close()


def test_spatial_index_on_unknown_column_fails(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql(LUGARES, catalog)
    with pytest.raises(QueryExecutionError, match="unknown column"):
        execute_sql("CREATE INDEX idx ON lugares (nope) TYPE RTREE", catalog)
    catalog.close()
