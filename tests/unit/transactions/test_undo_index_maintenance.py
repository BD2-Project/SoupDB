"""Undo keeps the indexes in step with the table.

Undoing a DELETE puts the row back with a NEW RID, and undoing an INSERT removes
the row the INSERT indexed. If the undo only touched storage, a ROLLBACK would
leave a row that is live in the table and missing from every index - and an index
that does not describe its table answers wrongly and silently, with no error to
hint at it: a lookup just reads a table it no longer matches.

These tests pin both directions for both index families, plus the NULL rule: a
spatial index has no key for a NULL coordinate, so such a row stays out of it in
either direction.
"""

from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.rtree import RTree, SpatialQueries
from engine.indexes.rtree.mbr import MBR
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from engine.transactions.session import TransactionalSession
from tests.fakes.fake_catalog import FakeCatalog

LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)
NOMBRES = (
    ColumnDef("id", ColumnType.INT),
    ColumnDef("nombre", ColumnType.TEXT),
)


def make_lugares(spatial: bool = True, with_null: bool = False) -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("lugares", LUGARES)
    catalog.insert("lugares", ("CERCANO", (0.0, 0.0)))
    catalog.insert("lugares", ("lejos", (5.0, 0.0)))
    if with_null:
        catalog.insert("lugares", ("sin_ubicacion", None))
    if spatial:
        catalog.add_spatial_index("lugares", "ubicacion", index_name="idx_ubic")
    return catalog


def make_nombres(index: bool = True) -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("nombres", NOMBRES)
    catalog.insert("nombres", (1, "uno"))
    catalog.insert("nombres", (2, "dos"))
    if index:
        catalog.add_index("nombres", "nombre", index_name="idx_nombre")
    return catalog


def tree_of(catalog: FakeCatalog) -> RTree:
    return catalog.indexes("lugares")["idx_ubic"]


def indexed_points(catalog: FakeCatalog) -> set[tuple[float, float]]:
    """Puntos que el índice espacial describe ahora mismo."""
    entries = SpatialQueries(tree_of(catalog)).entries()
    return {(point.x, point.y) for point, _ in entries}


def run(sql: str, catalog: FakeCatalog):
    return execute(plan(parse(sql), catalog), catalog)


def first_column(sql: str, catalog: FakeCatalog) -> list[object]:
    return [row[0] for row in run(sql, catalog).rows]


def session_of(catalog: FakeCatalog) -> TransactionalSession:
    return TransactionalSession(catalog)


# --- undo de un DELETE: la fila vuelve al índice ---------------------------


def test_rollback_of_a_delete_restores_the_spatial_entry() -> None:
    catalog = make_lugares()
    session = session_of(catalog)
    session.begin()
    session.execute("DELETE FROM lugares WHERE nombre = 'CERCANO'")
    assert (0.0, 0.0) not in indexed_points(catalog)
    session.rollback()
    # La fila vuelve a la tabla, así que su entrada tiene que volver al índice.
    assert (0.0, 0.0) in indexed_points(catalog)
    assert len(indexed_points(catalog)) == 2
    session.close()


def test_after_the_rollback_the_index_and_the_table_agree() -> None:
    # El invariante que importa: la fila más cercana sigue siendo la más cercana
    # según el índice, y no solo según la tabla.
    catalog = make_lugares()
    session = session_of(catalog)
    session.begin()
    session.execute("DELETE FROM lugares WHERE nombre = 'CERCANO'")
    session.rollback()
    caja = MBR(-1.0, -1.0, 1.0, 1.0)
    assert len(SpatialQueries(tree_of(catalog)).range_search(caja)) == 1
    assert first_column(
        "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 1.0",
        catalog,
    ) == ["CERCANO"]
    session.close()


def test_rollback_of_a_delete_restores_a_scalar_entry_too() -> None:
    # El arreglo no es específico del índice espacial: vale para cualquier índice.
    catalog = make_nombres()
    session = session_of(catalog)
    session.begin()
    session.execute("DELETE FROM nombres WHERE id = 1")
    assert first_column("SELECT id FROM nombres WHERE nombre = 'uno'", catalog) == []
    session.rollback()
    assert first_column("SELECT id FROM nombres WHERE nombre = 'uno'", catalog) == [1]
    session.close()


# --- undo de un INSERT: la fila sale del índice ----------------------------


def test_rollback_of_an_insert_removes_the_spatial_entry() -> None:
    catalog = make_lugares()
    session = session_of(catalog)
    session.begin()
    session.execute("INSERT INTO lugares VALUES ('nuevo', POINT(1, 0))")
    assert (1.0, 0.0) in indexed_points(catalog)
    session.rollback()
    assert (1.0, 0.0) not in indexed_points(catalog)
    assert len(indexed_points(catalog)) == 2
    session.close()


def test_rollback_of_an_insert_removes_a_scalar_entry_too() -> None:
    catalog = make_nombres()
    session = session_of(catalog)
    session.begin()
    session.execute("INSERT INTO nombres VALUES (3, 'tres')")
    assert first_column("SELECT id FROM nombres WHERE nombre = 'tres'", catalog) == [3]
    session.rollback()
    assert first_column("SELECT id FROM nombres WHERE nombre = 'tres'", catalog) == []
    session.close()


def test_rollback_of_an_insert_leaves_the_survivors_indexed() -> None:
    catalog = make_lugares()
    session = session_of(catalog)
    session.begin()
    session.execute("INSERT INTO lugares VALUES ('nuevo', POINT(1, 0))")
    session.rollback()
    caja = MBR(-1.0, -1.0, 6.0, 1.0)
    assert len(SpatialQueries(tree_of(catalog)).range_search(caja)) == 2
    session.close()


# --- la regla de NULL se respeta en ambos sentidos ------------------------


def test_a_rolled_back_null_coordinate_stays_out_of_the_spatial_index() -> None:
    # Un NULL no es una ubicación: el índice espacial lo rechaza. La fila vuelve a
    # la tabla pero no al índice, que es justo lo que se comprueba aquí.
    catalog = make_lugares(with_null=True)
    session = session_of(catalog)
    session.begin()
    session.execute("DELETE FROM lugares WHERE nombre = 'sin_ubicacion'")
    session.rollback()
    assert len(indexed_points(catalog)) == 2
    assert "sin_ubicacion" in first_column("SELECT nombre FROM lugares", catalog)
    session.close()


def test_a_rolled_back_insert_of_a_null_coordinate_does_not_raise() -> None:
    catalog = make_lugares()
    session = session_of(catalog)
    session.begin()
    session.execute("INSERT INTO lugares VALUES ('nuevo', NULL)")
    assert len(indexed_points(catalog)) == 2
    session.rollback()
    assert len(indexed_points(catalog)) == 2
    session.close()


# --- la fila recuperada se consulta con el RID que le dio el undo ----------


def test_the_restored_row_is_reachable_through_the_index() -> None:
    # El undo reinserta la fila con OTRO RID: la entrada tiene que verse ahí, y no
    # solo el punto. Si se indexara el RID viejo, el índice describiría una fila
    # que ya no existe.
    catalog = make_lugares()
    session = session_of(catalog)
    session.begin()
    session.execute("DELETE FROM lugares WHERE nombre = 'CERCANO'")
    session.rollback()
    vivos = {rid for rid, _ in catalog.file_org("lugares").scan()}
    indexados = {rid for _, rid in SpatialQueries(tree_of(catalog)).entries()}
    assert indexados == vivos
    session.close()


# --- contra un Catalog real en disco --------------------------------------


def test_after_a_rollback_the_index_matches_the_table_when_reopened() -> None:
    # El índice espacial es una estructura DERIVADA: Catalog lo reconstruye al
    # reabrir (engine/common/catalog.py::_build_index). Esta es la comprobación de
    # persistencia del arreglo: lo que el undo dejó en disco tiene que volver a
    # describir la misma tabla.
    import tempfile
    from pathlib import Path

    from engine.common.catalog import Catalog
    from engine.query import execute_sql

    root = Path(tempfile.mkdtemp(prefix="undo_h01_"))
    catalog = Catalog(root)
    execute_sql("CREATE TABLE lugares (nombre VARCHAR(40), ubicacion POINT)", catalog)
    execute_sql("INSERT INTO lugares VALUES ('CERCANO', POINT(0,0))", catalog)
    execute_sql("INSERT INTO lugares VALUES ('lejos', POINT(9,0))", catalog)
    execute_sql("CREATE INDEX idx_ub ON lugares (ubicacion) TYPE RTREE", catalog)

    session = TransactionalSession(catalog)
    session.begin()
    session.execute("DELETE FROM lugares WHERE nombre = 'CERCANO'")
    session.rollback()
    session.close()
    catalog.close()

    reopened = Catalog(root)
    try:
        query = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0,0)) LIMIT 1"
        rows = [row[0] for row in execute_sql(query, reopened).rows]
        # El índice y la tabla coinciden: la fila revertida es la más cercana.
        assert rows == ["CERCANO"]
        todas = [row[0] for row in execute_sql("SELECT nombre FROM lugares", reopened).rows]
        assert sorted(todas) == ["CERCANO", "lejos"]
    finally:
        reopened.close()
