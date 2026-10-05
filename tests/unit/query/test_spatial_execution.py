"""End-to-end spatial queries: POINT columns, INSERT POINT(), WHERE distance
and ORDER BY distance ... LIMIT through the full parse -> plan -> execute path.
"""

from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog

LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)


def make_catalog() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("lugares", LUGARES)
    for nombre, point in (("A", (0.0, 0.0)), ("B", (3.0, 4.0)), ("C", (6.0, 8.0))):
        catalog.insert("lugares", (nombre, point))
    return catalog


def run(sql: str, catalog: FakeCatalog):
    return execute(plan(parse(sql), catalog), catalog)


def test_insert_point_value_roundtrips() -> None:
    catalog = make_catalog()
    result = run("INSERT INTO lugares (nombre, ubicacion) VALUES ('D', POINT(1, 2))", catalog)
    assert result.affected == 1
    rows = [row for _, row in catalog.file_org("lugares").scan()]
    from engine.common.record import decode_row

    decoded = [decode_row(record.data, LUGARES) for record in rows]
    # POINT(1, 2) es latitud 1 y longitud 2: se guarda como Point(x=2, y=1).
    assert decoded[-1] == ("D", (2.0, 1.0))


def test_where_distance_filters_by_radius() -> None:
    catalog = make_catalog()
    result = run(
        "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 6.0",
        catalog,
    )
    assert result.rows == (("A",), ("B",))


def test_where_distance_accepts_int_literal() -> None:
    # El radio se escribe `< 6`, no `< 6.0`: el literal entero se promueve.
    catalog = make_catalog()
    entero = run("SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 6", catalog)
    flotante = run(
        "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 6.0", catalog
    )
    assert entero.rows == flotante.rows


def test_order_by_distance_limit_returns_knn() -> None:
    catalog = make_catalog()
    result = run(
        "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 2",
        catalog,
    )
    assert result.rows == (("A",), ("B",))


def test_order_by_distance_desc_limit() -> None:
    catalog = make_catalog()
    result = run(
        "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) DESC LIMIT 1",
        catalog,
    )
    assert result.rows == (("C",),)


def test_select_project_distance_column_type() -> None:
    catalog = make_catalog()
    result = run(
        "SELECT nombre, distance(ubicacion, POINT(0, 0)) FROM lugares LIMIT 1",
        catalog,
    )
    assert [column.name for column in result.columns] == ["nombre", "column_2"]
    assert result.columns[1].type_name is ColumnType.FLOAT


def test_limit_offset_e2e() -> None:
    catalog = FakeCatalog()
    catalog.create_table("t", (ColumnDef("id", ColumnType.INT),))
    for i in range(5):
        catalog.insert("t", (i,))
    assert run("SELECT id FROM t LIMIT 3", catalog).rows == ((0,), (1,), (2,))
    assert run("SELECT id FROM t LIMIT 3 OFFSET 1", catalog).rows == ((1,), (2,), (3,))
    assert run("SELECT id FROM t LIMIT 0", catalog).rows == ()
