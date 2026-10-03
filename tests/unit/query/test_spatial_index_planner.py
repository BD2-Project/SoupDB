"""Planner and executor tests for the spatial-index access path.

The planner must swap the full scan for a ``SpatialIndexScan`` only when it can
prove the bounding box contains every row of the radius, and must leave every
other predicate exactly as it was. The dataset is small enough to know the
answer by hand and has a point (3, 3) inside the box but outside the circle, so
a wrong box or a missing exact filter is visible.
"""

import random

from engine.common.record import decode_row
from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog

LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)
EXTRA = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
    ColumnDef("otra", ColumnType.POINT),
)

# (0, 0) y (-1, 0) están dentro del radio 4; (3, 3) está dentro de la caja
# [-4, 4]² pero fuera del círculo (distancia 4.24); (0, 4) cae justo en el
# borde de la caja y del círculo.
ROWS = (
    ("A", (0.0, 0.0)),
    ("B", (3.0, 4.0)),
    ("C", (6.0, 8.0)),
    ("D", (-1.0, 0.0)),
    ("E", (0.0, 4.0)),
    ("F", (3.0, 3.0)),
)

RADIUS_SQL = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 4.0"
MATCHES = ["A", "D"]


def make_catalog(index: bool = True, rows=ROWS, columns=LUGARES) -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("lugares", columns)
    for row in rows:
        catalog.insert("lugares", row)
    if index:
        catalog.add_spatial_index("lugares", "ubicacion", index_name="idx_ubic")
    return catalog


def names(result) -> list[str]:
    return sorted(row[0] for row in result.rows)


def run(sql: str, catalog: FakeCatalog):
    return execute(plan(parse(sql), catalog), catalog)


def find(node, op: str):
    if node.op == op:
        return node
    for child in node.children:
        found = find(child, op)
        if found is not None:
            return found
    return None


def drain(plan_tree) -> list[tuple[object, ...]]:
    plan_tree.root.open()
    try:
        rows = []
        while True:
            record = plan_tree.root.next()
            if record is None:
                break
            rows.append(decode_row(record.data, plan_tree.root.schema))
    finally:
        plan_tree.root.close()
    return rows


# --- caso feliz -----------------------------------------------------------


def test_index_path_returns_exactly_the_full_scan_rows() -> None:
    with_index = run(RADIUS_SQL, make_catalog(index=True))
    without_index = run(RADIUS_SQL, make_catalog(index=False))
    assert names(with_index) == names(without_index) == MATCHES


def test_explain_shows_the_spatial_index_and_its_parameters() -> None:
    plan_tree = plan(parse(RADIUS_SQL), make_catalog(index=True))
    node = find(plan_tree.root.explain(), "SpatialIndexScan")
    assert node is not None
    assert node.detail == {
        "index": "idx_ubic",
        "column": "ubicacion",
        "center": (0.0, 0.0),
        "radius": 4.0,
        "metric": "euclidean",
    }


def test_without_index_the_plan_stays_a_full_scan() -> None:
    plan_tree = plan(parse(RADIUS_SQL), make_catalog(index=False))
    root = plan_tree.root.explain()
    assert find(root, "TableScan") is not None
    assert find(root, "SpatialIndexScan") is None


def test_index_node_is_a_leaf_below_the_filter() -> None:
    plan_tree = plan(parse(RADIUS_SQL), make_catalog(index=True))
    root = plan_tree.root.explain()
    filter_node = find(root, "Filter")
    assert filter_node is not None
    assert [child.op for child in filter_node.children] == ["SpatialIndexScan"]


def test_box_candidates_are_more_than_the_circle_matches() -> None:
    # El índice devuelve 5 candidatos de la caja y el filtro exacto deja 2: la
    # diferencia es el trabajo que la caja no recorta y el predicado resuelve.
    plan_tree = plan(parse(RADIUS_SQL), make_catalog(index=True))
    drain(plan_tree)
    root = plan_tree.root.explain()
    assert find(root, "SpatialIndexScan").rows == 5
    assert find(root, "Filter").rows == 2


def test_explain_analyze_renders_the_spatial_scan() -> None:
    result = run(f"EXPLAIN ANALYZE {RADIUS_SQL}", make_catalog(index=True))
    text = "\n".join(row[0] for row in result.rows)
    assert "SpatialIndexScan" in text
    assert "'index': 'idx_ubic'" in text


# --- casos borde ----------------------------------------------------------


def test_radius_zero_matches_only_the_same_point() -> None:
    catalog = make_catalog(index=True)
    empty = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 0.0"
    exact = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 1.0"
    assert run(empty, catalog).rows == ()
    assert names(run(exact, catalog)) == ["A"]


def test_point_on_the_box_border_is_not_lost() -> None:
    catalog = make_catalog(index=True)
    # (0, 4) está en el borde de la caja [-4, 4]² y a distancia exactamente 4.
    strict = run("SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 4.0", catalog)
    inclusive = run(
        "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) <= 4.0", catalog
    )
    assert "E" not in names(strict)
    assert "E" in names(inclusive)
    assert names(inclusive) == names(
        run(
            "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) <= 4.0",
            make_catalog(index=False),
        )
    )


def test_lte_uses_the_index_with_the_same_box() -> None:
    plan_tree = plan(
        parse("SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) <= 4.0"),
        make_catalog(index=True),
    )
    node = find(plan_tree.root.explain(), "SpatialIndexScan")
    assert node is not None
    assert node.detail["radius"] == 4.0


def test_negative_center_and_negative_coordinates() -> None:
    rows = (
        ("A", (-3.0, -4.0)),
        ("B", (-1.0, -2.0)),
        ("C", (5.0, 5.0)),
    )
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(-1, -2)) < 2.0"
    assert names(run(sql, make_catalog(index=True, rows=rows))) == ["B"]
    assert names(run(sql, make_catalog(index=False, rows=rows))) == ["B"]


def test_duplicate_points_are_returned_once_per_row() -> None:
    rows = (
        ("A", (1.0, 1.0)),
        ("B", (1.0, 1.0)),
        ("C", (9.0, 9.0)),
    )
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(1, 1)) < 0.5"
    assert names(run(sql, make_catalog(index=True, rows=rows))) == ["A", "B"]
    assert names(run(sql, make_catalog(index=False, rows=rows))) == ["A", "B"]


def test_null_point_is_not_indexed_and_does_not_break_the_build() -> None:
    rows = (
        ("A", (0.0, 0.0)),
        ("N", None),
    )
    catalog = make_catalog(index=True, rows=rows)
    # El índice no puede guardar un NULL (no es una ubicación) y la fila sin
    # punto no es candidata de un radio; el INSERT tampoco debe fallar.
    catalog.insert("lugares", ("N2", None))
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 1.0"
    assert names(run(sql, catalog)) == ["A"]


def test_limit_and_offset_combine_with_the_index() -> None:
    base = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < 6.0"
    sql = f"{base} LIMIT 2 OFFSET 1"
    indexed = run(sql, make_catalog(index=True))
    scanned = run(sql, make_catalog(index=False))
    assert len(indexed.rows) == 2
    assert len(scanned.rows) == 2
    # Sin ORDER BY el orden no está garantizado, así que ambos caminos pueden
    # elegir un subconjunto distinto del mismo conjunto de coincidencias; lo que
    # sí se fija es el tamaño y que ninguna fila de fuera del radio se cuele.
    matches = set(names(run(base, make_catalog(index=False))))
    assert set(names(indexed)) <= matches
    assert set(names(scanned)) <= matches


# --- la métrica y la forma del predicado deciden el camino ----------------


def test_haversine_falls_back_to_the_full_scan() -> None:
    catalog = make_catalog(index=True)
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0), 'haversine') < 1000.0"
    plan_tree = plan(parse(sql), catalog)
    assert find(plan_tree.root.explain(), "SpatialIndexScan") is None
    assert names(run(sql, catalog)) == names(run(sql, make_catalog(index=False)))


def test_haversine_falls_back_but_keeps_comparable_results() -> None:
    catalog = make_catalog(index=True)
    sql = (
        "SELECT nombre FROM lugares "
        "WHERE distance(ubicacion, POINT(-56.16, -34.90), 'haversine') < 5000.0"
    )
    assert names(run(sql, catalog)) == names(run(sql, make_catalog(index=False)))


def test_greater_than_falls_back_to_the_full_scan() -> None:
    catalog = make_catalog(index=True)
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) > 4.0"
    for statement in (plan(parse(sql), catalog), plan(parse(f"EXPLAIN {sql}"), catalog)):
        root = statement.root
        assert find(root.explain(), "SpatialIndexScan") is None
    assert names(run(sql, catalog)) == names(run(sql, make_catalog(index=False)))


def test_greater_or_equal_falls_back_to_the_full_scan() -> None:
    catalog = make_catalog(index=True)
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) >= 4.0"
    assert find(plan(parse(sql), catalog).root.explain(), "SpatialIndexScan") is None
    assert names(run(sql, catalog)) == names(run(sql, make_catalog(index=False)))


def test_negative_radius_falls_back_to_the_full_scan() -> None:
    catalog = make_catalog(index=True)
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, POINT(0, 0)) < -1.0"
    assert find(plan(parse(sql), catalog).root.explain(), "SpatialIndexScan") is None
    assert run(sql, catalog).rows == ()


def test_non_literal_center_falls_back_to_the_full_scan() -> None:
    catalog = FakeCatalog()
    catalog.create_table("lugares", EXTRA)
    catalog.insert("lugares", ("A", (0.0, 0.0), (1.0, 1.0)))
    catalog.add_spatial_index("lugares", "ubicacion", index_name="idx_ubic")
    # El centro es una columna, no un POINT literal: no se puede construir la caja.
    sql = "SELECT nombre FROM lugares WHERE distance(ubicacion, otra) < 2.0"
    assert find(plan(parse(sql), catalog).root.explain(), "SpatialIndexScan") is None


def test_non_point_column_falls_back_to_the_full_scan() -> None:
    catalog = make_catalog(index=True)
    sql = "SELECT nombre FROM lugares WHERE distance(nombre, POINT(0, 0)) < 2.0"
    assert find(plan(parse(sql), catalog).root.explain(), "SpatialIndexScan") is None


def test_spatial_index_is_never_used_for_a_scalar_range() -> None:
    # Una columna POINT con solo un índice espacial no puede servir un BETWEEN
    # escalar: el planner la deja en escaneo completo.
    catalog = make_catalog(index=True)
    plan_tree = plan(parse("SELECT nombre FROM lugares WHERE ubicacion BETWEEN 1 AND 2"), catalog)
    assert find(plan_tree.root.explain(), "SpatialIndexScan") is None
    assert find(plan_tree.root.explain(), "IndexRangeScan") is None


# --- equivalencia sobre datos aleatorios ----------------------------------


def test_index_matches_full_scan_on_random_data() -> None:
    rng = random.Random(41)
    rows = [
        (
            f"fila{i}",
            (
                round(rng.uniform(-50.0, 50.0), 3),
                round(rng.uniform(-50.0, 50.0), 3),
            ),
        )
        for i in range(200)
    ]
    for _ in range(10):
        cx = round(rng.uniform(-50.0, 50.0), 3)
        cy = round(rng.uniform(-50.0, 50.0), 3)
        radius = round(rng.uniform(0.0, 25.0), 3)
        sql = f"SELECT nombre FROM lugares WHERE distance(ubicacion, POINT({cx}, {cy})) < {radius}"
        indexed = run(sql, make_catalog(index=True, rows=rows))
        scanned = run(sql, make_catalog(index=False, rows=rows))
        assert names(indexed) == names(scanned)
