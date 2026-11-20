"""Tests for the SpatialKnnScan volcano operator."""

import pytest

from engine.algorithms.spatial import euclidean_metric
from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.rtree import RTree
from engine.query.operators import SpatialKnnScan, TableScan
from tests.fakes.fake_index import FakeIndex
from tests.fakes.fake_storage import FakeFileOrganization

LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)

# Tres filas a distancias 1, 4 y 9 del centro (0, 0): sin empates, el k-NN es
# inequívoco. Las cuatro filas de TIED_ROWS están todas a distancia 1.
DISTINCT_ROWS = (
    ("A", (0.0, 1.0)),
    ("B", (0.0, 4.0)),
    ("C", (0.0, 9.0)),
)
TIED_ROWS = (
    ("A", (1.0, 0.0)),
    ("B", (-1.0, 0.0)),
    ("C", (0.0, 1.0)),
    ("D", (0.0, -1.0)),
)
CENTER = (0.0, 0.0)


def build(rows=DISTINCT_ROWS) -> tuple[FakeFileOrganization, RTree]:
    fake = FakeFileOrganization()
    tree = RTree(order=4)
    for row in rows:
        rid = fake.insert(Record(data=encode_row(row, LUGARES)))
        tree.insert(row[1], rid)
    return fake, tree


def scan_of(
    fake: FakeFileOrganization,
    tree: RTree,
    k: int = 2,
    center: tuple[float, float] = CENTER,
) -> SpatialKnnScan:
    return SpatialKnnScan(
        tree,
        fake,
        center=center,
        k=k,
        schema=LUGARES,
        index_name="idx_ubic",
        column="ubicacion",
        metric=euclidean_metric(),
    )


def drain(operator) -> list[tuple[object, ...]]:
    operator.open()
    try:
        result = []
        while True:
            record = operator.next()
            if record is None:
                return result
            result.append(decode_row(record.data, operator.schema))
    finally:
        operator.close()


def names(operator) -> list[str]:
    return [row[0] for row in drain(operator)]


# --- el índice se usa cuando las k filas son inequívocas ------------------


def test_reads_exactly_the_k_nearest_rows_when_distances_differ() -> None:
    fake, tree = build()
    assert names(scan_of(fake, tree, k=1)) == ["A"]
    assert names(scan_of(fake, tree, k=2)) == ["A", "B"]
    assert names(scan_of(fake, tree, k=3)) == ["A", "B", "C"]


def test_the_index_answers_instead_of_scanning_the_table() -> None:
    # La cuarta fila existe solo en la tabla y distingue escaneo de índice.
    fake, tree = build()
    fake.insert(Record(data=encode_row(("D", (0.0, 100.0)), LUGARES)))
    assert names(scan_of(fake, tree, k=3)) == ["A", "B", "C"]


def test_a_far_center_still_answers_from_the_index() -> None:
    # Se comparan conjuntos porque Sort define el orden en el plan.
    fake, tree = build()
    assert sorted(names(scan_of(fake, tree, k=2, center=(0.0, 100.0)))) == ["B", "C"]


def test_explain_reports_index_center_k_and_metric() -> None:
    fake, tree = build()
    node = scan_of(fake, tree, k=2).explain()
    assert node.op == "SpatialKnnScan"
    assert node.detail == {
        "index": "idx_ubic",
        "column": "ubicacion",
        "center": (0.0, 0.0),
        "k": 2,
        "metric": "euclidean",
    }


def test_the_operator_is_a_leaf() -> None:
    fake, tree = build()
    assert scan_of(fake, tree, k=2).explain().children == []


# --- los empates caen al escaneo completo ----------------------------------


def test_a_tie_falls_back_to_the_full_scan_in_table_order() -> None:
    # El escaneo preserva el orden de entrada entre las cuatro distancias iguales.
    fake, tree = build(TIED_ROWS)
    assert names(scan_of(fake, tree, k=2)) == ["A", "B", "C", "D"]


def test_a_tie_below_the_kth_distance_also_falls_back() -> None:
    # El empate está entre A y B (distancia 1) y k=3 lo alcanza: aunque d_k sea
    # único, las filas entregadas no están ordenadas de forma inequívoca.
    fake, tree = build(TIED_ROWS)
    assert names(scan_of(fake, tree, k=3)) == ["A", "B", "C", "D"]


def test_a_tie_outside_the_k_nearest_also_falls_back() -> None:
    # k=2 sobre las cuatro filas empatadas: d_k = 1 y hay cuatro filas a esa
    # distancia, no dos.
    fake, tree = build(TIED_ROWS)
    assert names(scan_of(fake, tree, k=2)) == ["A", "B", "C", "D"]


def test_duplicate_points_fall_back_instead_of_guessing() -> None:
    fake, tree = build((("A", (2.0, 2.0)), ("B", (2.0, 2.0)), ("C", (9.0, 9.0))))
    assert names(scan_of(fake, tree, k=2)) == ["A", "B", "C"]


def test_k_larger_than_the_index_falls_back_and_returns_everything() -> None:
    # El índice no tiene k filas: puede tener menos entradas que filas (un índice
    # viejo tras un rollback), así que no se fía de él.
    fake, tree = build()
    assert names(scan_of(fake, tree, k=50)) == ["A", "B", "C"]


def test_a_stale_rid_falls_back_instead_of_dropping_the_row() -> None:
    # El RID muerto tiene una distancia única para aislar la recuperación sin empates.
    fake, tree = build()
    tree.insert((0.0, 0.0), RID(page_id=99, slot=0))
    assert names(scan_of(fake, tree, k=1)) == ["A", "B", "C"]


def test_empty_index_falls_back_to_an_empty_table() -> None:
    fake = FakeFileOrganization()
    tree = RTree(order=4)
    assert names(scan_of(fake, tree, k=2)) == []


def test_the_fallback_reads_the_same_rows_a_scan_would() -> None:
    fake, tree = build(TIED_ROWS)
    assert drain(scan_of(fake, tree, k=2)) == drain(TableScan(fake, LUGARES))


# --- errores --------------------------------------------------------------


def test_index_that_is_not_an_rtree_raises() -> None:
    fake = FakeFileOrganization()
    operator = SpatialKnnScan(
        FakeIndex(),
        fake,
        center=CENTER,
        k=2,
        schema=LUGARES,
        index_name="idx_ubic",
        column="ubicacion",
        metric=euclidean_metric(),
    )
    with pytest.raises(QueryExecutionError, match="not an R-Tree"):
        operator.open()
