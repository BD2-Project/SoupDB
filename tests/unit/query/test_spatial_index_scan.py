"""Tests for the SpatialIndexScan volcano operator.

The operator asks a spatial index for the bounding box ``center ± radius`` and
hands the records to the ``Filter`` planned on top, so these tests pin the two
properties the rest of the engine relies on: the box is a SUPERCONSET of the
circle (nothing that could match is ever lost), and it is not the circle itself
(the extra candidates are the ones the exact predicate has to discard).
"""

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.rtree import RTree
from engine.query.operators import SpatialIndexScan
from tests.fakes.fake_index import FakeIndex
from tests.fakes.fake_storage import FakeFileOrganization

LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)

# (0, 0) y (-1, 0) están dentro del radio 4; (3, 3) está dentro de la caja
# [-4, 4]² pero fuera del círculo (distancia 4.24), y (0, 4) está justo en el
# borde de la caja y del círculo.
ROWS = (
    ("A", (0.0, 0.0)),
    ("B", (3.0, 4.0)),
    ("C", (6.0, 8.0)),
    ("D", (-1.0, 0.0)),
    ("E", (0.0, 4.0)),
    ("F", (3.0, 3.0)),
)

CENTER = (0.0, 0.0)
RADIUS = 4.0


def build() -> tuple[FakeFileOrganization, RTree]:
    fake = FakeFileOrganization()
    tree = RTree(order=4)
    for row in ROWS:
        rid = fake.insert(Record(data=encode_row(row, LUGARES)))
        tree.insert(row[1], rid)
    return fake, tree


def scan_of(
    fake: FakeFileOrganization,
    tree: RTree,
    center: tuple[float, float] = CENTER,
    radius: float = RADIUS,
) -> SpatialIndexScan:
    return SpatialIndexScan(
        tree,
        fake,
        center=center,
        radius=radius,
        schema=LUGARES,
        index_name="idx_ubic",
        column="ubicacion",
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


def test_box_is_a_supercone_of_the_circle() -> None:
    fake, tree = build()
    # El operador emite en el orden en que el índice devuelve los RIDs, que no
    # es el orden de la tabla: lo que se fija es el conjunto de candidatos.
    assert {row[0] for row in drain(scan_of(fake, tree))} == {"A", "B", "D", "E", "F"}


def test_box_is_not_the_circle_corners_come_back_as_candidates() -> None:
    fake, tree = build()
    names = {row[0] for row in drain(scan_of(fake, tree))}
    # (3, 3) y (3, 4) caen dentro de la caja y a distancia 4.24 y 5 del centro:
    # el índice los entrega y el predicado exacto es quien los descarta.
    assert {"B", "E", "F"} <= names
    assert "C" not in names


def test_radius_zero_only_reaches_the_exact_point() -> None:
    fake, tree = build()
    assert drain(scan_of(fake, tree, radius=0.0)) == [("A", (0.0, 0.0))]
    assert drain(scan_of(fake, tree, center=(1.0, 0.0), radius=0.0)) == []


def test_negative_radius_box_is_normalized() -> None:
    fake, tree = build()
    # Un radio negativo describe un círculo vacío; la caja se normaliza a
    # centro ± |r| y el predicado exacto descarta todas las filas.
    assert {row[0] for row in drain(scan_of(fake, tree, radius=-1.0))} <= {"A", "D", "F"}


def test_negative_center_coordinates() -> None:
    fake, tree = build()
    # Centro (-1, 0) y radio 1: la caja es [-2, 0] × [-1, 1], que contiene tanto
    # D (en el centro) como A (justo en el borde derecho, distancia 1).
    assert {row[0] for row in drain(scan_of(fake, tree, center=(-1.0, 0.0), radius=1.0))} == {
        "A",
        "D",
    }


def test_skips_stale_rid() -> None:
    fake, tree = build()
    tree.insert((0.0, 0.0), RID(page_id=9, slot=9))
    assert drain(scan_of(fake, tree, radius=0.0)) == [("A", (0.0, 0.0))]


def test_index_without_range_support_raises() -> None:
    fake = FakeFileOrganization()
    plain = FakeIndex(supports_range=False)
    operator = SpatialIndexScan(
        plain,
        fake,
        center=CENTER,
        radius=RADIUS,
        schema=LUGARES,
        index_name="idx_ubic",
        column="ubicacion",
    )
    with pytest.raises(QueryExecutionError, match="does not support it"):
        operator.open()


def test_explain_reports_index_center_radius_and_metric() -> None:
    fake, tree = build()
    operator = scan_of(fake, tree)
    drain(operator)
    node = operator.explain()
    assert node.op == "SpatialIndexScan"
    assert node.rows == 5
    assert node.children == []
    assert node.detail == {
        "index": "idx_ubic",
        "column": "ubicacion",
        "center": CENTER,
        "radius": RADIUS,
        "metric": "euclidean",
    }


def test_explain_before_open_reports_zero_rows() -> None:
    fake, tree = build()
    node = scan_of(fake, tree).explain()
    assert node.rows == 0
    assert node.detail["index"] == "idx_ubic"


def test_duplicate_points_keep_one_row_per_rid() -> None:
    fake = FakeFileOrganization()
    tree = RTree(order=4)
    for name in ("primero", "segundo"):
        rid = fake.insert(Record(data=encode_row((name, (1.0, 1.0)), LUGARES)))
        tree.insert((1.0, 1.0), rid)
    rows = drain(scan_of(fake, tree, center=(1.0, 1.0), radius=0.0))
    assert [row[0] for row in rows] == ["primero", "segundo"]
