"""Tests de Point y MBR del R-Tree."""

from engine.indexes.rtree.mbr import MBR
from engine.indexes.rtree.point import Point


def test_point_from_scalar_maps_to_x_axis() -> None:
    assert Point.from_key(10) == Point(10.0, 0.0)
    assert Point.from_key(-3) == Point(-3.0, 0.0)


def test_point_from_tuple() -> None:
    assert Point.from_key((4, 7)) == Point(4.0, 7.0)


def test_point_from_string_is_stable() -> None:
    assert Point.from_key("a") == Point.from_key("a")
    assert Point.from_key("a") != Point.from_key("b")


def test_mbr_from_point() -> None:
    mbr = MBR.from_point(Point(3, 4))
    assert mbr == MBR(3, 4, 3, 4)


def test_mbr_union() -> None:
    a = MBR.from_point(Point(1, 1))
    b = MBR.from_point(Point(5, 4))
    assert a.union(b) == MBR(1, 1, 5, 4)


def test_mbr_area() -> None:
    assert MBR(0, 0, 10, 10).area() == 100
    assert MBR.from_point(Point(2, 2)).area() == 0


def test_mbr_enlargement() -> None:
    box = MBR(0, 0, 10, 10)
    inside = MBR(2, 2, 3, 3)
    assert box.enlargement(inside) == 0
    growth = MBR(0, 0, 12, 12)
    assert box.enlargement(growth) == 44  # 144 - 100


def test_mbr_contains_point_inclusive() -> None:
    box = MBR(0, 0, 10, 10)
    assert box.contains(Point(0, 0))
    assert box.contains(Point(10, 10))
    assert box.contains(Point(5, 5))
    assert not box.contains(Point(11, 5))
    assert not box.contains(Point(5, -1))


def test_mbr_intersects() -> None:
    a = MBR(0, 0, 10, 10)
    assert a.intersects(MBR(5, 5, 20, 20))
    assert a.intersects(MBR(10, 10, 20, 20))  # bordes tocados
    assert not a.intersects(MBR(11, 11, 20, 20))
    assert not a.intersects(MBR(-20, -20, -1, -1))


def test_point_ordering_for_1d_range() -> None:
    low = Point.from_key(20)
    high = Point.from_key(30)
    assert low.x < high.x
    assert low.y == high.y == 0.0
