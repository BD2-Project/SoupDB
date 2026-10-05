"""Tests for the pure spatial metrics module: euclidean and haversine.

``engine.query.spatial_metrics`` holds the convention (x = lon, y = lat, Earth
radius, invalid coordinates rejected) that both the query engine and the
benchmark baseline share, so it is tested as a unit with an independent
spherical-law-of-cosines oracle.
"""

import math

import pytest

from engine.indexes.rtree.point import Point
from engine.query.spatial_metrics import (
    EARTH_RADIUS_M,
    EUCLIDEAN,
    HAVERSINE,
    METRIC_NAMES,
    distance,
    euclidean_distance,
    haversine_distance,
    normalize_metric,
)

# Ciudades (lon, lat) para valores de referencia conocidos.
MONTEVIDEO = (-56.1645, -34.9011)
BUENOS_AIRES = (-58.3816, -34.6037)
MADRID = (-3.7038, 40.4168)
LISBOA = (-9.1393, 38.7223)


def antipode(point: tuple[float, float]) -> tuple[float, float]:
    """Antipodal point, keeping the longitude inside [-180, 180]."""
    lon = point[0] + 180.0
    return (lon - 360.0 if lon > 180.0 else lon, -point[1])


def spherical_law_of_cosines(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Independent geodesic oracle: cos(d/R) = sin(f1)sin(f2)+cos(f1)cos(f2)cos(l2-l1)."""
    lon1, lat1 = math.radians(a[0]), math.radians(a[1])
    lon2, lat2 = math.radians(b[0]), math.radians(b[1])
    cos_central = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(
        lon2 - lon1
    )
    return EARTH_RADIUS_M * math.acos(min(1.0, max(-1.0, cos_central)))


def test_metric_names_are_the_documented_pair() -> None:
    assert METRIC_NAMES == (EUCLIDEAN, HAVERSINE) == ("euclidean", "haversine")


def test_earth_radius_matches_the_iugg_mean_radius() -> None:
    assert EARTH_RADIUS_M == 6_371_008.8


# --- euclidean -------------------------------------------------------------


def test_euclidean_is_the_plane_hypot() -> None:
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert euclidean_distance((1.5, -2.25), (-4.0, 1.0)) == pytest.approx(math.hypot(5.5, -3.25))


def test_euclidean_is_zero_only_for_the_same_point() -> None:
    assert euclidean_distance((12.5, -30.0), (12.5, -30.0)) == 0.0
    assert euclidean_distance((0.0, 0.0), (0.0, 0.001)) > 0.0


def test_euclidean_is_symmetric() -> None:
    assert euclidean_distance((1.0, 2.0), (-3.0, 7.5)) == euclidean_distance(
        (-3.0, 7.5), (1.0, 2.0)
    )


def test_euclidean_accepts_any_finite_plane_coordinates() -> None:
    # La euclidiana es una métrica de plano: no exige rango geográfico.
    assert euclidean_distance((0.0, 0.0), (900.0, 250.0)) == pytest.approx(math.hypot(900.0, 250.0))


# --- haversine -------------------------------------------------------------


def test_haversine_is_zero_for_the_same_point() -> None:
    assert haversine_distance(MONTEVIDEO, MONTEVIDEO) == 0.0
    assert haversine_distance((0.0, 0.0), (0.0, 0.0)) == 0.0


def test_haversine_antipodal_pair_is_half_the_circumference() -> None:
    # Exacto, no aproximado: el par antípodal da pi * R bit a bit, así que el
    # borde se puede comparar con ese valor sin sorpresas de redondeo.
    assert haversine_distance((0.0, 0.0), (180.0, 0.0)) == math.pi * EARTH_RADIUS_M
    assert haversine_distance((0.0, 0.0), (-180.0, 0.0)) == math.pi * EARTH_RADIUS_M
    assert haversine_distance(MONTEVIDEO, antipode(MONTEVIDEO)) == pytest.approx(
        math.pi * EARTH_RADIUS_M
    )
    assert math.pi * EARTH_RADIUS_M == pytest.approx(20_015_114.442035925)


def test_haversine_one_degree_of_latitude_is_the_meridian_arc() -> None:
    assert haversine_distance((0.0, 0.0), (0.0, 1.0)) == pytest.approx(
        EARTH_RADIUS_M * math.pi / 180
    )
    assert haversine_distance((0.0, 0.0), (0.0, 1.0)) == pytest.approx(111_195.0802335329)


def test_haversine_known_city_pair_value() -> None:
    # Montevideo -> Buenos Aires, ~205 km on the real Earth.
    assert haversine_distance(MONTEVIDEO, BUENOS_AIRES) == pytest.approx(
        205_232.35938356873, rel=1e-9
    )
    assert haversine_distance(MADRID, LISBOA) == pytest.approx(502_447.91631964506, rel=1e-9)


def test_haversine_is_symmetric() -> None:
    assert haversine_distance(MONTEVIDEO, LISBOA) == haversine_distance(LISBOA, MONTEVIDEO)


def test_haversine_takes_the_short_way_across_the_antimeridian() -> None:
    across = haversine_distance((179.0, 0.0), (-179.0, 0.0))
    assert across == pytest.approx(EARTH_RADIUS_M * math.radians(2.0))
    assert across < haversine_distance((179.0, 0.0), (179.0, 0.0)) + 1_000_000.0


def test_haversine_grows_with_latitude_gap() -> None:
    gaps = [haversine_distance((10.0, 0.0), (10.0, lat)) for lat in (0.0, 5.0, 20.0, 80.0)]
    assert gaps == sorted(gaps)
    assert haversine_distance((10.0, 0.0), (10.0, -20.0)) == pytest.approx(gaps[2])


def test_haversine_matches_independent_legal_cosines_oracle() -> None:
    samples = [
        (-56.1645, -34.9011),
        (151.2093, -33.8688),
        (-0.1276, 51.5072),
        (139.6917, 35.6895),
        (-43.1729, -22.9068),
        (18.4241, -33.9249),
    ]
    for a in samples:
        for b in samples:
            assert haversine_distance(a, b) == pytest.approx(
                spherical_law_of_cosines(a, b), abs=1e-6
            )


# --- dispatch --------------------------------------------------------------


def test_distance_defaults_to_euclidean() -> None:
    assert distance((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert distance((0.0, 0.0), (3.0, 4.0), EUCLIDEAN) == 5.0
    assert distance((0.0, 0.0), (3.0, 4.0)) == euclidean_distance((0.0, 0.0), (3.0, 4.0))


def test_distance_selects_haversine_case_insensitively() -> None:
    expected = haversine_distance(MONTEVIDEO, BUENOS_AIRES)
    assert distance(MONTEVIDEO, BUENOS_AIRES, "haversine") == expected
    assert distance(MONTEVIDEO, BUENOS_AIRES, "Haversine") == expected
    assert distance(MONTEVIDEO, BUENOS_AIRES, "  HAVersine  ") == expected
    assert distance(MONTEVIDEO, BUENOS_AIRES, "Haversine") == expected


def test_normalize_metric_returns_canonical_names() -> None:
    assert normalize_metric("Haversine") == HAVERSINE
    assert normalize_metric("EUCLIDEAN") == EUCLIDEAN


@pytest.mark.parametrize("bad", ["manhattan", "haversin", "", "euclidian"])
def test_normalize_metric_rejects_unknown_names(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown distance metric"):
        normalize_metric(bad)


def test_unknown_metric_error_lists_the_valid_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_metric("manhattan")
    message = str(excinfo.value)
    assert "manhattan" in message
    assert "euclidean" in message and "haversine" in message


def test_distance_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unknown distance metric"):
        distance((0.0, 0.0), (1.0, 1.0), "manhattan")


def test_metrics_work_on_rtree_points() -> None:
    # El baseline pasa Point (NamedTuple) del R-Tree: mismo contrato 2D.
    a = Point(-56.1645, -34.9011)
    b = Point(-58.3816, -34.6037)
    assert haversine_distance(a, b) == pytest.approx(haversine_distance(tuple(a), tuple(b)))
    assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(tuple(a), tuple(b)))


def test_int_coordinates_are_accepted_as_degrees() -> None:
    assert euclidean_distance((0, 0), (3, 4)) == 5.0
    assert haversine_distance((0, 0), (0, 1)) == pytest.approx(
        haversine_distance((0.0, 0.0), (0.0, 1.0))
    )


# --- invalid coordinates ---------------------------------------------------


def test_haversine_rejects_longitude_out_of_range() -> None:
    for bad in ((180.001, 0.0), (-200.0, 45.0), (540.0, -45.0)):
        with pytest.raises(ValueError, match=r"longitude in \[-180, 180\]"):
            haversine_distance(bad, (0.0, 0.0))
        with pytest.raises(ValueError, match=r"longitude in \[-180, 180\]"):
            haversine_distance((0.0, 0.0), bad)


def test_haversine_rejects_latitude_out_of_range() -> None:
    for bad in ((0.0, 90.5), (45.0, -91.0), (-120.0, 1000.0)):
        with pytest.raises(ValueError, match=r"latitude in \[-90, 90\]"):
            haversine_distance(bad, (0.0, 0.0))
        with pytest.raises(ValueError, match=r"latitude in \[-90, 90\]"):
            haversine_distance((0.0, 0.0), bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_metrics_reject_non_finite_coordinates(bad: float) -> None:
    with pytest.raises(ValueError, match="finite coordinates"):
        haversine_distance((bad, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="finite coordinates"):
        haversine_distance((0.0, 0.0), (0.0, bad))
    with pytest.raises(ValueError, match="finite coordinates"):
        euclidean_distance((bad, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="finite coordinates"):
        euclidean_distance((0.0, 0.0), (bad, 0.0))


def test_metrics_require_two_finite_numeric_coordinates() -> None:
    with pytest.raises(ValueError, match="requires 2D points"):
        euclidean_distance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="numeric coordinates"):
        euclidean_distance(("a", 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="numeric coordinates"):
        haversine_distance((True, 0.0), (0.0, 0.0))


def test_haversine_error_messages_show_the_rejected_value() -> None:
    with pytest.raises(ValueError, match=r"got 190.0"):
        haversine_distance((190.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match=r"got 95.5"):
        haversine_distance((0.0, 95.5), (0.0, 0.0))
