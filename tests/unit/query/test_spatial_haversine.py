"""End-to-end haversine queries over a real persistent Catalog.

Covers ``distance(a, b, 'haversine')`` in WHERE and in ORDER BY (ASC/DESC with
LIMIT), the parity with the default euclidean metric, the known-value check and
the error surface (unknown metric, non-literal metric, non-POINT operand, strict
int/float comparison, coordinates out of range) through the full
parse -> plan -> execute path.
"""

import math
from pathlib import Path

import pytest

from engine.common.catalog import Catalog
from engine.common.errors import QueryExecutionError, QueryParseError
from engine.query import execute_sql
from engine.query.spatial_metrics import EARTH_RADIUS_KM

# (lon, lat) de ciudades: la convención del módulo es x = longitud, y = latitud.
MONTEVIDEO = (-56.1645, -34.9011)
BUENOS_AIRES = (-58.3816, -34.6037)
SANTIAGO = (-70.6483, -33.4489)
PARIS = (2.3522, 48.8566)
LISBOA = (-9.1393, 38.7223)

# Distancia de referencia Montevideo -> Buenos Aires en km (radio medio 6371.0088).
MVD_BUE_KM = 205.23235938356873

MERCADORES = (
    ("Montevideo", MONTEVIDEO),
    ("Buenos Aires", BUENOS_AIRES),
    ("Santiago", SANTIAGO),
    ("Paris", PARIS),
    ("Lisboa", LISBOA),
)


def make_catalog(tmp_path: Path) -> Catalog:
    """Real catalog in a temp dir with a POINT column and one row per capital."""
    catalog = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    execute_sql("CREATE TABLE ciudades (nombre TEXT, ubicacion POINT)", catalog)
    for nombre, (lon, lat) in MERCADORES:
        execute_sql(
            f"INSERT INTO ciudades VALUES ('{nombre}', POINT({lon}, {lat}))",
            catalog,
        )
    return catalog


def nombres(result) -> tuple[str, ...]:
    return tuple(row[0] for row in result.rows)


# --- WHERE -----------------------------------------------------------------


def test_where_haversine_radius_inside_and_outside(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    # Radio de barrio: solo Montevideo (0 km) y Buenos Aires (205.2 km).
    inside = execute_sql(
        "SELECT nombre FROM ciudades "
        "WHERE distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') < 300.0",
        catalog,
    )
    assert nombres(inside) == ("Montevideo", "Buenos Aires")
    # Santiago (1341 km), Lisboa (9508 km) y Paris (10961 km) quedan fuera.
    outside = execute_sql(
        "SELECT nombre FROM ciudades "
        "WHERE distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') >= 1000.0",
        catalog,
    )
    assert nombres(outside) == ("Santiago", "Paris", "Lisboa")
    # Santiago entra al ampliar el radio a 1500 km.
    wider = execute_sql(
        "SELECT nombre FROM ciudades "
        "WHERE distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') < 1500.0",
        catalog,
    )
    assert nombres(wider) == ("Montevideo", "Buenos Aires", "Santiago")
    catalog.close()


def test_where_haversine_radius_zero_matches_only_the_same_point(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    result = execute_sql(
        "SELECT nombre FROM ciudades "
        "WHERE distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') < 1.0",
        catalog,
    )
    assert nombres(result) == ("Montevideo",)
    catalog.close()


def test_where_haversine_threshold_at_the_antipodal_pair(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    execute_sql("CREATE TABLE extremos (nombre TEXT, ubicacion POINT)", catalog)
    execute_sql("INSERT INTO extremos VALUES ('origen', POINT(0, 0))", catalog)
    execute_sql("INSERT INTO extremos VALUES ('antipodo', POINT(180, 0))", catalog)
    half_circumference = math.pi * EARTH_RADIUS_KM
    inside = execute_sql(
        "SELECT nombre FROM extremos "
        "WHERE distance(ubicacion, POINT(0, 0), 'haversine') < 20015.0",
        catalog,
    )
    assert nombres(inside) == ("origen",)
    # Justo por encima de media circunferencia entra el par antípodal.
    result = execute_sql(
        "SELECT nombre FROM extremos "
        f"WHERE distance(ubicacion, POINT(0, 0), 'haversine') < {half_circumference + 0.001}",
        catalog,
    )
    assert nombres(result) == ("origen", "antipodo")
    # El límite exacto de pi * R también los incluye, sin margen.
    exact = execute_sql(
        "SELECT nombre FROM extremos "
        f"WHERE distance(ubicacion, POINT(0, 0), 'haversine') <= {half_circumference}",
        catalog,
    )
    assert nombres(exact) == ("origen", "antipodo")
    # Y en la euclidiana el mismo par está a 180 grados, no a 20015 km.
    plane = execute_sql(
        "SELECT nombre FROM extremos WHERE distance(ubicacion, POINT(0, 0)) < 180.0",
        catalog,
    )
    assert nombres(plane) == ("origen",)
    catalog.close()


def test_where_haversine_is_monotonic_in_the_radius(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    counts = [
        len(
            execute_sql(
                "SELECT nombre FROM ciudades "
                f"WHERE distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') < {radius}",
                catalog,
            ).rows
        )
        for radius in (1.0, 210.0, 1500.0, 10000.0)
    ]
    assert counts == [1, 2, 3, 4]
    catalog.close()


def test_where_haversine_and_euclidean_are_different_metrics(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    euclidean = nombres(
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(-56.1645, -34.9011)) < 3.0",
            catalog,
        )
    )
    haversine = nombres(
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') < 3.0",
            catalog,
        )
    )
    # En grados el vecindario es Montevideo y Buenos Aires; en km, solo Montevideo.
    assert euclidean == ("Montevideo", "Buenos Aires")
    assert haversine == ("Montevideo",)
    catalog.close()


def test_default_metric_matches_explicit_euclidean(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    for clause in (
        "SELECT nombre, distance(ubicacion, POINT(0, 0)) FROM ciudades",
        "SELECT nombre, distance(ubicacion, POINT(0, 0), 'euclidean') FROM ciudades",
        "SELECT nombre, distance(ubicacion, POINT(0, 0), 'EUCLIDEAN') FROM ciudades",
    ):
        assert execute_sql(clause + " ORDER BY nombre", catalog).rows == execute_sql(
            "SELECT nombre, distance(ubicacion, POINT(0, 0), 'euclidean') "
            "FROM ciudades ORDER BY nombre",
            catalog,
        ).rows
    catalog.close()


def test_metric_name_is_case_insensitive(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    rows = execute_sql(
        "SELECT nombre FROM ciudades "
        "WHERE distance(ubicacion, POINT(-56.1645, -34.9011), 'HaVeRsInE') < 1500.0",
        catalog,
    )
    assert nombres(rows) == ("Montevideo", "Buenos Aires", "Santiago")
    catalog.close()


# --- reference values ------------------------------------------------------


def test_projected_haversine_matches_the_reference_value(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    result = execute_sql(
        "SELECT nombre, distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') "
        "FROM ciudades WHERE nombre = 'Buenos Aires'",
        catalog,
    )
    assert result.rows[0][0] == "Buenos Aires"
    assert result.rows[0][1] == pytest.approx(MVD_BUE_KM)
    assert result.rows[0][1] == pytest.approx(205.23235938356873)
    catalog.close()


def test_projected_haversine_of_the_same_point_is_exactly_zero(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    result = execute_sql(
        "SELECT distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') "
        "FROM ciudades WHERE nombre = 'Montevideo'",
        catalog,
    )
    assert result.rows[0][0] == 0.0
    catalog.close()


def test_distance_column_type_is_float_for_both_metrics(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    for metric in ("", ", 'euclidean'", ", 'haversine'"):
        result = execute_sql(
            f"SELECT distance(ubicacion, POINT(0, 0){metric}) FROM ciudades LIMIT 1",
            catalog,
        )
        assert result.columns[0].type_name is not None
        assert result.columns[0].type_name.name == "FLOAT"
    catalog.close()


# --- ORDER BY --------------------------------------------------------------


def test_order_by_haversine_ascending_returns_knn(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    result = execute_sql(
        "SELECT nombre FROM ciudades "
        "ORDER BY distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') LIMIT 3",
        catalog,
    )
    assert nombres(result) == ("Montevideo", "Buenos Aires", "Santiago")
    catalog.close()


def test_order_by_haversine_descending_returns_farthest_first(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    result = execute_sql(
        "SELECT nombre FROM ciudades "
        "ORDER BY distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') DESC LIMIT 2",
        catalog,
    )
    assert nombres(result) == ("Paris", "Lisboa")
    catalog.close()


def test_order_by_haversine_matches_ascending_reversed(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    asc = nombres(
        execute_sql(
            "SELECT nombre FROM ciudades "
            "ORDER BY distance(ubicacion, POINT(0, 0), 'haversine') LIMIT 5",
            catalog,
        )
    )
    desc = nombres(
        execute_sql(
            "SELECT nombre FROM ciudades "
            "ORDER BY distance(ubicacion, POINT(0, 0), 'haversine') DESC LIMIT 5",
            catalog,
        )
    )
    assert asc == tuple(reversed(desc))
    catalog.close()


def test_order_by_haversine_k_greater_than_rows_and_limit_offsets(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    result = execute_sql(
        "SELECT nombre FROM ciudades "
        "ORDER BY distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') LIMIT 10",
        catalog,
    )
    assert len(result.rows) == len(MERCADORES)
    window = execute_sql(
        "SELECT nombre FROM ciudades "
        "ORDER BY distance(ubicacion, POINT(-56.1645, -34.9011), 'haversine') LIMIT 2 OFFSET 1",
        catalog,
    )
    assert nombres(window) == ("Buenos Aires", "Santiago")
    catalog.close()


def test_order_by_haversine_orders_at_high_latitude_where_degrees_do_not(
    tmp_path: Path,
) -> None:
    """At 80 degrees longitude a degree is ~193 km, so the order really flips."""
    catalog = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    execute_sql("CREATE TABLE artico (nombre TEXT, ubicacion POINT)", catalog)
    execute_sql("INSERT INTO artico VALUES ('este', POINT(10, 80))", catalog)
    execute_sql("INSERT INTO artico VALUES ('norte', POINT(0, 88))", catalog)
    center = "POINT(0, 80)"
    in_degrees = nombres(
        execute_sql(
            f"SELECT nombre FROM artico ORDER BY distance(ubicacion, {center}) LIMIT 2",
            catalog,
        )
    )
    in_kilometres = nombres(
        execute_sql(
            f"SELECT nombre FROM artico "
            f"ORDER BY distance(ubicacion, {center}, 'haversine') LIMIT 2",
            catalog,
        )
    )
    assert in_degrees == ("norte", "este")
    assert in_kilometres == ("este", "norte")
    catalog.close()


# --- errors ----------------------------------------------------------------


def test_unknown_metric_is_a_parse_error(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryParseError, match="unknown distance metric"):
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(0, 0), 'manhattan') < 5.0",
            catalog,
        )
    catalog.close()


def test_unknown_metric_error_lists_the_valid_names(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryParseError) as excinfo:
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(0, 0), 'sinhattan') < 5.0",
            catalog,
        )
    message = str(excinfo.value)
    assert "sinhattan" in message
    assert "euclidean" in message and "haversine" in message
    catalog.close()


@pytest.mark.parametrize(
    "metric_argument",
    ["metrica", "1", "metrica_columna", "POINT(1, 2)", "NULL"],
)
def test_metric_must_be_a_string_literal(tmp_path: Path, metric_argument: str) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryParseError, match="metric must be a string literal"):
        execute_sql(
            "SELECT nombre FROM ciudades "
            f"WHERE distance(ubicacion, POINT(0, 0), {metric_argument}) < 5.0",
            catalog,
        )
    catalog.close()


def test_too_many_arguments_is_a_parse_error(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryParseError):
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(0, 0), 'haversine', 'euclidean') < 5.0",
            catalog,
        )
    catalog.close()


def test_non_point_operand_still_fails_with_the_same_message(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    execute_sql("CREATE TABLE otros (nombre TEXT, ubicacion INT)", catalog)
    execute_sql("INSERT INTO otros VALUES ('x', 7)", catalog)
    with pytest.raises(
        QueryExecutionError, match=r"distance operands must be POINT values, got int"
    ):
        execute_sql(
            "SELECT nombre FROM otros WHERE distance(ubicacion, POINT(0, 0), 'haversine') < 5.0",
            catalog,
        )
    with pytest.raises(
        QueryExecutionError, match=r"distance operands must be POINT values, got str"
    ):
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(nombre, POINT(0, 0), 'haversine') < 5.0",
            catalog,
        )
    catalog.close()


def test_int_literal_radius_still_fails_with_strict_typing(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryExecutionError, match="cannot mix float and int values"):
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(0, 0), 'haversine') < 100",
            catalog,
        )
    catalog.close()


def test_coordinates_out_of_range_are_rejected_at_execution(tmp_path: Path) -> None:
    catalog = make_catalog(tmp_path)
    with pytest.raises(QueryExecutionError, match=r"latitude in \[-90, 90\]"):
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(0, 200), 'haversine') < 5.0",
            catalog,
        )
    with pytest.raises(QueryExecutionError, match=r"longitude in \[-180, 180\]"):
        execute_sql(
            "SELECT nombre FROM ciudades "
            "WHERE distance(ubicacion, POINT(400, 0), 'haversine') < 5.0",
            catalog,
        )
    # La euclidiana es métrica de plano: las mismas coordenadas no se validan.
    plano = execute_sql(
        "SELECT nombre FROM ciudades WHERE distance(ubicacion, POINT(0, 200)) < 5.0",
        catalog,
    )
    assert nombres(plano) == ()
    catalog.close()


def test_stored_point_out_of_range_fails_only_for_haversine(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    execute_sql("CREATE TABLE raros (nombre TEXT, ubicacion POINT)", catalog)
    execute_sql("INSERT INTO raros VALUES ('fuera', POINT(999, 0))", catalog)
    en_grados = execute_sql(
        "SELECT nombre FROM raros WHERE distance(ubicacion, POINT(0, 0)) < 1000.0",
        catalog,
    )
    assert nombres(en_grados) == ("fuera",)
    with pytest.raises(QueryExecutionError, match=r"longitude in \[-180, 180\]"):
        execute_sql(
            "SELECT nombre FROM raros "
            "WHERE distance(ubicacion, POINT(0, 0), 'haversine') < 1000.0",
            catalog,
        )
    catalog.close()