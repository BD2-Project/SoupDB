"""Planner tests for the k-NN spatial access path.

``ORDER BY distance(columna, POINT(...)) LIMIT k`` must reach the spatial index
only when the index can answer the same question the full scan answers, and the
result must be the scan's result - including its ORDER OF rows, because a ``Sort``
is stable and a tie is resolved by table order.

The planner refuses the index (full scan) for: a WHERE, a GROUP BY / aggregate /
DISTINCT, no LIMIT, ``LIMIT 0``, a DESC distance, a non-euclidean metric, a
non-literal centre, a non-POINT column and a column with no spatial index.
"""

import math
import random

from engine.common.schema import ColumnDef, ColumnType
from engine.query.executor import execute
from engine.query.parser import parse
from engine.query.planner import plan
from tests.fakes.fake_catalog import FakeCatalog

LUGARES = (
    ColumnDef("nombre", ColumnType.TEXT),
    ColumnDef("ubicacion", ColumnType.POINT),
)

# Las cuatro primeras filas están empatadas a distancia 1 del centro (0, 0) y las
# dos últimas a distancia 5. Con datos sin empate el operador saca las k filas del
# índice; con este empate cae al escaneo, y ambos caminos deben coincidir igual.
ROWS = (
    ("A", (1.0, 0.0)),
    ("B", (-1.0, 0.0)),
    ("C", (0.0, 1.0)),
    ("D", (0.0, -1.0)),
    ("E", (3.0, 4.0)),
    ("F", (-3.0, 4.0)),
)

# Las mismas filas sobre un eje, a distancias 1, 4 y 9 del centro (0, 0): sin
# empates, así que el k-NN del índice es inequívoco y sí se usa.
UNTIED_ROWS = (
    ("A", (0.0, 1.0)),
    ("B", (0.0, 4.0)),
    ("C", (0.0, 9.0)),
    ("D", (0.0, 16.0)),
)

KNN = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT {}"


def make_catalog(index: bool = True, rows=ROWS) -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table("lugares", LUGARES)
    for row in rows:
        catalog.insert("lugares", row)
    if index:
        catalog.add_spatial_index("lugares", "ubicacion", index_name="idx_ubic")
    return catalog


def run(sql: str, catalog: FakeCatalog):
    return execute(plan(parse(sql), catalog), catalog)


def names(result) -> list[str]:
    """Rows IN ORDER: for a k-NN the order is part of the answer, not a detail."""
    return [row[0] for row in result.rows]


def find(node, op: str):
    if node.op == op:
        return node
    for child in node.children:
        found = find(child, op)
        if found is not None:
            return found
    return None


def explained(sql: str, index: bool = True) -> list[str]:
    catalog = make_catalog(index=index)
    return [row[0] for row in run(f"EXPLAIN {sql}", catalog).rows]


def analyzed(sql: str, index: bool = True, rows=ROWS) -> dict[str, int]:
    """Filas que produjo cada nodo, según EXPLAIN ANALYZE.

    Sirve para observar qué camino se EJECUTÓ de verdad: la forma del plan dice lo
    que se planeó, el conteo de filas dice lo que leyó cada operador. Si el operador
    cae al escaneo, entrega todas las filas de la tabla en vez de las k del índice.
    """
    catalog = make_catalog(index=index, rows=rows)
    result = run(f"EXPLAIN ANALYZE {sql}", catalog)
    counts: dict[str, int] = {}
    for (line,) in result.rows:
        parts = line.split()
        if parts and parts[0] == "->" and len(parts) > 1:
            counts[parts[1]] = int(parts[2].split("=")[1])
    return counts


# --- camino con índice ----------------------------------------------------


def test_order_by_distance_limit_uses_the_spatial_index() -> None:
    lines = explained(KNN.format(3))
    assert any("SpatialKnnScan" in line for line in lines)
    assert not any("TableScan" in line for line in lines)


def test_explain_shows_the_index_its_parameters_and_where_it_sits() -> None:
    node = find(plan(parse(KNN.format(3)), make_catalog()).root.explain(), "SpatialKnnScan")
    assert node is not None
    assert node.detail == {
        "index": "idx_ubic",
        "column": "ubicacion",
        "center": (0.0, 0.0),
        "k": 3,
        "metric": "euclidean",
    }
    # El índice es la hoja: el Sort y el Limit siguen decidiendo las k filas.
    assert [child.op for child in node.children] == []
    lines = explained(KNN.format(3))
    assert lines[0].startswith("-> Limit")
    assert "-> SpatialKnnScan" in lines[-1]


def test_index_and_full_scan_agree_on_a_tie() -> None:
    # El caso que separa este camino de un k-NN ingenuo: las cuatro filas están a
    # distancia 1 y el índice las desempata por coordenada, no por orden de tabla.
    assert names(run(KNN.format(2), make_catalog(index=True))) == ["A", "B"]
    assert names(run(KNN.format(2), make_catalog(index=False))) == ["A", "B"]


def test_index_and_full_scan_agree_for_every_k() -> None:
    for k in range(1, 8):
        sql = KNN.format(k)
        assert names(run(sql, make_catalog(index=True))) == names(
            run(sql, make_catalog(index=False))
        ), f"k={k}"


def test_untied_data_is_answered_by_the_index_and_agrees_with_the_scan() -> None:
    # Datos sin empates: el índice sí resuelve y aun así coincide con el escaneo.
    for k in (1, 2, 3, 4):
        sql = KNN.format(k)
        assert any("SpatialKnnScan" in line for line in explained(sql))
        assert names(run(sql, make_catalog(index=True, rows=UNTIED_ROWS))) == names(
            run(sql, make_catalog(index=False, rows=UNTIED_ROWS))
        )


# --- el camino indexado se EJECUTA, no solo se planeó ----------------------


def test_without_ties_the_index_really_supplies_the_rows() -> None:
    # La forma del plan no prueba que el índice produjera las filas: el conteo de
    # EXPLAIN ANALYZE sí. Lee 2 (las k filas), no las 4 de la tabla.
    counts = analyzed(KNN.format(2), rows=UNTIED_ROWS)
    assert counts["SpatialKnnScan"] == 2
    assert "TableScan" not in counts


def test_with_ties_the_operator_falls_back_to_reading_the_table() -> None:
    # ROWS está empatado: el operador entrega las 4 filas de la tabla, no 2.
    counts = analyzed(KNN.format(2), rows=ROWS)
    assert counts["SpatialKnnScan"] == len(ROWS)


def test_the_offset_rows_also_come_from_the_index() -> None:
    # Con OFFSET el k del índice sube: LIMIT 2 OFFSET 2 necesita las 4 filas más
    # cercanas para devolver 2.
    sql = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 2 OFFSET 2"
    assert analyzed(sql, rows=UNTIED_ROWS)["SpatialKnnScan"] == 4


# --- el filtro de distancias desconocidas ----------------------------------


def test_both_paths_filter_out_unknown_distances() -> None:
    # Sin este filtro, una fila con distancia NULL llegaría al Sort. Se inyecta en
    # los DOS caminos: indexado y escaneo completo.
    for index in (True, False):
        root = plan(parse(KNN.format(3)), make_catalog(index=index)).root.explain()
        node = find(root, "Filter")
        assert node is not None, f"sin filtro implícito con index={index}"
        predicate = str(node.detail["predicate"])
        assert "op='>='" in predicate and "value=0.0" in predicate
        # El filtro va sobre la hoja y por debajo del Sort.
        assert find(root, "Sort") is not None
        assert node.children[0].op in {"SpatialKnnScan", "TableScan"}


def test_offset_counts_towards_the_k() -> None:
    sql = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 2 OFFSET 2"
    node = find(plan(parse(sql), make_catalog()).root.explain(), "SpatialKnnScan")
    # El OFFSET también consume filas ordenadas: k = limit + offset.
    assert node is not None and node.detail["k"] == 4
    assert names(run(sql, make_catalog(index=True))) == names(run(sql, make_catalog(index=False)))


def test_a_second_sort_key_falls_back_to_the_full_scan() -> None:
    # Con dos criterios, un empate de distancia lo resuelve el segundo, y si ése
    # también empata vuelve a mandar el orden del escaneo: el operador no puede
    # garantizarlo, así que no se le pregunta al índice.
    sql = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)), nombre LIMIT 3"
    assert not any("SpatialKnnScan" in line for line in explained(sql))
    assert names(run(sql, make_catalog(index=True))) == names(run(sql, make_catalog(index=False)))


def test_order_by_a_projected_distance_alias_uses_the_index() -> None:
    sql = "SELECT nombre, distance(ubicacion, POINT(0, 0)) AS d FROM lugares ORDER BY d LIMIT 3"
    assert any("SpatialKnnScan" in line for line in explained(sql))
    assert names(run(sql, make_catalog(index=True))) == names(run(sql, make_catalog(index=False)))


def test_index_survives_rows_inserted_after_it_was_created() -> None:
    catalog = make_catalog()
    catalog.insert("lugares", ("G", (0.2, 0.0)))
    assert names(run(KNN.format(1), catalog)) == ["G"]


# --- caminhos que deben seguir en escaneo completo -------------------------


def test_without_index_the_plan_stays_a_full_scan() -> None:
    lines = explained(KNN.format(3), index=False)
    assert not any("SpatialKnnScan" in line for line in lines)
    assert any("TableScan" in line for line in lines)


def test_desc_falls_back_to_the_full_scan() -> None:
    # DESC quiere las k filas MÁS lejanas: un k-NN devuelve las más cercanas.
    sql = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) DESC LIMIT 3"
    assert not any("SpatialKnnScan" in line for line in explained(sql))
    # Lo más lejano primero: E y F a distancia 5, y luego el empate a 1 en orden
    # de tabla (A).
    assert [row[0] for row in run(sql, make_catalog()).rows] == ["E", "F", "A"]


def test_haversine_falls_back_to_the_full_scan() -> None:
    sql = (
        "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0), 'haversine') LIMIT 3"
    )
    assert not any("SpatialKnnScan" in line for line in explained(sql))
    assert names(run(sql, make_catalog())) == ["A", "B", "C"]


def test_where_falls_back_to_the_full_scan() -> None:
    # Las k filas más cercanas de la tabla no son las k más cercanas de las que
    # sobreviven al filtro: el índice podaría filas que deben competir.
    sql = (
        "SELECT nombre FROM lugares WHERE nombre >= 'C' "
        "ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 2"
    )
    assert not any("SpatialKnnScan" in line for line in explained(sql))
    assert names(run(sql, make_catalog())) == ["C", "D"]


def test_group_by_falls_back_to_the_full_scan() -> None:
    sql = (
        "SELECT ubicacion, COUNT(nombre) FROM lugares "
        "GROUP BY ubicacion ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 1"
    )
    assert not any("SpatialKnnScan" in line for line in explained(sql))


def test_distinct_falls_back_to_the_full_scan() -> None:
    sql = "SELECT DISTINCT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 3"
    assert not any("SpatialKnnScan" in line for line in explained(sql))


def test_without_limit_falls_back_to_the_full_scan() -> None:
    # Sin LIMIT hacen falta todas las filas: no hay k que preguntarle al índice.
    sql = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0))"
    assert not any("SpatialKnnScan" in line for line in explained(sql))


def test_limit_zero_falls_back_to_the_full_scan() -> None:
    sql = "SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT(0, 0)) LIMIT 0"
    assert not any("SpatialKnnScan" in line for line in explained(sql))
    assert run(sql, make_catalog()).rows == ()


def test_non_literal_center_falls_back_to_the_full_scan() -> None:
    # El centro es una columna, no un literal: el planner no puede ni calcular la
    # distancia en tiempo de planning, así que no abre el camino del índice.
    schema = (
        ColumnDef("nombre", ColumnType.TEXT),
        ColumnDef("ubicacion", ColumnType.POINT),
        ColumnDef("centro", ColumnType.POINT),
    )
    catalog = FakeCatalog()
    catalog.create_table("con_centro", schema)
    catalog.insert("con_centro", ("A", (1.0, 0.0), (0.0, 0.0)))
    catalog.add_spatial_index("con_centro", "ubicacion", index_name="idx_ubic")
    sql = "SELECT nombre FROM con_centro ORDER BY distance(ubicacion, centro) LIMIT 1"
    lines = [row[0] for row in run(f"EXPLAIN {sql}", catalog).rows]
    assert not any("SpatialKnnScan" in line for line in lines)
    assert names(run(sql, catalog)) == ["A"]


def test_non_point_column_falls_back_to_the_full_scan() -> None:
    sql = "SELECT nombre FROM lugares ORDER BY distance(nombre, POINT(0, 0)) LIMIT 2"
    assert not any("SpatialKnnScan" in line for line in explained(sql))


def test_leading_key_other_than_distance_falls_back_to_the_full_scan() -> None:
    sql = "SELECT nombre FROM lugares ORDER BY nombre, distance(ubicacion, POINT(0, 0)) LIMIT 3"
    assert not any("SpatialKnnScan" in line for line in explained(sql))


# --- equivalencia contra oráculo brute-force ------------------------------


def brute_force_knn(rows, center: tuple[float, float], k: int) -> list[str]:
    """Oráculo: distancia euclidiana, empates por orden de tabla, se toman k."""
    cx, cy = center
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (math.hypot(item[1][1][0] - cx, item[1][1][1] - cy), item[0]),
    )
    return [row[0] for _position, row in ordered[:k]]


def test_matches_brute_force_oracle_on_tied_data() -> None:
    centers = ((0.0, 0.0), (1.0, 1.0), (-2.0, 0.5), (100.0, 100.0))
    for center in centers:
        for k in range(1, len(ROWS) + 2):
            sql = (
                "SELECT nombre FROM lugares "
                f"ORDER BY distance(ubicacion, POINT({center[0]}, {center[1]})) LIMIT {k}"
            )
            indexed = names(run(sql, make_catalog(index=True)))
            scanned = names(run(sql, make_catalog(index=False)))
            expected = brute_force_knn(ROWS, center, k)
            assert indexed == expected, f"center={center} k={k}"
            assert scanned == expected, f"center={center} k={k}"


def test_matches_brute_force_oracle_on_random_data() -> None:
    rng = random.Random(7)
    rows = [
        (
            f"fila{i}",
            (round(rng.uniform(-50.0, 50.0), 3), round(rng.uniform(-50.0, 50.0), 3)),
        )
        for i in range(200)
    ]
    for _ in range(20):
        cx = round(rng.uniform(-50.0, 50.0), 3)
        cy = round(rng.uniform(-50.0, 50.0), 3)
        k = rng.randint(1, 12)
        sql = (
            f"SELECT nombre FROM lugares ORDER BY distance(ubicacion, POINT({cx}, {cy})) LIMIT {k}"
        )
        expected = brute_force_knn(rows, (cx, cy), k)
        assert names(run(sql, make_catalog(index=True, rows=rows))) == expected
        assert names(run(sql, make_catalog(index=False, rows=rows))) == expected


def test_matches_brute_force_oracle_with_many_exact_duplicates() -> None:
    # Todos los puntos iguales: la distancia es idéntica para todas las filas y
    # sólo el orden de tabla puede desempatar.
    rows = tuple((f"f{i}", (2.0, 2.0)) for i in range(12))
    for k in range(1, 13):
        sql = KNN.format(k)
        assert names(run(sql, make_catalog(index=True, rows=rows))) == [row[0] for row in rows[:k]]


# --- bloqueo: el camino con índice no hace dirty read ----------------------


def test_the_knn_path_blocks_like_a_scan_instead_of_reading_uncommitted_rows() -> None:
    # El lector por índice toma el lock de tabla antes de fetch. Sin él devolvería
    # la fila que otra transacción tiene escrita y sin commitear, porque al INSERT
    # le basta con el lock de tabla: la fila nueva no tiene lock de registro que la
    # detenga.
    import threading

    from engine.common.errors import TransactionError
    from engine.transactions.session import TransactionalSession
    from engine.transactions.transaction_manager import TransactionManager

    # Datos SIN empates: así el operador responde con el índice en vez de caer al
    # escaneo, que es lo que haría la prueba vacuamente cierta (un escaneo ya
    # bloquea la tabla, con o sin el lock del lector por índice).
    catalog = make_catalog(index=True, rows=UNTIED_ROWS)
    assert find(plan(parse(KNN.format(2)), catalog).root.explain(), "SpatialKnnScan")
    manager = TransactionManager()
    escritor = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    lector = TransactionalSession(catalog, transaction_manager=manager, lock_timeout_ms=40)
    escrito = threading.Event()
    intentado = threading.Event()

    def escribir() -> None:
        escritor.begin()
        escritor.execute("INSERT INTO lugares VALUES ('SUELTA', POINT(0.5, 0.5))")
        escrito.set()
        # El escritor mantiene su lock de tabla hasta que el lector lo intente: si
        # revirtiera antes, la prueba dependeria del orden de los hilos.
        intentado.wait(timeout=5)
        escritor.rollback()

    errores: list[BaseException] = []

    def leer() -> None:
        escrito.wait(timeout=5)
        try:
            lector.begin()
            lector.execute(KNN.format(2))
        except BaseException as exc:
            errores.append(exc)
        finally:
            intentado.set()

    hilos = [threading.Thread(target=escribir), threading.Thread(target=leer)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join(timeout=10)
    for hilo in hilos:
        assert not hilo.is_alive(), "un hilo se quedo esperando: deadlock"
    # El lector tiene que BLOQUEAR: expirar por timeout es lo correcto, leer la fila
    # sin commitear sería el dirty read.
    assert errores, "el lector NO se bloqueo: devolvio la fila sin commitear"
    assert all(isinstance(exc, TransactionError) for exc in errores), errores
    for session in (escritor, lector):
        if session.current_transaction() is not None:
            session.rollback()
        session.close()
