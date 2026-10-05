"""Comparación 2.2.4 contra PostgreSQL con índice GiST.

Corre las mismas consultas por radio y k-NN que
``benchmarks.scripts.compare_spatial_indexes``, sobre los mismos puntos y los
mismos centros, para que las tres técnicas sean comparables fila a fila.

Requiere el servicio ``postgres`` del ``docker-compose.yml`` levantado:

    docker compose up -d postgres
    uv run python -m benchmarks.scripts.compare_postgres_gist

Se usa PostGIS con ``geography`` porque el tipo ``point`` nativo de PostgreSQL
mide distancia euclidiana en grados, que no es comparable con el haversine en
metros del motor. Con ``geography`` los conteos coinciden con los del baseline
secuencial, y esa coincidencia es la verificación de que la comparación es válida.

``ST_DWithin`` se llama con ``use_spheroid => false``: por defecto PostGIS mide
sobre el elipsoide WGS84 mientras que el motor usa una esfera, y esa diferencia
de medio por mil movía los puntos que caen justo en el borde del radio.

No se agrega ninguna dependencia de Python: se habla con la base por ``psql``
dentro del contenedor, igual que lo haría una consola.
"""

import json
import subprocess
import time

from benchmarks.scripts.compare_spatial_indexes import (
    BENCHMARK_RADIUS_VALUES_DEG,
    QUERIES,
    SEED,
)
from benchmarks.spatial_report import SpatialResult, summarize, write_spatial_results
from benchmarks.spatial_workload import (
    BENCHMARK_K_VALUES,
    BENCHMARK_RADIUS_VALUES_M,
    generate_geographic_entries,
    generate_query_centers,
)

DATASET_SIZES = (1_000, 10_000, 100_000)

SERVICE = "postgres"
DATABASE = "soupdb"
USER = "soupdb"
TABLE = "lugares_bench"

STRUCTURE = "postgres_gist"


class PostgresUnavailableError(RuntimeError):
    pass


def _psql(sql: str, *, stdin: str | None = None) -> str:
    """Ejecuta SQL con psql dentro del contenedor y devuelve su salida."""
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        SERVICE,
        "psql",
        "-U",
        USER,
        "-d",
        DATABASE,
        "-v",
        "ON_ERROR_STOP=1",
        "-qAt",
        "-c",
        sql,
    ]
    try:
        completed = subprocess.run(
            command,
            input=stdin,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise PostgresUnavailableError("docker no está disponible en el PATH") from error
    except subprocess.CalledProcessError as error:
        raise PostgresUnavailableError(
            f"psql falló: {error.stderr.strip() or error.stdout.strip()}"
        ) from error

    return completed.stdout


def _copy(entries) -> None:
    """Carga las entradas con COPY: insertar una por una no termina con 100 000."""
    rows = "\n".join(f"{page}\t{slot}\t{point.x}\t{point.y}" for point, (page, slot) in entries)
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        SERVICE,
        "psql",
        "-U",
        USER,
        "-d",
        DATABASE,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        f"COPY {TABLE} (rid_page, rid_slot, lon, lat) FROM STDIN",
    ]
    subprocess.run(command, input=rows, capture_output=True, text=True, check=True)


def _prepare(entries) -> None:
    _psql("CREATE EXTENSION IF NOT EXISTS postgis")
    _psql(f"DROP TABLE IF EXISTS {TABLE}")
    _psql(
        f"CREATE TABLE {TABLE} ("
        "  rid_page int NOT NULL,"
        "  rid_slot int NOT NULL,"
        "  lon double precision NOT NULL,"
        "  lat double precision NOT NULL,"
        "  ubicacion geography(Point, 4326),"
        # La euclidiana del motor opera sobre grados, así que necesita `geometry`:
        # `geography` siempre devuelve metros sobre la esfera.
        "  plano geometry(Point, 4326)"
        ")"
    )
    _copy(entries)
    _psql(
        f"UPDATE {TABLE} SET ubicacion = ST_MakePoint(lon, lat)::geography, "
        f"plano = ST_SetSRID(ST_MakePoint(lon, lat), 4326)"
    )


def _create_index(column: str) -> tuple[float, int]:
    """Crea el índice GiST sobre una columna y devuelve (ms, bytes)."""
    name = f"{TABLE}_{column}_gix"
    _psql(f"DROP INDEX IF EXISTS {name}")
    started = time.perf_counter()
    _psql(f"CREATE INDEX {name} ON {TABLE} USING gist ({column})")
    elapsed = (time.perf_counter() - started) * 1000
    size = int(_psql(f"SELECT pg_relation_size('{name}')").strip() or 0)
    _psql(f"VACUUM ANALYZE {TABLE}")
    return elapsed, size


def _explain(sql: str) -> tuple[float, int, int]:
    """Devuelve (ms de ejecución, páginas leídas, filas) según el plan de Postgres.

    `shared_read` cuenta las páginas que Postgres trajo de disco; `shared_hit` las
    que ya tenía en su propio cache. Se suman porque del lado del motor se cuentan
    las páginas tocadas, estén o no en el buffer pool.
    """
    plan = _psql(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {sql}")
    root = json.loads(plan)[0]["Plan"]
    elapsed = json.loads(plan)[0]["Execution Time"]
    pages = root.get("Shared Hit Blocks", 0) + root.get("Shared Read Blocks", 0)
    return elapsed, pages, root.get("Actual Rows", 0)


def _measure(centers, build_sql) -> tuple[list[float], int, int]:
    """Una medición por consulta: el protocolo pide media, mediana y p95."""
    samples: list[float] = []
    pages = 0
    rows = 0
    for center in centers:
        elapsed, one_pages, one_rows = _explain(build_sql(center))
        samples.append(elapsed)
        pages += one_pages
        rows += one_rows
    return samples, pages, rows


def _result(
    operation: str,
    metric: str,
    parameter: float,
    records: int,
    samples: list[float],
    pages: int,
    rows: int,
    build_ms: float,
    size_bytes: int,
) -> SpatialResult:
    mean, median, p95 = summarize(samples)
    return SpatialResult(
        structure=STRUCTURE,
        operation=operation,
        metric=metric,
        parameter=parameter,
        records=records,
        queries=len(samples),
        build_ms=build_ms,
        elapsed_ms=median,
        mean_ms=mean,
        median_ms=median,
        p95_ms=p95,
        disk_reads=pages,
        disk_writes=0,
        size_bytes=size_bytes,
        # El proceso de PostgreSQL corre en su contenedor: su RSS no es comparable
        # con el del motor y se deja en cero en vez de inventar una equivalencia.
        peak_rss_bytes=0,
        result_count=rows,
    )


def _literal(column: str, center) -> str:
    """El punto de consulta con el mismo SRID que la columna que se compara."""
    point = f"ST_SetSRID(ST_MakePoint({center.x}, {center.y}), 4326)"
    return f"{point}::geography" if column == "ubicacion" else point


def _radius_sql(column: str, center, radius: float, spheroid: str) -> str:
    return (
        f"SELECT rid_page, rid_slot FROM {TABLE} "
        f"WHERE ST_DWithin({column}, {_literal(column, center)}, {radius}{spheroid})"
    )


def _knn_sql(column: str, center, k: int) -> str:
    return (
        f"SELECT rid_page, rid_slot FROM {TABLE} "
        f"ORDER BY {column} <-> {_literal(column, center)} LIMIT {k}"
    )


def run(sizes: tuple[int, ...] = DATASET_SIZES) -> list[SpatialResult]:
    results: list[SpatialResult] = []

    for records in sizes:
        entries = generate_geographic_entries(records, seed=SEED)
        centers = generate_query_centers(QUERIES, seed=SEED)
        _prepare(entries)

        for metric, column, radii, spheroid in (
            # `use_spheroid => false`: PostGIS mide sobre el elipsoide WGS84 por
            # defecto y el motor usa una esfera; esa diferencia movía los puntos
            # que caen justo sobre el borde del radio.
            ("haversine", "ubicacion", BENCHMARK_RADIUS_VALUES_M, ", false"),
            ("euclidean", "plano", BENCHMARK_RADIUS_VALUES_DEG, ""),
        ):
            build_ms, index_bytes = _create_index(column)

            for label, radius in zip(BENCHMARK_RADIUS_VALUES_M, radii, strict=True):
                samples, pages, rows = _measure(
                    centers,
                    lambda center, r=radius, c=column, sp=spheroid: _radius_sql(c, center, r, sp),
                )
                results.append(
                    _result(
                        f"radius_{int(label)}m",
                        metric,
                        radius,
                        records,
                        samples,
                        pages,
                        rows,
                        build_ms,
                        index_bytes,
                    )
                )

            for k in BENCHMARK_K_VALUES:
                samples, pages, rows = _measure(
                    centers,
                    lambda center, kk=k, c=column: _knn_sql(c, center, kk),
                )
                results.append(
                    _result(
                        f"knn_{k}",
                        metric,
                        k,
                        records,
                        samples,
                        pages,
                        rows,
                        build_ms,
                        index_bytes,
                    )
                )

    return results


def main() -> None:
    try:
        results = run()
    except PostgresUnavailableError as error:
        raise SystemExit(
            f"No se pudo hablar con PostgreSQL: {error}\n"
            "Levantalo con: docker compose up -d postgres"
        ) from error

    # Misma suite que compare_spatial_indexes: las tres técnicas son el mismo
    # experimento 2.2.4 y tienen que poder leerse juntas.
    path = write_spatial_results("spatial_indexes", results)
    print(f"{len(results)} mediciones escritas en {path}")


if __name__ == "__main__":
    main()
