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
import statistics
import subprocess
import time

from benchmarks.harness import BenchmarkResult, write_results
from benchmarks.spatial_workload import (
    BENCHMARK_K_VALUES,
    BENCHMARK_RADIUS_VALUES_M,
    build_spatial_workload,
)

DATASET_SIZES = (1_000, 10_000, 100_000)
RUNS = 5
QUERIES_PER_PARAMETER = 20
SEED = 20260404

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
        "  ubicacion geography(Point, 4326)"
        ")"
    )
    _copy(entries)
    _psql(f"UPDATE {TABLE} SET ubicacion = ST_MakePoint(lon, lat)::geography")


def _create_index() -> tuple[float, int]:
    """Crea el índice GiST y devuelve (ms, bytes). Es el equivalente a `build`."""
    _psql(f"DROP INDEX IF EXISTS {TABLE}_geo")
    started = time.perf_counter()
    _psql(f"CREATE INDEX {TABLE}_geo ON {TABLE} USING gist (ubicacion)")
    elapsed = (time.perf_counter() - started) * 1000
    size = int(_psql(f"SELECT pg_relation_size('{TABLE}_geo')").strip() or 0)
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


def _measure(centers, build_sql) -> list[tuple[float, int, int]]:
    samples = []
    for _ in range(RUNS):
        elapsed = 0.0
        pages = 0
        rows = 0
        for center in centers:
            one_elapsed, one_pages, one_rows = _explain(build_sql(center))
            elapsed += one_elapsed
            pages += one_pages
            rows += one_rows
        samples.append((elapsed, pages, rows))
    return samples


def _result(operation: str, records: int, samples, size_bytes: int) -> BenchmarkResult:
    elapsed = statistics.median(sample[0] for sample in samples)
    pages = int(statistics.median(sample[1] for sample in samples))
    rows = int(statistics.median(sample[2] for sample in samples))

    return BenchmarkResult(
        structure=STRUCTURE,
        operation=operation,
        records=records,
        operations=QUERIES_PER_PARAMETER,
        elapsed_ms=elapsed,
        ops_per_second=(QUERIES_PER_PARAMETER / (elapsed / 1000)) if elapsed > 0 else 0.0,
        disk_reads=pages,
        disk_writes=0,
        size_bytes=size_bytes,
        result_count=rows,
    )


def run(sizes: tuple[int, ...] = DATASET_SIZES) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for records in sizes:
        workload = build_spatial_workload(
            entry_count=records,
            query_count=QUERIES_PER_PARAMETER,
            seed=SEED,
        )
        _prepare(workload.entries)
        build_ms, index_bytes = _create_index()

        results.append(
            BenchmarkResult(
                structure=STRUCTURE,
                operation="build",
                records=records,
                operations=records,
                elapsed_ms=build_ms,
                ops_per_second=(records / (build_ms / 1000)) if build_ms > 0 else 0.0,
                disk_reads=0,
                disk_writes=0,
                size_bytes=index_bytes,
                result_count=records,
            )
        )

        centers = [query.center for query in workload.radius_queries][:QUERIES_PER_PARAMETER]

        for radius in BENCHMARK_RADIUS_VALUES_M:
            samples = _measure(
                centers,
                lambda center, radius=radius: (
                    f"SELECT rid_page, rid_slot FROM {TABLE} "
                    f"WHERE ST_DWithin(ubicacion, "
                    f"ST_MakePoint({center.x}, {center.y})::geography, {radius}, false)"
                ),
            )
            results.append(_result(f"radius_{int(radius)}m", records, samples, index_bytes))

        for k in BENCHMARK_K_VALUES:
            samples = _measure(
                centers,
                lambda center, k=k: (
                    f"SELECT rid_page, rid_slot FROM {TABLE} "
                    f"ORDER BY ubicacion <-> ST_MakePoint({center.x}, {center.y})::geography "
                    f"LIMIT {k}"
                ),
            )
            results.append(_result(f"knn_{k}", records, samples, index_bytes))

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
    path = write_results("spatial_indexes", results)
    print(f"{len(results)} mediciones escritas en {path}")


if __name__ == "__main__":
    main()
