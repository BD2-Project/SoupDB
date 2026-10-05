"""Genera las tablas LaTeX de la sección experimental del informe.

Los números se leen de los CSV de `benchmarks/results/`, no se transcriben a
mano: una tabla escrita a mano se desactualiza en cuanto alguien vuelve a correr
el benchmark, y un dígito mal copiado en el informe es indefendible.

    uv run python -m benchmarks.scripts.export_informe_tablas > tablas.tex
"""

import csv
import glob
from collections import defaultdict
from pathlib import Path

RESULTS = "benchmarks/results/spatial_indexes_*.csv"

STRUCTURE_LABEL = {
    "sequential_scan": "Secuencial",
    "rtree": "R-Tree",
    "postgres_gist": "PostgreSQL GiST",
}
ORDER = ("sequential_scan", "rtree", "postgres_gist")
OPERATIONS = (
    ("radius_1000m", "Radio 1 km"),
    ("radius_5000m", "Radio 5 km"),
    ("radius_10000m", "Radio 10 km"),
    ("knn_10", "k-NN, k=10"),
    ("knn_50", "k-NN, k=50"),
    ("knn_100", "k-NN, k=100"),
)


def load() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in glob.glob(RESULTS):
        with Path(path).open(encoding="utf-8") as file:
            reader = csv.DictReader(file)
            # Solo el esquema nuevo trae media/p95; los CSV viejos se descartan.
            if "p95_ms" not in (reader.fieldnames or []):
                continue
            rows.extend(reader)
    return rows


def number(value: float, decimals: int = 2) -> str:
    """Formato es-PE con separador de miles fino, legible en LaTeX."""
    text = f"{value:,.{decimals}f}".replace(",", "\\,").replace(".", ",")
    return text


def table_times(rows, records: int, metric: str) -> str:
    index = defaultdict(dict)
    for row in rows:
        if int(row["records"]) == records and row["metric"] == metric:
            index[row["operation"]][row["structure"]] = row

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Tiempo por consulta con {number(records, 0)} registros, "
        f"métrica {metric}. Media / mediana / p95 sobre 100 consultas.}}",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Operación & Técnica & Media (ms) & Mediana (ms) & p95 (ms) \\\\",
        "\\midrule",
    ]

    for operation, label in OPERATIONS:
        present = [s for s in ORDER if s in index.get(operation, {})]
        for position, structure in enumerate(present):
            row = index[operation][structure]
            name = label if position == 0 else ""
            lines.append(
                f"{name} & {STRUCTURE_LABEL[structure]} & "
                f"{number(float(row['mean_ms']))} & "
                f"{number(float(row['median_ms']))} & "
                f"{number(float(row['p95_ms']))} \\\\"
            )
        lines.append("\\addlinespace")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def table_reads(rows, metric: str) -> str:
    index = defaultdict(dict)
    for row in rows:
        if row["metric"] == metric:
            index[(row["operation"], int(row["records"]))][row["structure"]] = row

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Páginas leídas en 100 consultas, métrica {metric}.}}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Operación & N & Secuencial & R-Tree & GiST \\\\",
        "\\midrule",
    ]

    for operation, label in OPERATIONS:
        for records in (1_000, 10_000, 100_000):
            entry = index.get((operation, records), {})
            if not entry:
                continue
            cells = []
            for structure in ORDER:
                row = entry.get(structure)
                cells.append(number(float(row["disk_reads"]), 0) if row else "---")
            name = label if records == 1_000 else ""
            lines.append(f"{name} & {number(records, 0)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def table_build(rows) -> str:
    index = defaultdict(dict)
    for row in rows:
        index[int(row["records"])][row["structure"]] = row

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Construcción y huella física.}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Técnica & N & Construcción (ms) & Tamaño (MB) \\\\",
        "\\midrule",
    ]

    for structure in ORDER:
        for records in (1_000, 10_000, 100_000):
            row = index.get(records, {}).get(structure)
            if not row:
                continue
            size = int(row["size_bytes"]) / 1_048_576
            lines.append(
                f"{STRUCTURE_LABEL[structure]} & {number(records, 0)} & "
                f"{number(float(row['build_ms']))} & {number(size, 2)} \\\\"
            )
        lines.append("\\addlinespace")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit("No hay CSV con el esquema del protocolo en benchmarks/results/")

    print("% Generado por benchmarks/scripts/export_informe_tablas.py")
    print(table_build(rows))
    for metric in ("haversine", "euclidean"):
        print(table_times(rows, 100_000, metric))
    for metric in ("haversine", "euclidean"):
        print(table_reads(rows, metric))


if __name__ == "__main__":
    main()
