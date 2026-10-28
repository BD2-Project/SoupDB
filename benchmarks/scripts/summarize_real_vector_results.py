"""Consolida resultados ya medidos del benchmark vectorial real (#129).

No construye ni consulta ningún índice: solo lee los CSV que escribió
``compare_vector_indexes --real-data-dir`` (una o varias ejecuciones), los valida y
produce los artefactos finales:

- ``<prefijo>.csv``: CSV maestro con el encabezado de siempre.
- ``<prefijo>_{image,audio}_{build,knn}.svg``: tiempo contra tamaño (``write_charts``).
- ``<prefijo>_summary.md``: tablas calculadas desde el CSV maestro, anomalías detectadas
  con reglas fijas e interpretación separada de los datos.

Si junto a un CSV de entrada está su galería (``<csv>_gallery.html``), el resumen la
cita y lee de ella si el top-10 exacto euclidiano coincide con el coseno. Con
``--real-data-dir`` cuenta además los vectores duplicados de cada ``vectores.bin``.

Uso::

    uv run python -m benchmarks.scripts.summarize_real_vector_results \\
        --inputs benchmarks/results/vector_indexes_real_A.csv \\
                 benchmarks/results/vector_indexes_real_B.csv \\
        --output-prefix benchmarks/results/vector_indexes_real_final
"""

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from benchmarks.scripts.compare_vector_indexes import (
    BRUTE_FORCE,
    BUFFER_CAPACITY,
    HNSW,
    IVF,
    MEDIA_TYPES,
    METRICS,
    PAGE_SIZE,
    QUERIES,
    REAL_SIZES,
    RUNS,
    TECHNIQUES,
    K,
    VectorResult,
    append_results,
    summarize,
    vectors_for,
    write_charts,
)
from engine.multimedia.vector_file import iter_vectors

#: Dimensión de los vectores reales: K = 128 palabras del codebook (#120).
REAL_DIM = 128
#: Corridas cuyo máximo/mínimo supera esto se señalan como dispersas.
DISPERSION_LIMIT = 1.5
#: Razones coseno/euclidiana fuera de [1/x, x] se señalan.
METRIC_RATIO_LIMIT = 2.0

HEADER = [f.name for f in fields(VectorResult)]
_TYPES = {f.name: f.type for f in fields(VectorResult)}
OPERATIONS = ("build", "knn")


class ValidationError(ValueError):
    """Los CSV no forman un experimento completo; lleva todos los problemas encontrados."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class Expected:
    """Lo que tiene que contener el experimento completo."""

    media: tuple[str, ...] = MEDIA_TYPES
    sizes: tuple[int, ...] = REAL_SIZES
    techniques: tuple[str, ...] = TECHNIQUES
    metrics: tuple[str, ...] = METRICS
    k: int = K
    queries: int = QUERIES
    dim: int = REAL_DIM
    runs: int = RUNS

    @property
    def rows(self) -> int:
        combos = len(self.media) * len(self.sizes) * len(self.techniques) * len(self.metrics)
        return combos * (1 + self.runs)


#: El experimento #129 completo: 2 medios × 3 tamaños × 3 técnicas × 2 métricas × 6 filas.
EXPECTED = Expected()


@dataclass
class Source:
    path: Path
    results: list[VectorResult]
    gallery: Path | None

    @property
    def datasets(self) -> list[tuple[str, int]]:
        return sorted({(r.media_type, r.dataset_size) for r in self.results})


# --- lectura y validación ---------------------------------------------------------------


def read_results(path: str | Path) -> list[VectorResult]:
    """Filas de un CSV del benchmark, con los tipos de ``VectorResult``."""
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise ValidationError([f"{Path(path).name}: encabezado {reader.fieldnames}"])
        return [VectorResult(**{c: _parse(c, v) for c, v in row.items()}) for row in reader]


def _parse(column: str, value: str) -> object:
    # Un float escrito como entero ("144" en disk_reads de build) se queda entero, así el
    # CSV maestro reproduce cada celda tal como la escribió el benchmark.
    kind = _TYPES[column]
    if kind is float and value.lstrip("-").isdigit():
        return int(value)
    return kind(value)


def _key(r: VectorResult) -> tuple[str, int, str, str]:
    return (r.media_type, r.dataset_size, r.technique, r.metric)


def validate(results: Sequence[VectorResult], expected: Expected = EXPECTED) -> None:
    """Lanza ValidationError con todos los problemas, o no hace nada si el experimento cuadra."""
    problems = []
    if len(results) != expected.rows:
        problems.append(f"{len(results)} filas, se esperaban {expected.rows}")

    measures = Counter((*_key(r), r.operation, r.run) for r in results)
    for measure, count in sorted(measures.items()):
        if count > 1:
            problems.append(f"medición duplicada {measure} ({count} veces)")

    for column, wanted in (
        ("media_type", set(expected.media)),
        ("dataset_size", set(expected.sizes)),
        ("technique", set(expected.techniques)),
        ("metric", set(expected.metrics)),
        ("operation", set(OPERATIONS)),
        ("k", {expected.k}),
        ("queries", {expected.queries}),
        ("dim", {expected.dim}),
    ):
        found = {getattr(r, column) for r in results}
        if found != wanted:
            problems.append(f"{column} = {sorted(found)}, se esperaba {sorted(wanted)}")

    groups = _groups(results)
    for media in expected.media:
        for size in expected.sizes:
            for technique in expected.techniques:
                for metric in expected.metrics:
                    key = (media, size, technique, metric)
                    group = groups.get(key, [])
                    builds = [r.run for r in group if r.operation == "build"]
                    runs = sorted(r.run for r in group if r.operation == "knn")
                    if builds != [1]:
                        problems.append(f"{key}: build en corridas {builds}, se esperaba [1]")
                    if runs != list(range(1, expected.runs + 1)):
                        problems.append(f"{key}: knn en corridas {runs}")
                    if len({r.params for r in group}) > 1:
                        problems.append(f"{key}: params distintos entre filas")
    if problems:
        raise ValidationError(problems)


def _groups(results: Iterable[VectorResult]) -> dict[tuple, list[VectorResult]]:
    groups: dict[tuple, list[VectorResult]] = defaultdict(list)
    for r in results:
        groups[_key(r)].append(r)
    return groups


def _order(r: VectorResult, expected: Expected) -> tuple:
    def position(values: Sequence, value: object) -> int:
        return values.index(value) if value in values else len(values)

    return (
        position(expected.media, r.media_type),
        r.dataset_size,
        position(expected.techniques, r.technique),
        position(expected.metrics, r.metric),
        position(OPERATIONS, r.operation),
        r.run,
    )


# --- datos derivados (solo lectura) -------------------------------------------------------


def gallery_agreement(path: Path) -> tuple[int, int, int, int]:
    """(consultas, HIT, MISS, consultas con top-10 exacto euclidiana = coseno) de una galería."""
    text = path.read_text(encoding="utf-8")
    queries = same = 0
    for section in re.split(r"(?=<h2>)", text)[1:]:
        queries += 1
        exact = {}
        for block in re.split(r"(?=<h3>)", section)[1:]:
            metric = re.match(r"<h3>([^<]+)</h3>", block).group(1)
            row = re.search(
                rf'<div class="row"><div class="label">{BRUTE_FORCE}<.*?(?=<div class="row">|$)',
                block,
                re.S,
            )
            if row:
                exact[metric] = re.findall(r'<div class="cap">id (\d+)', row.group(0))
        if len(set(map(tuple, exact.values()))) == 1 and len(exact) > 1:
            same += 1
    return queries, text.count(" HIT</div>"), text.count(" MISS</div>"), same


def count_duplicates(vectors_path: Path) -> int:
    """Vectores que repiten exactamente a otro anterior del mismo archivo."""
    seen = Counter(
        hashlib.blake2b(repr(record.vector).encode(), digest_size=16).digest()
        for record in iter_vectors(vectors_path)
    )
    return sum(count - 1 for count in seen.values())


# --- resumen ------------------------------------------------------------------------------


def _label(size: int) -> str:
    return f"{size // 1000}k" if size % 1000 == 0 else str(size)


def _num(value: float) -> str:
    # Sin notación científica: 5755151 lecturas no deben verse como 5.75515e+06.
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"


def _seconds(value_ms: float) -> str:
    seconds = value_ms / 1000
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds:,.1f} s ({seconds / 60:.1f} min)"


def _sentence(text: str) -> str:
    # Solo la primera letra en mayúscula (str.capitalize bajaría "HNSW" a "hnsw").
    return (text[:1].upper() + text[1:] + ".") if text else ""


def _ratio(a: float, b: float) -> str:
    return f"{a / b:.1f}×" if b else "∞"


class _Data:
    """Acceso por combinación a medianas (``summarize``) y filas del CSV maestro."""

    def __init__(self, results: Sequence[VectorResult], expected: Expected) -> None:
        self.expected = expected
        self.groups = _groups(results)
        self.medians = {
            (s["media_type"], s["dataset_size"], s["technique"], s["metric"], s["operation"]): s
            for s in summarize(results)
        }

    def build(self, *key: object) -> VectorResult:
        return next(r for r in self.groups[key] if r.operation == "build")

    def knn(self, *key: object) -> list[VectorResult]:
        return [r for r in self.groups[key] if r.operation == "knn"]

    def recall(self, *key: object) -> float:
        return self.medians[(*key, "knn")]["recall_at_k"]

    def query_ms(self, *key: object) -> float:
        return self.medians[(*key, "knn")]["elapsed_ms"]

    def query_reads(self, *key: object) -> float:
        return self.medians[(*key, "knn")]["disk_reads"]

    def build_ms(self, *key: object) -> float:
        return self.build(*key).elapsed_ms

    def table(self, title: str, note: str, cell: Callable[..., str]) -> list[str]:
        sizes = self.expected.sizes
        lines = [f"### {title}", "", note, ""]
        lines.append("| medio | técnica | métrica | " + " | ".join(map(_label, sizes)) + " |")
        lines.append("|---|---|---|" + "---:|" * len(sizes))
        for media in self.expected.media:
            for technique in self.expected.techniques:
                for metric in self.expected.metrics:
                    cells = " | ".join(cell(media, size, technique, metric) for size in sizes)
                    lines.append(f"| {media} | {technique} | {metric} | {cells} |")
        return [*lines, ""]

    def every(self) -> Iterable[tuple[str, int, str, str]]:
        e = self.expected
        for media in e.media:
            for size in e.sizes:
                for technique in e.techniques:
                    for metric in e.metrics:
                        yield media, size, technique, metric


def _ivf_fraction(params: str) -> float | None:
    values = dict(part.split("=") for part in params.split(";") if "=" in part)
    if "nlist" in values and "nprobe" in values:
        return min(1.0, int(values["nprobe"]) / int(values["nlist"]))
    return None


def anomalies(data: _Data) -> list[str]:
    """Hechos que rompen la tendencia general, detectados con reglas fijas."""
    e = data.expected
    found = []
    reference, other = e.metrics[0], e.metrics[-1]

    for media, size, technique, metric in data.every():
        values = [r.elapsed_ms for r in data.knn(media, size, technique, metric)]
        if min(values) > 0 and max(values) / min(values) > DISPERSION_LIMIT:
            found.append(
                f"Corridas dispersas: {media} {_label(size)} {technique} {metric},"
                f" {min(values):.2f}–{max(values):.2f} ms por consulta."
            )

    if reference != other:
        for media in e.media:
            for size in e.sizes:
                for technique in e.techniques:
                    a, b = (media, size, technique, other), (media, size, technique, reference)
                    for what, ta, tb, ra, rb in (
                        ("consulta", data.query_ms(*a), data.query_ms(*b),
                         data.query_reads(*a), data.query_reads(*b)),
                        ("construcción", data.build_ms(*a), data.build_ms(*b),
                         data.build(*a).disk_reads, data.build(*b).disk_reads),
                    ):  # fmt: skip
                        if tb and not 1 / METRIC_RATIO_LIMIT <= ta / tb <= METRIC_RATIO_LIMIT:
                            same_io = (
                                "lecturas iguales"
                                if ra == rb
                                else f"lecturas {_num(ra)} / {_num(rb)}"
                            )
                            found.append(
                                f"{what.capitalize()} {other}/{reference} = {_ratio(ta, tb)} en"
                                f" {media} {_label(size)} {technique} ({same_io})."
                            )

    for media in e.media:
        for technique in e.techniques:
            for metric in e.metrics:
                for small, large in zip(e.sizes, e.sizes[1:], strict=False):
                    t_small = data.query_ms(media, small, technique, metric)
                    t_large = data.query_ms(media, large, technique, metric)
                    if t_large < t_small:
                        pages = data.build(media, small, technique, metric).bytes_on_disk
                        pages //= PAGE_SIZE
                        fits = (
                            f"; el índice de {_label(small)} ({pages} páginas) cabe en el buffer"
                            f" de {BUFFER_CAPACITY}"
                            if pages <= BUFFER_CAPACITY
                            else ""
                        )
                        found.append(
                            f"Consulta más lenta con menos datos: {media} {technique} {metric},"
                            f" {t_small:.3f} ms a {_label(small)} frente a {t_large:.3f} ms a"
                            f" {_label(large)}{fits}."
                        )
                    if technique == BRUTE_FORCE:
                        continue
                    r_small = data.recall(media, small, technique, metric)
                    r_large = data.recall(media, large, technique, metric)
                    if r_large > r_small:
                        found.append(
                            f"Recall que sube con N: {media} {technique} {metric},"
                            f" {r_small:.3f} a {_label(small)} → {r_large:.3f} a {_label(large)}."
                        )
    return found


def interpretation(data: _Data, galleries: Sequence[tuple[int, int, int, int]]) -> list[str]:
    """Párrafos con cifras calculadas; cada afirmación se comprueba antes de escribirse."""
    e = data.expected
    top = e.sizes[-1]
    prev = e.sizes[-2] if len(e.sizes) > 1 else top
    ref = e.metrics[0]
    approx = [t for t in e.techniques if t != BRUTE_FORCE]
    lines = [
        "---",
        "",
        "## Interpretación",
        "",
        "Esta sección resume tendencias visibles en las tablas anteriores. Donde se ofrece una"
        " explicación, es una hipótesis compatible con los datos, no algo que este experimento"
        " demuestre.",
        "",
    ]

    # Escalabilidad.
    parts = []
    for media in e.media:
        if HNSW in e.techniques and IVF in e.techniques:
            hnsw = data.build_ms(media, top, HNSW, ref)
            ivf = data.build_ms(media, top, IVF, ref)
            parts.append(
                f"en {media}, construir HNSW ({ref}) a {_label(top)} costó {_seconds(hnsw)}"
                f" frente a {_seconds(ivf)} de IVF ({_ratio(hnsw, ivf)})"
            )
    sentence = "**Escalabilidad IVF frente a HNSW.** " + _sentence("; ".join(parts))
    if BRUTE_FORCE in e.techniques:
        for media in e.media:
            reads = " → ".join(f"{data.query_reads(media, s, BRUTE_FORCE, ref):g}" for s in e.sizes)
            growth = data.query_ms(media, top, BRUTE_FORCE, ref) / data.query_ms(
                media, prev, BRUTE_FORCE, ref
            )
            sentence += (
                f" La fuerza bruta en {media} lee {reads} páginas por consulta y su tiempo crece"
                f" {growth:.1f}× de {_label(prev)} a {_label(top)}."
            )
        for media in e.media:
            times = " / ".join(f"{data.query_ms(media, top, t, ref):.1f}" for t in e.techniques)
            sentence += (
                f" A {_label(top)} ({ref}, {media}) una consulta tardó {times} ms"
                f" ({' / '.join(e.techniques)})."
            )
    lines += [sentence, ""]

    # Trade-off.
    def winners(pick: Callable[[str, str], str]) -> str:
        by_winner: dict[str, list[str]] = defaultdict(list)
        for media in e.media:
            for metric in e.metrics:
                by_winner[pick(media, metric)].append(f"{media}·{metric}")
        if len(by_winner) == 1:
            return f"{next(iter(by_winner))} en todos los casos"
        return "; ".join(f"{w} en {', '.join(cases)}" for w, cases in sorted(by_winner.items()))

    slowest_build = winners(
        lambda m, mt: max(e.techniques, key=lambda t: data.build_ms(m, top, t, mt))
    )
    fastest_query = winners(
        lambda m, mt: min(e.techniques, key=lambda t: data.query_ms(m, top, t, mt))
    )
    best_recall = winners(lambda m, mt: max(approx, key=lambda t: data.recall(m, top, t, mt)))
    sentence = (
        f"**Trade-off construcción / consulta / recall.** A {_label(top)}: construcción más cara:"
        f" {slowest_build}. Consulta más rápida: {fastest_query}. Mayor recall entre las"
        f" aproximadas: {best_recall}."
    )
    if IVF in e.techniques:
        for media in e.media:
            fractions = [_ivf_fraction(data.build(media, s, IVF, ref).params) for s in e.sizes]
            recalls = [data.recall(media, s, IVF, ref) for s in e.sizes]
            if None in fractions:
                continue
            falling = all(a > b for a, b in zip(recalls, recalls[1:], strict=False))
            sentence += (
                f" IVF en {media}: fracción de listas visitadas (nprobe/nlist)"
                f" {' → '.join(f'{f:.1%}' for f in fractions)}, recall"
                f" {' → '.join(f'{r:.3f}' for r in recalls)}"
                + (
                    "; la caída es compatible con visitar una fracción menor, aunque el"
                    " experimento no varía nprobe para confirmarlo."
                    if falling
                    else "; el recall no cae de forma monótona con N."
                )
            )
    lines += [sentence, ""]

    # Imagen frente a audio.
    if len(e.media) > 1:
        cells = "; ".join(
            f"{t}: " + " / ".join(f"{data.recall(m, top, t, ref):.3f}" for m in e.media)
            for t in approx
        )
        sentence = (
            f"**{_sentence(' frente a '.join(e.media))}** Recall a {_label(top)} ({ref},"
            f" {' / '.join(e.media)}) con los mismos parámetros: {cells}."
        )
        if HNSW in e.techniques:
            reads = " / ".join(f"{data.build(m, top, HNSW, ref).disk_reads:,}" for m in e.media)
            sentence += f" Lecturas al construir HNSW a {_label(top)}: {reads} páginas."
        sentence += " Los datos muestran estas diferencias, no su causa."
        lines += [sentence, ""]

    # Normalización L2.
    sentence = (
        "**Vectores L2-normalizados.** Los vectores TF-IDF del pipeline tienen norma 1"
        " (`tfidf_vector`), y para vectores unitarios `||a − b||² = 2 − 2·cos(a, b)`. La distancia"
        " euclidiana es entonces una función monótona decreciente del coseno, y la fuerza bruta"
        " debe ordenar igual con las dos métricas salvo empates y redondeo float32."
    )
    shown = sum(g[0] for g in galleries)
    if shown:
        same = sum(g[3] for g in galleries)
        sentence += (
            f" En las galerías, el top-10 exacto euclidiano coincide con el coseno en {same} de"
            f" {shown} consultas mostradas."
        )
    if BRUTE_FORCE in e.techniques and len(e.metrics) > 1:
        equal_io = all(
            len({data.query_reads(m, s, BRUTE_FORCE, mt) for mt in e.metrics}) == 1
            for m in e.media
            for s in e.sizes
        )
        if equal_io:
            sentence += (
                " La fuerza bruta lee las mismas páginas con las dos métricas, así que su"
                " diferencia de tiempo entre métricas no viene del I/O. Estos datos no permiten"
                " separar el costo de cada fórmula del ruido de medición."
            )
    lines += [sentence, ""]

    # IVF y la métrica.
    if IVF in e.techniques and len(e.metrics) > 1:
        cases = [(m, s) for m in e.media for s in e.sizes]
        differ = [
            (m, s)
            for m, s in cases
            if len({data.query_reads(m, s, IVF, mt) for mt in e.metrics}) > 1
        ]
        gap = max(
            max(data.recall(m, s, IVF, mt) for mt in e.metrics)
            - min(data.recall(m, s, IVF, mt) for mt in e.metrics)
            for m, s in cases
        )
        lines += [
            "**IVF y la métrica.** Los centroides de IVF son promedios de vectores unitarios y"
            " no tienen norma 1. Ordenarlos por euclidiana o por coseno puede elegir listas"
            f" distintas. Las lecturas por consulta de IVF difieren entre métricas en"
            f" {len(differ)} de {len(cases)} datasets, y la mayor diferencia de recall entre"
            f" métricas es {gap:.3f}.",
            "",
        ]

    # HNSW y la métrica.
    if HNSW in e.techniques and len(e.metrics) > 1:
        other = e.metrics[-1]
        recall_diff = [
            f"{m} {_label(s)} ("
            + " / ".join(f"{data.recall(m, s, HNSW, mt):.3f}" for mt in e.metrics)
            + ")"
            for m in e.media
            for s in e.sizes
            if len({data.recall(m, s, HNSW, mt) for mt in e.metrics}) > 1
        ]
        ratios = [
            data.build_ms(m, s, HNSW, other) / data.build_ms(m, s, HNSW, ref)
            for m in e.media
            for s in e.sizes
        ]
        reads_gap = max(
            abs(data.build(m, s, HNSW, other).disk_reads - data.build(m, s, HNSW, ref).disk_reads)
            / max(1, data.build(m, s, HNSW, ref).disk_reads)
            for m in e.media
            for s in e.sizes
        )
        lines += [
            "**HNSW y la métrica.** HNSW guarda la métrica y se construyó un grafo por métrica."
            + (
                f" El recall difiere entre métricas en: {'; '.join(recall_diff)}."
                if recall_diff
                else " El recall coincide entre métricas en todos los datasets."
            )
            + f" La construcción con {other} tardó entre {min(ratios):.2f}× y {max(ratios):.2f}×"
            f" lo de {ref}, con lecturas que difieren como mucho un {reads_gap:.2%}. Los datos"
            " no explican esa diferencia de tiempo.",
            "",
        ]
    return lines


def render_summary(
    results: Sequence[VectorResult],
    sources: Sequence[Source],
    master: Path,
    svgs: Sequence[Path],
    expected: Expected = EXPECTED,
    duplicates: dict[tuple[str, int], int] | None = None,
) -> str:
    data = _Data(results, expected)
    galleries = [(s, gallery_agreement(s.gallery)) for s in sources if s.gallery is not None]
    md = [
        "# Benchmark vectorial real (#129): resultados finales",
        "",
        f"Generado por `benchmarks/scripts/summarize_real_vector_results.py` desde"
        f" `{master.name}`. Ningún número de este documento se escribió a mano, y no se volvió"
        " a ejecutar ningún índice. Técnicas: "
        + ", ".join(expected.techniques)
        + "; métricas: "
        + ", ".join(expected.metrics)
        + f"; k = {expected.k}; {expected.queries} consultas por dataset (objetos del propio"
        f" dataset, sin contarse a sí mismos); {expected.runs} corridas de consulta por"
        f" combinación; dimensión {expected.dim}.",
        "",
        "## Origen de los datos",
        "",
        "| CSV de origen | datasets | filas | galería |",
        "|---|---|---:|---|",
    ]
    for source in sources:
        datasets = ", ".join(f"{m} {_label(s)}" for m, s in source.datasets)
        gallery = f"`{source.gallery.name}`" if source.gallery else "—"
        md.append(f"| `{source.path.name}` | {datasets} | {len(source.results)} | {gallery} |")
    params = {
        t: " · ".join(
            f"{_label(s)}: `{data.build(expected.media[0], s, t, expected.metrics[0]).params}`"
            for s in expected.sizes
        )
        for t in expected.techniques
        if t != BRUTE_FORCE
    }
    md += [
        "",
        f"Consolidado: `{master.name}`, **{len(results)} filas** = {len(expected.media)} medios ×"
        f" {len(expected.sizes)} tamaños × {len(expected.techniques)} técnicas ×"
        f" {len(expected.metrics)} métricas × (1 build + {expected.runs} knn).",
        "",
        "Validado: sin mediciones duplicadas; medios, tamaños, técnicas y métricas exactamente"
        f" los esperados; k = {expected.k}; queries = {expected.queries}; dim = {expected.dim};"
        f" 1 build y {expected.runs} knn por combinación, con los mismos `params`.",
        "",
        "Parámetros: " + "; ".join(f"{t.upper()} {p}" for t, p in params.items()) + ".",
        "",
        "## Datos",
        "",
    ]
    stable = all(len({r.recall_at_k for r in data.knn(*c)}) == 1 for c in data.every())
    md += data.table(
        "Recall@10",
        f"Mediana de las {expected.runs} corridas"
        + (" (en este experimento todas dan el mismo valor)" if stable else "")
        + ". Contra la fuerza bruta de la misma métrica; la fuerza bruta es 1 por definición.",
        lambda *c: f"{data.recall(*c):.3f}",
    )
    md += data.table(
        "Tiempo de construcción",
        "Una construcción por combinación (`operation = build`).",
        lambda *c: _seconds(data.build_ms(*c)),
    )

    def query_cell(*c: object) -> str:
        values = [r.elapsed_ms for r in data.knn(*c)]
        return f"{data.query_ms(*c):.3f} ({min(values):.2f}–{max(values):.2f})"

    md += data.table(
        "Tiempo de consulta",
        f"ms por consulta: mediana de las {expected.runs} corridas (mínimo–máximo entre corridas).",
        query_cell,
    )
    md += data.table(
        "Almacenamiento",
        "`bytes_on_disk` del índice en MiB (2^20 bytes); el valor exacto en bytes está en el CSV.",
        lambda *c: f"{data.build(*c).bytes_on_disk / 2**20:.2f}",
    )
    md += data.table(
        "I/O de construcción",
        f"Páginas de {PAGE_SIZE // 1024} KiB: lecturas / escrituras durante la construcción.",
        lambda *c: f"{data.build(*c).disk_reads:,.0f} / {data.build(*c).disk_writes:,}",
    )
    md += data.table(
        "I/O de consulta",
        "Lecturas de página por consulta (mediana de las corridas; cada corrida reabre el índice"
        f" con el buffer pool vacío, {BUFFER_CAPACITY} páginas).",
        lambda *c: f"{data.query_reads(*c):g}",
    )
    md += data.table(
        "Memoria",
        "KiB: retenida al reabrir el índice (build) / pico de Python durante las consultas (knn).",
        lambda *c: (
            f"{data.build(*c).memory_bytes / 1024:.1f} / {data.knn(*c)[0].memory_bytes / 1024:.1f}"
        ),
    )
    if duplicates:
        md += [
            "### Vectores duplicados",
            "",
            "Vectores de cada `vectores.bin` idénticos a otro anterior del mismo archivo.",
            "",
            "| medio | " + " | ".join(map(_label, expected.sizes)) + " |",
            "|---|" + "---:|" * len(expected.sizes),
        ]
        for media in expected.media:
            cells = " | ".join(str(duplicates.get((media, s), "—")) for s in expected.sizes)
            md.append(f"| {media} | {cells} |")
        md.append("")

    found = anomalies(data)
    md += [
        "### Anomalías observadas",
        "",
        "Hechos que no siguen la tendencia general, detectados con reglas fijas del script"
        f" (corridas con máximo/mínimo > {DISPERSION_LIMIT}; razón entre métricas fuera de"
        f" [1/{METRIC_RATIO_LIMIT:g}, {METRIC_RATIO_LIMIT:g}]; consulta más lenta con menos datos;"
        " recall que sube con N). Se dejan como están, sin corregir."
        + (
            " Los tamaños vienen de ejecuciones distintas, así que las comparaciones de tiempo"
            " entre tamaños mezclan ejecuciones."
            if len(sources) > 1
            else ""
        ),
        "",
    ]
    md += [f"- {item}" for item in found] or ["- Ninguna."]
    md += [
        "",
        "## Gráficas",
        "",
        "Generadas desde el CSV maestro con `write_charts` (log-log; color = técnica, trazo"
        " continuo = euclidiana, discontinuo = coseno):",
        "",
        *[f"- `{svg.name}`" for svg in svgs],
        "",
        "## Galerías (evidencia de las ejecuciones originales)",
        "",
    ]
    if galleries:
        md += [
            "No se regeneran. Enlazan por `file://` a los datasets fuente, así que las imágenes"
            " y audios solo se ven en la máquina que los tiene. No se copió ningún archivo"
            " multimedia.",
            "",
            "| galería | datasets | consultas | HIT | MISS | top-10 exacto euclidiana = coseno |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for source, (queries, hits, misses, same) in galleries:
            datasets = ", ".join(f"{m} {_label(s)}" for m, s in source.datasets)
            md.append(
                f"| `{source.gallery.name}` | {datasets} | {queries} | {hits} | {misses} |"
                f" {same}/{queries} |"
            )
    else:
        md.append("Ningún CSV de entrada tiene su galería al lado (`<csv>_gallery.html`).")
    md += ["", *interpretation(data, [g for _s, g in galleries])]

    sentence = (
        "**Limitación del Recall@10 con empates.** Entre vectores empatados (duplicados), el"
        " orden depende de la técnica, y una técnica aproximada puede devolver un empatado distinto"
        " del que eligió la fuerza bruta. Esa respuesta es igual de cercana pero cuenta como MISS,"
        " así que el Recall@10 puede quedar algo subestimado; el experimento no cuantifica cuánto."
    )
    if duplicates:
        with_dups = [f"{m} {_label(s)}: {n}" for (m, s), n in sorted(duplicates.items()) if n]
        sentence += (
            f" Datasets con duplicados: {', '.join(with_dups)}." if with_dups else
            " Ningún dataset contiene vectores duplicados."
        )  # fmt: skip
    else:
        sentence += " (Para contar los duplicados, pasa `--real-data-dir`.)"
    md += [sentence, ""]
    return "\n".join(md)


# --- orquestación -------------------------------------------------------------------------


def consolidate(
    inputs: Sequence[str | Path],
    output_prefix: str | Path,
    expected: Expected = EXPECTED,
    real_data_dir: str | Path | None = None,
) -> tuple[Path, list[Path], Path]:
    """Valida los CSV y escribe CSV maestro, SVG y resumen. No escribe nada si no validan."""
    sources = []
    for path in map(Path, inputs):
        gallery = path.with_name(f"{path.stem}_gallery.html")
        sources.append(Source(path, read_results(path), gallery if gallery.is_file() else None))
    results = [r for source in sources for r in source.results]
    validate(results, expected)
    results.sort(key=lambda r: _order(r, expected))

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    master = prefix.with_name(f"{prefix.name}.csv")
    if master.resolve() in {p.resolve() for p in map(Path, inputs)}:
        raise ValidationError([f"{master.name} es también una entrada"])
    master.unlink(missing_ok=True)
    append_results(master, results)
    if read_results(master) != results:
        raise ValidationError([f"{master.name} no reproduce las filas de entrada"])

    svgs = write_charts(results, master)
    duplicates = None
    if real_data_dir is not None:
        duplicates = {
            (m, s): count_duplicates(vectors_for(real_data_dir, m, s))
            for m in expected.media
            for s in expected.sizes
            if vectors_for(real_data_dir, m, s).is_file()
        }
    summary = prefix.with_name(f"{prefix.name}_summary.md")
    text = render_summary(results, sources, master, svgs, expected, duplicates)
    summary.write_text(text, encoding="utf-8")
    return master, svgs, summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="CSV ya medidos")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help="p. ej. benchmarks/results/vector_indexes_real_final",
    )
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        help="opcional: carpeta con images/ y audio/ para contar vectores duplicados",
    )
    args = parser.parse_args(argv)
    try:
        master, svgs, summary = consolidate(
            args.inputs, args.output_prefix, real_data_dir=args.real_data_dir
        )
    except ValidationError as error:
        print("Validación fallida; no se escribió nada:", file=sys.stderr)
        for problem in error.problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"{master}")
    for svg in svgs:
        print(f"{svg}")
    print(f"{summary}")


if __name__ == "__main__":
    main()
