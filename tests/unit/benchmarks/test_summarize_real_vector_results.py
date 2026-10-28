"""Consolidador de resultados reales, con CSV artificiales en tmp_path (sin índices)."""

import csv
import math
import re
from dataclasses import replace

import pytest

from benchmarks.scripts import compare_vector_indexes as bench
from benchmarks.scripts import summarize_real_vector_results as summary
from benchmarks.scripts.compare_vector_indexes import VectorResult, append_results
from benchmarks.scripts.summarize_real_vector_results import (
    EXPECTED,
    ValidationError,
    consolidate,
    gallery_agreement,
    main,
    read_results,
    validate,
)
from engine.multimedia.vector_file import VectorRecord, write_vectors

BUILD_MS = {"brute_force": 50.0, "ivf": 2000.0, "hnsw": 20000.0}
QUERY_MS = {"brute_force": 0.02, "ivf": 0.004, "hnsw": 0.002}


def _params(technique, size):
    if technique == "ivf":
        return f"nlist={round(math.sqrt(size))};nprobe=8"
    if technique == "hnsw":
        return "M=16;ef_construction=100;ef_search=64"
    return ""


def _recall(media, size, technique):
    if technique == "brute_force":
        return 1.0
    step = (1000, 10000, 100000).index(size)
    base = {"ivf": 0.98, "hnsw": 1.0}[technique]
    return round(
        base - (0.04 if technique == "ivf" else 0.01) * step - (media == "audio") * 0.002, 3
    )


def _rows(media, size):
    rows = []
    for technique in ("brute_force", "ivf", "hnsw"):
        for metric in ("euclidean", "cosine"):
            slower = 1.5 if metric == "cosine" else 1.0
            common = {
                "media_type": media,
                "dataset_size": size,
                "technique": technique,
                "metric": metric,
                "bytes_on_disk": size * 600,
                "dim": 128,
                "queries": 50,
                "k": 10,
                "params": _params(technique, size),
            }
            rows.append(
                VectorResult(
                    operation="build",
                    run=1,
                    elapsed_ms=BUILD_MS[technique] * size / 1000 * slower,
                    disk_reads=float(size // 7),
                    disk_writes=size // 4,
                    memory_bytes=15000,
                    recall_at_k=0.0,
                    **common,
                )
            )
            for run in range(1, 6):
                rows.append(
                    VectorResult(
                        operation="knn",
                        run=run,
                        elapsed_ms=QUERY_MS[technique] * size * slower * (1 + run / 100),
                        disk_reads=size / 70,
                        disk_writes=0,
                        memory_bytes=650000,
                        recall_at_k=_recall(media, size, technique),
                        **common,
                    )
                )
    return rows


def _write(path, rows):
    append_results(path, rows)
    return path


@pytest.fixture
def inputs(tmp_path):
    """Cuatro CSV como las ejecuciones reales: 1k, 10k, 100k imagen, 100k audio."""
    results = tmp_path / "results"
    results.mkdir()
    return [
        _write(results / "run_a.csv", _rows("image", 1000) + _rows("audio", 1000)),
        _write(results / "run_b.csv", _rows("image", 10000) + _rows("audio", 10000)),
        _write(results / "run_c.csv", _rows("image", 100000)),
        _write(results / "run_d.csv", _rows("audio", 100000)),
    ]


def _all_rows(inputs):
    return [r for path in inputs for r in read_results(path)]


def _outputs(prefix):
    return sorted(prefix.parent.glob(f"{prefix.name}*"))


# --- consolidación ------------------------------------------------------------------------


def test_valid_inputs_give_a_master_csv_with_every_row(tmp_path, inputs):
    prefix = tmp_path / "out" / "final"

    master, svgs, summary_path = consolidate(inputs, prefix)

    rows = read_results(master)
    assert len(rows) == EXPECTED.rows == 216
    assert sorted(map(repr, rows)) == sorted(map(repr, _all_rows(inputs)))
    with master.open(encoding="utf-8") as f:
        assert next(csv.reader(f)) == summary.HEADER
    # Orden estable: medio, tamaño, técnica, métrica, build antes que knn, corrida.
    assert (rows[0].media_type, rows[0].dataset_size, rows[0].technique, rows[0].operation) == (
        "image",
        1000,
        "brute_force",
        "build",
    )
    assert [r.run for r in rows[:6]] == [1, 1, 2, 3, 4, 5]
    assert master.name == "final.csv"
    assert summary_path.name == "final_summary.md"
    assert len(svgs) == 4


def test_every_cell_is_copied_exactly(tmp_path, inputs):
    # Floats con muchos decimales y floats escritos como entero ("144") salen idénticos.
    rows = read_results(inputs[0])
    rows[0] = replace(rows[0], disk_reads=144)
    rows[1] = replace(rows[1], recall_at_k=0.9940000000000001, elapsed_ms=8.180123456789)
    inputs[0].unlink()
    _write(inputs[0], rows)

    master, _svgs, _summary = consolidate(inputs, tmp_path / "final")

    def lines(path):
        return path.read_text(encoding="utf-8").splitlines()[1:]

    assert sorted(lines(master)) == sorted(line for path in inputs for line in lines(path))
    text = master.read_text(encoding="utf-8")
    assert "0.9940000000000001" in text and "8.180123456789" in text
    assert ",144,250," in text


def _rewrite(path, transform):
    rows = transform(read_results(path))
    path.unlink()
    _write(path, rows)


def test_duplicated_measure_is_rejected_and_nothing_is_written(tmp_path, inputs):
    _rewrite(inputs[2], lambda rows: rows + rows[:1])
    prefix = tmp_path / "final"

    with pytest.raises(ValidationError) as error:
        consolidate(inputs, prefix)

    assert any("duplicada" in p for p in error.value.problems)
    assert any("217 filas" in p for p in error.value.problems)
    assert _outputs(prefix) == []


def test_same_input_twice_is_a_duplicate(tmp_path, inputs):
    with pytest.raises(ValidationError, match="duplicada"):
        consolidate([*inputs, inputs[3]], tmp_path / "final")


def test_missing_run_is_reported(tmp_path, inputs):
    def drop_run_5(rows):
        return [
            r
            for r in rows
            if not (r.technique == "ivf" and r.metric == "cosine" and r.run == 5)
            or r.media_type != "audio"
        ]

    _rewrite(inputs[3], drop_run_5)

    with pytest.raises(ValidationError) as error:
        consolidate(inputs, tmp_path / "final")

    assert (
        "('audio', 100000, 'ivf', 'cosine'): knn en corridas [1, 2, 3, 4]" in error.value.problems
    )


def test_missing_dataset_is_reported(tmp_path, inputs):
    with pytest.raises(ValidationError) as error:
        consolidate(inputs[:3], tmp_path / "final")

    problems = "\n".join(error.value.problems)
    assert "180 filas, se esperaban 216" in problems
    assert "('audio', 100000, 'brute_force', 'euclidean'): build en corridas []" in problems


@pytest.mark.parametrize(
    ("column", "value"),
    [("dim", 64), ("k", 5), ("queries", 49), ("media_type", "synthetic"), ("metric", "inner")],
)
def test_inconsistent_parameters_are_reported(tmp_path, inputs, column, value):
    _rewrite(inputs[1], lambda rows: [replace(rows[0], **{column: value}), *rows[1:]])

    with pytest.raises(ValidationError) as error:
        consolidate(inputs, tmp_path / "final")

    assert any(p.startswith(f"{column} = ") for p in error.value.problems)


def test_wrong_header_is_rejected(tmp_path, inputs):
    inputs[0].write_text("technique,operation\nivf,knn\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="encabezado"):
        consolidate(inputs, tmp_path / "final")


def test_master_cannot_overwrite_an_input(tmp_path, inputs):
    with pytest.raises(ValidationError, match="también una entrada"):
        consolidate(inputs, inputs[0].with_suffix(""))


def test_validate_accepts_the_complete_experiment(inputs):
    validate(_all_rows(inputs))


# --- gráficas -----------------------------------------------------------------------------


def test_four_svgs_with_every_size_technique_and_metric(tmp_path, inputs):
    _master, svgs, _summary = consolidate(inputs, tmp_path / "final")

    assert sorted(p.name for p in svgs) == [
        "final_audio_build.svg",
        "final_audio_knn.svg",
        "final_image_build.svg",
        "final_image_knn.svg",
    ]
    for svg in svgs:
        text = svg.read_text(encoding="utf-8")
        polylines = re.findall(r'<polyline points="([^"]+)"', text)
        assert len(polylines) == 6
        assert all(len(points.split()) == 3 for points in polylines)
        for size in (1000, 10000, 100000):
            assert f">{size}</text>" in text
        for technique in ("brute_force", "ivf", "hnsw"):
            for metric in ("euclidean", "cosine"):
                assert f">{technique} · {metric}</text>" in text


# --- resumen ------------------------------------------------------------------------------


def _summary(tmp_path, inputs, **kwargs):
    _master, _svgs, path = consolidate(inputs, tmp_path / "final", **kwargs)
    return path.read_text(encoding="utf-8")


def _cell(text, title, media, technique, metric, column):
    """Celda de una tabla del resumen: column 0 = 1k, 1 = 10k, 2 = 100k."""
    section = text.split(f"### {title}")[1].split("###")[0]
    row = next(
        line
        for line in section.splitlines()
        if line.startswith(f"| {media} | {technique} | {metric} |")
    )
    return row.strip("|").split("|")[3 + column].strip()


def test_summary_has_every_section_and_interpretation_apart(tmp_path, inputs):
    text = _summary(tmp_path, inputs)

    for title in (
        "Recall@10",
        "Tiempo de construcción",
        "Tiempo de consulta",
        "Almacenamiento",
        "I/O de construcción",
        "I/O de consulta",
        "Memoria",
        "Anomalías observadas",
    ):
        assert f"### {title}" in text
    data, interpretation = text.split("\n---\n")
    assert "## Datos" in data and "## Interpretación" not in data
    assert "## Interpretación" in interpretation and "|---" not in interpretation
    assert "2 − 2·cos(a, b)" in interpretation
    assert "**216 filas**" in data
    for name in ("run_a.csv", "run_b.csv", "run_c.csv", "run_d.csv"):
        assert f"`{name}`" in data


def test_summary_numbers_come_from_the_data(tmp_path, inputs):
    text = _summary(tmp_path, inputs)
    assert _cell(text, "Recall@10", "image", "ivf", "euclidean", 2) == "0.900"
    assert _cell(text, "Tiempo de construcción", "image", "hnsw", "cosine", 2) == (
        "3,000.0 s (50.0 min)"
    )
    # Mediana de 5 corridas con 1 %…5 % de más: la tercera, 3 %.
    assert _cell(text, "Tiempo de consulta", "audio", "brute_force", "euclidean", 0).startswith(
        "20.600 (20.20–21.00)"
    )
    assert _cell(text, "Almacenamiento", "audio", "hnsw", "euclidean", 1) == "5.72"

    # Si cambia el dato, cambia el resumen.
    _rewrite(
        inputs[2],
        lambda rows: [
            replace(r, recall_at_k=0.5)
            if (r.technique, r.metric, r.operation) == ("ivf", "euclidean", "knn")
            else r
            for r in rows
        ],
    )
    changed = _summary(tmp_path, inputs)
    assert _cell(changed, "Recall@10", "image", "ivf", "euclidean", 2) == "0.500"


def test_anomalies_are_detected_from_the_data(tmp_path, inputs):
    clean = _summary(tmp_path, inputs).split("### Anomalías observadas")[1].split("##")[0]
    assert "- Ninguna." in clean

    def inject(rows):
        out = []
        for r in rows:
            key = (r.media_type, r.technique, r.metric, r.operation)
            if key == ("audio", "hnsw", "cosine", "knn") and r.run == 2:
                r = replace(r, elapsed_ms=r.elapsed_ms * 3)  # corrida dispersa
            if key[1:] == ("ivf", "euclidean", "knn"):
                r = replace(r, recall_at_k=0.99)  # recall que sube de 10k a 100k
            out.append(r)
        return out

    _rewrite(inputs[3], inject)
    found = _summary(tmp_path, inputs).split("### Anomalías observadas")[1].split("##")[0]

    assert "Corridas dispersas: audio 100k hnsw cosine" in found
    assert "Recall que sube con N: audio ivf euclidean, 0.938 a 10k → 0.990 a 100k" in found
    assert "Ninguna" not in found


def test_summary_reads_existing_galleries_without_searching(tmp_path, inputs):
    info = [bench.ObjectInfo(i, f"{i:012d}.jpg", "image") for i in range(12)]
    exact = info[1:11]
    entry = bench.GalleryQuery(
        "image",
        1000,
        info[0],
        {
            ("euclidean", "brute_force"): exact,
            ("cosine", "brute_force"): exact,
            ("euclidean", "ivf"): [*exact[:9], info[11]],
            ("cosine", "ivf"): exact,
            ("euclidean", "hnsw"): exact,
            ("cosine", "hnsw"): exact,
        },
    )
    gallery = inputs[0].with_name("run_a_gallery.html")
    gallery.write_text(bench.render_gallery([entry], {}), encoding="utf-8")

    assert gallery_agreement(gallery) == (1, 39, 1, 1)
    text = _summary(tmp_path, inputs)
    assert "| `run_a_gallery.html` | audio 1k, image 1k | 1 | 39 | 1 | 1/1 |" in text
    assert "coincide con el coseno en 1 de 1 consultas" in text


def test_duplicate_vectors_are_counted_with_real_data_dir(tmp_path, inputs):
    data = tmp_path / "data"
    for media, folder in bench.MEDIA_DIRS.items():
        path = bench.vectors_for(data, media, 1000)
        path.parent.mkdir(parents=True)
        vectors = [[float(i % (3 if media == "audio" else 9))] * 128 for i in range(9)]
        records = [VectorRecord(i, f"{folder}/{i}", v) for i, v in enumerate(vectors)]
        write_vectors(path, records, dimension=128, count=len(records))

    text = _summary(tmp_path, inputs, real_data_dir=data)

    assert "### Vectores duplicados" in text
    assert "| image | 0 | — | — |" in text
    assert "| audio | 6 | — | — |" in text
    assert "Datasets con duplicados: audio 1k: 6." in text


def test_command_line(tmp_path, inputs, capsys):
    prefix = tmp_path / "final"
    main(["--inputs", *map(str, inputs), "--output-prefix", str(prefix)])

    assert len(_outputs(prefix)) == 6
    assert "final_summary.md" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exit_info:
        main(["--inputs", *map(str, inputs[:2]), "--output-prefix", str(tmp_path / "bad")])
    assert exit_info.value.code == 1
    assert "Validación fallida" in capsys.readouterr().err
    assert _outputs(tmp_path / "bad") == []


def test_large_counts_are_not_written_in_scientific_notation():
    assert summary._num(5755151.0) == "5,755,151"
    assert summary._num(156.02) == "156.02"
