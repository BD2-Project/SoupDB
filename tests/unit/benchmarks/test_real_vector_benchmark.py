"""Modo de datos reales del benchmark vectorial, con vectores.bin diminutos en tmp_path."""

import csv
import json
import math
import random
import re

import pytest

from benchmarks.scripts import compare_vector_indexes as bench
from benchmarks.scripts.compare_vector_indexes import (
    AUDIO,
    BRUTE_FORCE,
    HNSW,
    IMAGE,
    IVF,
    Config,
    VectorFileItems,
    knn_ids,
    load_real_dataset,
    main,
    recall_at_k,
    render_gallery,
    rid_for,
    run_real,
    select_query_positions,
    write_charts,
    write_results,
)
from engine.common.rid import RID
from engine.indexes.vector.brute_force import BruteForceIndex
from engine.indexes.vector.metrics import COSINE, EUCLIDEAN
from engine.multimedia.vector_file import VectorRecord, iter_vectors, write_vectors

DIM = 128
N = 150
SMALL = Config(queries=6, runs=1, m=4, ef_construction=16, ef_search=16, nprobe=2)


def _unit_vectors(n, seed):
    # Grupos alrededor de 8 centros y normalizados a norma 1, como los TF-IDF reales.
    rng = random.Random(seed)
    centers = [[rng.random() for _ in range(DIM)] for _ in range(8)]
    vectors = []
    for _ in range(n):
        v = [max(0.0, c + rng.gauss(0, 0.2)) for c in rng.choice(centers)]
        length = math.hypot(*v)
        vectors.append([x / length for x in v])
    return vectors


def _image_path(i):
    return f"{9 + 3 * i:012d}.jpg"


def _audio_path(i):
    return f"{i // 6:03d}/{i // 6:06d}.mp3#segment={i % 6:06d}"


def _write_dataset(data_dir, media_type, size, seed):
    folder = data_dir / bench.MEDIA_DIRS[media_type] / str(size)
    folder.mkdir(parents=True)
    path_of = _image_path if media_type == IMAGE else _audio_path
    # IDs del manifest distintos de la posición, para comprobar que se conservan.
    records = [
        VectorRecord(1000 + i, path_of(i), v) for i, v in enumerate(_unit_vectors(size, seed))
    ]
    write_vectors(folder / "vectores.bin", records, dimension=DIM, count=size)
    manifests = data_dir / "manifests"
    manifests.mkdir(exist_ok=True)
    if media_type == AUDIO:
        rows = [
            {
                "id": r.id,
                "type": "audio_segment",
                "path": r.path.split("#")[0],
                "segment": i % 6,
                "start": 5.0 * (i % 6),
                "end": 5.0 * (i % 6) + 5.0,
            }
            for i, r in enumerate(records)
        ]
        lines = "".join(json.dumps(row) + "\n" for row in rows)
        (manifests / f"audio_{size}.jsonl").write_text(lines, encoding="utf-8")
    return folder / "vectores.bin"


@pytest.fixture
def data_dir(tmp_path):
    data = tmp_path / "multimedia"
    _write_dataset(data, IMAGE, N, seed=1)
    _write_dataset(data, AUDIO, N, seed=2)
    return data


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    data = tmp_path_factory.mktemp("multimedia")
    for size in (100, 200):
        _write_dataset(data, IMAGE, size, seed=size)
        _write_dataset(data, AUDIO, size, seed=size + 1)
    results, gallery = run_real(
        data, sizes=[100, 200], config=SMALL, progress=lambda _m: None, gallery_count=2
    )
    return data, results, gallery


# --- carga, RIDs y consultas ------------------------------------------------------------


def test_rids_are_assigned_by_ordinal_position():
    assert [rid_for(i) for i in (0, 1, 63, 64, 130)] == [
        RID(0, 0),
        RID(0, 1),
        RID(0, 63),
        RID(1, 0),
        RID(2, 2),
    ]


def test_real_dataset_reads_vectores_bin_with_stable_rids_and_metadata(data_dir):
    path = bench.vectors_for(data_dir, AUDIO, N)

    dataset = load_real_dataset(path, AUDIO, queries=5)
    again = load_real_dataset(path, AUDIO, queries=5)
    records = list(iter_vectors(path))
    items = list(dataset.items)

    assert dataset.size == N
    assert dataset.dim == DIM
    assert [rid for _v, rid in items] == [rid_for(i) for i in range(N)]
    assert [tuple(v) for v, _rid in items] == [r.vector for r in records]
    # Reiterable: IVF lo recorre dos veces y debe ver lo mismo.
    assert [rid for _v, rid in dataset.items] == [rid for _v, rid in items]
    assert dataset.metadata == again.metadata
    assert dataset.metadata[rid_for(7)] == (1007, _audio_path(7), AUDIO, None, None)


def test_items_are_not_a_one_shot_generator(data_dir):
    items = VectorFileItems(bench.vectors_for(data_dir, IMAGE, N))

    assert iter(items) is not items


def test_query_positions_are_deterministic_and_distinct():
    positions = select_query_positions(1000, 50)

    assert positions == select_query_positions(1000, 50)
    assert positions != select_query_positions(1000, 50, seed=1)
    assert len(set(positions)) == 50
    assert positions == sorted(positions)
    assert all(0 <= p < 1000 for p in positions)
    assert select_query_positions(5, 50) == [0, 1, 2, 3, 4]


def test_queries_are_real_objects_of_the_dataset(data_dir):
    path = bench.vectors_for(data_dir, IMAGE, N)
    dataset = load_real_dataset(path, IMAGE, queries=5)
    records = list(iter_vectors(path))

    positions = select_query_positions(N, 5)
    assert dataset.query_rids == [rid_for(p) for p in positions]
    assert [tuple(q) for q in dataset.queries] == [records[p].vector for p in positions]


# --- self-exclusion y recall ------------------------------------------------------------


class _FakeIndex:
    def __init__(self, ranking):
        self.ranking = ranking
        self.asked = []

    def top_k(self, query, k, **opts):
        self.asked.append(k)
        return [(rid, 1.0) for rid in self.ranking[:k]]


def test_knn_ids_asks_k_plus_one_and_drops_the_query_wherever_it_is():
    ranking = [RID(0, i) for i in range(20)]

    middle = _FakeIndex(ranking)
    assert knn_ids(middle, [0.0], 10, RID(0, 3)) == [r for r in ranking[:11] if r != RID(0, 3)]
    assert middle.asked == [11]

    # Si la consulta no aparece (p. ej. empates), quedan los k primeros de los k + 1.
    absent = _FakeIndex(ranking)
    assert knn_ids(absent, [0.0], 10, RID(9, 9)) == ranking[:10]

    # Sin RID que excluir (modo sintético) se piden exactamente k.
    synthetic = _FakeIndex(ranking)
    assert knn_ids(synthetic, [0.0], 10, None) == ranking[:10]
    assert synthetic.asked == [10]


def test_brute_force_answers_exclude_the_query_itself(tmp_path, data_dir):
    dataset = load_real_dataset(bench.vectors_for(data_dir, IMAGE, N), IMAGE, queries=6)
    index = BruteForceIndex.open(tmp_path / "bf", DIM)
    for vector, rid in dataset.items:
        index.insert(vector, rid)

    for query, rid in zip(dataset.queries, dataset.query_rids, strict=True):
        found = knn_ids(index, query, 10, rid, metric=EUCLIDEAN)
        full = [r for r, _s in index.top_k(query, 11, metric=EUCLIDEAN)]
        assert full[0] == rid  # el objeto es su propio vecino más cercano...
        assert found == full[1:]  # ...y no cuenta como vecino.
        assert rid not in found
        assert len(found) == 10
    index.close()


def test_recall_at_10():
    exact = [RID(0, i) for i in range(10)]

    assert recall_at_k(exact, exact) == 1.0
    assert recall_at_k([*exact[:7], RID(1, 0), RID(1, 1), RID(1, 2)], exact) == 0.7
    assert recall_at_k([RID(2, i) for i in range(10)], exact) == 0.0
    # Orden distinto, mismo conjunto: recall 1.
    assert recall_at_k(list(reversed(exact)), exact) == 1.0


# --- benchmark completo sobre datos pequeños --------------------------------------------


def test_smoke_runs_every_media_technique_metric_and_size(real_run):
    _data, results, _gallery = real_run

    combos = {(r.media_type, r.technique, r.metric, r.dataset_size) for r in results}
    assert combos == {
        (media, technique, metric, size)
        for media in (IMAGE, AUDIO)
        for technique in (BRUTE_FORCE, IVF, HNSW)
        for metric in (EUCLIDEAN, COSINE)
        for size in (100, 200)
    }
    assert {r.operation for r in results} == {"build", "knn"}
    knn = [r for r in results if r.operation == "knn"]
    assert all(r.dim == DIM for r in results)  # de los datos, no de Config.dim (64)
    assert all(r.k == 10 and r.queries == 6 for r in results)
    assert all(0.0 <= r.recall_at_k <= 1.0 for r in knn)
    assert all(r.disk_reads > 0 and r.elapsed_ms > 0 for r in knn)
    assert all(r.bytes_on_disk > 0 for r in results)
    assert all(r.disk_writes > 0 for r in results if r.operation == "build")


def test_brute_force_against_itself_has_recall_1(real_run):
    _data, results, _gallery = real_run

    brute = [r for r in results if r.technique == BRUTE_FORCE and r.operation == "knn"]
    assert brute
    assert all(r.recall_at_k == 1.0 for r in brute)


def test_every_technique_and_metric_gets_the_same_queries(monkeypatch, data_dir):
    calls = []
    real = bench.knn_ids

    def spy(index, query, k, exclude, **opts):
        calls.append((type(index).__name__, opts["metric"], exclude, tuple(query)))
        return real(index, query, k, exclude, **opts)

    monkeypatch.setattr(bench, "knn_ids", spy)
    run_real(data_dir, media_types=[IMAGE], sizes=[N], config=SMALL, progress=lambda _m: None)

    dataset = load_real_dataset(bench.vectors_for(data_dir, IMAGE, N), IMAGE, SMALL.queries)
    expected = [(rid, tuple(q)) for q, rid in zip(dataset.queries, dataset.query_rids, strict=True)]
    groups = {}
    for name, metric, exclude, query in calls:
        groups.setdefault((name, metric), []).append((exclude, query))
    assert set(groups) == {
        (name, metric)
        for name in ("BruteForceIndex", "IVFIndex", "HNSWIndex")
        for metric in (EUCLIDEAN, COSINE)
    }
    # runs + 1 pasadas de memoria, siempre las mismas consultas en el mismo orden.
    for sequence in groups.values():
        assert sequence == expected * (SMALL.runs + 1)


def test_recall_uses_the_exact_truth_of_the_same_metric():
    # Vectores sin normalizar: aquí euclidiana y coseno sí ordenan distinto.
    config = Config(dim=8, queries=10, runs=1, nprobe=1)
    dataset = bench.synthetic_dataset(400, config)
    seen = {}

    results = bench.run_datasets(
        [dataset],
        techniques=[BRUTE_FORCE, IVF],
        config=config,
        progress=lambda _m: None,
        on_answers=lambda _d, answers: seen.update(answers),
    )

    assert seen[(EUCLIDEAN, BRUTE_FORCE)] != seen[(COSINE, BRUTE_FORCE)]
    for metric in (EUCLIDEAN, COSINE):
        reported = next(
            r.recall_at_k
            for r in results
            if (r.technique, r.metric, r.operation) == (IVF, metric, "knn")
        )
        pairs = zip(seen[(metric, IVF)], seen[(metric, BRUTE_FORCE)], strict=True)
        expected = sum(recall_at_k(found, exact) for found, exact in pairs) / 10
        assert reported == pytest.approx(expected)


def test_unit_vectors_rank_the_same_with_euclidean_and_cosine(tmp_path, data_dir):
    # Documentado: con norma 1, |a-b|² = 2 - 2·cos, así que el orden coincide.
    dataset = load_real_dataset(bench.vectors_for(data_dir, IMAGE, N), IMAGE, queries=6)
    index = BruteForceIndex.open(tmp_path / "bf", DIM)
    for vector, rid in dataset.items:
        index.insert(vector, rid)

    for query, rid in zip(dataset.queries, dataset.query_rids, strict=True):
        euclidean = knn_ids(index, query, 10, rid, metric=EUCLIDEAN)
        cosine = knn_ids(index, query, 10, rid, metric=COSINE)
        assert euclidean == cosine
    index.close()


def test_csv_has_the_new_fields_and_one_row_per_measure(tmp_path, real_run):
    _data, results, _gallery = real_run

    path = write_results(results, tmp_path, prefix="vector_indexes_real")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert path.name.startswith("vector_indexes_real_")
    assert list(rows[0]) == [
        "media_type",
        "dataset_size",
        "technique",
        "metric",
        "operation",
        "run",
        "elapsed_ms",
        "disk_reads",
        "disk_writes",
        "bytes_on_disk",
        "memory_bytes",
        "recall_at_k",
        "dim",
        "queries",
        "k",
        "params",
    ]
    assert len(rows) == len(results)
    assert {row["media_type"] for row in rows} == {IMAGE, AUDIO}
    ivf = next(row for row in rows if row["technique"] == IVF)
    assert re.fullmatch(r"nlist=\d+;nprobe=2", ivf["params"])


def test_charts_are_split_by_media_and_operation(tmp_path, real_run):
    _data, results, _gallery = real_run

    paths = write_charts(results, tmp_path / "vector_indexes_real_x.csv")

    assert sorted(p.name for p in paths) == [
        "vector_indexes_real_x_audio_build.svg",
        "vector_indexes_real_x_audio_knn.svg",
        "vector_indexes_real_x_image_build.svg",
        "vector_indexes_real_x_image_knn.svg",
    ]
    image_knn = (tmp_path / "vector_indexes_real_x_image_knn.svg").read_text(encoding="utf-8")
    assert "imagen" in image_knn
    # Una línea por técnica · métrica.
    assert image_knn.count("<polyline") == 6
    for label in ("brute_force · euclidean", "ivf · cosine", "hnsw · euclidean"):
        assert label in image_knn


# --- galería -----------------------------------------------------------------------------


def test_gallery_entries_have_every_technique_and_metric(real_run):
    _data, _results, gallery = real_run

    assert [(g.media_type, g.size) for g in gallery] == [
        (IMAGE, 100),
        (IMAGE, 100),
        (IMAGE, 200),
        (IMAGE, 200),
        (AUDIO, 100),
        (AUDIO, 100),
        (AUDIO, 200),
        (AUDIO, 200),
    ]
    for entry in gallery:
        assert set(entry.results) == {
            (metric, technique)
            for metric in (EUCLIDEAN, COSINE)
            for technique in (BRUTE_FORCE, IVF, HNSW)
        }
        assert all(len(found) == 10 for found in entry.results.values())
        assert all(entry.query.id not in {i.id for i in f} for f in entry.results.values())
    audio = next(g for g in gallery if g.media_type == AUDIO)
    assert audio.query.start is not None and audio.query.end == audio.query.start + 5.0


def test_gallery_html_marks_hits_and_misses_with_real_paths(tmp_path, real_run):
    _data, _results, gallery = real_run
    coco, fma = tmp_path / "coco", tmp_path / "fma"

    page = render_gallery(gallery, {IMAGE: coco, AUDIO: fma})

    assert page.startswith("<!doctype html>")
    for text in ("image · N=100", "audio · N=200", "consulta id", "brute_force", "ivf", "hnsw"):
        assert text in page
    assert "<h3>euclidean</h3>" in page and "<h3>cosine</h3>" in page
    assert coco.resolve().as_uri() + "/" in page
    assert re.search(r'<audio controls preload="none" src="file:///[^"]+\.mp3#t=\d+,\d+"', page)
    assert "segmento" in page
    # Una tarjeta por resultado aproximado: 2 técnicas × 2 métricas × 10 por consulta.
    hits, misses = page.count(" HIT</div>"), page.count(" MISS</div>")
    assert hits + misses == len(gallery) * 2 * 2 * 10
    for entry in gallery:
        exact = {i.id for i in entry.results[(EUCLIDEAN, BRUTE_FORCE)]}
        found = entry.results[(EUCLIDEAN, IVF)]
        hit_count = sum(i.id in exact for i in found)
        assert f"recall {hit_count}/10" in page


def test_gallery_without_media_roots_still_shows_paths(real_run):
    _data, _results, gallery = real_run

    page = render_gallery(gallery[:1], {})

    assert "<img" not in page and "<audio" not in page
    assert gallery[0].query.path in page


def test_command_line_writes_csv_charts_and_gallery(tmp_path, data_dir, capsys):
    out = tmp_path / "results"
    main(
        [
            "--real-data-dir",
            str(data_dir),
            "--sizes",
            str(N),
            "--queries",
            "4",
            "--runs",
            "1",
            "--m",
            "4",
            "--ef-construction",
            "16",
            "--ef-search",
            "16",
            "--image-root",
            str(tmp_path / "coco"),
            "--audio-root",
            str(tmp_path / "fma"),
            "--results-dir",
            str(out),
        ]
    )

    names = sorted(p.name for p in out.iterdir())
    assert len([n for n in names if n.endswith(".csv")]) == 1
    assert len([n for n in names if n.endswith(".svg")]) == 4
    assert len([n for n in names if n.endswith("_gallery.html")]) == 1
    assert "galería:" in capsys.readouterr().out


def test_gallery_prefers_queries_with_errors_deterministically():
    rids = [rid_for(i) for i in range(40)]
    metadata = {rid: bench.ObjectInfo(i, _image_path(i), IMAGE) for i, rid in enumerate(rids)}
    query_rids = rids[:5]
    dataset = bench.Dataset(IMAGE, 40, DIM, [], [[0.0]] * 5, query_rids, metadata)
    exact = [rids[10 + 2 * q : 12 + 2 * q] for q in range(5)]
    approx = [list(found) for found in exact]
    approx[3] = [rids[30], rids[31]]  # consulta 3: 2 MISS
    approx[1] = [exact[1][0], rids[32]]  # consulta 1: 1 MISS
    answers = {
        (EUCLIDEAN, BRUTE_FORCE): exact,
        (EUCLIDEAN, IVF): approx,
        (EUCLIDEAN, HNSW): exact,
    }

    chosen = bench.gallery_queries(dataset, answers, 3)
    again = bench.gallery_queries(dataset, answers, 3)

    # Las dos con errores y, a igualdad (0 MISS), la de menor posición; en orden de posición.
    assert [g.query.id for g in chosen] == [0, 1, 3]
    assert chosen == again
    errorless = {key: exact for key in answers}
    assert [g.query.id for g in bench.gallery_queries(dataset, errorless, 3)] == [0, 1, 2]


def test_rows_of_finished_datasets_survive_a_crash(monkeypatch, tmp_path, data_dir):
    out = tmp_path / "results"
    real = bench.run_real

    def crashes_on_audio(data, media_types, *args, **kwargs):
        if media_types == [AUDIO]:
            raise KeyboardInterrupt
        return real(data, media_types, *args, **kwargs)

    monkeypatch.setattr(bench, "run_real", crashes_on_audio)
    with pytest.raises(KeyboardInterrupt):
        main(
            ["--real-data-dir", str(data_dir), "--sizes", str(N), "--results-dir", str(out)]
            + ["--queries", "3", "--runs", "1", "--m", "4", "--ef-construction", "16"]
        )

    (csv_file,) = out.glob("*.csv")
    with csv_file.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {row["media_type"] for row in rows} == {IMAGE}
