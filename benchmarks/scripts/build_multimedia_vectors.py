"""codebook.npz + vectores.bin para cada manifest de 1k / 10k / 100k, y su validación.

Llama a ``engine.multimedia.build_vectors.build`` sin cambiar nada del pipeline.
Todos los tamaños comparten la misma caché de descriptores: como 1k ⊂ 10k ⊂ 100k,
al pasar de 1k a 10k solo se extraen los 9.000 objetos nuevos, y si se corta una
ejecución, repetir el comando lee de la caché lo ya calculado.

Cada tamaño entrena su propio codebook con su propio manifest.

Salida (por defecto bajo benchmarks/data/multimedia, ignorada por Git)::

    manifests/images_1k.jsonl ...      (prepare_multimedia_datasets)
    cache/image/..., cache/audio_segment/...
    images/1000/codebook.npz, vectores.bin, params.json
    audio/100000/...

Uso::

    uv run --group multimedia python -m benchmarks.scripts.build_multimedia_vectors \\
        image --root D:/datasets/coco/train2017 --sizes 1000 10000 100000
"""

import argparse
import json
import math
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path

from benchmarks.scripts.prepare_multimedia_datasets import DATA_DIR, SIZES, manifest_path
from engine.multimedia import build_vectors
from engine.multimedia.audio_features import SAMPLE_RATE, SEGMENT_SECONDS
from engine.multimedia.build_vectors import record_path
from engine.multimedia.image_features import MAX_IMAGE_SIDE
from engine.multimedia.manifest import read_manifest
from engine.multimedia.vector_file import iter_vectors

# Parámetros fijos del experimento (docs/datasets_multimedia.md).
K = 128
MAX_DESCRIPTORS = 50_000
SEED = 20261024
OUTPUT_DIRS = {"image": "images", "audio": "audio"}


def output_dir(data_dir: str | Path, kind: str, size: int) -> Path:
    return Path(data_dir) / OUTPUT_DIRS[kind] / str(size)


def validate_vector_file(vectors_path: str | Path, manifest: str | Path, k: int) -> int:
    """Comprueba vectores.bin contra su manifest, registro a registro, sin cargarlos todos."""
    seen_ids = set()
    count = 0
    pairs = zip_longest(read_manifest(manifest), iter_vectors(vectors_path))
    for count, (row, record) in enumerate(pairs, start=1):
        if row is None or record is None:
            expected = sum(1 for _ in read_manifest(manifest))
            raise ValueError(f"{vectors_path}: count mismatch, manifest has {expected} rows")
        if record.id in seen_ids:
            raise ValueError(f"{vectors_path}: duplicated id {record.id}")
        seen_ids.add(record.id)
        if record.id != row["id"]:
            raise ValueError(f"record {count}: id {record.id}, expected {row['id']}")
        if record.path != record_path(row):
            raise ValueError(f"record {count}: path {record.path!r}, expected {record_path(row)!r}")
        if len(record.vector) != k:
            raise ValueError(f"record {count}: dimension {len(record.vector)}, expected {k}")
        if not all(math.isfinite(x) for x in record.vector):
            raise ValueError(f"record {count}: vector has NaN or infinity")
    return count


def build_size(
    kind: str,
    root: str | Path,
    size: int,
    data_dir: str | Path = DATA_DIR,
    k: int = K,
    max_descriptors: int = MAX_DESCRIPTORS,
    seed: int = SEED,
) -> Path:
    manifest = manifest_path(Path(data_dir) / "manifests", kind, size)
    if not manifest.is_file():
        raise SystemExit(f"falta {manifest}: ejecuta antes prepare_multimedia_datasets")
    out = output_dir(data_dir, kind, size)
    out.mkdir(parents=True, exist_ok=True)
    vectors = out / "vectores.bin"

    build_vectors.build(
        manifest,
        root,
        Path(data_dir) / "cache",
        out / "codebook.npz",
        vectors,
        k,
        max_descriptors,
        seed,
    )
    count = validate_vector_file(vectors, manifest, k)
    if count != size:
        raise ValueError(f"{vectors}: {count} records, expected {size}")

    params = {
        "kind": kind,
        "size": size,
        "manifest": manifest.as_posix(),
        "k": k,
        "max_descriptors": max_descriptors,
        "seed": seed,
        "image_max_side": MAX_IMAGE_SIDE,
        "audio_segment_seconds": SEGMENT_SECONDS,
        "audio_sample_rate": SAMPLE_RATE,
    }
    (out / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    return vectors


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="vectores.bin de 1k/10k/100k desde manifests")
    parser.add_argument("type", choices=list(OUTPUT_DIRS))
    parser.add_argument("--root", type=Path, required=True, help="carpeta del dataset fuente")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--max-descriptors", type=int, default=MAX_DESCRIPTORS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    # De menor a mayor: cada tamaño reaprovecha la caché que dejó el anterior.
    for size in sorted(args.sizes):
        vectors = build_size(
            args.type, args.root, size, args.data_dir, args.k, args.max_descriptors, args.seed
        )
        print(f"{args.type} {size}: {vectors} validado")


if __name__ == "__main__":
    main()
