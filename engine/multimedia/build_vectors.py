"""Del manifest a ``codebook.npz`` y ``vectores.bin``.

manifest -> descriptores (caché) -> muestra -> K-Means -> histogramas -> idf -> codebook
manifest -> descriptores (caché) -> palabras -> histograma -> tf-idf -> L2 -> vectores.bin

Un manifest es de un solo tipo (todo ``image`` o todo ``audio_segment``): imagen y
audio nunca comparten codebook. Los descriptores salen de la caché en disco, así que
repetir el comando tras un corte no recalcula SIFT/MFCC ya guardados.

Uso:
    python -m engine.multimedia.build_vectors MANIFEST ROOT CACHE CODEBOOK VECTORES \\
        --k 128 --max-descriptors 50000 --seed 42
"""

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from engine.multimedia.audio_features import extract_mfcc_cached, segment_id
from engine.multimedia.codebook import (
    assign_words,
    compute_idf,
    count_words,
    sample_descriptors,
    save_codebook,
    tfidf_vector,
    train_codebook,
)
from engine.multimedia.image_features import extract_sift_cached
from engine.multimedia.manifest import read_manifest
from engine.multimedia.vector_file import VectorRecord, write_vectors

REQUIRED_FIELDS = {
    "image": ("id", "type", "path"),
    "audio_segment": ("id", "type", "path", "segment"),
}
DEFAULT_MAX_DESCRIPTORS = 50_000


def validate_manifest(manifest_path: str | Path) -> tuple[str, int]:
    """Comprueba campos y tipo único; devuelve (tipo, número de filas).

    Solo lee JSON, sin extraer nada: un manifest mal formado falla al principio y no
    tras horas de extracción.
    """
    kind = None
    count = 0
    for line, row in enumerate(read_manifest(manifest_path), start=1):
        row_kind = row.get("type")
        if row_kind not in REQUIRED_FIELDS:
            raise ValueError(f"line {line}: unknown type {row_kind!r}")
        if kind is None:
            kind = row_kind
        elif row_kind != kind:
            raise ValueError(
                f"line {line}: mixed manifest ({kind!r} and {row_kind!r}); "
                "images and audio need separate codebooks"
            )
        missing = [field for field in REQUIRED_FIELDS[kind] if field not in row]
        if missing:
            raise ValueError(f"line {line}: missing fields {missing}")
        count += 1
    if kind is None:
        raise ValueError(f"empty manifest {manifest_path}")
    return kind, count


def descriptor_for_row(row: dict, root: str | Path, cache_dir: str | Path) -> np.ndarray:
    path = Path(root) / row["path"]
    if row["type"] == "image":
        return extract_sift_cached(path, cache_dir)
    if row["type"] == "audio_segment":
        # Solo el segmento de la fila: cada segmento es un objeto independiente.
        return extract_mfcc_cached(path, row["segment"], cache_dir)
    raise ValueError(f"unknown type {row['type']!r}")


def record_path(row: dict) -> str:
    if row["type"] == "audio_segment":
        return segment_id(row["path"], row["segment"])
    return row["path"]


def _histogram(
    row: dict, root: str | Path, cache_dir: str | Path, centroids: np.ndarray
) -> np.ndarray:
    words = assign_words(descriptor_for_row(row, root, cache_dir), centroids)
    return count_words(words, len(centroids))


def train_from_manifest(
    manifest_path: str | Path,
    root: str | Path,
    cache_dir: str | Path,
    k: int,
    max_descriptors: int | None = DEFAULT_MAX_DESCRIPTORS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    _, count = validate_manifest(manifest_path)

    # 1ª pasada: generador de descriptores -> muestra acotada por max_descriptors.
    blocks = (descriptor_for_row(row, root, cache_dir) for row in read_manifest(manifest_path))
    sample = sample_descriptors(blocks, max_descriptors, seed)
    centroids = train_codebook(sample, k, seed)
    del sample

    # 2ª pasada: histograma de cada objeto. Lo único que crece con N es esta matriz N × K.
    histograms = np.zeros((count, k), dtype=np.int64)
    for i, row in enumerate(read_manifest(manifest_path)):
        histograms[i] = _histogram(row, root, cache_dir, centroids)
    return centroids, compute_idf(histograms)


def vector_for_row(
    row: dict, root: str | Path, cache_dir: str | Path, centroids: np.ndarray, idf: np.ndarray
) -> list[float]:
    return tfidf_vector(_histogram(row, root, cache_dir, centroids), idf)


def iter_vector_records(
    manifest_path: str | Path,
    root: str | Path,
    cache_dir: str | Path,
    centroids: np.ndarray,
    idf: np.ndarray,
) -> Iterator[VectorRecord]:
    # Generador: cada vector se calcula cuando write_vectors lo pide y se escribe al momento.
    for row in read_manifest(manifest_path):
        vector = vector_for_row(row, root, cache_dir, centroids, idf)
        yield VectorRecord(row["id"], record_path(row), vector)


def build_vector_file(
    manifest_path: str | Path,
    root: str | Path,
    cache_dir: str | Path,
    centroids: np.ndarray,
    idf: np.ndarray,
    output: str | Path,
) -> int:
    _, count = validate_manifest(manifest_path)
    records = iter_vector_records(manifest_path, root, cache_dir, centroids, idf)
    write_vectors(output, records, dimension=len(centroids), count=count)
    return count


def build(
    manifest_path: str | Path,
    root: str | Path,
    cache_dir: str | Path,
    codebook_path: str | Path,
    vectors_path: str | Path,
    k: int,
    max_descriptors: int | None = DEFAULT_MAX_DESCRIPTORS,
    seed: int = 0,
) -> int:
    centroids, idf = train_from_manifest(manifest_path, root, cache_dir, k, max_descriptors, seed)
    save_codebook(codebook_path, centroids, idf)
    return build_vector_file(manifest_path, root, cache_dir, centroids, idf, vectors_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manifest -> codebook.npz + vectores.bin")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path, help="carpeta a la que son relativas las rutas")
    parser.add_argument("cache", type=Path, help="carpeta de la caché de descriptores")
    parser.add_argument("codebook", type=Path, help="salida codebook.npz")
    parser.add_argument("vectors", type=Path, help="salida vectores.bin")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--max-descriptors", type=int, default=DEFAULT_MAX_DESCRIPTORS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    count = build(
        args.manifest,
        args.root,
        args.cache,
        args.codebook,
        args.vectors,
        args.k,
        args.max_descriptors,
        args.seed,
    )
    print(f"{count} vectores de dimensión {args.k} -> {args.vectors} (codebook {args.codebook})")


if __name__ == "__main__":
    main()
