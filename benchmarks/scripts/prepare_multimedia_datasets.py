"""Manifests reproducibles de 1k / 10k / 100k objetos para COCO 2017 y FMA Medium.

No descarga nada: recibe las carpetas ya extraídas (ver docs/datasets_multimedia.md).

Se genera una sola vez el manifest más grande con ``engine.multimedia.manifest`` y
los menores se cortan de él línea a línea, así ``1k ⊂ 10k ⊂ 100k`` byte a byte: los
primeros 1.000 IDs y rutas de los tres coinciden exactamente.

Uso::

    uv run --group multimedia python -m benchmarks.scripts.prepare_multimedia_datasets \\
        --coco D:/datasets/coco/train2017 --fma D:/datasets/fma_medium
"""

import argparse
import os
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from pathlib import Path

from engine.multimedia.manifest import (
    IMAGE_EXTENSIONS,
    audio_rows,
    image_rows,
    list_media_files,
    write_manifest,
)

DATA_DIR = Path("benchmarks/data/multimedia")
SIZES = (1_000, 10_000, 100_000)
PREFIXES = {"image": "images", "audio": "audio"}


def size_label(size: int) -> str:
    return f"{size // 1000}k" if size % 1000 == 0 else str(size)


def manifest_path(manifests_dir: str | Path, kind: str, size: int) -> Path:
    # images_1k.jsonl, audio_100k.jsonl...
    return Path(manifests_dir) / f"{PREFIXES[kind]}_{size_label(size)}.jsonl"


def require_dir(path: str | Path, name: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise SystemExit(f"{name}: {path} no existe o no es una carpeta")
    return path


def _copy_prefix(source: Path, target: Path, lines: int) -> None:
    with (
        open(source, encoding="utf-8") as src,
        open(target, "w", encoding="utf-8", newline="\n") as f,
    ):
        f.writelines(islice(src, lines))


def write_nested_manifests(
    kind: str, rows: Iterable[dict], manifests_dir: str | Path, sizes: Sequence[int]
) -> list[Path]:
    """Escribe el manifest mayor y corta de él los demás como prefijos exactos."""
    sizes = sorted(set(sizes))
    largest = sizes[-1]
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    target = manifest_path(manifests_dir, kind, largest)
    # Temporal + rename: si se corta (el de audio tarda), no queda un manifest a medias.
    tmp = target.with_name(target.name + ".tmp")
    written = write_manifest(tmp, rows, limit=largest)
    if written < largest:
        tmp.unlink()
        raise SystemExit(f"{kind}: solo hay {written} objetos, se necesitan {largest}")
    os.replace(tmp, target)

    for size in sizes[:-1]:
        _copy_prefix(target, manifest_path(manifests_dir, kind, size), size)
    return [manifest_path(manifests_dir, kind, size) for size in sizes]


def prepare_images(coco: str | Path, manifests_dir: str | Path, sizes: Sequence[int]) -> list[Path]:
    coco = require_dir(coco, "COCO")
    # Contar imágenes es barato (solo listar), así que se comprueba antes de empezar.
    available = len(list_media_files(coco, IMAGE_EXTENSIONS))
    print(f"COCO: {available} imágenes en {coco}")
    if available < max(sizes):
        raise SystemExit(f"COCO: {available} imágenes, se necesitan {max(sizes)}")
    return write_nested_manifests("image", image_rows(coco), manifests_dir, sizes)


def prepare_audio(fma: str | Path, manifests_dir: str | Path, sizes: Sequence[int]) -> list[Path]:
    fma = require_dir(fma, "FMA")
    # Los segmentos solo se conocen decodificando; se para al llegar al mayor tamaño.
    rows: Iterator[dict] = audio_rows(fma, skip_unreadable=True)
    paths = write_nested_manifests("audio", rows, manifests_dir, sizes)
    print(f"FMA: {max(sizes)} segmentos de {fma}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manifests 1k/10k/100k de COCO y FMA")
    parser.add_argument("--coco", type=Path, help="carpeta train2017 extraída")
    parser.add_argument("--fma", type=Path, help="carpeta fma_medium extraída")
    parser.add_argument("--manifests", type=Path, default=DATA_DIR / "manifests")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args(argv)
    if args.coco is None and args.fma is None:
        parser.error("indica --coco, --fma o ambos")

    if args.coco is not None:
        for path in prepare_images(args.coco, args.manifests, args.sizes):
            print(f"  {path}")
    if args.fma is not None:
        for path in prepare_audio(args.fma, args.manifests, args.sizes):
            print(f"  {path}")


if __name__ == "__main__":
    main()
