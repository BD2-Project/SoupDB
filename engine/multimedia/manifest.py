"""Manifest del dataset multimedia: una línea JSON por objeto, con IDs reproducibles.

Imagen:            {"id": 0, "type": "image", "path": "gatos/001.jpg"}
Segmento de audio: {"id": 0, "type": "audio_segment", "path": "rock/a.mp3",
                    "segment": 1, "start": 5.0, "end": 10.0}

``path`` es relativa a la carpeta raíz y con "/", así el manifest es el mismo en
cualquier máquina. Los archivos se ordenan antes de numerar: mismos archivos, mismo
manifest. Para un dataset de 1k/10k/100k basta con ``limit``.

Uso: python -m engine.multimedia.manifest {image,audio} RAIZ SALIDA.jsonl [--limit N]
"""

import argparse
import json
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

from engine.multimedia.audio_features import SAMPLE_RATE, SEGMENT_SECONDS, iter_audio_segments

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".au"})


def list_media_files(root: str | Path, extensions: Iterable[str]) -> list[str]:
    root = Path(root)
    extensions = {e.lower() for e in extensions}
    files = (
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    # Orden por la ruta relativa en texto: no depende del orden en que el SO lista.
    return sorted(files)


def image_rows(root: str | Path) -> Iterator[dict]:
    for i, path in enumerate(list_media_files(root, IMAGE_EXTENSIONS)):
        yield {"id": i, "type": "image", "path": path}


def audio_rows(root: str | Path, duration: float = SEGMENT_SECONDS) -> Iterator[dict]:
    # Hay que decodificar cada pista para saber cuántos segmentos tiene; una a la vez.
    next_id = 0
    for path in list_media_files(root, AUDIO_EXTENSIONS):
        for segment, samples in iter_audio_segments(Path(root) / path, duration):
            start = segment * duration
            yield {
                "id": next_id,
                "type": "audio_segment",
                "path": path,
                "segment": segment,
                "start": round(start, 3),
                "end": round(start + len(samples) / SAMPLE_RATE, 3),
            }
            next_id += 1


def write_manifest(path: str | Path, rows: Iterable[dict], limit: int | None = None) -> int:
    written = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in islice(rows, limit):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def read_manifest(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Construye el manifest JSONL de un dataset.")
    parser.add_argument("type", choices=["image", "audio"])
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="máximo de objetos (1k, 10k...)")
    parser.add_argument("--duration", type=float, default=SEGMENT_SECONDS)
    args = parser.parse_args(argv)

    if args.type == "image":
        rows = image_rows(args.root)
    else:
        rows = audio_rows(args.root, args.duration)
    count = write_manifest(args.output, rows, args.limit)
    print(f"{count} objetos -> {args.output}")


if __name__ == "__main__":
    main()
