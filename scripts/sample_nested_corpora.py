"""Create nested, reproducible TREC-COVID corpora in one streaming pass."""

import argparse
import hashlib
import heapq
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.sample_inverted_corpus import CORPUS_URL


def _document(line: str | bytes, line_number: int) -> dict[str, object]:
    try:
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or type(row.get("_id")) not in (str, int)
            or not str(row["_id"]).strip()
            or not isinstance(row.get("title"), str)
            or not isinstance(row.get("text"), str)
        ):
            raise ValueError("expected _id, title and text")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid corpus document at line {line_number}") from exc
    return {key: row[key] for key in ("_id", "title", "text")}


def _priority(seed: int, source_id: str) -> int:
    payload = f"{seed}\0{source_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def nested_sample(
    lines: Iterable[str | bytes],
    *,
    sizes: tuple[int, ...] = (1_000, 10_000, 100_000),
    seed: int = 7,
) -> tuple[list[dict[str, object]], int]:
    """Keep the lowest stable ID hashes so each smaller sample nests in larger.

    The selection heap retains at most max(sizes) documents. A separate set of
    normalized IDs detects duplicate source IDs, so total auxiliary memory is
    O(max(sizes) + source document count). Ranking by a hash of seed and source
    ID makes selection repeatable for identical corpus IDs. Results are ordered
    by sampling priority.
    """
    if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("sizes must be positive integers")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("sizes must be unique and sorted ascending")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    capacity = sizes[-1]
    heap: list[tuple[int, str, dict[str, object]]] = []
    seen_ids: set[str] = set()
    document_count = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        document = _document(line, line_number)
        source_id = str(document["_id"]).strip()
        if source_id in seen_ids:
            raise ValueError(f"duplicate corpus ID {source_id!r} at line {line_number}")
        seen_ids.add(source_id)
        document_count += 1
        priority = _priority(seed, source_id)
        entry = (-priority, source_id, document)
        if len(heap) < capacity:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)

    if document_count < capacity:
        raise ValueError(f"corpus contains {document_count} documents; requested {capacity}")
    selected = sorted(heap, key=lambda item: (-item[0], item[1]))
    return [document for _, _, document in selected], document_count


def write_nested_corpora(
    lines: Iterable[str | bytes],
    output_dir: str | Path,
    *,
    sizes: tuple[int, ...] = (1_000, 10_000, 100_000),
    seed: int = 7,
    source: str = CORPUS_URL,
) -> dict[str, object]:
    """Publish all JSONL sizes and a SHA-256 manifest as one directory rename."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    sample, total = nested_sample(lines, sizes=sizes, seed=seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nested-corpus-", dir=output_dir.parent))
    try:
        samples: dict[str, object] = {}
        for size in sizes:
            label = f"{size // 1_000}k" if size % 1_000 == 0 else str(size)
            filename = f"corpus_{label}.jsonl"
            digest = hashlib.sha256()
            with (temporary / filename).open("xb") as file:
                for document in sample[:size]:
                    payload = (
                        json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n"
                    ).encode("utf-8")
                    file.write(payload)
                    digest.update(payload)
            samples[str(size)] = {"file": filename, "documents": size, "sha256": digest.hexdigest()}
        manifest: dict[str, object] = {
            "source": source,
            "source_documents": total,
            "seed": seed,
            "selection": "ascending sha256(seed + NUL + string(_id))",
            "nested": True,
            "samples": samples,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
        return manifest
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local source JSONL instead of downloading")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/data/trec_covid_nested")
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.input is None:
        request = Request(CORPUS_URL, headers={"User-Agent": "SoupDB-nested-corpus/1"})
        with urlopen(request, timeout=60) as source:
            manifest = write_nested_corpora(source, args.output_dir, seed=args.seed)
    else:
        with args.input.open("rb") as source:
            manifest = write_nested_corpora(
                source, args.output_dir, seed=args.seed, source=str(args.input)
            )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
