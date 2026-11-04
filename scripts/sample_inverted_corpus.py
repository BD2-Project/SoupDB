"""Export a seeded reservoir sample from the corpus used by scripts/corpus.py."""

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.request import Request, urlopen

CORPUS_URL = "https://huggingface.co/datasets/mteb/trec-covid/resolve/main/corpus.jsonl"


def reservoir_sample(
    lines: Iterable[str | bytes], *, size: int = 1000, seed: int = 7,
) -> tuple[list[dict], int]:
    """Uniform sampling without replacement in O(size) retained documents.

    Reproducibility requires identical input order, size and seed. Blank lines
    are ignored; invalid JSON/schema fails rather than biasing the sample.
    """
    if type(size) is not int or size <= 0 or type(seed) is not int:
        raise ValueError("size must be positive and seed must be an integer")
    rng = random.Random(seed)
    sample: list[dict] = []
    count = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if (not isinstance(row, dict) or type(row.get("_id")) not in (str, int)
                    or not str(row["_id"]).strip()
                    or not isinstance(row.get("title"), str)
                    or not isinstance(row.get("text"), str)):
                raise ValueError("expected _id, title and text")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid corpus document at line {line_number}") from exc
        document = {key: row[key] for key in ("_id", "title", "text")}
        count += 1
        if len(sample) < size:
            sample.append(document)
        else:
            position = rng.randrange(count)
            if position < size:
                sample[position] = document
    if count < size:
        raise ValueError(f"corpus contains {count} documents; requested {size}")
    return sample, count


def write_sample(output: str | Path, sample: list[dict]) -> str:
    """Publish complete UTF-8 JSONL atomically and return its SHA-256."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
            for row in sample:
                data = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                file.write(data)
                digest.update(data)
        os.replace(temporary, output)
        return digest.hexdigest()
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local source JSONL instead of downloading")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/data/trec_covid_1k.jsonl"))
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.input is not None:
        with args.input.open("rb") as lines:
            sample, count = reservoir_sample(lines, size=args.size, seed=args.seed)
        source = str(args.input)
    else:
        request = Request(CORPUS_URL, headers={"User-Agent": "SoupDB-corpus-sampler/1"})
        with urlopen(request, timeout=60) as lines:
            sample, count = reservoir_sample(lines, size=args.size, seed=args.seed)
        source = CORPUS_URL
    digest = write_sample(args.output, sample)
    print(json.dumps({"source": source, "source_documents": count, "sample_size": len(sample),
                      "seed": args.seed, "sha256": digest, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
