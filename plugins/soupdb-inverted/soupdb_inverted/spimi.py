"""Build SPIMI blocks and sorted intermediate runs with a soft memory budget.

The budget estimates UTF-8 terms and dictionary postings, not Python RSS.
Tokenization and a per-document Counter may exceed it for one large document;
the block itself flushes incrementally while consuming that Counter.
"""

import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from engine.algorithms.external_sort import write_sorted_run
from engine.common.rid import RID
from soupdb_inverted.merge import compare_postings, encode_posting
from soupdb_inverted.tokenizer import Tokenizer


@dataclass(frozen=True)
class SPIMIBuild:
    """Caller owns successful runs; remove their build directory after use."""

    runs: tuple[Path, ...]
    doc_lengths: dict[int, int]
    rids: dict[int, RID]


def spimi_runs(
    items: Iterable[tuple[str, RID]],
    directory: Path,
    *,
    tokenizer: Tokenizer,
    memory_limit_bytes: int,
) -> SPIMIBuild:
    """Assign dense internal IDs to real RIDs and spill term/doc TF blocks.

    A unique directory prevents collisions. On failure all files created by
    this call are removed; existing files beneath directory are preserved.
    """
    if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be a positive integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="spimi-", dir=directory))
    runs: list[Path] = []
    lengths: dict[int, int] = {}
    rids: dict[int, RID] = {}
    seen_rids: set[RID] = set()
    block: dict[str, dict[int, int]] = {}
    used_bytes = 0

    def flush() -> None:
        nonlocal used_bytes
        if block:
            path = workdir / f"block-{len(runs)}.run"
            records = (
                encode_posting(term, doc_id, tf)
                for term, postings in sorted(block.items())
                for doc_id, tf in sorted(postings.items())
            )
            runs.append(write_sorted_run(path, records, compare_postings))
            block.clear()
            used_bytes = 0

    try:
        for doc_id, (text, rid) in enumerate(items):
            if (
                not isinstance(rid, RID)
                or type(rid.page_id) is not int
                or rid.page_id < 0
                or type(rid.slot) is not int
                or rid.slot < 0
            ):
                raise ValueError("documents require valid storage RIDs")
            if rid in seen_rids:
                raise ValueError(f"duplicate document RID {rid!r}")
            seen_rids.add(rid)
            counts = Counter(tokenizer(text))
            lengths[doc_id] = sum(counts.values())
            rids[doc_id] = rid
            for term, tf in counts.items():
                cost = 96 + (len(term.encode("utf-8")) + 64 if term not in block else 0)
                if block and used_bytes + cost > memory_limit_bytes:
                    flush()
                    cost = 160 + len(term.encode("utf-8"))
                block.setdefault(term, {})[doc_id] = tf
                used_bytes += cost
                if used_bytes >= memory_limit_bytes:
                    flush()
        flush()
        if not runs:
            workdir.rmdir()
        return SPIMIBuild(tuple(runs), lengths, rids)
    except BaseException:
        shutil.rmtree(workdir)
        raise
