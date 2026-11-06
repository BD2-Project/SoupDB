"""DML-maintained inverted index: base postings plus an exact delta layer.

The SPIMI index is immutable once built. This wrapper keeps added documents and
removed RIDs in a small sidecar so SQL ``INSERT``/``DELETE`` are reflected by
``SEARCH`` without rebuilding the corpus, and survive reopening the database.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from soupdb_inverted.scoring import bm25_term, idf
from soupdb_inverted.tokenizer import DEFAULT_TOKENIZER, Tokenizer

_RID = tuple[int, int]


def _rid_key(rid: Any) -> _RID:
    return (int(rid[0]), int(rid[1]))


class MutableInvertedIndex(RankedIndex):
    """RankedIndex with atomic, idempotent per-RID mutations."""

    atomic_mutations = True

    def __init__(self, base: Any, sidecar: Path):
        self._base = base
        self._sidecar = Path(sidecar)
        self._tokenizer: Tokenizer = getattr(
            getattr(base, "_store", None), "tokenizer", DEFAULT_TOKENIZER
        )
        self._added: dict[_RID, str] = {}
        self._removed: set[_RID] = set()
        self._load()

    def __len__(self) -> int:
        return len(self._base) + len(self._added) - len(self._removed)

    def insert_document(self, rid: Any, value: object) -> None:
        if value is None:
            return
        key = _rid_key(rid)
        self._added[key] = str(value)
        self._removed.discard(key)
        self._persist()

    def remove_document(self, rid: Any, value: object) -> None:
        key = _rid_key(rid)
        self._added.pop(key, None)
        self._removed.add(key)
        self._persist()

    def top_k(self, query: str, k: int, **opts: Any) -> list[tuple[Any, float]]:
        method = str(opts.get("method", "tf_idf")).lower()
        needed = k + len(self._removed) + len(self._added)
        merged = {
            _rid_key(rid): score
            for rid, score in self._base.top_k(query, needed, **opts)
            if _rid_key(rid) not in self._removed
        }
        scores = self._score_added(query, method)
        for key in self._added:
            if key in self._removed:
                continue
            merged[key] = max(merged.get(key, 0.0), scores.get(key, 0.0))
        return self._sorted(merged, k)

    def match_top_k(self, query: str, k: int, **opts: Any) -> list[tuple[Any, float]]:
        """Only documents sharing analyzed query terms, as the SQL contract requires."""
        method = str(opts.get("method", "tf_idf")).lower()
        needed = k + len(self._removed) + len(self._added)
        match_search = getattr(self._base, "match_top_k", None)
        hits = (
            match_search(query, needed, **opts)
            if callable(match_search)
            else self._base.top_k(query, needed, **opts)
        )
        merged = {_rid_key(rid): score for rid, score in hits if _rid_key(rid) not in self._removed}
        for key, score in self._score_added(query, method).items():
            if key in self._removed or score <= 0.0:
                continue
            merged[key] = max(merged.get(key, 0.0), score)
        return self._sorted(merged, k)

    def _sorted(self, merged: dict[_RID, float], k: int) -> list[tuple[Any, float]]:
        return [
            (RID(page, slot), score)
            for (page, slot), score in sorted(merged.items(), key=lambda item: (-item[1], item[0]))[
                :k
            ]
        ]

    def close(self) -> None:
        self._base.close()

    # --- delta scoring and persistence ----------------------------------

    def _score_added(self, query: str, method: str) -> dict[_RID, float]:
        if not self._added:
            return {}
        terms = self._tokenizer(query)
        if not terms:
            return {}
        store = getattr(self._base, "_store", None)
        base_n = int(getattr(store, "N", 0) or 0)
        base_avgdl = float(getattr(store, "avgdl", 0.0) or 0.0)
        base_df = {}
        if store is not None:
            wanted = set(terms)
            for term, entry in store.terms():
                if term in wanted:
                    base_df[term] = entry.df
        counts = {key: Counter(self._tokenizer(text)) for key, text in self._added.items()}
        lengths = {key: sum(counter.values()) for key, counter in counts.items()}
        delta_n = len(counts)
        effective_n = base_n + delta_n
        effective_avgdl = (
            (base_avgdl * base_n + sum(lengths.values())) / effective_n if effective_n else 0.0
        )
        delta_df = {
            term: sum(1 for counter in counts.values() if term in counter) for term in set(terms)
        }
        scores: dict[_RID, float] = {}
        for key, counter in counts.items():
            length = lengths[key]
            score = 0.0
            for term in terms:
                tf = counter.get(term, 0)
                if not tf:
                    continue
                df = base_df.get(term, 0) + delta_df.get(term, 0)
                if method == "bm25":
                    score += bm25_term(
                        tf,
                        n=effective_n,
                        df=df,
                        doc_len=length,
                        avgdl=effective_avgdl,
                    )
                else:
                    score += tf * idf(effective_n, df)
            if method != "bm25" and length:
                score /= math.sqrt(length)
            scores[key] = score
        return scores

    def _load(self) -> None:
        if not self._sidecar.exists():
            return
        try:
            data = json.loads(self._sidecar.read_text(encoding="utf-8"))
            self._added = {
                (int(page), int(slot)): str(text) for page, slot, text in data.get("added", [])
            }
            self._removed = {(int(page), int(slot)) for page, slot in data.get("removed", [])}
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid inverted-index mutation sidecar: {exc}") from exc

    def _persist(self) -> None:
        self._sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "added": [[page, slot, text] for (page, slot), text in sorted(self._added.items())],
            "removed": [[page, slot] for page, slot in sorted(self._removed)],
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=self._sidecar.parent, delete=False, encoding="utf-8"
            ) as file:
                temporary = Path(file.name)
                json.dump(payload, file, ensure_ascii=False, sort_keys=True)
            os.replace(temporary, self._sidecar)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def rid_key(rid: Sequence[int]) -> _RID:
    """Public helper for tests and callers."""
    return _rid_key(rid)
