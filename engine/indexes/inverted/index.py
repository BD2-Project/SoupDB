"""Exact disk-backed text ranking, with a separate MATCH candidate mode."""

import heapq
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from engine.common.rid import RID
from engine.indexes.base import RankedIndex
from engine.indexes.inverted.scoring import bm25_term, cosine, idf, validate_bm25
from engine.indexes.inverted.store import InvertedStore
from engine.indexes.inverted.tokenizer import DEFAULT_TOKENIZER, Tokenizer


class InvertedIndex(RankedIndex):
    """Immutable exact index; ties use ascending real RID for reproducibility.

    top_k covers all indexed documents, including zero scores, as required by
    RankedIndex. match_top_k covers only documents sharing analyzed query terms.
    Cosine norms are derived from persisted postings on open, without corpus
    access; documents/lengths/norms use O(N) memory, not an in-memory term index.
    """

    def __init__(self, store: InvertedStore) -> None:
        self._store = store
        self._closed = False
        squared = dict.fromkeys(store.rids, 0.0)
        try:
            for term, entry in store.terms():
                inverse_frequency = idf(store.N, entry.df)
                for doc_id, tf in store.postings(term):
                    squared[doc_id] += (tf * inverse_frequency) ** 2
        except BaseException:
            store.close()
            raise
        self._norms = {doc_id: math.sqrt(value) for doc_id, value in squared.items()}

    @classmethod
    def build(
        cls, path: str | Path, items: Iterable[tuple[str, RID]], *,
        tokenizer: Tokenizer = DEFAULT_TOKENIZER, memory_limit_bytes: int = 8 * 1024 * 1024,
        page_size: int = 4096, fan_in: int = 8,
    ) -> "InvertedIndex":
        return cls(InvertedStore.build(
            path, items, tokenizer=tokenizer, memory_limit_bytes=memory_limit_bytes,
            page_size=page_size, fan_in=fan_in,
        ))

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        return cls(InvertedStore.load(path))

    def __len__(self) -> int:
        return self._store.N

    def top_k(self, query: str, k: int, **opts: Any) -> list[tuple[RID, float]]:
        """Rank with method=tf_idf|bm25; matches_only optionally restricts hits.

        BM25 accepts finite k1>0 and b in [0,1]. Defaults: k1=1.2, b=0.75.
        Empty/OOV queries have zero scores over the full universe by default.
        """
        if self._closed:
            raise ValueError("inverted index is closed")
        if type(k) is not int or k < 0:
            raise ValueError("k must be a non-negative integer")
        if not isinstance(query, str):
            raise TypeError("text query must be a string")
        unknown = opts.keys() - {"method", "matches_only", "k1", "b"}
        if unknown:
            raise ValueError(f"unknown search options: {sorted(unknown)}")
        method = opts.get("method", "tf_idf")
        if method not in ("tf_idf", "bm25"):
            raise ValueError("method must be tf_idf or bm25")
        matches_only = opts.get("matches_only", False)
        if type(matches_only) is not bool:
            raise ValueError("matches_only must be a boolean")
        k1, b = opts.get("k1", 1.2), opts.get("b", 0.75)
        if method == "bm25":
            validate_bm25(k1, b)
        elif "k1" in opts or "b" in opts:
            raise ValueError("k1 and b require method=bm25")
        if k == 0:
            return []

        store = self._store
        query_counts = Counter(store.tokenizer(query))
        scores: dict[int, float] = {}
        query_squared_norm = 0.0
        for term, query_tf in query_counts.items():
            postings = store.postings(term)
            if not postings:
                continue
            if method == "tf_idf":
                inverse_frequency = idf(store.N, len(postings))
                query_weight = query_tf * inverse_frequency
                query_squared_norm += query_weight ** 2
            for doc_id, tf in postings:
                if method == "tf_idf":
                    contribution = query_weight * tf * inverse_frequency
                else:
                    contribution = query_tf * bm25_term(
                        tf, n=store.N, df=len(postings), doc_len=store.doc_lengths[doc_id],
                        avgdl=store.avgdl, k1=k1, b=b,
                    )
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution
        if method == "tf_idf":
            query_norm = math.sqrt(query_squared_norm)
            scores = {doc_id: cosine(dot, query_norm, self._norms[doc_id])
                      for doc_id, dot in scores.items()}
        candidates = scores if matches_only else store.rids
        hits = ((store.rids[doc_id], float(scores.get(doc_id, 0.0))) for doc_id in candidates)
        return heapq.nsmallest(k, hits, key=lambda hit: (-hit[1], hit[0]))

    def match_top_k(self, query: str, k: int, **opts: Any) -> list[tuple[RID, float]]:
        """SQL MATCH: empty/OOV queries return []; zero-score matches remain."""
        return self.top_k(query, k, **{**opts, "matches_only": True})

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._store.close()
