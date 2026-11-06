"""DML-maintained ANN index: native inserts plus persistent tombstones.

IVF and HNSW support incremental insertion but not physical deletion. This
wrapper records removed RIDs in a sidecar and filters them from results, so
``DELETE`` is reflected immediately and survives reopening the database.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from engine.common.rid import RID
from engine.indexes.base import RankedIndex

_RID = tuple[int, int]


def _rid_key(rid: Any) -> _RID:
    return (int(rid[0]), int(rid[1]))


class MutableVectorIndex(RankedIndex):
    """RankedIndex with atomic, idempotent per-RID mutations."""

    atomic_mutations = True

    def __init__(
        self,
        base: Any,
        sidecar: Path,
        *,
        vectorizer: Any,
        profile_id: str,
    ):
        self._base = base
        self._sidecar = Path(sidecar)
        self._vectorizer = vectorizer
        self._profile_id = profile_id
        self._removed: set[_RID] = set()
        self._load()

    def __len__(self) -> int:
        return len(self._base)

    def insert_document(self, rid: Any, value: object) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError("vector-indexed reference must be text")
        vector = self._vectorizer.vectorize(value, profile_id=self._profile_id)
        key = _rid_key(rid)
        self._removed.discard(key)
        self._base.insert(vector, rid)
        self._persist()

    def remove_document(self, rid: Any, value: object) -> None:
        key = _rid_key(rid)
        self._removed.add(key)
        self._persist()

    def top_k(self, query: Any, k: int, **opts: Any) -> list[tuple[Any, float]]:
        needed = k + len(self._removed)
        hits = self._base.top_k(query, needed, **opts)
        return [
            (RID(*_rid_key(rid)), score)
            for rid, score in hits
            if _rid_key(rid) not in self._removed
        ][:k]

    def close(self) -> None:
        self._base.close()

    def _load(self) -> None:
        if not self._sidecar.exists():
            return
        try:
            data = json.loads(self._sidecar.read_text(encoding="utf-8"))
            self._removed = {(int(page), int(slot)) for page, slot in data.get("removed", [])}
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid vector-index mutation sidecar: {exc}") from exc

    def _persist(self) -> None:
        self._sidecar.parent.mkdir(parents=True, exist_ok=True)
        payload = {"removed": [[page, slot] for page, slot in sorted(self._removed)]}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", dir=self._sidecar.parent, delete=False, encoding="utf-8"
            ) as file:
                temporary = Path(file.name)
                json.dump(payload, file, sort_keys=True)
            os.replace(temporary, self._sidecar)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
