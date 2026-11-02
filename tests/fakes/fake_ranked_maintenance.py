"""Deterministic reversible ranked index for transactional integration tests."""

from engine.common.rid import RID
from engine.indexes.base import RankedIndex


class FakeReversibleRankedIndex(RankedIndex):
    """Idempotent, atomic dictionary-based mutations; no real search ranking."""

    atomic_mutations = True

    def __init__(self) -> None:
        self.documents: dict[RID, object] = {}
        self.closed = False
        self.fail_next: str | None = None
        self.calls: list[tuple[str, RID, object]] = []

    def top_k(self, query, k: int, **opts) -> list[tuple[RID, float]]:
        return []

    def insert_document(self, rid: RID, value: object) -> None:
        if self.fail_next == "insert":
            self.fail_next = None
            raise RuntimeError("simulated insert failure")
        self.calls.append(("insert", rid, value))
        self.documents[rid] = value

    def remove_document(self, rid: RID, value: object) -> None:
        if self.fail_next == "remove":
            self.fail_next = None
            raise RuntimeError("simulated remove failure")
        self.calls.append(("remove", rid, value))
        self.documents.pop(rid, None)

    def close(self) -> None:
        self.closed = True
