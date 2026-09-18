"""In-memory index oracle implementing the frozen Index contract."""

from engine.common.errors import UnsupportedOperation
from engine.common.rid import RID
from engine.indexes.base import Index, Key


class FakeIndex(Index):
    """In-memory index backed by a dict, used as oracle and in planner tests."""

    def __init__(self, supports_range: bool = True) -> None:
        self._entries: dict[Key, list[RID]] = {}
        self._supports_range = supports_range

    @property
    def supports_range(self) -> bool:
        return self._supports_range

    def insert(self, key: Key, rid: RID) -> None:
        self._entries.setdefault(key, []).append(rid)

    def search(self, key: Key) -> list[RID]:
        return list(self._entries.get(key, []))

    def range_search(self, lo: Key, hi: Key) -> list[RID]:
        if not self._supports_range:
            raise UnsupportedOperation("FakeIndex without range support")
        result: list[RID] = []
        for key in sorted(self._entries):
            if lo <= key <= hi:
                result.extend(self._entries[key])
        return result

    def remove(self, key: Key, rid: RID | None = None) -> int:
        if key not in self._entries:
            return 0
        if rid is None:
            return len(self._entries.pop(key))
        before = len(self._entries[key])
        self._entries[key] = [item for item in self._entries[key] if item != rid]
        return before - len(self._entries[key])

    def close(self) -> None:
        self._entries.clear()