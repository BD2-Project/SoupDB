"""In-process ranked index factory used only by DDL tests.

The fake's shared state models persisted providers for reopen tests; it does
not claim that real IVF/HNSW or inverted-index structures persist.
"""

from dataclasses import replace

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.ranked_catalog import RankedMetadata
from tests.fakes.fake_ranked_search import ScriptedRankedIndex


class FakeRankedFactory:
    def __init__(self) -> None:
        self._stored: dict[str, RankedBinding] = {}
        self.built: list[str] = []
        self.loaded: list[str] = []

    def build(self, metadata: RankedMetadata, file_org, schema) -> RankedBinding:
        kind = {"INVERTED": "text", "IVF": "ivf", "HNSW": "hnsw"}[metadata.algorithm]
        index = ScriptedRankedIndex(kind)
        binding = RankedBinding(
            metadata.name,
            metadata.table,
            metadata.column,
            metadata.algorithm,
            index,
            metadata.metric,
            metadata.dimension,
            metadata.profile_id,
            item_count=sum(1 for _ in file_org.scan()),
            exact=True,
            exhaustive=True,
            mutable=False,
        )
        self._stored[metadata.name] = binding
        self.built.append(metadata.name)
        return binding

    def load(self, metadata: RankedMetadata, file_org, schema) -> RankedBinding:
        if metadata.name not in self._stored:
            raise QueryExecutionError(f"ranked provider missing for {metadata.name!r}")
        original = self._stored[metadata.name]
        # A new instance is returned, like a genuine provider reopening disk data.
        index = ScriptedRankedIndex(
            {"INVERTED": "text", "IVF": "ivf", "HNSW": "hnsw"}[metadata.algorithm]
        )
        index._responses.update(original.index._responses)
        self.loaded.append(metadata.name)
        return replace(original, index=index)
