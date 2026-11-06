"""Disk-backed SQL factory for real IVF and HNSW over multimedia references.

The catalog owns a RankedBinding; this factory creates, persists, reloads and
removes the physical vector index without changing IVF/HNSW internals.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding
from engine.common.ranked_catalog import RankedMetadata
from engine.common.record import decode_row
from engine.common.schema import ColumnDef
from engine.storage.base import FileOrganization
from soupdb_vector.indexes.hnsw import HNSWIndex
from soupdb_vector.indexes.ivf import IVFIndex
from soupdb_vector.multimedia.sql_profiles import MultimediaProfile, MultimediaQueryVectorizer

_MANIFEST = "sql_index.json"
_VERSION = 1


class OnDiskVectorFactory:
    """Opt-in factory that makes SQL IVF/HNSW indices from SIFT/MFCC profiles.

    ``storage_root`` must be a persistent path in the same DB deployment.
    Indexes are immutable from SQL: INSERT/DELETE remain blocked until a
    reversible ANN mutation contract is implemented by the underlying index.
    """

    def __init__(
        self,
        storage_root: str | Path,
        profiles: Mapping[str, MultimediaProfile],
        *,
        page_size: int = 4096,
        nlist: int = 4,
        buffer_capacity: int = 16,
    ) -> None:
        if page_size < 1024 or nlist < 1 or buffer_capacity < 1:
            raise ValueError("invalid IVF/HNSW storage parameters")
        self.storage_root = Path(storage_root)
        self.query_vectorizer = MultimediaQueryVectorizer(profiles)
        self.page_size = page_size
        self.nlist = nlist
        self.buffer_capacity = buffer_capacity

    def _directory(self, name: str) -> Path:
        # A stable digest prevents index names from escaping the storage root.
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return self.storage_root / digest

    def _descriptor(self, meta: RankedMetadata, count: int) -> dict:
        profile = self.query_vectorizer.profile(meta.profile_id)
        return {
            "version": _VERSION,
            "name": meta.name,
            "table": meta.table,
            "column": meta.column,
            "algorithm": meta.algorithm,
            "metric": meta.metric,
            "dimension": meta.dimension,
            "profile_id": meta.profile_id,
            "kind": profile.kind,
            "root": str(Path(profile.root).resolve()),
            "codebook_sha256": hashlib.sha256(Path(profile.codebook).read_bytes()).hexdigest(),
            "count": count,
        }

    def _check_metadata(self, meta: RankedMetadata) -> None:
        if meta.algorithm not in ("IVF", "HNSW"):
            raise QueryExecutionError("real multimedia factory handles IVF and HNSW only")
        actual = self.query_vectorizer.dimension(meta.profile_id)
        if actual != meta.dimension:
            raise QueryExecutionError(
                f"index dimension {meta.dimension} differs from profile {meta.profile_id!r} "
                f"dimension {actual}"
            )

    def _binding(self, meta: RankedMetadata, index, count: int) -> RankedBinding:
        return RankedBinding(
            name=meta.name,
            table=meta.table,
            column=meta.column,
            algorithm=meta.algorithm,
            index=index,
            metric=meta.metric,
            dimension=meta.dimension,
            profile_id=meta.profile_id,
            item_count=count,
            exact=False,
            exhaustive=False,
            mutable=False,
            reversible=False,
        )

    def build(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        """Stage and publish a physical index, then return its reopened binding."""
        self._check_metadata(metadata)
        final = self._directory(metadata.name)
        if final.exists():
            raise QueryExecutionError(
                f"physical ranked index {metadata.name!r} already exists at {final}"
            )
        position = next(i for i, col in enumerate(schema) if col.name == metadata.column)

        class RowSource:
            # IVF calls iter(source) twice (training and insertion).
            # This keeps memory bounded even for large multimedia datasets.
            def __iter__(self):
                for rid, record in file_org.scan():
                    reference = decode_row(record.data, schema)[position]
                    if reference is None:
                        continue
                    if not isinstance(reference, str):
                        raise QueryExecutionError("multimedia indexed reference must be text")
                    try:
                        vector = self_vectorizer.vectorize(
                            reference, profile_id=metadata.profile_id
                        )
                    except Exception as exc:
                        raise QueryExecutionError(
                            f"cannot vectorize indexed reference {reference!r}: {exc}"
                        ) from exc
                    yield vector, rid

        self_vectorizer = self.query_vectorizer
        source = RowSource()

        self.storage_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".building_", dir=self.storage_root))
        index = None
        try:
            if metadata.algorithm == "IVF":
                index = IVFIndex.build(
                    staging / "ivf.db",
                    source,
                    dim=metadata.dimension,
                    nlist=self.nlist,
                    page_size=self.page_size,
                    buffer_capacity=self.buffer_capacity,
                )
            else:
                index = HNSWIndex(
                    staging / "hnsw",
                    dim=metadata.dimension,
                    metric=metadata.metric,
                    page_size=self.page_size,
                    buffer_capacity=self.buffer_capacity,
                )
                for vector, rid in source:
                    index.insert(vector, rid)
            count = len(index)
            if count == 0:
                raise QueryExecutionError("IVF/HNSW CREATE INDEX needs at least one non-NULL row")
            index.close()
            index = None
            (staging / _MANIFEST).write_text(
                json.dumps(self._descriptor(metadata, count), sort_keys=True),
                encoding="utf-8",
            )
            staging.rename(final)
        except BaseException:
            if index is not None:
                index.close()
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return self.load(metadata, file_org, schema)

    def load(
        self,
        metadata: RankedMetadata,
        file_org: FileOrganization,
        schema: tuple[ColumnDef, ...],
    ) -> RankedBinding:
        """Reopen an existing index after checking its profile fingerprint."""
        self._check_metadata(metadata)
        directory = self._directory(metadata.name)
        try:
            stored = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
            expected = self._descriptor(metadata, stored["count"])
            if stored != expected:
                raise QueryExecutionError(
                    f"ranked index {metadata.name!r} profile/configuration changed; "
                    "rebuild before querying"
                )
            if metadata.item_count is not None and metadata.item_count != stored["count"]:
                raise QueryExecutionError("ranked index cardinality metadata does not match")
            if metadata.algorithm == "IVF":
                index = IVFIndex.open(
                    directory / "ivf.db",
                    dim=metadata.dimension,
                    page_size=self.page_size,
                    buffer_capacity=self.buffer_capacity,
                )
            else:
                index = HNSWIndex(
                    directory / "hnsw",
                    dim=metadata.dimension,
                    page_size=self.page_size,
                    buffer_capacity=self.buffer_capacity,
                )
            if len(index) != stored["count"]:
                index.close()
                raise QueryExecutionError("physical vector count differs from SQL metadata")
            return self._binding(metadata, index, stored["count"])
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise QueryExecutionError(
                f"failed to reopen ranked index {metadata.name!r}: {exc}"
            ) from exc

    def drop(self, metadata: RankedMetadata) -> None:
        """Delete physical data after the catalog forgets the ranked metadata."""
        directory = self._directory(metadata.name)
        if directory.exists():
            shutil.rmtree(directory)
