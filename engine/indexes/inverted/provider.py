"""Build and reopen immutable inverted providers over real SQL table RIDs."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding, validate_ranked_binding
from engine.common.ranked_catalog import RankedMetadata, check_ranked_metadata
from engine.common.record import decode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef
from engine.indexes.inverted.index import InvertedIndex
from engine.indexes.inverted.tokenizer import DEFAULT_TOKENIZER, Tokenizer
from engine.storage.base import FileOrganization

_MANIFEST = "sql_index.json"


class OnDiskInvertedFactory:
    """Opt-in factory; NULL rows are omitted and empty text rows are included.

    Pass this provider directly to Catalog, or compose it by algorithm using
    DelegatingRankedFactory. SQL writes are blocked after index creation.
    """

    def __init__(
        self, storage_root: str | Path, *, tokenizer: Tokenizer = DEFAULT_TOKENIZER,
        memory_limit_bytes: int = 8 * 1024 * 1024, page_size: int = 4096, fan_in: int = 8,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.tokenizer = tokenizer
        self.memory_limit_bytes = memory_limit_bytes
        self.page_size = page_size
        self.fan_in = fan_in

    def _directory(self, name: str) -> Path:
        return self.storage_root / hashlib.sha256(name.encode("utf-8")).hexdigest()

    def _check(self, metadata: RankedMetadata, schema: tuple[ColumnDef, ...]) -> None:
        check_ranked_metadata(metadata, schema)
        if metadata.algorithm != "INVERTED":
            raise QueryExecutionError("inverted factory handles INVERTED only")

    def _descriptor(self, metadata: RankedMetadata, count: int) -> dict[str, object]:
        return {
            "version": 1, "name": metadata.name, "table": metadata.table,
            "column": metadata.column, "algorithm": metadata.algorithm, "count": count,
        }

    def _binding(self, metadata: RankedMetadata, index: InvertedIndex) -> RankedBinding:
        return RankedBinding(
            name=metadata.name, table=metadata.table, column=metadata.column,
            algorithm="INVERTED", index=index, item_count=len(index),
            exact=True, exhaustive=True, mutable=False, reversible=False,
        )

    def build(
        self, metadata: RankedMetadata, file_org: FileOrganization, schema: tuple[ColumnDef, ...]
    ) -> RankedBinding:
        """Stage complete physical data before making the directory visible."""
        self._check(metadata, schema)
        final = self._directory(metadata.name)
        if final.exists():
            raise QueryExecutionError(f"physical ranked index {metadata.name!r} already exists")
        position = next(i for i, col in enumerate(schema) if col.name == metadata.column)

        def documents() -> Iterator[tuple[str, RID]]:
            for rid, record in file_org.scan():
                text = decode_row(record.data, schema)[position]
                if text is None:
                    continue
                if not isinstance(text, str):
                    raise QueryExecutionError("inverted indexed value must be text")
                yield text, rid

        self.storage_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".building_", dir=self.storage_root))
        index: InvertedIndex | None = None
        published = False
        try:
            index = InvertedIndex.build(
                staging / "index", documents(), tokenizer=self.tokenizer,
                memory_limit_bytes=self.memory_limit_bytes, page_size=self.page_size,
                fan_in=self.fan_in,
            )
            count = len(index)
            index.close()
            index = None
            (staging / _MANIFEST).write_text(
                json.dumps(self._descriptor(metadata, count), sort_keys=True), encoding="utf-8"
            )
            staging.rename(final)
            published = True
            return self.load(metadata, file_org, schema)
        except BaseException:
            if index is not None:
                index.close()
            shutil.rmtree(staging, ignore_errors=True)
            if published:
                shutil.rmtree(final, ignore_errors=True)
            raise

    def load(
        self, metadata: RankedMetadata, file_org: FileOrganization, schema: tuple[ColumnDef, ...]
    ) -> RankedBinding:
        """Load durable index files without scanning table rows or rebuilding."""
        self._check(metadata, schema)
        directory = self._directory(metadata.name)
        index: InvertedIndex | None = None
        try:
            stored = json.loads((directory / _MANIFEST).read_text(encoding="utf-8"))
            if not isinstance(stored, dict):
                raise ValueError("ranked SQL descriptor must be an object")
            count = stored["count"]
            if type(stored.get("version")) is not int or stored["version"] != 1:
                raise ValueError("unsupported ranked SQL descriptor version")
            if type(count) is not int or count < 0 or stored != self._descriptor(metadata, count):
                raise ValueError("ranked SQL descriptor differs from metadata")
            if metadata.item_count is not None and metadata.item_count != count:
                raise ValueError("ranked index cardinality metadata does not match")
            index = InvertedIndex.load(directory / "index")
            if len(index) != count:
                raise ValueError("physical document count differs from SQL metadata")
            binding = self._binding(metadata, index)
            validate_ranked_binding(binding, schema)
            return binding
        except BaseException as exc:
            if index is not None:
                index.close()
            if isinstance(exc, (OSError, KeyError, ValueError, TypeError)):
                raise QueryExecutionError(
                    f"failed to reopen ranked index {metadata.name!r}: {exc}"
                ) from exc
            raise

    def drop(self, metadata: RankedMetadata) -> None:
        """Catalog closes the provider before removing its physical files."""
        directory = self._directory(metadata.name)
        if directory.exists():
            shutil.rmtree(directory)
