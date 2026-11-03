import json
from dataclasses import replace

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked_catalog import RankedMetadata
from engine.common.record import Record, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.inverted.provider import OnDiskInvertedFactory


SCHEMA = (ColumnDef("body", ColumnType.TEXT),)
METADATA = RankedMetadata("idx", "docs", "body", "INVERTED")


class Source:
    def scan(self):
        yield RID(7, 3), Record(encode_row(("alpha",), SCHEMA))


class NoScan:
    def scan(self):
        raise AssertionError("load must not scan or rebuild the corpus")


def test_provider_load_uses_artifacts_and_real_rids_without_scanning(tmp_path):
    factory = OnDiskInvertedFactory(tmp_path)
    factory.build(METADATA, Source(), SCHEMA).index.close()
    binding = factory.load(METADATA, NoScan(), SCHEMA)
    try:
        assert binding.item_count == 1
        assert binding.index.match_top_k("alpha", 5) == [(RID(7, 3), 0.0)]
    finally:
        binding.index.close()


@pytest.mark.parametrize("version", [True, 1.0, 2])
def test_provider_rejects_invalid_descriptor_version(tmp_path, version):
    factory = OnDiskInvertedFactory(tmp_path)
    factory.build(METADATA, Source(), SCHEMA).index.close()
    path = next(tmp_path.rglob("sql_index.json"))
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    descriptor["version"] = version
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(QueryExecutionError, match="reopen"):
        factory.load(METADATA, NoScan(), SCHEMA)


def test_provider_rejects_identity_count_and_missing_artifacts(tmp_path):
    factory = OnDiskInvertedFactory(tmp_path)
    factory.build(METADATA, Source(), SCHEMA).index.close()
    for metadata in (replace(METADATA, table="other"), replace(METADATA, item_count=2)):
        with pytest.raises(QueryExecutionError):
            factory.load(metadata, NoScan(), SCHEMA)
    hash_path = next(tmp_path.rglob("terms.hash"))
    hash_path.unlink()
    with pytest.raises(QueryExecutionError):
        factory.load(METADATA, NoScan(), SCHEMA)
    assert not hash_path.exists()


def test_provider_cleans_failed_input_and_post_publication_reopen(tmp_path, monkeypatch):
    factory = OnDiskInvertedFactory(tmp_path, memory_limit_bytes=1)

    class FailingSource:
        def scan(self):
            yield from Source().scan()
            raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        factory.build(METADATA, FailingSource(), SCHEMA)
    assert list(tmp_path.iterdir()) == []

    def failed_load(*args):
        raise RuntimeError("load failed")

    monkeypatch.setattr(factory, "load", failed_load)
    with pytest.raises(RuntimeError, match="load failed"):
        factory.build(METADATA, Source(), SCHEMA)
    assert list(tmp_path.iterdir()) == []
