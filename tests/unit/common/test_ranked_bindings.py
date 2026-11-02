"""Tests for ranked metadata and the isolated SQL catalog fake."""

from dataclasses import FrozenInstanceError, replace

import pytest

from engine.common.errors import QueryExecutionError
from engine.common.ranked import RankedBinding, validate_ranked_binding
from engine.common.schema import ColumnDef, ColumnType
from tests.fakes.fake_catalog import FakeCatalog
from tests.fakes.fake_ranked_search import ScriptedRankedIndex


@pytest.fixture
def catalog() -> FakeCatalog:
    catalog = FakeCatalog()
    catalog.create_table(
        "documentos",
        (
            ColumnDef("id", ColumnType.INT),
            ColumnDef("contenido", ColumnType.TEXT),
            ColumnDef("archivo", ColumnType.VARCHAR, 255),
        ),
    )
    return catalog


@pytest.fixture
def text_binding() -> RankedBinding:
    return RankedBinding(
        name="idx_contenido",
        table="documentos",
        column="contenido",
        algorithm="INVERTED",
        index=ScriptedRankedIndex("text"),
        item_count=3,
        exact=True,
        exhaustive=True,
    )


@pytest.fixture
def vector_binding() -> RankedBinding:
    return RankedBinding(
        name="idx_archivo_hnsw",
        table="documentos",
        column="archivo",
        algorithm="HNSW",
        index=ScriptedRankedIndex("hnsw"),
        metric="cosine",
        dimension=64,
        profile_id="image-v1",
    )


def test_register_text_and_vector_without_polluting_scalar_indexes(
    catalog: FakeCatalog, text_binding: RankedBinding, vector_binding: RankedBinding
) -> None:
    catalog.add_ranked_index(text_binding)
    catalog.add_ranked_index(vector_binding)

    assert catalog.indexes("documentos") == {}
    assert catalog.indexes_for("documentos", "contenido") == {}
    assert catalog.ranked_indexes_for("documentos") == {
        text_binding.name: text_binding,
        vector_binding.name: vector_binding,
    }
    assert catalog.ranked_indexes_for("documentos", "contenido") == {
        text_binding.name: text_binding
    }
    assert catalog.ranked_indexes_for("documentos", "id") == {}
    assert catalog.index_location(text_binding.name) == ("documentos", "contenido")
    assert catalog.index_location(vector_binding.name) == ("documentos", "archivo")


def test_bindings_are_immutable(text_binding: RankedBinding) -> None:
    with pytest.raises(FrozenInstanceError):
        text_binding.name = "otro"


def test_registration_does_not_search_or_maintain_index(
    catalog: FakeCatalog, text_binding: RankedBinding
) -> None:
    catalog.insert("documentos", (1, "base de datos", "imagen.jpg"))
    catalog.add_ranked_index(text_binding)
    catalog.insert("documentos", (2, "sistemas distribuidos", "otra.jpg"))

    assert text_binding.index.calls == []


def test_scalar_and_ranked_indices_coexist(
    catalog: FakeCatalog, text_binding: RankedBinding
) -> None:
    catalog.add_index("documentos", "id", index_name="idx_id")
    catalog.add_ranked_index(text_binding)

    assert "idx_id" in catalog.indexes("documentos")
    assert text_binding.name not in catalog.indexes("documentos")
    assert catalog.index_location("idx_id") == ("documentos", "id")


def test_index_names_are_unique_across_families(
    catalog: FakeCatalog, text_binding: RankedBinding
) -> None:
    catalog.add_index("documentos", "id", index_name="nombre_usado")
    with pytest.raises(QueryExecutionError, match="already exists"):
        catalog.add_ranked_index(replace(text_binding, name="nombre_usado"))

    catalog.add_ranked_index(text_binding)
    with pytest.raises(QueryExecutionError, match="already exists"):
        catalog.add_index("documentos", "id", index_name=text_binding.name)


def test_duplicate_ranked_name_is_rejected(
    catalog: FakeCatalog, text_binding: RankedBinding
) -> None:
    catalog.add_ranked_index(text_binding)
    with pytest.raises(QueryExecutionError, match="already exists"):
        catalog.add_ranked_index(text_binding)
    assert len(catalog.ranked_indexes_for("documentos")) == 1


def test_drop_ranked_index_keeps_scalar_indices(
    catalog: FakeCatalog, text_binding: RankedBinding
) -> None:
    catalog.add_index("documentos", "id", index_name="idx_id")
    catalog.add_ranked_index(text_binding)
    catalog.drop_index(text_binding.name)

    assert catalog.ranked_indexes_for("documentos") == {}
    assert catalog.index_location(text_binding.name) is None
    assert "idx_id" in catalog.indexes("documentos")


def test_drop_table_unregisters_all_index_families(
    catalog: FakeCatalog, text_binding: RankedBinding
) -> None:
    catalog.add_index("documentos", "id", index_name="idx_id")
    catalog.add_ranked_index(text_binding)
    catalog.drop_table("documentos")

    assert catalog.index_location("idx_id") is None
    assert catalog.index_location(text_binding.name) is None


def test_unknown_tables_are_rejected(catalog: FakeCatalog, text_binding: RankedBinding) -> None:
    with pytest.raises(QueryExecutionError, match="unknown table"):
        catalog.ranked_indexes_for("inexistente")
    with pytest.raises(QueryExecutionError, match="unknown table"):
        catalog.add_ranked_index(replace(text_binding, table="inexistente"))


def test_ranked_binding_can_use_ivf(catalog: FakeCatalog, vector_binding: RankedBinding) -> None:
    binding = replace(
        vector_binding,
        name="idx_archivo_ivf",
        algorithm="IVF",
        index=ScriptedRankedIndex("ivf"),
        metric="euclidean",
    )
    catalog.add_ranked_index(binding)
    assert catalog.ranked_indexes_for("documentos", "archivo") == {binding.name: binding}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": ""}, "name"),
        ({"table": " "}, "table"),
        ({"column": ""}, "column"),
        ({"algorithm": "BM25"}, "unsupported ranked algorithm"),
        ({"index": object()}, "must implement RankedIndex"),
        ({"column": "no_existe"}, "unknown column"),
        ({"column": "id"}, "must be TEXT or VARCHAR"),
        ({"metric": "cosine"}, "cannot have vector metadata"),
        ({"dimension": 64}, "cannot have vector metadata"),
        ({"profile_id": "image-v1"}, "cannot have vector metadata"),
        ({"item_count": -1}, "item_count"),
        ({"item_count": True}, "item_count"),
        ({"exact": 1}, "exact"),
        ({"exhaustive": "si"}, "exhaustive"),
        ({"mutable": None}, "mutable"),
    ],
)
def test_invalid_text_bindings_are_rejected(
    catalog: FakeCatalog,
    text_binding: RankedBinding,
    changes: dict[str, object],
    message: str,
) -> None:
    invalid = replace(text_binding, **changes)
    with pytest.raises(QueryExecutionError, match=message):
        catalog.add_ranked_index(invalid)
    assert catalog.ranked_indexes_for("documentos") == {}
    assert catalog.index_location(text_binding.name) is None


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"metric": None}, "supported vector metric"),
        ({"metric": "haversine"}, "supported vector metric"),
        ({"dimension": 0}, "positive integer"),
        ({"dimension": -1}, "positive integer"),
        ({"dimension": True}, "positive integer"),
        ({"dimension": 1.5}, "positive integer"),
        ({"profile_id": None}, "profile_id"),
        ({"profile_id": ""}, "profile_id"),
    ],
)
def test_invalid_vector_bindings_are_rejected(
    catalog: FakeCatalog,
    vector_binding: RankedBinding,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(QueryExecutionError, match=message):
        catalog.add_ranked_index(replace(vector_binding, **changes))
    assert catalog.ranked_indexes_for("documentos") == {}


def test_validation_function_accepts_compatible_schema(
    catalog: FakeCatalog, text_binding: RankedBinding
) -> None:
    assert validate_ranked_binding(text_binding, catalog.schema("documentos")) is None
