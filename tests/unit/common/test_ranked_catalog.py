"""C17: metadata persisted separately; no phantom ranked index on reopen."""

from engine.common.catalog import Catalog
from engine.common.ranked import RankedBinding
from engine.common.schema import ColumnDef, ColumnType
from tests.fakes.fake_ranked_search import ScriptedRankedIndex


def test_persist_metadata_without_advertising_missing_provider(tmp_path):
    c = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    c.create_table("docs", (ColumnDef("id", ColumnType.INT), ColumnDef("texto", ColumnType.TEXT)))
    fake = ScriptedRankedIndex("text")
    c.register_ranked_index(
        RankedBinding(
            name="idx_doc",
            table="docs",
            column="texto",
            algorithm="INVERTED",
            index=fake,
            item_count=0,
            exact=True,
            exhaustive=True,
        )
    )
    assert c.ranked_indexes_for("docs")["idx_doc"].index is fake
    assert c.ranked_metadata_for("docs")["idx_doc"].item_count == 0
    assert c.index_location("idx_doc") == ("docs", "texto")
    c.close()

    reopened = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    assert reopened.ranked_indexes_for("docs") == {}
    assert reopened.ranked_metadata_for("docs")["idx_doc"].algorithm == "INVERTED"
    assert reopened.index_location("idx_doc") == ("docs", "texto")
    assert reopened.tables()[:3] == ["SysTables", "SysColumns", "SysIndexes"]
    assert "SysRankedIndexes" not in reopened.tables()
    reopened.drop_index("idx_doc")
    assert reopened.ranked_metadata_for("docs") == {}
    reopened.close()

    final = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    assert final.ranked_metadata_for("docs") == {}
    final.close()


def test_metadata_and_table_drop_cleanup(tmp_path):
    c = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    c.create_table("docs", (ColumnDef("texto", ColumnType.TEXT),))
    c.register_ranked_index(
        RankedBinding(
            name="idx",
            table="docs",
            column="texto",
            algorithm="INVERTED",
            index=ScriptedRankedIndex("text"),
        )
    )
    c.drop_table("docs")
    assert c.index_location("idx") is None
    c.close()
    reopened = Catalog(tmp_path, page_size=256, buffer_capacity=4)
    assert reopened.index_location("idx") is None
    reopened.close()
