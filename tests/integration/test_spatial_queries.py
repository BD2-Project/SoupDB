from collections import Counter

from engine.algorithms.spatial import Polygon2D, euclidean_metric
from engine.common.record import Record, decode_row, encode_row
from engine.common.rid import RID
from engine.common.schema import ColumnDef, ColumnType
from engine.indexes.rtree import MBR, Point, RTree, SpatialQueries
from engine.indexes.rtree._page_store import RTreePageStore
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.heap_file import HeapFile

PAGE_SIZE = 4096

COLUMNS = (
    ColumnDef("name", ColumnType.TEXT),
    ColumnDef("x", ColumnType.FLOAT),
    ColumnDef("y", ColumnType.FLOAT),
)


def _entries() -> list[tuple[Point, RID]]:
    return [
        (Point(0, 0), RID(0, 0)),
        (Point(1, 0), RID(0, 1)),
        (Point(1, 1), RID(0, 2)),
        (Point(1, 1), RID(0, 3)),
        (Point(3, 4), RID(0, 4)),
        (Point(10, 10), RID(0, 5)),
        (Point(-2, -1), RID(0, 6)),
    ]


def _build_tree(
    entries: list[tuple[Point, RID]] | None = None,
) -> RTree:
    tree = RTree(order=2)

    for point, rid in entries or _entries():
        tree.insert(point, rid)

    return tree


def _polygon() -> Polygon2D:
    return Polygon2D(
        (
            Point(-0.5, -0.5),
            Point(2, -0.5),
            Point(2, 2),
            Point(-0.5, 2),
        )
    )


def _query_results(tree: RTree) -> dict[str, object]:
    queries = SpatialQueries(tree)
    metric = euclidean_metric()

    return {
        "range": Counter(
            queries.range_search(
                MBR(-1, -1, 2, 2),
            )
        ),
        "radius": Counter(
            queries.radius_search(
                Point(0, 0),
                1.5,
                metric,
            )
        ),
        "knn": queries.knn(
            Point(0, 0),
            4,
            metric,
        ),
        "polygon": Counter(
            queries.polygon_search(
                _polygon(),
            )
        ),
    }


def _open_page_store(
    path,
    *,
    capacity: int = 1,
) -> tuple[DiskManager, BufferManager, RTreePageStore]:
    disk = DiskManager(path, PAGE_SIZE)
    buffer = BufferManager(disk, capacity)
    store = RTreePageStore(disk, buffer)

    return disk, buffer, store


def test_advanced_queries_survive_monolithic_roundtrip(
    tmp_path,
) -> None:
    path = tmp_path / "spatial.bin"

    tree = _build_tree()
    expected = _query_results(tree)

    tree.save(path)

    loaded = RTree.load(path)

    assert _query_results(loaded) == expected


def test_advanced_queries_survive_paged_roundtrip_with_buffer_one(
    tmp_path,
) -> None:
    path = tmp_path / "spatial.db"

    tree = _build_tree()
    expected = _query_results(tree)

    disk, _buffer, store = _open_page_store(
        path,
        capacity=1,
    )
    store.save_tree(tree)
    disk.close()

    disk, _buffer, store = _open_page_store(
        path,
        capacity=1,
    )
    loaded = store.load_tree()
    disk.close()

    assert _query_results(loaded) == expected


def test_paged_queries_survive_reopen_modify_and_resave(
    tmp_path,
) -> None:
    path = tmp_path / "spatial.db"

    tree = _build_tree(_entries()[:4])

    disk, _buffer, store = _open_page_store(path)
    store.save_tree(tree)
    disk.close()

    disk, _buffer, store = _open_page_store(path)
    loaded = store.load_tree()

    loaded.insert(
        Point(0.5, 0.5),
        RID(9, 9),
    )
    loaded.insert(
        Point(20, 20),
        RID(9, 10),
    )

    expected = _query_results(loaded)

    store.save_tree(loaded)
    disk.close()

    disk, _buffer, store = _open_page_store(path)
    reloaded = store.load_tree()
    disk.close()

    assert _query_results(reloaded) == expected
    assert RID(9, 9) in SpatialQueries(reloaded).range_search(MBR(0, 0, 1, 1))


def test_spatial_rids_resolve_heap_records_after_reopen(
    tmp_path,
) -> None:
    data_path = tmp_path / "records.db"
    index_path = tmp_path / "spatial.db"

    rows = [
        ("origin", 0.0, 0.0),
        ("east", 1.0, 0.0),
        ("duplicate-a", 1.0, 1.0),
        ("duplicate-b", 1.0, 1.0),
        ("far", 10.0, 10.0),
    ]

    data_disk = DiskManager(data_path, PAGE_SIZE)
    data_buffer = BufferManager(data_disk, capacity=2)
    heap = HeapFile(data_disk, data_buffer)

    tree = RTree(order=2)

    for row in rows:
        rid = heap.insert(
            Record(
                encode_row(
                    row,
                    COLUMNS,
                )
            )
        )

        tree.insert(
            Point(row[1], row[2]),
            rid,
        )

    data_buffer.flush_all()
    data_disk.close()

    index_disk, _index_buffer, store = _open_page_store(
        index_path,
        capacity=1,
    )
    store.save_tree(tree)
    index_disk.close()

    data_disk = DiskManager(data_path, PAGE_SIZE)
    data_buffer = BufferManager(data_disk, capacity=2)
    reopened_heap = HeapFile(data_disk, data_buffer)

    index_disk, _index_buffer, store = _open_page_store(
        index_path,
        capacity=1,
    )
    reopened_tree = store.load_tree()

    result_rids = SpatialQueries(reopened_tree).range_search(MBR(0.5, 0.5, 1.5, 1.5))

    decoded_rows = []

    for rid in result_rids:
        record = reopened_heap.fetch(rid)

        assert record is not None

        decoded_rows.append(
            decode_row(
                record.data,
                COLUMNS,
            )
        )

    assert Counter(row[0] for row in decoded_rows) == Counter(
        {
            "duplicate-a": 1,
            "duplicate-b": 1,
        }
    )

    assert all(row[1:] == (1.0, 1.0) for row in decoded_rows)

    index_disk.close()
    data_disk.close()
