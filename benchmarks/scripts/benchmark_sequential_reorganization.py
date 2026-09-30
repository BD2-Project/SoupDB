"""Measure SequentialFile._reorganize_globally() cost in isolation.

Fragmentation is prepared by tombstoning whole pages directly via _slotted_page
internals, bypassing SequentialFile.remove() -- remove() itself now triggers
_reorganize_globally() once the 30% threshold is crossed, which would collapse
the fragmentation before we get to measure it separately.
"""

import argparse
import csv
import statistics
import tempfile
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from engine.common.record import Record
from engine.common.rid import RID
from engine.storage import _sequential_page as seqpage
from engine.storage import _slotted_page as slotted
from engine.storage.buffer_manager import BufferManager
from engine.storage.disk_manager import DiskManager
from engine.storage.sequential_file import SequentialFile

PAGE_SIZE = 4096
BUFFER_CAPACITY = 32
SEED = 42
KEY_SIZE = 8
RECORD_SIZE = 64
PAYLOAD_SIZE = RECORD_SIZE - KEY_SIZE


def make_record(key: int) -> Record:
    payload = bytes((key + i) % 256 for i in range(PAYLOAD_SIZE))
    return Record(data=key.to_bytes(KEY_SIZE, "big") + payload)


def record_key(record: Record) -> int:
    return int.from_bytes(record.data[:KEY_SIZE], "big")


@dataclass
class ReorgResult:
    N: int
    elapsed_ms: float
    disk_reads: int
    disk_writes: int
    reclaimable_before: int
    reclaimable_after: int
    occupied_before: int
    waste_ratio_before: float
    page_count: int
    size_bytes: int
    live_records: int
    rid_checks_total: int
    rid_checks_ok: int


def _iter_pages(sf: SequentialFile) -> Iterator[int]:
    bm = sf._buffer_manager
    main_id: int | None = 0
    while main_id is not None:
        yield main_id
        frame = bm.pin(main_id)
        overflow_id = seqpage.first_overflow_page_id(frame)
        next_main = seqpage.next_page_id(frame)
        bm.unpin(main_id, dirty=False)

        while overflow_id is not None:
            yield overflow_id
            frame = bm.pin(overflow_id)
            next_overflow = seqpage.next_page_id(frame)
            bm.unpin(overflow_id, dirty=False)
            overflow_id = next_overflow

        main_id = next_main


def _fragment_until_threshold(sf: SequentialFile) -> tuple[int, int, set[int]]:
    bm = sf._buffer_manager
    reclaimable_total, occupied_total = sf._global_waste_stats()
    tombstoned_pages: set[int] = set()

    for page_id in _iter_pages(sf):
        frame = bm.pin(page_id)
        body = seqpage.body(frame)
        page_occupied = sf._occupied_payload_bytes(body)
        for slot in range(slotted.slot_count(body)):
            slotted.delete_record(body, slot)
        bm.unpin(page_id, dirty=True)

        reclaimable_total += page_occupied
        tombstoned_pages.add(page_id)

        if sf._should_reorganize_globally(reclaimable_total, occupied_total):
            return reclaimable_total, occupied_total, tombstoned_pages

    raise RuntimeError("exhausted all pages without exceeding 30% global waste")


def _count_live_records(sf: SequentialFile) -> int:
    bm = sf._buffer_manager
    live = 0
    for page_id in _iter_pages(sf):
        frame = bm.pin(page_id)
        body = seqpage.body(frame)
        for slot in range(slotted.slot_count(body)):
            if slotted.read_record(body, slot) is not None:
                live += 1
        bm.unpin(page_id, dirty=False)
    return live


def _run_once(record_count: int, workdir: Path) -> ReorgResult:
    path = workdir / f"seq-{record_count}.db"
    dm = DiskManager(path, page_size=PAGE_SIZE)
    bm = BufferManager(dm, capacity=BUFFER_CAPACITY)
    sf = SequentialFile(dm, bm, key_fn=record_key)

    all_rids: list[tuple[RID, int]] = []
    for key in range(record_count):
        rid = sf.insert(make_record(key))
        all_rids.append((rid, key))
    bm.flush_all()

    reclaimable_before, occupied_before, tombstoned_pages = _fragment_until_threshold(sf)
    live_records = _count_live_records(sf)

    bm.flush_all()

    reads_before, writes_before = dm.reads, dm.writes
    started = time.perf_counter()

    sf._reorganize_globally()
    bm.flush_all()

    elapsed = time.perf_counter() - started
    reads_after, writes_after = dm.reads, dm.writes

    reclaimable_after, _occupied_after = sf._global_waste_stats()
    if reclaimable_after != 0:
        raise AssertionError(f"reclaimable_after={reclaimable_after}, expected 0")

    live_rids = [(rid, key) for rid, key in all_rids if rid.page_id not in tombstoned_pages]
    rid_checks_ok = 0
    for rid, key in live_rids:
        record = sf.fetch(rid)
        if record is not None and record_key(record) == key:
            rid_checks_ok += 1

    result = ReorgResult(
        N=record_count,
        elapsed_ms=elapsed * 1000,
        disk_reads=reads_after - reads_before,
        disk_writes=writes_after - writes_before,
        reclaimable_before=reclaimable_before,
        reclaimable_after=reclaimable_after,
        occupied_before=occupied_before,
        waste_ratio_before=reclaimable_before / occupied_before * 100,
        page_count=dm.page_count,
        size_bytes=path.stat().st_size,
        live_records=live_records,
        rid_checks_total=len(live_rids),
        rid_checks_ok=rid_checks_ok,
    )

    dm.close()
    return result


def _median(results: list[ReorgResult]) -> ReorgResult:
    sample = results[0]
    return ReorgResult(
        N=sample.N,
        elapsed_ms=statistics.median(r.elapsed_ms for r in results),
        disk_reads=round(statistics.median(r.disk_reads for r in results)),
        disk_writes=round(statistics.median(r.disk_writes for r in results)),
        reclaimable_before=round(statistics.median(r.reclaimable_before for r in results)),
        reclaimable_after=round(statistics.median(r.reclaimable_after for r in results)),
        occupied_before=round(statistics.median(r.occupied_before for r in results)),
        waste_ratio_before=statistics.median(r.waste_ratio_before for r in results),
        page_count=round(statistics.median(r.page_count for r in results)),
        size_bytes=round(statistics.median(r.size_bytes for r in results)),
        live_records=round(statistics.median(r.live_records for r in results)),
        rid_checks_total=round(statistics.median(r.rid_checks_total for r in results)),
        rid_checks_ok=round(statistics.median(r.rid_checks_ok for r in results)),
    )


def run_benchmark(sizes: list[int], repetitions: int) -> list[ReorgResult]:
    medians: list[ReorgResult] = []
    for record_count in sizes:
        runs: list[ReorgResult] = []
        for _ in range(repetitions):
            with tempfile.TemporaryDirectory() as tmp:
                runs.append(_run_once(record_count, Path(tmp)))
        medians.append(_median(runs))
    return medians


def _print_results(results: list[ReorgResult]) -> None:
    header = (
        f"{'N':>10} "
        f"{'ms':>12} "
        f"{'reads':>8} "
        f"{'writes':>8} "
        f"{'reclaim_before':>15} "
        f"{'reclaim_after':>14} "
        f"{'occupied_before':>16} "
        f"{'waste_%':>8} "
        f"{'pages':>8} "
        f"{'bytes':>10} "
        f"{'live':>8} "
        f"{'rid_ok':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.N:>10} "
            f"{r.elapsed_ms:>12.3f} "
            f"{r.disk_reads:>8} "
            f"{r.disk_writes:>8} "
            f"{r.reclaimable_before:>15} "
            f"{r.reclaimable_after:>14} "
            f"{r.occupied_before:>16} "
            f"{r.waste_ratio_before:>8.2f} "
            f"{r.page_count:>8} "
            f"{r.size_bytes:>10} "
            f"{r.live_records:>8} "
            f"{r.rid_checks_ok:>4}/{r.rid_checks_total:<5}"
        )


def _write_csv(
    results: list[ReorgResult], *, results_dir: str | Path = "benchmarks/results"
) -> Path:
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"sequential_reorganization_{timestamp}.csv"

    fieldnames = list(ReorgResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure SequentialFile reorganization cost.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10_000, 100_000])
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    results = run_benchmark(args.sizes, args.repetitions)

    _print_results(results)

    path = _write_csv(results)
    print(f"\nCSV: {path}")


if __name__ == "__main__":
    main()
