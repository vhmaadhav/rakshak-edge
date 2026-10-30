"""PhreshPhish extraction job (kaggle/jobs/phreshphish_extract.py) on a synthetic local repo.

No network: LocalSource stands in for hf://datasets/phreshphish/phreshphish. The fixture has the
same layout (data/*.parquet, benchmark/benign.txt, phish.txt, benchmark-*/N.txt) and an html column.
"""

from __future__ import annotations

import datetime
import json
import random
import string
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import phreshphish_extract as job  # noqa: E402


class FakeCtx:
    def __init__(self):
        self.results: dict = {}
        self.lines: list[str] = []
        self.saves = 0

    def log(self, msg):
        self.lines.append(str(msg))

    def save(self):
        self.saves += 1


def sha(tag: str, i: int) -> str:
    return f"{tag}{i:02d}".ljust(64, "0")


def random_html(rng: random.Random, n: int = 8000) -> str:
    return "".join(rng.choice(string.ascii_letters + " <>/=") for _ in range(n))


def write_shard(path: Path, rows: list[dict], rng: random.Random, drop: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "sha256": [r["sha256"] for r in rows],
        "url": [r["url"] for r in rows],
        "label": [r["label"] for r in rows],
        "target": [r["target"] for r in rows],
        "date": [r["date"] for r in rows],
        "lang": [r["lang"] for r in rows],
        "lang_score": [0.9 for _ in rows],
        "html": [random_html(rng) for _ in rows],
    }
    if drop:
        del cols[drop]
    table = pa.table(cols)
    pq.write_table(table, path, row_group_size=4)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    rng = random.Random(0)
    root = tmp_path / "repo"
    test_rows = []
    for i in range(12):
        phish = i % 3 == 0
        test_rows.append({
            "sha256": sha("t", i),
            "url": f"https://site{i}.example/path",
            "label": "phish" if phish else "benign",
            "target": "SBI" if phish else None,
            "date": datetime.date(2025, 10, 1 + i),
            "lang": "en",
        })
    write_shard(root / "data" / "test-000.parquet", test_rows, rng)
    train_rows = [{
        "sha256": sha("r", i),
        "url": f"http://train{i}.example/",
        "label": "phish" if i % 2 else "benign",
        "target": None,
        "date": datetime.date(2025, 6, 1 + i),
        "lang": "hi",
    } for i in range(6)]
    write_shard(root / "data" / "train-000.parquet", train_rows, rng)
    # A shard without the label column: its unit must fail and the others must still land.
    write_shard(root / "data" / "train-001.parquet", train_rows, rng, drop="label")

    benign_ids = [sha("t", i) for i in (1, 2, 4, 5, 7, 8, 10, 11)]
    phish_ids = [sha("t", i) for i in (0, 3, 6, 9)]
    (root / "benchmark").mkdir()
    (root / "benchmark" / "benign.txt").write_text("\n".join(benign_ids) + "\n")
    (root / "benchmark" / "phish.txt").write_text("\n".join(phish_ids) + "\n")
    (root / "benchmark" / "benchmark-005").mkdir()
    (root / "benchmark" / "benchmark-005" / "0.txt").write_text(sha("t", 0) + "\n")
    (root / "benchmark" / "benchmark-500").mkdir()
    (root / "benchmark" / "benchmark-500" / "0.txt").write_text(
        sha("t", 0) + "\n" + sha("t", 3) + "\n")

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("RAKSHAK_WORK", str(work))
    return root, work


class RecordingSource(job.LocalSource):
    """LocalSource that logs (offset, length) of every read the Parquet reader makes."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.reads: dict[str, list[tuple[int, int]]] = {}

    def open(self, rel: str, meter: dict):
        log = self.reads.setdefault(rel, [])
        handle = open(self.root / rel, "rb")

        class Recorder:
            def read(self, n=-1):
                pos = handle.tell()
                data = handle.read(n)
                log.append((pos, len(data)))
                return data

            def seek(self, offset, whence=0):
                return handle.seek(offset, whence)

            def tell(self):
                return handle.tell()

            def close(self):
                handle.close()

        return job._Meter(Recorder(), meter)


def html_chunks(path: Path) -> list[tuple[int, int]]:
    """Byte ranges of the html column chunks in a Parquet file."""
    md = pq.ParquetFile(path).metadata
    ranges = []
    for rg in range(md.num_row_groups):
        group = md.row_group(rg)
        for j in range(group.num_columns):
            col = group.column(j)
            if col.path_in_schema != "html":
                continue
            start = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
            ranges.append((start, col.total_compressed_size))
    return ranges


def run_job(root: Path, work: Path, argv: list[str], started: float | None = None):
    ctx = FakeCtx()
    args = job.parse_args(argv)
    status = job.extract_corpus(ctx, RecordingSource(root), args, work,
                                time.time() if started is None else started)
    return ctx, status


def test_projection_skips_html_and_outputs_flags(repo):
    root, work = repo
    ctx, status = run_job(root, work, ["--splits", "test", "train"])

    # train-001 has no label column, so it fails; the other two shards land.
    assert status == "partial"
    assert [f["file"] for f in ctx.results["failures"]] == ["data/train-001.parquet"]

    table = pq.read_table(work / "phreshphish_urls.parquet")
    assert table.column_names == list(job.OUT_SCHEMA.names)
    assert "html" not in table.column_names
    rows = {r["sha256"]: r for r in table.to_pylist()}
    assert len(rows) == 18

    benign_ids = {sha("t", i) for i in (1, 2, 4, 5, 7, 8, 10, 11)}
    for i in range(12):
        r = rows[sha("t", i)]
        assert r["split"] == "test"
        assert r["bench_benign"] is (i in (1, 2, 4, 5, 7, 8, 10, 11))
        assert r["bench_full"] is (r["bench_benign"] or i in (0, 3, 6, 9))
        expected = {0: ["benchmark-005/0", "benchmark-500/0"], 3: ["benchmark-500/0"],
                    6: [], 9: []}.get(i, [])
        assert r["bench_subsets"] == expected
    assert rows[sha("t", 0)]["target"] == "SBI"
    assert rows[sha("t", 1)]["target"] is None
    assert all(not r["bench_benign"] and not r["bench_full"] and r["bench_subsets"] == []
               for r in rows.values() if r["split"] == "train")
    assert benign_ids <= {sha for sha, r in rows.items() if r["bench_benign"]}


def test_stats_and_json_outputs(repo):
    root, work = repo
    ctx, _ = run_job(root, work, ["--splits", "test", "train"])
    stats = json.loads((work / "phreshphish_stats.json").read_text())

    assert stats["columns_not_read"] == ["html", "lang_score"]
    assert stats["rows_by_split_label"]["test"] == {"benign": 8, "phish": 4}
    assert stats["rows_by_split_label"]["train"] == {"benign": 3, "phish": 3}
    assert stats["rows_total"] == 18
    assert stats["date_range"]["min"] == "2025-06-01"
    assert stats["date_range"]["max"] == "2025-10-12"
    assert stats["checks"]["merge_row_count_matches_parts"] is True
    assert stats["benchmark"]["benign_ids_not_extracted"] == 0
    assert stats["benchmark"]["full_phish_ids_not_extracted"] == 0
    assert stats["benchmark"]["subsets"] == 2
    assert stats["benchmark"]["subsets_per_rate"] == {"0.05%": 1, "5%": 1}
    assert stats["bytes"]["wire_total"] is None  # local source: not measurable


    bench = json.loads((work / "benchmark_subsets.json").read_text())
    assert sorted(bench["subsets"]) == ["benchmark-005/0", "benchmark-500/0"]
    assert bench["subsets"]["benchmark-005/0"]["phish_sha256"] == [sha("t", 0)]
    assert bench["subsets"]["benchmark-005/0"]["base_rate"] == pytest.approx(1 / (8 + 1))
    assert len(bench["benign_sha256"]) == 8
    assert not (work / "parts").exists()
    assert ctx.results["summary"].startswith("18 URL rows")


def test_html_bytes_are_never_requested(repo):
    root, work = repo
    source = RecordingSource(root)
    args = job.parse_args(["--splits", "test", "train", "--batch-rows", "5"])
    job.extract_corpus(FakeCtx(), source, args, work, time.time())

    shard = root / "data" / "test-000.parquet"
    html = html_chunks(shard)
    assert len(html) == 3  # 12 rows in row groups of 4
    reads = source.reads["data/test-000.parquet"]
    # The first read is the footer prefetch (the last 64 KB, or the whole file when smaller).
    # It is metadata, not column data, so it is left out of the check.
    column_reads = reads[1:]
    assert column_reads, "the reader made no column reads"
    for pos, n in column_reads:
        for start, length in html:
            assert pos + n <= start or pos >= start + length, (pos, n, start, length)


def test_time_limit_stops_before_any_shard(repo):
    root, work = repo
    ctx, status = run_job(root, work, ["--max-hours", "0"], started=time.time() - 1)
    stats = json.loads((work / "phreshphish_stats.json").read_text())
    assert stats["time_limit_hit"] is True
    assert stats["shards_done"] == 0
    assert status == "fail"
    assert not (work / "phreshphish_urls.parquet").exists()


def test_missing_benchmark_file_is_an_error(repo):
    root, work = repo
    (root / "benchmark" / "phish.txt").unlink()
    with pytest.raises(FileNotFoundError):
        run_job(root, work, [])
