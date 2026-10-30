"""PhreshPhish URL extraction for the URL benchmark, with the HTML column left out.

Source: hf://datasets/phreshphish/phreshphish. Parquet shards live under data/ (train-*.parquet,
test-*.parquet). The benchmark subsets are id lists under benchmark/: benign.txt (shared),
phish.txt (the full benchmark) and benchmark-{005,010,050,100,500}/N.txt (phish ids for one subset
at a 0.05%, 0.1%, 0.5%, 1% or 5% base rate). Ids are whitespace-separated sha256 values.

Launch: kaggle/launch.sh phreshphish_extract --internet   (CPU; no datasets attached)

Only sha256, url, label, target, date and lang are requested from each shard, so the html column
(and lang_score) is never read. Each shard is one Units unit and writes its own part file under
/kaggle/working/parts, so a partial run keeps every shard it finished. The parts are merged into
phreshphish_urls.parquet at the end, including after a stop at the time limit (10 h by default,
under the 12 h session). A stop is recorded as status partial.

Outputs in /kaggle/working:
- phreshphish_urls.parquet: sha256, url, label, target, date, lang, split, bench_benign (in
  benign.txt), bench_full (in benign.txt or phish.txt), bench_subsets (list of subset ids such as
  "benchmark-005/0" that contain the row as a phish sample).
- benchmark_subsets.json: benign and full-phish id lists, and every subset's phish ids with its
  base rate. A subset is benign_sha256 plus its phish_sha256.
- phreshphish_stats.json: rows per split and label, date range, bytes read, wall time and checks.

Flags go through kaggle/launch.sh phreshphish_extract --job-args="...":
  --splits {test,train}...   (default test train)
  --max-hours FLOAT          stop cleanly after this many hours (default 10)
  --batch-rows INT           rows per Arrow batch (default 65536)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from _common import Units, env_report, out_dir, write_metrics

REPO_ROOT = 'datasets/phreshphish/phreshphish'
COLUMNS = ('sha256', 'url', 'label', 'target', 'date', 'lang')
NOT_READ = ('html', 'lang_score')
SPLITS = ('test', 'train')
DEFAULT_MAX_HOURS = 10.0
DEFAULT_BATCH_ROWS = 65536
HF_BLOCK_BYTES = 256 * 1024
OUT_NAME = 'phreshphish_urls.parquet'
BENCH_NAME = 'benchmark_subsets.json'
STATS_NAME = 'phreshphish_stats.json'
SUBSET_RE = re.compile(r'benchmark/(benchmark-(\d+))/(\d+)\.txt')
RATE_LABELS = {'005': '0.05%', '010': '0.1%', '050': '0.5%', '100': '1%', '500': '5%'}
OUT_SCHEMA = pa.schema([
    ('sha256', pa.string()),
    ('url', pa.string()),
    ('label', pa.string()),
    ('target', pa.string()),
    ('date', pa.date32()),
    ('lang', pa.string()),
    ('split', pa.string()),
    ('bench_benign', pa.bool_()),
    ('bench_full', pa.bool_()),
    ('bench_subsets', pa.list_(pa.string())),
])


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog='phreshphish_extract')
    p.add_argument('--splits', nargs='+', default=list(SPLITS), choices=SPLITS)
    p.add_argument('--max-hours', type=float, default=DEFAULT_MAX_HOURS)
    p.add_argument('--batch-rows', type=int, default=DEFAULT_BATCH_ROWS)
    return p.parse_args(argv or [])


class _Meter(io.RawIOBase):
    """Read-only wrapper that counts the bytes the Parquet reader takes from the file."""

    def __init__(self, raw, meter: dict):
        self._raw = raw
        self._meter = meter

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    def readinto(self, buf) -> int:
        data = self._raw.read(len(buf))
        n = len(data)
        buf[:n] = data
        self._meter['reader_bytes'] += n
        return n

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


class LocalSource:
    """The repo layout on local disk. Used by the tests; wire bytes are not measurable here."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def find(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.is_dir():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob('*') if p.is_file())

    def open(self, rel: str, meter: dict):
        return _Meter(open(self.root / rel, 'rb'), meter)

    def wire_bytes(self) -> int | None:
        return None


_WIRE = {'bytes': 0, 'patched': False}


def _count_wire() -> None:
    """Count the bytes HfFileSystem fetches over HTTP (block read-ahead included)."""
    if _WIRE['patched']:
        return
    from huggingface_hub.hf_file_system import HfFileSystemFile

    original = HfFileSystemFile._fetch_range

    def counted(self, start, end):
        data = original(self, start, end)
        _WIRE['bytes'] += len(data)
        return data

    HfFileSystemFile._fetch_range = counted
    _WIRE['patched'] = True


class HfSource:
    """hf://datasets/phreshphish/phreshphish, read through HfFileSystem (anonymous, public)."""

    def __init__(self):
        from huggingface_hub import HfFileSystem

        _count_wire()
        self.fs = HfFileSystem()
        self.root = REPO_ROOT

    def find(self, prefix: str) -> list[str]:
        found = self.fs.find(f'{self.root}/{prefix}', withdirs=False)
        if found and not all(p.startswith(self.root + '/') for p in found):
            raise RuntimeError(f'unexpected Hub path format, e.g. {found[0]!r}')
        return sorted(p[len(self.root) + 1:] for p in found)

    def open(self, rel: str, meter: dict):
        raw = self.fs.open(f'{self.root}/{rel}', 'rb', block_size=HF_BLOCK_BYTES)
        return _Meter(raw, meter)

    def wire_bytes(self) -> int:
        return _WIRE['bytes']


@dataclass
class IdSets:
    benign: set = field(default_factory=set)
    full_phish: set = field(default_factory=set)
    subsets: dict = field(default_factory=dict)  # "benchmark-005/0" -> {rate_label, phish}
    phish_subsets: dict = field(default_factory=dict)  # sha256 -> [subset ids]


def _read_ids(src, rel: str, meter: dict) -> list[str]:
    with src.open(rel, meter) as f:
        return f.read().decode('utf-8').split()


def load_id_sets(src, meter: dict) -> IdSets:
    """Read benign.txt, phish.txt and every subset file. Ids are returned as sets and lists."""
    ids = IdSets()
    txt = [name for name in src.find('benchmark') if name.endswith('.txt')]
    if 'benchmark/benign.txt' not in txt or 'benchmark/phish.txt' not in txt:
        raise FileNotFoundError('benchmark/benign.txt and benchmark/phish.txt are required')
    ids.benign = set(_read_ids(src, 'benchmark/benign.txt', meter))
    ids.full_phish = set(_read_ids(src, 'benchmark/phish.txt', meter))
    for rel in txt:
        match = SUBSET_RE.fullmatch(rel)
        if match is None:
            continue
        sid = f'{match.group(1)}/{match.group(3)}'
        phish = _read_ids(src, rel, meter)
        ids.subsets[sid] = {'rate_label': RATE_LABELS.get(match.group(2), match.group(2)),
                            'phish': phish}
        for sha in phish:
            ids.phish_subsets.setdefault(sha, []).append(sid)
    if not ids.subsets:
        raise FileNotFoundError('no benchmark-*/N.txt subset files found under benchmark/')
    return ids


def write_benchmark_json(ids: IdSets, path: Path) -> dict:
    """Write the id lists as compact JSON, atomically. Returns the per-rate subset counts."""
    n_benign = len(ids.benign)
    subsets = {}
    per_rate: Counter = Counter()
    for sid in sorted(ids.subsets):
        entry = ids.subsets[sid]
        n_phish = len(entry['phish'])
        per_rate[entry['rate_label']] += 1
        subsets[sid] = {
            'rate_label': entry['rate_label'],
            'n_phish': n_phish,
            'base_rate': n_phish / (n_benign + n_phish) if n_benign + n_phish else None,
            'phish_sha256': entry['phish'],
        }
    payload = {
        'source': REPO_ROOT,
        'note': 'subset = benign_sha256 + phish_sha256 (benign is shared by every subset)',
        'benign_sha256': sorted(ids.benign),
        'full_phish_sha256': sorted(ids.full_phish),
        'subsets': subsets,
    }
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    tmp.replace(path)
    return dict(per_rate)


def extract_file(src, rel: str, split: str, ids: IdSets, state: dict, meter: dict,
                 parts: Path, batch_rows: int) -> dict:
    """One shard: read the projected columns in batches and write its part file.

    State (seen ids, counters) is changed only after the whole shard succeeds, so a failed shard
    leaves no partial counts behind.
    """
    began = time.perf_counter()
    reader_before = meter['reader_bytes']
    wire_before = src.wire_bytes()
    part = parts / f'{split}-{Path(rel).stem}.parquet'
    tmp = part.with_name(part.name + '.tmp')
    by_label: Counter = Counter()
    checks: Counter = Counter()
    new_shas: list[str] = []
    in_file: set[str] = set()
    date_min = date_max = None
    rows = 0
    try:
        with src.open(rel, meter) as raw:
            pf = pq.ParquetFile(raw)
            names = pf.schema_arrow.names
            missing = [c for c in COLUMNS if c not in names]
            if missing:
                raise ValueError(f'{rel}: missing columns {missing}; the file has {names}')
            with pq.ParquetWriter(tmp, OUT_SCHEMA, compression='zstd') as writer:
                for batch in pf.iter_batches(batch_size=batch_rows, columns=list(COLUMNS)):
                    cols = {name: batch.column(name).to_pylist() for name in COLUMNS}
                    sha = cols['sha256']
                    benign = [s in ids.benign for s in sha]
                    full = [b or (s in ids.full_phish) for b, s in zip(benign, sha, strict=True)]
                    subsets = [ids.phish_subsets.get(s, []) for s in sha]
                    for s, lab, is_benign, day in zip(
                            sha, cols['label'], benign, cols['date'], strict=True):
                        by_label[lab] += 1
                        if s in state['seen_all'] or s in in_file:
                            checks['duplicate_sha256'] += 1
                        in_file.add(s)
                        new_shas.append(s)
                        if day is not None:
                            if date_min is None or day < date_min:
                                date_min = day
                            if date_max is None or day > date_max:
                                date_max = day
                        if lab == 'phish' and is_benign:
                            checks['phish_label_in_benign_list'] += 1
                        if lab == 'benign' and s in ids.full_phish:
                            checks['benign_label_in_phish_list'] += 1
                    table = pa.Table.from_pydict({
                        'sha256': sha,
                        'url': cols['url'],
                        'label': cols['label'],
                        'target': cols['target'],
                        'date': cols['date'],
                        'lang': cols['lang'],
                        'split': [split] * len(sha),
                        'bench_benign': benign,
                        'bench_full': full,
                        'bench_subsets': subsets,
                    }, schema=OUT_SCHEMA)
                    writer.write_table(table)
                    rows += len(sha)
        tmp.replace(part)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    state['seen_all'].update(new_shas)
    state['seen_split'][split].update(new_shas)
    for key, n in checks.items():
        state['checks'][key] += n
    wire_after = src.wire_bytes()
    return {
        'split': split,
        'file': rel,
        'rows': rows,
        'by_label': dict(by_label),
        'date_min': date_min.isoformat() if date_min else None,
        'date_max': date_max.isoformat() if date_max else None,
        'reader_bytes': meter['reader_bytes'] - reader_before,
        'wire_bytes': None if wire_after is None else wire_after - wire_before,
        'wall_s': round(time.perf_counter() - began, 2),
    }


def merge_parts(parts: Path, out_path: Path) -> int:
    """Concatenate the part files into out_path, one batch at a time, then remove the parts."""
    files = sorted(parts.glob('*.parquet'))
    if not files:
        return 0
    tmp = out_path.with_name(out_path.name + '.tmp')
    total = 0
    with pq.ParquetWriter(tmp, OUT_SCHEMA, compression='zstd') as writer:
        for part in files:
            with pq.ParquetFile(part) as pf:
                for batch in pf.iter_batches(batch_size=DEFAULT_BATCH_ROWS):
                    writer.write_table(pa.Table.from_batches([batch]).cast(OUT_SCHEMA))
                    total += batch.num_rows
    tmp.replace(out_path)
    shutil.rmtree(parts, ignore_errors=True)
    return total


def extract_corpus(ctx, src, args: argparse.Namespace, work: Path, started: float) -> str:
    """Run the whole extraction into work. Returns the notification status."""
    limit = args.max_hours * 3600.0
    meter = {'reader_bytes': 0}
    ids = load_id_sets(src, meter)
    ctx.log(f'benchmark ids: {len(ids.benign)} benign, {len(ids.full_phish)} phish, '
            f'{len(ids.subsets)} subsets, reader bytes so far {meter["reader_bytes"]}')
    rates = write_benchmark_json(ids, work / BENCH_NAME)
    ctx.log(f'wrote {BENCH_NAME}: subsets per rate {rates}')

    parts = work / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    state = {
        'seen_all': set(),
        'seen_split': {s: set() for s in args.splits},
        'checks': Counter(),
    }
    units = Units(ctx)
    shards = src.find('data')
    if not any(n.endswith('.parquet') for n in shards):
        raise FileNotFoundError('no data/*.parquet shards listed in the source')
    file_stats: list[dict] = []
    stopped = False
    for split in args.splits:
        for rel in [n for n in shards if n.startswith(f'data/{split}-') and n.endswith('.parquet')]:
            if time.time() - started > limit:
                stopped = True
                ctx.log(f'time limit of {args.max_hours} h reached; stopping before {rel}')
                break
            value = units.run({'split': split, 'file': rel}, extract_file, src, rel, split, ids,
                              state, meter, parts, args.batch_rows)
            if value is not None:
                file_stats.append(value)
                ctx.log(f'{split} {rel}: {value["rows"]} rows, {value["reader_bytes"]} reader '
                        f'bytes, {value["wall_s"]} s')
        if stopped:
            break

    rows = merge_parts(parts, work / OUT_NAME)
    ctx.log(f'merged {rows} rows into {OUT_NAME}')

    by_split_label: dict[str, dict[str, int]] = {}
    date_min = date_max = None
    for fs in file_stats:
        bucket = by_split_label.setdefault(fs['split'], {})
        for lab, n in fs['by_label'].items():
            bucket[lab] = bucket.get(lab, 0) + n
        if fs['date_min'] and (date_min is None or fs['date_min'] < date_min):
            date_min = fs['date_min']
        if fs['date_max'] and (date_max is None or fs['date_max'] > date_max):
            date_max = fs['date_max']

    seen = state['seen_all']
    wire = src.wire_bytes()
    stats = {
        'dataset': REPO_ROOT,
        'columns_read': list(COLUMNS),
        'columns_not_read': list(NOT_READ),
        'splits_requested': args.splits,
        'shards_listed': len(shards),
        'shards_done': len(file_stats),
        'time_limit_hit': stopped,
        'max_hours': args.max_hours,
        'batch_rows': args.batch_rows,
        'rows_total': rows,
        'rows_by_split_label': by_split_label,
        'date_range': {'min': date_min, 'max': date_max},
        'bytes': {
            'reader_total': meter['reader_bytes'],
            'wire_total': wire,
            'note': ('wire_total counts every HTTP range fetch, including read-ahead; '
                     'reader_total counts bytes handed to the Parquet reader, re-reads included'),
        },
        'wall_s': round(time.time() - started, 1),
        'benchmark': {
            'subsets': len(ids.subsets),
            'subsets_per_rate': rates,
            'benign_ids': len(ids.benign),
            'full_phish_ids': len(ids.full_phish),
            'subset_phish_ids_total': sum(len(v['phish']) for v in ids.subsets.values()),
            'benign_ids_not_extracted': len(ids.benign - seen),
            'full_phish_ids_not_extracted': len(ids.full_phish - seen),
            'subset_phish_ids_not_in_full_list': sum(
                1 for sha in ids.phish_subsets if sha not in ids.full_phish),
            'subset_phish_ids_in_benign_list': sum(
                1 for sha in ids.phish_subsets if sha in ids.benign),
        },
        'checks': {**dict(state['checks']), 'merge_row_count_matches_parts':
                   rows == sum(fs['rows'] for fs in file_stats)},
        'files': file_stats,
        'outputs': {'parquet': OUT_NAME, 'benchmark_json': BENCH_NAME},
    }
    write_metrics(STATS_NAME, stats)
    ctx.results['stats'] = {k: stats[k] for k in ('rows_total', 'rows_by_split_label',
                                                  'date_range', 'bytes', 'wall_s')}
    ctx.results['summary'] = (f'{rows} URL rows from {len(file_stats)}/{len(shards)} shards, '
                              f'{units.ok} units ok, {units.failed} failed, '
                              f'{meter["reader_bytes"] / 1e6:.1f} MB read')
    if rows == 0:
        return 'fail'
    if stopped or units.failed:
        return 'partial'
    return 'ok'


def run(ctx) -> bool | str:
    args = parse_args(ctx.argv)
    started = time.time()
    env = env_report()
    ctx.log(f'env: {env}')
    ctx.results['env'] = env
    ctx.log(f'splits {args.splits}, max {args.max_hours} h, batch {args.batch_rows} rows; '
            f'not read: {list(NOT_READ)}')
    src = HfSource()
    return extract_corpus(ctx, src, args, out_dir(), started)
