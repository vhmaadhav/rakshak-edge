"""Helpers for kaggle/jobs/*.py, used on the Kaggle kernel. Importing this module does no
computation.

Kernel output goes to RAKSHAK_WORK, which run.py sets to /kaggle/working. Metrics are JSON files
there.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import subprocess
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PACKAGES = ('numpy', 'scipy', 'scikit-learn', 'pandas', 'lightgbm', 'regex', 'tldextract')


def out_dir() -> Path:
    path = Path(os.environ.get('RAKSHAK_WORK', '/kaggle/working'))
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics(name: str, payload: dict) -> Path:
    """Write payload as JSON into the kernel output, atomically."""
    path = out_dir() / name
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n')
    tmp.replace(path)
    return path


def log_input_tree(log, root: str = '/kaggle/input', depth: int = 3, per_dir: int = 50) -> None:
    """Write the first ``depth`` levels of the Kaggle input mount through ``log``, so a missing
    dataset shows where Kaggle actually mounted things. Never raises."""
    base = Path(root)
    log(f'input tree under {root} (top {depth} levels):')
    if not base.is_dir():
        log(f'  {root} does not exist or is not a directory')
        return

    def walk(folder: Path, level: int) -> None:
        try:
            entries = sorted(folder.iterdir())
        except OSError as exc:
            log('  ' * level + f'<unreadable: {exc}>')
            return
        for entry in entries[:per_dir]:
            is_dir = entry.is_dir()
            log('  ' * level + entry.name + ('/' if is_dir else ''))
            if is_dir and level < depth:
                walk(entry, level + 1)
        if len(entries) > per_dir:
            log('  ' * level + f'... {len(entries) - per_dir} more')

    try:
        walk(base, 1)
    except Exception as exc:  # diagnostics only; the original error is what matters
        log(f'  listing failed: {type(exc).__name__}: {exc}')


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


@contextmanager
def timed(label: str, sink: dict):
    """Store the wall-clock seconds of the block under sink[label]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        sink[label] = round(time.perf_counter() - start, 3)


def _nvidia_smi():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True,
                              timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.stdout.splitlines():
        if line.strip():
            name, driver, memory = (part.strip() for part in line.split(',', 2))
            rows.append({'name': name, 'driver': driver, 'memory': memory})
    return rows or None


def _version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch_info():
    if importlib.util.find_spec('torch') is None:
        return None
    try:
        import torch
    except Exception as exc:  # an unusable torch install is reported, not fatal
        return {'import_error': f'{type(exc).__name__}: {exc}'}
    cuda = torch.cuda.is_available()
    return {'version': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': cuda,
            'device': torch.cuda.get_device_name(0) if cuda else None}


def env_report() -> dict:
    """Python, platform, CPU count, GPU (nvidia-smi and torch), and the versions of the
    project's packages."""
    return {
        'python': platform.python_version(),
        'implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'cpus': os.cpu_count(),
        'gpu': _nvidia_smi(),
        'torch': _torch_info(),
        'packages': {name: _version(name) for name in PACKAGES},
    }


def single_class_reason(y_test, split: str) -> str | None:
    """None when the test labels hold both classes. Otherwise the 'skipped: ...' line a job
    records instead of crashing in evaluate_scores."""
    present = sorted({int(v) for v in y_test})
    if len(present) >= 2:
        return None
    only = present[0] if present else 'none'
    return (f'skipped: {split} test split has only label {only} (n={len(y_test)}); '
            'a test split needs both classes')


class NonFiniteError(RuntimeError):
    """A loss, logits or gradient norm held NaN or inf. The unit that raised it fails."""


def non_finite_message(what: str, *, step=None, lr=None, dtype=None, last_finite_loss=None,
                       detail: str = '') -> str:
    """One diagnostic line for a non-finite value. Unknown fields print as n/a."""

    def fmt(value, spec: str) -> str:
        return 'n/a' if value is None else format(value, spec)

    text = (f'non-finite {what} at step {fmt(step, "d")}: lr {fmt(lr, ".3g")}, '
            f'dtype {dtype or "n/a"}, last finite loss {fmt(last_finite_loss, ".6g")}')
    return f'{text} ({detail})' if detail else text


def require_finite(values, what: str, **diagnostics) -> None:
    """Raise NonFiniteError when ``values`` (a float or any array) holds NaN or inf. The keyword
    diagnostics (step, lr, dtype, last_finite_loss) go into the message."""
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    bad = int((~np.isfinite(arr)).sum())
    if bad:
        raise NonFiniteError(non_finite_message(what, detail=f'{bad} of {arr.size} values',
                                                **diagnostics))


def record_skip(ctx, split: str, reason: str) -> None:
    """Log a skipped split and keep the reason in results.json under 'skipped'."""
    ctx.results.setdefault('skipped', {})[split] = reason
    ctx.log(reason)


def job_status(ok: int, failed: int) -> str:
    """The notification status of a job with some units passed and some failed.

    'ok': at least one unit passed and none failed. 'partial': some passed, some failed.
    'fail': no unit passed (including a job with no units at all).
    """
    if ok == 0:
        return 'fail'
    return 'partial' if failed else 'ok'


class Units:
    """Runs one (split, model, variant) unit at a time.

    A unit that raises is recorded with its traceback in results['failures'] and the job goes on.
    results.json is saved after every unit, so partial results survive a later crash.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.ok = 0
        self.failed = 0
        ctx.results['failures'] = []

    def run(self, unit: dict, fn: Callable[..., Any], *args: Any) -> Any:
        """Call fn(*args). Returns its value, or None when the unit failed."""
        label = ' '.join(f'{k}={v}' for k, v in unit.items())
        try:
            value = fn(*args)
        except Exception as exc:  # one bad unit must not stop the job
            self.failed += 1
            error = f'{type(exc).__name__}: {exc}'
            trace = traceback.format_exc()
            self.ctx.results['failures'].append({**unit, 'error': error, 'traceback': trace})
            self.ctx.log(f'FAILED {label}: {error}')
            self.ctx.log(trace)
            self.ctx.save()
            return None
        self.ok += 1
        self.ctx.save()
        return value

    def status(self) -> str:
        return job_status(self.ok, self.failed)
