"""Environment capture for result.json, and the `doctor` report."""

from __future__ import annotations

import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pi_common as pc

INTERESTING_FLAGS = ("asimd", "asimddp", "i8mm", "sve", "bf16", "fp16", "crc32", "aes")


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def cpu_summary(cpuinfo: Path = Path("/proc/cpuinfo")) -> dict[str, Any]:
    """Model name and a few feature flags of the first CPU (aarch64 or x86 names)."""
    text = pc.read_text(cpuinfo) or ""
    model = None
    flags: list[str] = []
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in ("model name", "cpu part") and model is None:
            model = value.strip()
        if key in ("features", "flags") and not flags:
            flags = value.split()
    present = sorted(f for f in INTERESTING_FLAGS if f in flags)
    return {"model": model, "cpus": os.cpu_count(), "flags_of_interest": present}


def mem_total_mb(meminfo: Path = Path("/proc/meminfo")) -> int | None:
    text = pc.read_text(meminfo) or ""
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
    return int(match.group(1)) // 1024 if match else None


def git_sha(root: Path) -> str | None:
    """Git sha of the synced code, written by sync_bench.sh into SYNC_INFO.json."""
    info = pc.read_json(root / "SYNC_INFO.json") or {}
    sha = info.get("git_sha")
    return sha if isinstance(sha, str) else None


def capture(root: Path) -> dict[str, Any]:
    """The env block of result.json."""
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "kernel": platform.release(),
        "platform": platform.platform(),
        "onnxruntime": _pkg_version("onnxruntime"),
        "executorch": _pkg_version("executorch"),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "pyarrow": _pkg_version("pyarrow"),
        "git_sha": git_sha(root),
        "cpu": cpu_summary(),
        "ram_total_mb": mem_total_mb(),
    }


def doctor(root: Path) -> dict[str, Any]:
    """Health of the machine: versions, imports, temperature, throttling, memory."""
    thr = pc.read_throttled_hex()
    from rbench.telemetry import read_temp_now

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "imports": {
            name: _importable(name)
            for name in (
                "onnxruntime",
                "numpy",
                "pandas",
                "pyarrow",
                "regex",
                "tldextract",
                "executorch",
            )
        },
        "versions": {
            name: _pkg_version(name)
            for name in (
                "onnxruntime",
                "executorch",
                "numpy",
                "pandas",
                "pyarrow",
                "regex",
                "tldextract",
            )
        },
        "threads_cpu_count": os.cpu_count(),
        "ram_total_mb": mem_total_mb(),
        "temp_c": read_temp_now(),
        "throttled": thr,
        "throttled_decoded": pc.decode_throttled(thr) if thr else None,
        "git_sha": git_sha(root),
        "bench_root": str(root),
    }
