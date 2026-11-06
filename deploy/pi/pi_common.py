"""Shared helpers for the Rakshak Pi 5 thermal tools.

Stdlib only. Used by rakshak_thermald.py, thermal_run.py and pi_dashboard.py.
Every vcgencmd call goes through run_vcgencmd(), which applies a short timeout
and returns None instead of raising, so a missing or hung firmware tool never
kills a daemon.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

VCGENCMD_TIMEOUT_S = 2.0

# bit -> (label, "now" | "occurred"), from `vcgencmd get_throttled`.
THROTTLED_BITS: dict[int, tuple[str, str]] = {
    0: ("under-voltage", "now"),
    1: ("arm freq capped", "now"),
    2: ("throttled", "now"),
    3: ("soft temp limit", "now"),
    16: ("under-voltage", "occurred"),
    17: ("arm freq capped", "occurred"),
    18: ("throttled", "occurred"),
    19: ("soft temp limit", "occurred"),
}

_THROTTLED_RE = re.compile(r"throttled=(0x[0-9a-fA-F]+)")
_PMIC_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s+(?:volt|current)\(\d+\)=(-?\d+(?:\.\d+)?)[VA]\s*$")


def run_vcgencmd(args: Sequence[str], timeout: float = VCGENCMD_TIMEOUT_S) -> str | None:
    """Run vcgencmd and return stdout, or None on any failure."""
    try:
        done = subprocess.run(
            ["vcgencmd", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout


def parse_throttled(text: str | None) -> str | None:
    """Extract the hex value from `throttled=0x...` output."""
    if not text:
        return None
    match = _THROTTLED_RE.search(text)
    return match.group(1).lower() if match else None


def read_throttled_hex() -> str | None:
    return parse_throttled(run_vcgencmd(["get_throttled"]))


def decode_throttled(value: str | int | None) -> dict[str, list[str]]:
    """Decode a throttled value into lists of active 'now' and 'occurred' flags."""
    result: dict[str, list[str]] = {"now": [], "occurred": []}
    if value is None or value == "":
        return result
    number = int(value, 16) if isinstance(value, str) else int(value)
    for bit, (label, kind) in sorted(THROTTLED_BITS.items()):
        if number >> bit & 1:
            result[kind].append(label)
    return result


def parse_pmic_adc(text: str | None) -> dict[str, float]:
    """Parse `vcgencmd pmic_read_adc` lines into {rail_name: value}.

    Keys keep the firmware spelling, e.g. 'VDD_CORE_A' (amps) and 'VDD_CORE_V' (volts).
    """
    rails: dict[str, float] = {}
    for line in (text or "").splitlines():
        match = _PMIC_RE.match(line)
        if match:
            rails[match.group(1)] = float(match.group(2))
    return rails


def pmic_power_w(rails: dict[str, float]) -> float | None:
    """Sum V*A over rails that report both a _V and an _A value (same prefix)."""
    volts = {name[:-2]: val for name, val in rails.items() if name.endswith("_V")}
    amps = {name[:-2]: val for name, val in rails.items() if name.endswith("_A")}
    common = volts.keys() & amps.keys()
    if not common:
        return None
    return round(sum(volts[key] * amps[key] for key in common), 3)


def read_pmic_rails() -> dict[str, float]:
    return parse_pmic_adc(run_vcgencmd(["pmic_read_adc"]))


def read_text(path: Path | str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def read_int(path: Path | str) -> int | None:
    text = read_text(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def read_temp_c(path: Path | str) -> float | None:
    """Read a thermal zone in millidegrees C and return degrees C."""
    milli = read_int(path)
    return None if milli is None else milli / 1000.0


def read_json(path: Path | str) -> dict | None:
    text = read_text(path)
    if not text:
        return None
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return None
    return doc if isinstance(doc, dict) else None


def write_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    """Write text via temp file + rename so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_json_atomic(path: Path, doc: dict) -> None:
    write_atomic(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")


def append_event(path: Path, line: str, keep: int = 500) -> None:
    """Append one line to an event log, keeping only the newest `keep` lines."""
    path = Path(path)
    existing = read_text(path) or ""
    lines = existing.splitlines() if existing else []
    lines.append(line)
    if len(lines) > keep:
        lines = lines[-keep:]
    write_atomic(path, "\n".join(lines) + "\n")


def tail_lines(path: Path | str, count: int, max_bytes: int = 65536) -> list[str]:
    """Return the last `count` lines of a file without reading all of it."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line may be cut mid-way
    return lines[-count:]


def cpufreq_root(sysfs_root: Path) -> Path:
    return Path(sysfs_root) / "devices" / "system" / "cpu" / "cpufreq"


def cpufreq_policies(sysfs_root: Path) -> list[Path]:
    """All cpufreq policy directories, sorted numerically."""
    root = cpufreq_root(sysfs_root)
    if not root.is_dir():
        return []

    def key(path: Path) -> int:
        digits = re.sub(r"\D", "", path.name)
        return int(digits) if digits else 0

    return sorted((p for p in root.glob("policy*") if p.is_dir()), key=key)
