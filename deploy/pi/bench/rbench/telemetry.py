"""Thermal and power telemetry for one run, from the thermal guard's per-second CSV.

The guard (thermal_run.Job) writes ``ts,temp,cap,freq,throttled,power_w`` rows while the
worker runs. This module turns them into the ``thermal`` block of result.json.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pi_common as pc


def read_rows(path: Path | str | None) -> list[dict[str, str]]:
    if path is None:
        return []
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _floats(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = (row.get(key) or "").strip()
        if value:
            try:
                out.append(float(value))
            except ValueError:
                continue
    return out


def summarise(
    rows: list[dict[str, str]],
    *,
    throttled_before: str | None,
    throttled_after: str | None,
    pauses: int,
    paused_s: float,
    kill_temp_c: float | None = None,
) -> dict[str, Any]:
    """The thermal block: temperatures, cap and power over the run plus throttle flags."""
    temps = _floats(rows, "temp")
    caps = _floats(rows, "cap")
    power = _floats(rows, "power_w")
    freqs = _floats(rows, "freq")
    return {
        "samples": len(rows),
        "temp_start_c": temps[0] if temps else None,
        "temp_max_c": max(temps) if temps else None,
        "temp_end_c": temps[-1] if temps else None,
        "cap_mhz_min": int(min(caps)) if caps else None,
        "freq_mhz_mean": round(sum(freqs) / len(freqs), 1) if freqs else None,
        "power_w_avg": round(sum(power) / len(power), 3) if power else None,
        "throttled_before": throttled_before,
        "throttled_before_decoded": pc.decode_throttled(throttled_before),
        "throttled_after": throttled_after,
        "throttled_after_decoded": pc.decode_throttled(throttled_after),
        "pauses": int(pauses),
        "paused_s": round(float(paused_s), 1),
        "guard_kill_temp_c": kill_temp_c,
    }


def read_temp_now() -> float | None:
    """Current SoC temperature from vcgencmd measure_temp (None off the Pi)."""
    text = pc.run_vcgencmd(["measure_temp"])
    if not text or "=" not in text:
        return None
    try:
        return float(text.split("=", 1)[1].strip().rstrip("'C").rstrip("'"))
    except ValueError:
        return None
