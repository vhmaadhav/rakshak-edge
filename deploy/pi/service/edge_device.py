"""Device telemetry for the edge service: temperature, throttle flags, PMIC power and energy.

Power is the sum of V x A over the Pi 5 PMIC ADC rails (``vcgencmd pmic_read_adc``, see
``pi_common.pmic_power_w``). It is the PMIC's own reading, not a wall or calibrated meter. Energy
for one request is the trapezoidal integral of samples taken every ``interval_s`` while the
request runs (``EnergyWindow``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pi_common as pc
from rbench.telemetry import read_temp_now


def read_power_w() -> float | None:
    return pc.pmic_power_w(pc.read_pmic_rails())


def read_snapshot(
    temp_fn: Callable[[], float | None] = read_temp_now,
    throttled_fn: Callable[[], str | None] = pc.read_throttled_hex,
    power_fn: Callable[[], float | None] = read_power_w,
) -> dict[str, Any]:
    """Temperature, throttle flags and PMIC power right now. Missing values are None."""
    return {
        "temp_c": temp_fn(),
        "throttled": throttled_fn(),
        "power_w": power_fn(),
        "ts": round(time.time(), 3),
    }


class EnergyWindow:
    """Samples power on a thread while a block runs, then integrates it.

    ``stop()`` returns ``{"energy_j", "mean_w", "samples", "duration_s"}``. With fewer than two
    usable samples the energy is mean power times duration (method "mean").
    """

    def __init__(
        self,
        power_fn: Callable[[], float | None] = read_power_w,
        interval_s: float = 0.25,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._power_fn = power_fn
        self._interval = interval_s
        self._clock = clock
        self._stop = threading.Event()
        self._samples: list[tuple[float, float]] = []
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._t1 = 0.0

    def start(self) -> None:
        self._t0 = self._clock()
        self._thread = threading.Thread(target=self._run, name="energy-window", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = self._clock()
            value = self._power_fn()
            if value is not None:
                self._samples.append((now, float(value)))
            self._stop.wait(self._interval)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self._t1 = self._clock()
        duration = max(self._t1 - self._t0, 0.0)
        samples = self._samples
        if len(samples) >= 2:
            energy = 0.0
            for (ta, pa), (tb, pb) in zip(samples, samples[1:], strict=False):
                energy += (pa + pb) / 2.0 * (tb - ta)
            method = "trapezoid"
            span = samples[-1][0] - samples[0][0]
            mean = energy / span if span > 0 else 0.0
        elif samples:
            mean = samples[0][1]
            energy = mean * duration
            method = "mean"
        else:
            mean, energy, method = 0.0, 0.0, "none"
        return {
            "energy_j": round(energy, 3),
            "mean_w": round(mean, 3),
            "samples": len(samples),
            "duration_s": round(duration, 3),
            "method": method,
        }
