"""Raspberry Pi 5 latency and memory benchmark for a sentence encoder exported to ONNX.

Standalone: onnxruntime, numpy, tokenizers and the standard library. transformers is not needed,
the tokenizer is read from tokenizer.json. Only --out is written. Nothing is downloaded.

For each sequence length (batch 1) it measures the tokenisation time and the model latency on a
built-in list of synthetic SMS strings padded or truncated to that length. It also measures the
realistic case: the same strings at their natural token length, truncated at 128 with no padding.
It records model load time, RSS before load, after load and the peak (VmHWM), the model file's bytes
and sha256, the onnxruntime version, the thread count, the SoC temperature (start, end, maximum) and
the throttle flags (start, end).

It refuses to start at or above --max-temp, and it stops early, recording why, if the SoC
reaches it.

Two input modes. Default: the built-in SMS strings, tokenised on the Pi with tokenizer.json
(needs the tokenizers package, which the bench venv has). --rows FILE.npz: pre-tokenised rows built
on the laptop by deploy/pi/bench/build_encoder_rows.py from the cal split, with the token length
distribution stored in the file; no tokenizer is needed on the Pi.

--companion-peak-mb records a companion process's peak (for example the Llama explainer, 1 839 MB
in docs/BASELINE_PI.md) as ADDED to the encoder's VmHWM. The sum is an estimate, labelled as such,
not a measurement of the two together.

Usage:
  encoder_bench.py --model PATH.onnx (--tokenizer DIR | --rows FILE.npz) --threads N
                   [--seq-lens 32 64 128] [--runs 50] [--warmup 5] [--max-temp 75]
                   [--companion-peak-mb MB] [--out FILE.json]

Exit codes: 0 done, 1 bad arguments or a missing dependency, 2 refused at start (too hot),
3 stopped early (temperature reached --max-temp; the JSON still holds what was measured).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

MAX_LEN = 128
DEFAULT_SEQ_LENS = (32, 64, 128)
REALISTIC_PASSES = 2
DEFAULT_MAX_TEMP = 75.0
DEFAULT_COOL_BELOW = 55.0
DEFAULT_MAX_COOL_WAIT = 120.0
DEFAULT_BURST_ABORT_C = 63.0
COOL_POLL_S = 1.0
MAX_COOL_FAILURES = 3
STALL_MS = 1000.0
LENGTH_BUCKETS = ((1, 32), (33, 64), (65, 96), (97, 128))
SCHEMA = 1
THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "arm_freq_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_occurred",
    17: "arm_freq_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temp_limit_occurred",
}

# 32 synthetic SMS-like strings (English, Hinglish, Devanagari, URLs and amounts). Invented for the
# benchmark; no real messages or personal data.
SMS_STRINGS: tuple[str, ...] = (
    "Your account will be blocked today. Verify KYC now at http://kyc-update-secure.example.com",
    "Congratulations! You won Rs 25,000 in lucky draw. Claim before 6 pm: bit.ly/claim-prize-x",
    "Aapka KYC expire ho raha hai. Turant update karein: https://kyc-verify.example.org",
    "Pay Rs 499 processing fee to receive your parcel. Track: http://dhl-redeliver.example.co",
    "You have been selected for a work from home job, earn Rs 5000 per day. Reply YES",
    "Your electricity connection will be disconnected tonight. Pay Rs 1,850 at http://bill-pay.example.in",
    "Your Aadhaar will be deactivated. Call the helpline and share OTP to avoid block",
    "Crypto doubling offer! Invest Rs 10,000 today and get Rs 40,000 in 24 hours.",
    "आपका बैंक खाता बंद हो सकता है। अभी KYC अपडेट करें: http://kyc-hindi.example.com",
    "Hi mum, my phone broke, please send Rs 15000 to this new UPI id urgently",
    "Your e-challan of Rs 2,500 is unpaid. Pay within 24 hours at http://challan.example.com",
    "Your OTP for login is 482913. Valid for 10 minutes. Do not share it with anyone.",
    "Your order #A1234 has been shipped and will arrive on Tuesday.",
    "Electricity bill of Rs 1,240 for March is generated. Due date 15 April.",
    "Reminder: your appointment is tomorrow at 10:30 AM at the clinic.",
    "Your salary of Rs 45,000 has been credited to your account ending 4521.",
    "Mom, I will reach home by 8 pm. Have dinner ready.",
    "Thanks for paying Rs 600 towards your mobile recharge. Validity till 20 Nov.",
    "Kal subah 7 baje meeting hai, please time pe aana.",
    "Your flight 6E 245 is on time. Boarding starts at 06:40 from gate 12.",
    "Your monthly statement is ready. View it in the app under statements.",
    "Your password was changed successfully. If this was not you, contact support.",
    "Bus ticket booked: Bengaluru to Chennai, 21 Nov, seat 14. Show this SMS at boarding.",
    "आपका ऑर्डर कल तक पहुँच जाएगा। धन्यवाद।",
    "Dentist appointment confirmed for Friday 5 pm. Reply C to cancel.",
    "Payment of Rs 2,350 to the grocery store was successful. Ref 88213.",
    "Your cab driver is 4 minutes away. Check the vehicle number in the app.",
    "Your insurance premium of Rs 8,400 is due on 30 Nov. Pay via the app.",
    "Your train PNR 4521678930 is confirmed for coach S4.",
    "Your library book is due on 25 Nov. Renew online on the library website.",
    "Gym membership renewed till March 2027. See you at the gym!",
    "Check the weather: light rain expected in Chennai this evening.",
)


# ---- pure helpers (no onnxruntime, no tokenizers) ----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="encoder_bench")
    p.add_argument("--model", required=True, type=Path, help="ONNX model file")
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="folder with tokenizer.json (and tokenizer_config.json); not needed with --rows",
    )
    p.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="pre-tokenised .npz of SMS rows (build_encoder_rows.py) instead of --tokenizer",
    )
    p.add_argument(
        "--companion-peak-mb",
        type=float,
        default=None,
        help="peak RSS of a companion process, added (not measured) into the combined estimate",
    )
    p.add_argument("--threads", required=True, type=int, help="intra-op threads")
    p.add_argument("--seq-lens", nargs="+", type=int, default=list(DEFAULT_SEQ_LENS))
    p.add_argument("--runs", type=int, default=50, help="timed runs per measurement")
    p.add_argument("--warmup", type=int, default=5, help="untimed runs before each measurement")
    p.add_argument(
        "--max-temp",
        type=float,
        default=DEFAULT_MAX_TEMP,
        help="refuse to start at or above this SoC temperature, and stop at it (C)",
    )
    p.add_argument(
        "--burst",
        type=int,
        default=0,
        help="duty cycle: run messages in bursts of N, cooling before each burst "
        "(0 = continuous, the original mode)",
    )
    p.add_argument(
        "--cool-below",
        type=float,
        default=DEFAULT_COOL_BELOW,
        help="burst mode: wait until the SoC is below this before each burst (C)",
    )
    p.add_argument(
        "--max-cool-wait",
        type=float,
        default=DEFAULT_MAX_COOL_WAIT,
        help="burst mode: bound on each cooldown wait (s); 3 waits in a row that do not cool "
        "abort the run",
    )
    p.add_argument(
        "--burst-abort-c",
        type=float,
        default=DEFAULT_BURST_ABORT_C,
        help="burst mode: a reading at or above this aborts and discards the burst (C)",
    )
    p.add_argument(
        "--only",
        choices=("all", "realistic", "sweep"),
        default="all",
        help="which measurements to run",
    )
    p.add_argument("--out", type=Path, default=None, help="write the JSON document here")
    args = p.parse_args(None if argv is None else list(argv))
    if args.threads < 1:
        p.error("--threads must be at least 1")
    if args.runs < 1 or args.warmup < 0:
        p.error("--runs must be at least 1 and --warmup must not be negative")
    if not args.seq_lens or any(not 1 <= n <= MAX_LEN for n in args.seq_lens):
        p.error(f"--seq-lens must be between 1 and {MAX_LEN}")
    if args.max_temp <= 0:
        p.error("--max-temp must be positive")
    if (args.tokenizer is None) == (args.rows is None):
        p.error("give exactly one of --tokenizer and --rows")
    if args.companion_peak_mb is not None and args.companion_peak_mb < 0:
        p.error("--companion-peak-mb must not be negative")
    if args.burst < 0:
        p.error("--burst must not be negative")
    if args.burst > 0:
        if args.cool_below <= 0 or args.max_cool_wait <= 0:
            p.error("--cool-below and --max-cool-wait must be positive")
        if args.burst_abort_c <= args.cool_below:
            p.error("--burst-abort-c must be above --cool-below")
    return args


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (the numpy default), q in [0, 100]."""
    if not values:
        raise ValueError("percentile of an empty sample")
    s = sorted(float(v) for v in values)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def summarise(samples_ms: Sequence[float]) -> dict[str, Any]:
    if not samples_ms:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None, "min": None}
    return {
        "n": len(samples_ms),
        "p50": round(percentile(samples_ms, 50), 3),
        "p95": round(percentile(samples_ms, 95), 3),
        "p99": round(percentile(samples_ms, 99), 3),
        "mean": round(sum(samples_ms) / len(samples_ms), 3),
        "min": round(min(samples_ms), 3),
    }


def fit_to_length(ids: Sequence[int], length: int, pad_id: int) -> tuple[list[int], list[int]]:
    """(ids, attention_mask) of exactly ``length``: truncated, or padded on the right with
    mask 0."""
    if length < 1:
        raise ValueError("length must be positive")
    real = list(ids)[:length]
    mask = [1] * len(real) + [0] * (length - len(real))
    return real + [pad_id] * (length - len(real)), mask


def truncate_keep_special(ids: Sequence[int], max_len: int) -> list[int]:
    """Truncate to max_len the way the HF tokenizer does with special tokens: the content is cut
    and the final special token (SEP) is kept."""
    ids = list(ids)
    if len(ids) <= max_len:
        return ids
    return ids[: max_len - 1] + [ids[-1]]


def resolve_pad_id(
    config: dict[str, Any], token_to_id: Callable[[str], int | None]
) -> tuple[int, str | None]:
    """(pad id, pad token). The pad token comes from tokenizer_config.json; 0 when missing."""
    token = config.get("pad_token")
    if isinstance(token, dict):
        token = token.get("content")
    if isinstance(token, str) and token:
        pid = token_to_id(token)
        if pid is not None:
            return int(pid), token
    return 0, token if isinstance(token, str) else None


def decode_throttled(hex_value: str | None) -> dict[str, Any]:
    """The vcgencmd get_throttled value ("0x50005") as a hex string and the set flag names."""
    if not hex_value:
        return {"hex": None, "flags": []}
    try:
        number = int(hex_value, 16)
    except ValueError:
        return {"hex": hex_value, "flags": []}
    flags = [name for bit, name in sorted(THROTTLE_BITS.items()) if number >> bit & 1]
    return {"hex": hex(number), "flags": flags}


def parse_temp_text(text: str | None) -> float | None:
    """'temp=41.2'C' from vcgencmd measure_temp to 41.2."""
    if not text or "=" not in text:
        return None
    try:
        return float(text.split("=", 1)[1].strip().rstrip("'C").rstrip("'"))
    except ValueError:
        return None


def start_refusal(temp_c: float | None, max_temp: float) -> str | None:
    """Why the run must not start, or None. An unreadable temperature (a laptop) is no refusal."""
    if temp_c is not None and temp_c >= max_temp:
        return f"SoC at {temp_c:.1f} C, at or above --max-temp {max_temp:g} C"
    return None


class TempGate:
    """Records the highest temperature seen and trips when a reading reaches max_temp. The reader
    is injected so the gate can be tested without a Pi."""

    def __init__(self, max_temp: float, reader: Callable[[], float | None]) -> None:
        self.max_temp = float(max_temp)
        self._reader = reader
        self.max_seen: float | None = None
        self.stop_reason: str | None = None

    def sample(self) -> float | None:
        temp = self._reader()
        if temp is None:
            return None
        if self.max_seen is None or temp > self.max_seen:
            self.max_seen = temp
        if self.stop_reason is None and temp >= self.max_temp:
            self.stop_reason = (
                f"SoC reached {temp:.1f} C, at or above --max-temp {self.max_temp:g} C"
            )
        return temp

    @property
    def tripped(self) -> bool:
        return self.stop_reason is not None


def token_length_stats(lengths: Sequence[int], truncated: int) -> dict[str, Any]:
    if not lengths:
        return {"n": 0}
    return {
        "n": len(lengths),
        "min": min(lengths),
        "p50": round(percentile(lengths, 50), 1),
        "p95": round(percentile(lengths, 95), 1),
        "max": max(lengths),
        "mean": round(sum(lengths) / len(lengths), 1),
        "truncated_at_max_len": truncated,
        "max_len": MAX_LEN,
    }


def load_rows(path: Path) -> tuple[list[list[int]], int, dict[str, Any]]:
    """Read a rows file written by build_encoder_rows.py: (rows, pad_id, meta). Each row is the
    unpadded token ids (already truncated to MAX_LEN, final SEP kept). meta holds the token length
    summary of the untruncated tokenisation, as stored by the builder."""
    import numpy as np

    with np.load(path, allow_pickle=False) as npz:
        ids = np.asarray(npz["ids"], dtype=np.int64)
        lengths = np.asarray(npz["lengths"], dtype=np.int64)
        full = np.asarray(npz["full_lengths"], dtype=np.int64)
        pad_id = int(npz["pad_id"])
        meta = json.loads(str(npz["meta_json"]))
    if ids.ndim != 2 or ids.shape[1] > MAX_LEN or lengths.shape != (ids.shape[0],):
        raise ValueError(f"{path}: ids {ids.shape} and lengths {lengths.shape} do not agree")
    if ids.shape[0] == 0:
        raise ValueError(f"{path}: no rows")
    if np.any(lengths < 1) or np.any(lengths > ids.shape[1]):
        raise ValueError(f"{path}: a length is outside 1..{ids.shape[1]}")
    if full.shape != lengths.shape:
        raise ValueError(f"{path}: full_lengths does not match lengths")
    rows = [ids[i, : lengths[i]].tolist() for i in range(ids.shape[0])]
    meta = {
        **meta,
        "rows": int(ids.shape[0]),
        "truncated_at_max_len": int(np.sum(full > MAX_LEN)),
        "full_token_len": token_length_stats(full.tolist(), int(np.sum(full > MAX_LEN))),
    }
    return rows, pad_id, meta


def length_bucket_stats(pairs: Sequence[tuple[int, float]]) -> dict[str, Any]:
    """Latency summary per token-length bucket from (token length, ms) pairs."""
    out: dict[str, Any] = {}
    for lo, hi in LENGTH_BUCKETS:
        out[f"{lo}-{hi}"] = summarise([ms for length, ms in pairs if lo <= length <= hi])
    return out


class Cooler:
    """Waits until the SoC is below ``cool_below`` before a burst. Each wait is bounded by
    ``max_wait_s``; a wait that ends without cooling is a failure, and ``max_failures`` failures in
    a row abort the run. A missing reading counts as cool (as in ``start_refusal``). Clock and sleep
    are injected so the logic can be tested without a Pi."""

    def __init__(
        self,
        read_temp: Callable[[], float | None],
        cool_below: float,
        max_wait_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_s: float = COOL_POLL_S,
        max_failures: int = MAX_COOL_FAILURES,
    ) -> None:
        self.read_temp = read_temp
        self.cool_below = float(cool_below)
        self.max_wait_s = float(max_wait_s)
        self.clock = clock
        self.sleep = sleep
        self.poll_s = poll_s
        self.max_failures = max_failures
        self.failures = 0
        self.waits: list[dict[str, Any]] = []

    def _cool(self, temp: float | None) -> bool:
        return temp is None or temp < self.cool_below

    def wait(self) -> dict[str, Any]:
        t_start = self.clock()
        first = self.read_temp()
        temp = first
        cooled = self._cool(temp)
        while not cooled and self.clock() - t_start < self.max_wait_s:
            self.sleep(self.poll_s)
            temp = self.read_temp()
            cooled = self._cool(temp)
        self.failures = 0 if cooled else self.failures + 1
        rec = {
            "temp_before": first,
            "temp_after": temp,
            "waited_s": round(self.clock() - t_start, 2),
            "cooled": cooled,
        }
        self.waits.append(rec)
        return rec

    @property
    def aborted(self) -> bool:
        return self.failures >= self.max_failures

    @property
    def non_cooling(self) -> int:
        return sum(1 for w in self.waits if not w["cooled"])


def _cooldown_stop(cooler: Cooler) -> str:
    return (
        f"cooldown: {cooler.max_failures} waits in a row did not bring the SoC below "
        f"{cooler.cool_below:g} C within {cooler.max_wait_s:g} s each"
    )


def cool_until_ready(cooler: Cooler) -> tuple[dict[str, Any], str | None]:
    """Waits until the SoC is below the cool threshold. Returns the last wait and a stop reason if
    the cooldown aborted the run."""
    while True:
        wait = cooler.wait()
        if wait["cooled"]:
            return wait, None
        if cooler.aborted:
            return wait, _cooldown_stop(cooler)


def run_warmup(
    session: Any,
    feeds: Sequence[dict[str, Any]],
    count: int,
    args: argparse.Namespace,
    gate: TempGate,
    cooler: Cooler,
) -> dict[str, Any]:
    """Untimed warm-up, gated like the bursts: the SoC must be below --cool-below before warm-up
    starts, and warm-up ends early at --burst-abort-c (the heat it adds is seen by the abort) or at
    the hard stop. A cooldown abort stops the run. Returns what ran and why it stopped."""
    _, stop = cool_until_ready(cooler)
    if stop is not None:
        return {"messages_run": 0, "aborted_at_c": None, "stop_reason": stop}
    ran = 0
    for feed in list(feeds)[:count]:
        session.run(None, feed)
        ran += 1
        temp = gate.sample()
        if gate.tripped:
            return {"messages_run": ran, "aborted_at_c": None, "stop_reason": gate.stop_reason}
        if temp is not None and temp >= args.burst_abort_c:
            return {"messages_run": ran, "aborted_at_c": temp, "stop_reason": None}
    return {"messages_run": ran, "aborted_at_c": None, "stop_reason": None}


def run_bursts(
    session: Any,
    feeds: Sequence[dict[str, Any]],
    lengths: Sequence[int],
    args: argparse.Namespace,
    gate: TempGate,
    cooler: Cooler,
    clock: Callable[[], float] = time.perf_counter,
    warmup: int = 0,
) -> dict[str, Any]:
    """Runs ``feeds`` in bursts of ``args.burst``, cooling before each. A burst is discarded (not
    retried) when a reading reaches ``--burst-abort-c`` or one message stalls past STALL_MS, which
    is how a guard pause shows up. Only kept bursts contribute latencies, as (token length, ms).
    ``warmup`` untimed messages run first, under the same temperature gate (see run_warmup)."""
    burst = int(args.burst)
    pairs: list[tuple[int, float]] = []
    bursts: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    stop: str | None = None
    t_start = clock()
    warm: dict[str, Any] | None = None
    if warmup > 0:
        warm = run_warmup(session, feeds, warmup, args, gate, cooler)
        stop = warm["stop_reason"]
    i = 0
    while i < len(feeds) and stop is None:
        wait, stop = cool_until_ready(cooler)
        if stop is not None:
            break
        if gate.tripped:
            stop = gate.stop_reason
            break
        end = min(i + burst, len(feeds))
        kept: list[tuple[int, float]] = []
        reason: str | None = None
        temp_max: float | None = None
        for j in range(i, end):
            t0 = clock()
            session.run(None, feeds[j])
            ms = (clock() - t0) * 1000.0
            kept.append((int(lengths[j]), ms))
            temp = gate.sample()
            if temp is not None and (temp_max is None or temp > temp_max):
                temp_max = temp
            if gate.tripped:
                stop = gate.stop_reason
                reason = "hard stop at --max-temp"
                break
            if ms > STALL_MS:
                reason = f"message took {ms:.0f} ms (stall, over {STALL_MS:g} ms)"
                break
            if temp is not None and temp >= args.burst_abort_c:
                reason = f"SoC {temp:.1f} C, at or above the {args.burst_abort_c:g} C burst abort"
                break
        rec: dict[str, Any] = {
            "index": len(bursts),
            "first_row": i,
            "messages_run": len(kept),
            "temp_before": wait["temp_before"],
            "cool_waited_s": wait["waited_s"],
            "temp_max": temp_max,
            "discarded": reason is not None,
        }
        bursts.append(rec)
        if reason is not None:
            discarded.append({**rec, "reason": reason})
        else:
            pairs.extend(kept)
        i = end
    wall = clock() - t_start
    return {
        "burst_size": burst,
        "pairs": pairs,
        "bursts": bursts,
        "discarded": discarded,
        "stop_reason": stop,
        "stopped": stop is not None,
        "measured": len(pairs),
        "rows_not_measured": len(feeds) - len(pairs),
        "warmup": warm,
        "cool_waits": cooler.waits,
        "non_cooling_waits": cooler.non_cooling,
        "wall_s": round(wall, 2),
        "msgs_per_min": round(len(pairs) / wall * 60.0, 2) if wall > 0 else None,
        "cooldown": {
            "cool_below_c": cooler.cool_below,
            "max_cool_wait_s": cooler.max_wait_s,
            "max_failures": cooler.max_failures,
            "burst_abort_c": args.burst_abort_c,
            "stall_ms": STALL_MS,
        },
    }


def _burst_summary(run: dict[str, Any]) -> dict[str, Any]:
    """The part of a run_bursts result that goes into the JSON (no raw pairs)."""
    return {k: v for k, v in run.items() if k != "pairs"}


def run_benchmark(
    session: Any,
    encode: Callable[[Any], list[int]],
    pad_id: int,
    texts: Sequence[Any],
    args: argparse.Namespace,
    gate: TempGate,
    input_names: tuple[str, str] = ("input_ids", "attention_mask"),
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Every measurement. Only ``session.run`` (and ``encode``) are called on the model side, so a
    fake session can drive the whole payload in tests. Stops at the first tripped gate. With
    ``args.burst`` the messages run as bursts with a cooldown before each (see run_bursts)."""
    import numpy as np

    id_name, mask_name = input_names
    seq_out: dict[str, Any] = {}
    stop = False
    do_sweep = args.only in ("all", "sweep")
    do_real = args.only in ("all", "realistic")
    burst_mode = args.burst > 0
    cooler = Cooler(gate.sample, args.cool_below, args.max_cool_wait) if burst_mode else None
    stop_reason: str | None = None

    def feed_for(ids: list[int], mask: list[int]) -> dict[str, Any]:
        return {
            id_name: np.asarray([ids], dtype=np.int64),
            mask_name: np.asarray([mask], dtype=np.int64),
        }

    for length in args.seq_lens if do_sweep else []:
        if stop:
            break
        if burst_mode:
            tok_ms = []
            feeds = []
            for i in range(args.runs):
                text = texts[i % len(texts)]
                t0 = clock()
                ids, mask = fit_to_length(encode(text), length, pad_id)
                tok_ms.append((clock() - t0) * 1000.0)
                feeds.append(feed_for(ids, mask))
            run = run_bursts(
                session, feeds, [length] * len(feeds), args, gate, cooler, clock, warmup=args.warmup
            )
            seq_out[str(length)] = {
                "seq_len": length,
                "tokenize_ms": summarise(tok_ms),
                "latency_ms": summarise([ms for _, ms in run["pairs"]]),
                "burst": _burst_summary(run),
            }
            if run["stopped"]:
                stop = True
                stop_reason = run["stop_reason"]
            continue
        tok_ms: list[float] = []
        feeds: list[dict[str, Any]] = []
        for i in range(args.runs):
            text = texts[i % len(texts)]
            t0 = clock()
            ids, mask = fit_to_length(encode(text), length, pad_id)
            tok_ms.append((clock() - t0) * 1000.0)
            if i < len(texts):
                feeds.append(feed_for(ids, mask))
        block: dict[str, Any] = {"tokenize_ms": summarise(tok_ms), "latency_ms": None}
        for i in range(args.warmup):
            session.run(None, feeds[i % len(feeds)])
        lat: list[float] = []
        for i in range(args.runs):
            feed = feeds[i % len(feeds)]
            t0 = clock()
            session.run(None, feed)
            lat.append((clock() - t0) * 1000.0)
            gate.sample()
            if gate.tripped:
                stop = True
                break
        block["latency_ms"] = summarise(lat)
        block["seq_len"] = length
        seq_out[str(length)] = block

    if not do_real:
        return {"seq_lens": seq_out, "realistic": None, "stopped": stop, "stop_reason": stop_reason}

    if burst_mode:
        full_b = [encode(t) for t in texts]
        natural_b = [truncate_keep_special(ids, MAX_LEN) for ids in full_b]
        token_lengths = token_length_stats(
            [len(x) for x in natural_b], sum(1 for ids in full_b if len(ids) > MAX_LEN)
        )
        realistic_b: dict[str, Any] = {
            "n_strings": len(texts),
            "passes": 1,
            "mode": "burst",
            "token_lengths": token_lengths,
            "skipped": stop,
        }
        if not stop:
            feeds_b = [feed_for(ids, [1] * len(ids)) for ids in natural_b]
            run = run_bursts(
                session,
                feeds_b,
                [len(x) for x in natural_b],
                args,
                gate,
                cooler,
                clock,
                warmup=args.warmup,
            )
            realistic_b["latency_ms"] = summarise([ms for _, ms in run["pairs"]])
            realistic_b["by_length"] = length_bucket_stats(run["pairs"])
            realistic_b["duty"] = {
                "measured": run["measured"],
                "wall_s": run["wall_s"],
                "msgs_per_min": run["msgs_per_min"],
            }
            realistic_b["burst"] = _burst_summary(run)
            realistic_b["skipped"] = run["stopped"]
            if run["stopped"]:
                stop_reason = run["stop_reason"]
            stop = stop or run["stopped"]
        else:
            realistic_b["latency_ms"] = None
        return {
            "seq_lens": seq_out,
            "realistic": realistic_b,
            "stopped": stop,
            "stop_reason": stop_reason,
        }

    realistic: dict[str, Any] = {"n_strings": len(texts), "passes": REALISTIC_PASSES}
    full = [encode(t) for t in texts]
    natural = [truncate_keep_special(ids, MAX_LEN) for ids in full]
    realistic["token_lengths"] = token_length_stats(
        [len(x) for x in natural], sum(1 for ids in full if len(ids) > MAX_LEN)
    )
    if not stop:
        feeds_nat = [feed_for(ids, [1] * len(ids)) for ids in natural]
        lat = []
        for i in range(min(args.warmup, len(feeds_nat))):
            session.run(None, feeds_nat[i])
        for _ in range(REALISTIC_PASSES):
            for feed in feeds_nat:
                t0 = clock()
                session.run(None, feed)
                lat.append((clock() - t0) * 1000.0)
                gate.sample()
                if gate.tripped:
                    stop = True
                    break
            if stop:
                break
        realistic["latency_ms"] = summarise(lat)
    else:
        realistic["latency_ms"] = None
    realistic["skipped"] = stop
    return {"seq_lens": seq_out, "realistic": realistic, "stopped": stop, "stop_reason": None}


def build_payload(
    *,
    status: str,
    args: argparse.Namespace,
    model_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
    runtime: dict[str, Any],
    memory_kb: dict[str, Any],
    thermal: dict[str, Any],
    measured: dict[str, Any] | None,
    started: str,
    finished: str,
    reason: str | None = None,
    combined: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The one JSON document written to --out."""
    measured = measured or {"seq_lens": {}, "realistic": None}
    return {
        "tool": "encoder_bench",
        "schema": SCHEMA,
        "status": status,
        "reason": reason,
        "started_utc": started,
        "finished_utc": finished,
        "model": model_info,
        "tokenizer": tokenizer_info,
        "runtime": runtime,
        "memory_kb": memory_kb,
        "thermal": thermal,
        "seq_lens": measured["seq_lens"],
        "realistic": measured["realistic"],
        "combined": combined,
        "args": {
            "model": str(args.model),
            "tokenizer": str(args.tokenizer),
            "threads": args.threads,
            "seq_lens": list(args.seq_lens),
            "runs": args.runs,
            "warmup": args.warmup,
            "max_temp_c": args.max_temp,
            "rows": str(args.rows) if args.rows else None,
            "companion_peak_mb": args.companion_peak_mb,
        },
    }


def human_summary(payload: dict[str, Any]) -> str:
    lines = [
        f"encoder_bench: {payload['status']}"
        + (f" ({payload['reason']})" if payload.get("reason") else "")
    ]
    model = payload["model"]
    if model.get("load_s") is not None:
        lines.append(
            f"model {model.get('bytes')} bytes, load {model['load_s']} s, "
            f"threads {payload['runtime']['threads']}"
        )
    for length, block in payload["seq_lens"].items():
        lat = block.get("latency_ms") or {}
        tok = block.get("tokenize_ms") or {}
        if lat.get("p50") is not None:
            lines.append(
                f"  seq {length:>4}: latency p50 {lat['p50']} ms, p95 {lat['p95']} ms, "
                f"p99 {lat['p99']} ms; "
                f"tokenise p50 {tok.get('p50')} ms"
            )
    real = payload.get("realistic") or {}
    rl = real.get("latency_ms") or {}
    if rl.get("p50") is not None:
        lines.append(
            f"  natural length: latency p50 {rl['p50']} ms, p95 {rl['p95']} ms, p99 {rl['p99']} ms"
        )
    duty = real.get("duty")
    if duty:
        lines.append(
            f"  bursts: {len(real['burst']['discarded'])} discarded of "
            f"{len(real['burst']['bursts'])}; "
            f"duty {duty['msgs_per_min']} msgs/min over {duty['wall_s']} s"
        )
    combined = payload.get("combined")
    if combined:
        lines.append(
            f"  encoder peak {combined['encoder_vmhwm_mb']} MB + companion "
            f"{combined['companion_peak_mb']} MB = {combined['estimated_sum_mb']} MB "
            "(estimate, not measured together)"
        )
    therm = payload["thermal"]
    lines.append(
        f"temp start {therm['start'].get('temp_c')} C, max {therm.get('max_c')} C, "
        f"end {therm['end'].get('temp_c')} C"
    )
    return "\n".join(lines)


# ---- device readers (vcgencmd first, as deploy/pi/service/edge_device.py does) ----


def _vcgencmd(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["vcgencmd", *args], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def read_soc_temp_c() -> float | None:
    temp = parse_temp_text(_vcgencmd(["measure_temp"]))
    if temp is not None:
        return temp
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def read_throttled() -> dict[str, Any]:
    text = _vcgencmd(["get_throttled"])
    value = text.split("=", 1)[1].strip() if text and "=" in text else None
    return decode_throttled(value)


def read_rss_kb() -> dict[str, int | None]:
    """VmRSS and VmHWM (peak) of this process, in kB, from /proc/self/status."""
    out: dict[str, int | None] = {"vmrss": None, "vmhwm": None}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                out["vmrss" if key == "VmRSS" else "vmhwm"] = int(rest.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = _now()
    temp_start = read_soc_temp_c()
    thermal_start = {"temp_c": temp_start, "throttled": read_throttled()}
    thermal: dict[str, Any] = {"max_temp_limit_c": args.max_temp, "start": thermal_start}
    refusal = start_refusal(temp_start, args.max_temp)
    if refusal:
        payload = build_payload(
            status="refused",
            args=args,
            model_info={},
            tokenizer_info={},
            runtime={},
            memory_kb={},
            thermal={**thermal, "end": None, "max_c": temp_start},
            measured=None,
            started=started,
            finished=_now(),
            reason=refusal,
        )
        _write(args.out, payload)
        print(f"encoder_bench: refused: {refusal}")
        return 2
    try:
        import onnxruntime as ort

        if args.rows is None:
            from tokenizers import Tokenizer
    except ImportError as exc:
        print(f"encoder_bench: missing dependency: {exc}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"encoder_bench: model file not found: {args.model}", file=sys.stderr)
        return 1
    model_info = {
        "file": args.model.name,
        "bytes": args.model.stat().st_size,
        "sha256": file_sha256(args.model),
    }
    rows_meta: dict[str, Any] | None = None
    if args.rows is not None:
        if not args.rows.is_file():
            print(f"encoder_bench: rows file not found: {args.rows}", file=sys.stderr)
            return 1
        try:
            rows, pad_id, rows_meta = load_rows(args.rows)
        except (ValueError, KeyError, OSError) as exc:
            print(f"encoder_bench: bad rows file: {exc}", file=sys.stderr)
            return 1
        tokenizer_info = {
            "rows": args.rows.name,
            "rows_sha256": file_sha256(args.rows),
            "pad_id": pad_id,
            "rows_meta": rows_meta,
        }
        texts: Sequence[Any] = rows

        def encode(row: Any) -> list[int]:
            return list(row)

    else:
        tok_json = args.tokenizer / "tokenizer.json"
        if not tok_json.is_file():
            print(f"encoder_bench: {tok_json} not found", file=sys.stderr)
            return 1
        tokenizer = Tokenizer.from_file(str(tok_json))
        config_path = args.tokenizer / "tokenizer_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        pad_id, pad_token = resolve_pad_id(config, tokenizer.token_to_id)
        tokenizer_info = {"dir": args.tokenizer.name, "pad_id": pad_id, "pad_token": pad_token}
        texts = SMS_STRINGS

        def encode(text: Any) -> list[int]:
            return tokenizer.encode(text).ids

    rss_before = read_rss_kb()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        str(args.model), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    load_s = round(time.perf_counter() - t0, 3)
    rss_after = read_rss_kb()
    names = [i.name for i in session.get_inputs()]
    if "input_ids" not in names or "attention_mask" not in names:
        print(f"encoder_bench: model inputs {names} lack input_ids/attention_mask", file=sys.stderr)
        return 1
    model_info["load_s"] = load_s

    runtime = {
        "onnxruntime": ort.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": args.threads,
        "inter_op_threads": 1,
        "execution": "sequential",
        "graph_optimization": "ORT_ENABLE_ALL",
        "provider": "CPUExecutionProvider",
    }
    gate = TempGate(args.max_temp, read_soc_temp_c)
    gate.max_seen = temp_start

    measured = run_benchmark(session, encode, pad_id, texts, args, gate)
    if rows_meta is not None and measured["realistic"] is not None:
        tl = measured["realistic"]["token_lengths"]
        tl["truncated_at_max_len"] = rows_meta["truncated_at_max_len"]
        tl["untruncated"] = rows_meta["full_token_len"]

    rss_end = read_rss_kb()
    thermal["end"] = {"temp_c": read_soc_temp_c(), "throttled": read_throttled()}
    thermal["max_c"] = gate.max_seen
    thermal["stop_reason"] = gate.stop_reason
    memory = {
        "rss_before_load": rss_before["vmrss"],
        "rss_after_load": rss_after["vmrss"],
        "vmhwm_peak": rss_end["vmhwm"],
        "rss_end": rss_end["vmrss"],
    }
    combined = None
    if args.companion_peak_mb is not None and rss_end["vmhwm"] is not None:
        enc_mb = round(rss_end["vmhwm"] / 1024, 1)
        combined = {
            "encoder_vmhwm_mb": enc_mb,
            "companion_peak_mb": args.companion_peak_mb,
            "estimated_sum_mb": round(enc_mb + args.companion_peak_mb, 1),
            "basis": "ADDED estimate: encoder VmHWM plus companion peak from a separate run; "
            "not measured together",
        }
    status = "stopped_early" if measured["stopped"] else "ok"
    payload = build_payload(
        status=status,
        args=args,
        model_info=model_info,
        tokenizer_info=tokenizer_info,
        runtime=runtime,
        memory_kb=memory,
        thermal=thermal,
        measured=measured,
        started=started,
        finished=_now(),
        reason=measured.get("stop_reason") or gate.stop_reason,
        combined=combined,
    )
    _write(args.out, payload)
    print(human_summary(payload))
    if args.out is not None:
        print(f"wrote {args.out}")
    return 3 if measured["stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
