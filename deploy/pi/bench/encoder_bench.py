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

Usage:
  encoder_bench.py --model PATH.onnx --tokenizer DIR --threads N [--seq-lens 32 64 128]
                   [--runs 50] [--warmup 5] [--max-temp 75] [--out FILE.json]

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
        required=True,
        type=Path,
        help="folder with tokenizer.json (and tokenizer_config.json)",
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
    p.add_argument("--out", type=Path, default=None, help="write the JSON document here")
    args = p.parse_args(list(argv or []))
    if args.threads < 1:
        p.error("--threads must be at least 1")
    if args.runs < 1 or args.warmup < 0:
        p.error("--runs must be at least 1 and --warmup must not be negative")
    if not args.seq_lens or any(not 1 <= n <= MAX_LEN for n in args.seq_lens):
        p.error(f"--seq-lens must be between 1 and {MAX_LEN}")
    if args.max_temp <= 0:
        p.error("--max-temp must be positive")
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
        return {"n": 0, "p50": None, "p95": None, "mean": None, "min": None}
    return {
        "n": len(samples_ms),
        "p50": round(percentile(samples_ms, 50), 3),
        "p95": round(percentile(samples_ms, 95), 3),
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


def run_benchmark(
    session: Any,
    encode: Callable[[str], list[int]],
    pad_id: int,
    texts: Sequence[str],
    args: argparse.Namespace,
    gate: TempGate,
    input_names: tuple[str, str] = ("input_ids", "attention_mask"),
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Every measurement. Only ``session.run`` (and ``encode``) are called on the model side, so a
    fake session can drive the whole payload in tests. Stops at the first tripped gate."""
    import numpy as np

    id_name, mask_name = input_names
    seq_out: dict[str, Any] = {}
    stop = False

    def feed_for(ids: list[int], mask: list[int]) -> dict[str, Any]:
        return {
            id_name: np.asarray([ids], dtype=np.int64),
            mask_name: np.asarray([mask], dtype=np.int64),
        }

    for length in args.seq_lens:
        if stop:
            break
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
    return {"seq_lens": seq_out, "realistic": realistic, "stopped": stop}


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
        "args": {
            "model": str(args.model),
            "tokenizer": str(args.tokenizer),
            "threads": args.threads,
            "seq_lens": list(args.seq_lens),
            "runs": args.runs,
            "warmup": args.warmup,
            "max_temp_c": args.max_temp,
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
                f"  seq {length:>4}: latency p50 {lat['p50']} ms, p95 {lat['p95']} ms; "
                f"tokenise p50 {tok.get('p50')} ms"
            )
    real = payload.get("realistic") or {}
    rl = real.get("latency_ms") or {}
    if rl.get("p50") is not None:
        lines.append(f"  natural length: latency p50 {rl['p50']} ms, p95 {rl['p95']} ms")
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
        from tokenizers import Tokenizer
    except ImportError as exc:
        print(f"encoder_bench: missing dependency: {exc}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"encoder_bench: model file not found: {args.model}", file=sys.stderr)
        return 1
    tok_json = args.tokenizer / "tokenizer.json"
    if not tok_json.is_file():
        print(f"encoder_bench: {tok_json} not found", file=sys.stderr)
        return 1

    model_info = {
        "file": args.model.name,
        "bytes": args.model.stat().st_size,
        "sha256": file_sha256(args.model),
    }
    tokenizer = Tokenizer.from_file(str(tok_json))
    config_path = args.tokenizer / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    pad_id, pad_token = resolve_pad_id(config, tokenizer.token_to_id)
    tokenizer_info = {"dir": args.tokenizer.name, "pad_id": pad_id, "pad_token": pad_token}

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

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text).ids

    measured = run_benchmark(session, encode, pad_id, SMS_STRINGS, args, gate)

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
        reason=gate.stop_reason,
    )
    _write(args.out, payload)
    print(human_summary(payload))
    if args.out is not None:
        print(f"wrote {args.out}")
    return 3 if measured["stopped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
