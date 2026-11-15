"""Tests for deploy/pi/bench/encoder_bench.py: the pure parts only. No onnxruntime, no tokenizers,
no Pi: the model side is a fake session that returns zeros."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "bench"
sys.path.insert(0, str(BENCH))

import encoder_bench as eb  # noqa: E402


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: returns zero logits for batch 1."""

    def __init__(self) -> None:
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def run(self, names, feed):
        self.calls += 1
        self.shapes.append(tuple(feed["input_ids"].shape))
        return [np.zeros((1, 2), dtype=np.float32)]


def _encode(text: str) -> list[int]:
    # a stable fake tokeniser: one id per word, plus CLS and SEP like the real one
    return [101, *range(1, len(text.split()) + 1), 102]


def _args(*extra: str):
    return eb.parse_args(["--model", "m.onnx", "--tokenizer", "tokdir", "--threads", "4", *extra])


# ---- percentile and summaries ----


def test_percentile_matches_linear_interpolation():
    assert eb.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert eb.percentile([1, 2, 3, 4], 0) == 1
    assert eb.percentile([1, 2, 3, 4], 100) == 4
    assert eb.percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48.0)
    assert eb.percentile([7.0], 95) == 7.0


def test_percentile_rejects_empty_sample():
    with pytest.raises(ValueError):
        eb.percentile([], 50)


def test_summarise_has_all_fields_and_handles_empty():
    s = eb.summarise([3.0, 1.0, 2.0])
    assert s == {"n": 3, "p50": 2.0, "p95": 2.9, "p99": 2.98, "mean": 2.0, "min": 1.0}
    assert eb.summarise([]) == {
        "n": 0,
        "p50": None,
        "p95": None,
        "p99": None,
        "mean": None,
        "min": None,
    }


# ---- padding and truncation ----


def test_fit_to_length_pads_with_mask_zero():
    ids, mask = eb.fit_to_length([5, 6, 7], 5, pad_id=0)
    assert ids == [5, 6, 7, 0, 0]
    assert mask == [1, 1, 1, 0, 0]


def test_fit_to_length_truncates_and_keeps_full_mask():
    ids, mask = eb.fit_to_length(list(range(10)), 4, pad_id=0)
    assert ids == [0, 1, 2, 3]
    assert mask == [1, 1, 1, 1]


def test_fit_to_length_exact_and_invalid_length():
    assert eb.fit_to_length([1, 2], 2, 0) == ([1, 2], [1, 1])
    with pytest.raises(ValueError):
        eb.fit_to_length([1], 0, 0)


def test_truncate_keep_special_keeps_final_token():
    ids = list(range(200))
    out = eb.truncate_keep_special(ids, 128)
    assert len(out) == 128
    assert out[:127] == ids[:127]
    assert out[-1] == ids[-1]
    assert eb.truncate_keep_special([1, 2, 3], 128) == [1, 2, 3]


def test_resolve_pad_id_from_config_and_fallbacks():
    lookup = {"[PAD]": 3}.get
    assert eb.resolve_pad_id({"pad_token": "[PAD]"}, lookup) == (3, "[PAD]")
    assert eb.resolve_pad_id({"pad_token": {"content": "[PAD]"}}, lookup) == (3, "[PAD]")
    assert eb.resolve_pad_id({}, lookup) == (0, None)
    assert eb.resolve_pad_id({"pad_token": "<pad>"}, lookup) == (0, "<pad>")


# ---- device text parsing ----


def test_parse_temp_text():
    assert eb.parse_temp_text("temp=41.2'C") == pytest.approx(41.2)
    assert eb.parse_temp_text(None) is None
    assert eb.parse_temp_text("garbage") is None


def test_decode_throttled_flags():
    out = eb.decode_throttled("0x50005")
    assert out["hex"] == "0x50005"
    assert out["flags"] == [
        "under_voltage_now",
        "throttled_now",
        "under_voltage_occurred",
        "throttled_occurred",
    ]
    assert eb.decode_throttled("0x0") == {"hex": "0x0", "flags": []}
    assert eb.decode_throttled(None) == {"hex": None, "flags": []}


# ---- argument parsing ----


def test_parse_args_defaults_and_values():
    args = _args()
    assert args.threads == 4
    assert args.seq_lens == [32, 64, 128]
    assert args.runs == 50 and args.warmup == 5
    assert args.max_temp == 75.0
    assert args.out is None
    custom = _args(
        "--seq-lens",
        "16",
        "8",
        "--runs",
        "3",
        "--warmup",
        "0",
        "--max-temp",
        "70",
        "--out",
        "x.json",
    )
    assert custom.seq_lens == [16, 8]
    assert custom.runs == 3 and custom.warmup == 0 and custom.max_temp == 70.0
    assert custom.out == Path("x.json")


@pytest.mark.parametrize(
    "extra",
    [
        ("--seq-lens", "0"),
        ("--seq-lens", "129"),
        ("--runs", "0"),
        ("--warmup", "-1"),
        ("--max-temp", "0"),
    ],
)
def test_parse_args_rejects_bad_values(extra):
    with pytest.raises(SystemExit):
        _args(*extra)


def test_parse_args_requires_threads_and_paths():
    with pytest.raises(SystemExit):
        eb.parse_args(["--model", "m.onnx", "--tokenizer", "d"])


# ---- temperature gate ----


def test_start_refusal_at_and_below_limit():
    assert eb.start_refusal(75.0, 75.0) is not None
    assert eb.start_refusal(74.9, 75.0) is None
    assert eb.start_refusal(None, 75.0) is None


def test_temp_gate_records_max_and_trips_at_limit():
    readings = iter([40.0, 70.0, 80.0, 60.0])
    gate = eb.TempGate(75.0, lambda: next(readings))
    gate.sample()
    gate.sample()
    assert not gate.tripped
    gate.sample()
    assert gate.tripped
    assert gate.max_seen == 80.0
    assert "80.0" in gate.stop_reason
    gate.sample()
    assert gate.max_seen == 80.0  # a later cooler reading does not lower the maximum


def test_temp_gate_without_readings_never_trips():
    gate = eb.TempGate(75.0, lambda: None)
    assert gate.sample() is None
    assert not gate.tripped and gate.max_seen is None


# ---- the measurement and the JSON payload, with a fake session ----


def test_run_benchmark_payload_shape_with_fake_session():
    session = FakeSession()
    args = _args("--seq-lens", "8", "16", "--runs", "3", "--warmup", "1")
    gate = eb.TempGate(75.0, lambda: 45.0)
    texts = eb.SMS_STRINGS[:5]
    measured = eb.run_benchmark(session, _encode, 0, texts, args, gate, clock=lambda: 0.0)
    assert measured["stopped"] is False
    assert set(measured["seq_lens"]) == {"8", "16"}
    block = measured["seq_lens"]["8"]
    assert block["seq_len"] == 8
    assert block["latency_ms"]["n"] == 3
    assert block["tokenize_ms"]["n"] == 3
    assert all(shape == (1, 8) for shape in session.shapes[:4])
    real = measured["realistic"]
    assert real["n_strings"] == 5
    assert real["token_lengths"]["n"] == 5
    assert real["latency_ms"]["n"] == 5 * eb.REALISTIC_PASSES
    assert real["skipped"] is False


def test_run_benchmark_stops_early_and_records_it():
    session = FakeSession()
    args = _args("--seq-lens", "8", "--runs", "10", "--warmup", "0")
    readings = iter([45.0, 45.0, 80.0])
    gate = eb.TempGate(75.0, lambda: next(readings, 80.0))
    measured = eb.run_benchmark(
        session, _encode, 0, eb.SMS_STRINGS[:3], args, gate, clock=lambda: 0.0
    )
    assert measured["stopped"] is True
    assert measured["realistic"]["skipped"] is True
    assert measured["realistic"]["latency_ms"] is None
    assert gate.stop_reason is not None


def test_payload_is_json_serialisable_and_complete():
    session = FakeSession()
    args = _args("--seq-lens", "8", "--runs", "2", "--warmup", "0")
    gate = eb.TempGate(75.0, lambda: 50.0)
    measured = eb.run_benchmark(
        session, _encode, 0, eb.SMS_STRINGS[:2], args, gate, clock=lambda: 0.0
    )
    payload = eb.build_payload(
        status="ok",
        args=args,
        model_info={"file": "m.onnx", "bytes": 10, "sha256": "0" * 64, "load_s": 0.1},
        tokenizer_info={"dir": "tokdir", "pad_id": 0, "pad_token": "[PAD]"},
        runtime={"onnxruntime": "1.31.1", "threads": 4},
        memory_kb={"rss_before_load": 1, "rss_after_load": 2, "vmhwm_peak": 3},
        thermal={
            "max_temp_limit_c": 75.0,
            "start": {"temp_c": 50.0, "throttled": {}},
            "end": {"temp_c": 50.0, "throttled": {}},
            "max_c": 50.0,
        },
        measured=measured,
        started="2026-11-11T00:00:00Z",
        finished="2026-11-11T00:00:01Z",
    )
    round_trip = json.loads(json.dumps(payload))
    for key in (
        "tool",
        "schema",
        "status",
        "model",
        "tokenizer",
        "runtime",
        "memory_kb",
        "thermal",
        "seq_lens",
        "realistic",
        "args",
    ):
        assert key in round_trip
    assert round_trip["args"]["threads"] == 4
    text = eb.human_summary(round_trip)
    assert "ok" in text and "seq" in text


def test_main_refuses_when_hot_and_writes_refusal_json(tmp_path, monkeypatch):
    monkeypatch.setattr(eb, "read_soc_temp_c", lambda: 80.0)
    out = tmp_path / "bench.json"
    code = eb.main(
        [
            "--model",
            str(tmp_path / "m.onnx"),
            "--tokenizer",
            str(tmp_path),
            "--threads",
            "4",
            "--out",
            str(out),
        ]
    )
    assert code == 2
    doc = json.loads(out.read_text())
    assert doc["status"] == "refused"
    assert "80.0" in doc["reason"]


def test_sms_strings_are_unique_and_non_empty():
    assert len(eb.SMS_STRINGS) == len(set(eb.SMS_STRINGS)) >= 16
    assert all(s.strip() for s in eb.SMS_STRINGS)


# ---- pre-tokenised rows, input exclusivity and the companion estimate ----


def _write_rows(path: Path, rows: list[list[int]], full: list[int] | None = None) -> Path:
    width = max(len(r) for r in rows)
    ids = np.zeros((len(rows), width), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
    lengths = np.asarray([len(r) for r in rows], dtype=np.int64)
    np.savez(
        path,
        ids=ids,
        lengths=lengths,
        full_lengths=np.asarray(full if full is not None else lengths, dtype=np.int64),
        pad_id=np.int64(0),
        meta_json=np.asarray(json.dumps({"split": "cal", "seed": 0})),
    )
    return path


def test_load_rows_round_trip_and_truncation_count(tmp_path):
    path = _write_rows(tmp_path / "r.npz", [[101, 5, 102], [101, 7, 8, 9, 102]], full=[3, 200])
    rows, pad_id, meta = eb.load_rows(path)
    assert rows == [[101, 5, 102], [101, 7, 8, 9, 102]]
    assert pad_id == 0
    assert meta["rows"] == 2 and meta["split"] == "cal"
    assert meta["truncated_at_max_len"] == 1
    assert meta["full_token_len"]["n"] == 2


def test_load_rows_rejects_inconsistent_files(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(
        bad,
        ids=np.zeros((2, 4), dtype=np.int64),
        lengths=np.asarray([9, 1], dtype=np.int64),
        full_lengths=np.asarray([9, 1], dtype=np.int64),
        pad_id=np.int64(0),
        meta_json=np.asarray("{}"),
    )
    with pytest.raises(ValueError):
        eb.load_rows(bad)


def test_parse_args_needs_exactly_one_input_mode(tmp_path):
    base = ["--model", "m.onnx", "--threads", "3"]
    with pytest.raises(SystemExit):
        eb.parse_args(base)
    with pytest.raises(SystemExit):
        eb.parse_args([*base, "--tokenizer", "t", "--rows", "r.npz"])
    args = eb.parse_args([*base, "--rows", "r.npz", "--companion-peak-mb", "1839.3"])
    assert args.tokenizer is None and args.rows == Path("r.npz")
    assert args.companion_peak_mb == pytest.approx(1839.3)
    with pytest.raises(SystemExit):
        eb.parse_args([*base, "--rows", "r.npz", "--companion-peak-mb", "-1"])


def test_run_benchmark_over_rows_uses_stored_ids(tmp_path):
    rows, _, _ = eb.load_rows(_write_rows(tmp_path / "r.npz", [[101, 4, 5, 102], [101, 6, 102]]))
    seen: list[int] = []

    class Recording(FakeSession):
        def run(self, names, feed):
            seen.append(int(feed["attention_mask"].sum()))
            return super().run(names, feed)

    args = _args("--seq-lens", "4", "--runs", "2", "--warmup", "0")
    gate = eb.TempGate(75.0, lambda: 45.0)
    measured = eb.run_benchmark(
        Recording(), lambda row: list(row), 0, rows, args, gate, clock=lambda: 0.0
    )
    assert measured["realistic"]["n_strings"] == 2
    assert measured["realistic"]["token_lengths"]["max"] == 4
    assert seen[:2] == [4, 3]


def test_payload_carries_added_combined_estimate_only_when_given():
    payload = eb.build_payload(
        status="ok",
        args=_args(),
        model_info={},
        tokenizer_info={},
        runtime={"threads": 3},
        memory_kb={},
        thermal={"start": {}, "end": {}},
        measured=None,
        started="2026-11-12T00:00:00Z",
        finished="2026-11-12T00:00:01Z",
        combined={
            "encoder_vmhwm_mb": 400.0,
            "companion_peak_mb": 1839.3,
            "estimated_sum_mb": 2239.3,
            "basis": "ADDED estimate",
        },
    )
    assert payload["combined"]["estimated_sum_mb"] == pytest.approx(2239.3)
    assert "ADDED" in payload["combined"]["basis"]
    plain = eb.build_payload(
        status="ok",
        args=_args(),
        model_info={},
        tokenizer_info={},
        runtime={},
        memory_kb={},
        thermal={"start": {}, "end": {}},
        measured=None,
        started="x",
        finished="y",
    )
    assert plain["combined"] is None


# ---- the laptop-side rows builder (pure parts; no tokenizers, no rakshak data) ----

import build_encoder_rows as ber  # noqa: E402


def test_pick_rows_is_seeded_sorted_and_checks_size():
    a = ber.pick_rows(100, 10, seed=0)
    assert np.array_equal(a, ber.pick_rows(100, 10, seed=0))
    assert len(set(a.tolist())) == 10 and list(a) == sorted(a)
    with pytest.raises(ValueError):
        ber.pick_rows(5, 10, seed=0)
    with pytest.raises(ValueError):
        ber.pick_rows(5, 0, seed=0)


def test_write_rows_round_trips_through_load_rows(tmp_path):
    full = [[101, 4, 102], [101, *range(1, 200), 102]]
    out = tmp_path / "rows.npz"
    summary = ber.write_rows(out, full, pad_id=0, meta={"split": "cal", "seed": 0})
    assert summary["rows"] == 2
    assert summary["token_len"]["truncated_at_max_len"] == 1
    rows, pad_id, meta = eb.load_rows(out)
    assert pad_id == 0
    assert rows[0] == [101, 4, 102]
    assert len(rows[1]) == eb.MAX_LEN and rows[1][-1] == 102
    assert meta["truncated_at_max_len"] == 1
    assert meta["full_token_len"]["max"] == 201


# ---- burst mode: cooldown, discard and abort (simulated SoC, no Pi) ----


class SimSoC:
    """A simulated SoC. Each session.run sets the next scheduled temperature; each sleep cools
    the SoC by ``cool_per_s`` (never below ``floor``)."""

    def __init__(self, sched=(), start=40.0, cool_per_s=5.0, floor=40.0):
        self.temp = float(start)
        self.sched = list(sched)
        self.calls = 0
        self.cool_per_s = cool_per_s
        self.floor = floor
        self.now = 0.0

    def read(self):
        return self.temp

    def sleep(self, seconds):
        self.now += seconds
        self.temp = max(self.floor, self.temp - self.cool_per_s * seconds)

    def clock(self):
        # latency clock: each call advances 10 ms, so each measured message is 10 ms
        self.now += 0.01
        return self.now

    def run(self, names, feed):
        i = self.calls
        self.calls += 1
        if i < len(self.sched):
            self.temp = float(self.sched[i])
        return [np.zeros((1, 2), dtype=np.float32)]


def _burst_args(*extra: str):
    return eb.parse_args(
        ["--model", "m.onnx", "--tokenizer", "tokdir", "--threads", "3", "--burst", "10", *extra]
    )


def _feeds(n: int):
    return [{"input_ids": np.zeros((1, 4), dtype=np.int64)} for _ in range(n)]


def test_cooler_waits_until_below_threshold_and_resets_failures():
    soc = SimSoC(start=60.0, cool_per_s=3.0)
    cooler = eb.Cooler(soc.read, 55.0, 120.0, clock=lambda: soc.now, sleep=soc.sleep)
    rec = cooler.wait()
    assert rec["cooled"] is True
    assert rec["temp_before"] == 60.0
    assert rec["waited_s"] == pytest.approx(2.0)  # 60 -> 57 -> 54
    assert cooler.failures == 0 and not cooler.aborted


def test_cooler_bounded_wait_counts_failures_and_aborts_after_three():
    soc = SimSoC(start=60.0, cool_per_s=0.0)
    cooler = eb.Cooler(soc.read, 55.0, 5.0, clock=lambda: soc.now, sleep=soc.sleep)
    for _ in range(2):
        rec = cooler.wait()
        assert rec["cooled"] is False and rec["waited_s"] == pytest.approx(5.0)
        assert not cooler.aborted
    cooler.wait()
    assert cooler.aborted
    assert cooler.non_cooling == 3


def test_cooler_missing_reading_counts_as_cool():
    cooler = eb.Cooler(lambda: None, 55.0, 120.0)
    assert cooler.wait()["cooled"] is True


def test_run_bursts_discards_burst_at_abort_temperature_and_continues():
    # the 13th message (index 12) reads 63 C: its burst (3 messages run) is discarded, not retried
    sched = [40.0] * 25
    sched[12] = 63.0
    soc = SimSoC(sched=sched, start=40.0)
    gate = eb.TempGate(64.0, soc.read)
    cooler = eb.Cooler(soc.read, 55.0, 120.0, clock=lambda: soc.now, sleep=soc.sleep)
    args = _burst_args()
    run = eb.run_bursts(soc, _feeds(25), [32] * 25, args, gate, cooler, clock=soc.clock)
    assert run["stopped"] is False
    assert len(run["bursts"]) == 3
    assert len(run["discarded"]) == 1
    assert run["discarded"][0]["messages_run"] == 3
    assert "burst abort" in run["discarded"][0]["reason"]
    assert run["measured"] == 15  # 10 kept + 5 kept; the discarded 3 are not counted
    assert run["rows_not_measured"] == 10
    assert all(ms == pytest.approx(10.0) for _, ms in run["pairs"])
    # the hot burst was followed by a cooldown wait below 55 C before the next burst
    assert run["cool_waits"][-1]["cooled"] is True
    assert run["non_cooling_waits"] == 0


def test_run_bursts_stall_discards_burst():
    class Stall(SimSoC):
        def clock(self):
            self.now += 0.01
            return self.now

    soc = Stall(start=40.0)
    gate = eb.TempGate(64.0, soc.read)
    cooler = eb.Cooler(soc.read, 55.0, 120.0, clock=lambda: soc.now, sleep=soc.sleep)
    calls = {"n": 0}
    real_clock = soc.clock

    def slow_clock():
        # the 2nd message of the first burst takes 2 s (as a guard pause would look)
        calls["n"] += 1
        if calls["n"] == 5:  # end of the 2nd message (call 1 starts the run)
            soc.now += 2.0
        return real_clock()

    run = eb.run_bursts(soc, _feeds(10), [32] * 10, _burst_args(), gate, cooler, clock=slow_clock)
    assert len(run["discarded"]) == 1
    assert "stall" in run["discarded"][0]["reason"]
    assert run["measured"] == 0


def test_run_bursts_aborts_after_three_non_cooling_waits():
    soc = SimSoC(start=60.0, cool_per_s=0.0)
    gate = eb.TempGate(64.0, soc.read)
    cooler = eb.Cooler(soc.read, 55.0, 3.0, clock=lambda: soc.now, sleep=soc.sleep)
    run = eb.run_bursts(soc, _feeds(10), [32] * 10, _burst_args(), gate, cooler, clock=soc.clock)
    assert run["stopped"] is True
    assert "cooldown" in run["stop_reason"]
    assert run["measured"] == 0
    assert run["non_cooling_waits"] == 3
    assert soc.calls == 0  # no burst ran while hot


def test_length_bucket_stats_split_by_token_length():
    pairs = [(10, 5.0), (32, 7.0), (40, 9.0), (100, 11.0)]
    out = eb.length_bucket_stats(pairs)
    assert out["1-32"]["n"] == 2
    assert out["33-64"]["n"] == 1
    assert out["65-96"]["n"] == 0 and out["65-96"]["p50"] is None
    assert out["97-128"]["p50"] == 11.0


def test_run_benchmark_burst_realistic_reports_duty_and_buckets():
    soc = SimSoC(start=40.0)
    gate = eb.TempGate(64.0, soc.read)
    args = _burst_args("--only", "realistic", "--warmup", "0")
    texts = list(eb.SMS_STRINGS)  # 32 strings -> 4 bursts of 10, 10, 10, 2
    out = eb.run_benchmark(soc, _encode, 0, texts, args, gate, clock=soc.clock)
    real = out["realistic"]
    assert out["stopped"] is False and out["seq_lens"] == {}
    assert real["mode"] == "burst" and real["passes"] == 1
    assert len(real["burst"]["bursts"]) == 4
    assert real["burst"]["discarded"] == []
    assert real["latency_ms"]["n"] == 32
    assert real["latency_ms"]["p50"] == pytest.approx(10.0)
    assert real["duty"]["measured"] == 32
    assert real["duty"]["msgs_per_min"] > 0
    assert sum(real["by_length"][k]["n"] for k in real["by_length"]) == 32
    assert "pairs" not in real["burst"]


def test_run_benchmark_burst_sweep_one_burst_per_length():
    soc = SimSoC(start=40.0)
    gate = eb.TempGate(64.0, soc.read)
    args = _burst_args(
        "--only",
        "sweep",
        "--seq-lens",
        "32",
        "64",
        "128",
        "--runs",
        "20",
        "--burst",
        "20",
        "--warmup",
        "0",
    )
    out = eb.run_benchmark(soc, _encode, 0, list(eb.SMS_STRINGS), args, gate, clock=soc.clock)
    assert out["realistic"] is None
    assert set(out["seq_lens"]) == {"32", "64", "128"}
    for block in out["seq_lens"].values():
        assert len(block["burst"]["bursts"]) == 1
        assert block["latency_ms"]["n"] == 20


def test_burst_argument_validation():
    with pytest.raises(SystemExit):
        _burst_args("--burst", "-1")
    with pytest.raises(SystemExit):
        _burst_args("--cool-below", "60", "--burst-abort-c", "55")
    with pytest.raises(SystemExit):
        _burst_args("--max-cool-wait", "0")
    args = _burst_args("--only", "sweep")
    assert args.burst == 10 and args.cool_below == 55.0 and args.max_cool_wait == 120.0
    assert args.burst_abort_c == 63.0 and args.only == "sweep"


def test_warmup_waits_for_cool_before_running():
    # starts at 60 C (hot): the warm-up must not run until the SoC is below 55 C
    soc = SimSoC(start=60.0, cool_per_s=3.0)
    gate = eb.TempGate(64.0, soc.read)
    cooler = eb.Cooler(soc.read, 55.0, 120.0, clock=lambda: soc.now, sleep=soc.sleep)
    run = eb.run_bursts(
        soc, _feeds(10), [32] * 10, _burst_args(), gate, cooler, clock=soc.clock, warmup=5
    )
    assert run["cool_waits"][0]["temp_before"] == 60.0
    assert run["cool_waits"][0]["cooled"] is True
    assert run["warmup"] == {"messages_run": 5, "aborted_at_c": None, "stop_reason": None}
    assert run["measured"] == 10 and run["discarded"] == []


def test_warmup_aborts_at_burst_abort_temperature_then_bursts_cool_first():
    # the 3rd warm-up message reads 63 C: warm-up stops there (2 more messages are not run)
    sched = [40.0] * 30
    sched[2] = 63.0
    soc = SimSoC(sched=sched, start=40.0)
    gate = eb.TempGate(64.0, soc.read)
    cooler = eb.Cooler(soc.read, 55.0, 120.0, clock=lambda: soc.now, sleep=soc.sleep)
    run = eb.run_bursts(
        soc, _feeds(20), [32] * 20, _burst_args(), gate, cooler, clock=soc.clock, warmup=5
    )
    assert run["warmup"]["messages_run"] == 3
    assert run["warmup"]["aborted_at_c"] == 63.0
    assert run["stopped"] is False
    # the first burst waited from 63 C back below 55 C before it ran
    assert run["cool_waits"][0]["temp_before"] == 40.0  # cooled before warm-up (already cool)
    assert run["cool_waits"][1]["temp_before"] == 63.0  # the burst waits from 63 C
    assert run["cool_waits"][1]["cooled"] is True
    assert run["measured"] == 20 and run["discarded"] == []


def test_warmup_cooldown_abort_runs_no_message_and_stops():
    soc = SimSoC(start=60.0, cool_per_s=0.0)
    gate = eb.TempGate(64.0, soc.read)
    cooler = eb.Cooler(soc.read, 55.0, 3.0, clock=lambda: soc.now, sleep=soc.sleep)
    run = eb.run_bursts(
        soc, _feeds(10), [32] * 10, _burst_args(), gate, cooler, clock=soc.clock, warmup=5
    )
    assert run["stopped"] is True and "cooldown" in run["stop_reason"]
    assert run["warmup"]["messages_run"] == 0
    assert soc.calls == 0
