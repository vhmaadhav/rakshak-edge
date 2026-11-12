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
    assert s == {"n": 3, "p50": 2.0, "p95": 2.9, "mean": 2.0, "min": 1.0}
    assert eb.summarise([]) == {"n": 0, "p50": None, "p95": None, "mean": None, "min": None}


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
