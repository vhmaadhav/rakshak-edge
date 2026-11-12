"""Explainer cache: CachedGenerator (LRU by exact prompt), its use in EdgeAnalyzer and in the
service health, and the eval report's handling of cached calls. No model is loaded."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "service"
REPO = Path(__file__).resolve().parents[1]
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import rakshak_edge_service as svc  # noqa: E402
from edge_pipeline import EdgeAnalyzer  # noqa: E402
from explainer_cache import CachedGenerator  # noqa: E402
from test_edge_service import (  # noqa: E402
    MEDIUM_TEXT,
    THRESHOLD,
    FakeDetector,
    FakeLLM,
    _device,
    _energy_stub,
)

SENTENCE = " may be a scam because it claims a prize."


class CountingLLM(FakeLLM):
    """FakeLLM with a sentence that depends on the prompt, so different prompts differ."""

    load_ms = 1234.5
    threads_reported = 3

    def __init__(self):
        super().__init__(SENTENCE)
        self.prompts = []

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.prompts.append(prompt)
        text, timing = super().generate(prompt, max_new_tokens=max_new_tokens, stop_when=stop_when)
        return f"{text.strip()} [{len(prompt)}]", timing


def stop_a(text):
    return True


def stop_b(text):
    return False


def test_miss_then_hit_returns_the_same_text_and_calls_inner_once():
    inner = CountingLLM()
    gen = CachedGenerator(inner, 4)
    text1, t1 = gen.generate("prompt one", max_new_tokens=64, stop_when=None)
    text2, t2 = gen.generate("prompt one", max_new_tokens=64, stop_when=None)
    assert text1 == text2
    assert inner.calls == 1
    assert gen.stats() == {"capacity": 4, "size": 1, "hits": 1, "misses": 1}


def test_miss_timing_keeps_inner_keys_and_says_not_cached():
    inner = CountingLLM()
    _, timing = CachedGenerator(inner, 4).generate("p")
    assert timing["cached"] is False
    assert timing["ttft_ms"] == 40.0
    assert timing["wall_ms"] == 120.0
    assert timing["prompt_tokens"] == 90
    assert timing["stop_hit"] is True


def test_miss_does_not_mutate_the_inner_timing_dict():
    class Inner:
        def __init__(self):
            self.timing = {"wall_ms": 1.0, "new_tokens": 2}

        def generate(self, prompt, max_new_tokens=64, stop_when=None):
            return "x", self.timing

    inner = Inner()
    _, timing = CachedGenerator(inner).generate("p")
    assert timing is not inner.timing
    assert "cached" not in inner.timing


def test_hit_timing_is_cached_with_no_decode_keys():
    gen = CachedGenerator(CountingLLM(), 4)
    gen.generate("p")
    _, hit = gen.generate("p")
    assert hit["cached"] is True
    assert "ttft_ms" not in hit
    assert "prefill_tok_s" not in hit
    assert "decode_tok_s" not in hit
    assert hit["new_tokens"] == 6
    assert hit["prompt_tokens"] == 90
    assert hit["stop_hit"] is True
    assert isinstance(hit["wall_ms"], float) and hit["wall_ms"] >= 0.0


def test_different_max_new_tokens_prompt_and_stop_are_separate_keys():
    inner = CountingLLM()
    gen = CachedGenerator(inner, 10)
    gen.generate("p", max_new_tokens=64)
    gen.generate("p", max_new_tokens=32)
    gen.generate("q", max_new_tokens=64)
    gen.generate("p", max_new_tokens=64, stop_when=stop_a)
    gen.generate("p", max_new_tokens=64, stop_when=stop_b)
    assert inner.calls == 5
    assert gen.stats()["size"] == 5
    gen.generate("p", max_new_tokens=64, stop_when=stop_a)
    assert inner.calls == 5


def test_lru_eviction_keeps_a_recently_read_entry():
    inner = CountingLLM()
    gen = CachedGenerator(inner, 2)
    gen.generate("a")
    gen.generate("b")
    gen.generate("a")  # a is now the most recent
    gen.generate("c")  # evicts b
    assert gen.stats()["size"] == 2
    gen.generate("a")
    assert inner.calls == 3  # a and c were hits, a is still cached
    gen.generate("b")
    assert inner.calls == 4  # b was evicted and is generated again


def test_inner_exception_is_not_cached_and_the_next_call_retries():
    class Flaky:
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, max_new_tokens=64, stop_when=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return "ok", {"new_tokens": 1}

    inner = Flaky()
    gen = CachedGenerator(inner, 4)
    with pytest.raises(RuntimeError, match="boom"):
        gen.generate("p")
    assert gen.stats()["size"] == 0
    text, timing = gen.generate("p")
    assert text == "ok" and timing["cached"] is False
    assert inner.calls == 2


@pytest.mark.parametrize("capacity", [0, -1])
def test_capacity_below_one_is_refused(capacity):
    with pytest.raises(ValueError):
        CachedGenerator(CountingLLM(), capacity)


def test_capacity_must_be_an_int():
    with pytest.raises(ValueError):
        CachedGenerator(CountingLLM(), 2.5)


def test_other_attributes_are_forwarded_to_the_inner_object():
    inner = CountingLLM()
    gen = CachedGenerator(inner, 4)
    assert gen.load_ms == 1234.5
    assert gen.threads_reported == 3
    with pytest.raises(AttributeError):
        _ = gen.no_such_attribute


def test_stats_and_clear_keep_the_counters():
    inner = CountingLLM()
    gen = CachedGenerator(inner, 4)
    gen.generate("a")
    gen.generate("a")
    assert gen.stats() == {"capacity": 4, "size": 1, "hits": 1, "misses": 1}
    gen.clear()
    assert gen.stats() == {"capacity": 4, "size": 0, "hits": 1, "misses": 1}
    gen.generate("a")
    assert inner.calls == 2
    assert gen.stats()["misses"] == 2


def _strip(result):
    out = {k: v for k, v in result.items() if k != "timings_ms"}
    out["llm_calls"] = [
        {
            k: v
            for k, v in c.items()
            if k not in ("wall_ms", "ttft_ms", "cached", "prompt_tokens", "new_tokens")
        }
        for c in result["llm_calls"]
    ]
    return out


def test_edge_analyzer_second_identical_request_is_served_from_the_cache():
    inner = CountingLLM()
    gen = CachedGenerator(inner, 256)
    analyzer = EdgeAnalyzer(FakeDetector(0.9), gen, THRESHOLD)
    first = analyzer.analyse(MEDIUM_TEXT)
    calls_after_first = inner.calls
    assert calls_after_first >= 1
    second = analyzer.analyse(MEDIUM_TEXT)
    assert inner.calls == calls_after_first
    assert _strip(first) == _strip(second)
    assert first["llm_calls"] and all(c["cached"] is False for c in first["llm_calls"])
    assert all(c["cached"] is True for c in second["llm_calls"])
    assert second["timings_ms"]["llm_cached"] == len(second["llm_calls"])
    assert first["timings_ms"]["llm_cached"] == 0


def test_detector_only_result_has_llm_cached_zero():
    analyzer = EdgeAnalyzer(FakeDetector(0.9), CountingLLM(), THRESHOLD)
    out = analyzer.analyse("Hello there, how are you doing today my friend")
    assert out["llm_calls"] == []
    assert out["timings_ms"]["llm_cached"] == 0


def _service_with(cache):
    return svc.Service(
        FakeDetector(0.0),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=_device(50.0),
        energy_factory=_energy_stub,
        explainer_cache=cache,
    )


def test_health_reports_explainer_cache_stats_with_and_without_a_cache():
    off = _service_with(None).health()["explainer_cache"]
    assert off == {"capacity": 0, "size": 0, "hits": 0, "misses": 0}

    gen = CachedGenerator(CountingLLM(), 8)
    gen.generate("p")
    gen.generate("p")
    on = _service_with(gen).health()["explainer_cache"]
    assert on == {"capacity": 8, "size": 1, "hits": 1, "misses": 1}


def _eval_body(calls, *, llm_skipped=False):
    timings = {
        "canonicalise": 1.0,
        "detect": 1.0,
        "evidence": 1.0,
        "llm_total": sum(c["wall_ms"] for c in calls),
        "verify": 0.0,
        "total": 10.0,
    }
    dev = {
        "start": {"temp_c": 50.0},
        "end": {"temp_c": 51.0},
        "energy": {"energy_j": 2.0, "mean_w": 3.0},
        "rss_mb": 100.0,
    }
    return {
        "decision": {"risk": "HIGH", "action": "WARN", "lang_detected": "en"},
        "advisory": {
            "lang": "en",
            "mode": "llm",
            "path": "llm",
            "llm_skipped_low_risk": llm_skipped,
            "points": [],
        },
        "verifier": {"final_contract_ok": True, "g2_rejected": []},
        "timings_ms": timings,
        "llm_calls": calls,
        "device": dev,
    }


def test_eval_report_excludes_cached_calls_from_latency_and_counts_them(tmp_path):
    import eval_service

    live = {"evidence_id": "e1", "wall_ms": 1800.0, "cached": False}
    hit = {"evidence_id": "e2", "wall_ms": 0.1, "cached": True}
    rows = [
        {"http_status": 200, "group": "evidence", "body": _eval_body([live, hit])},
        {"http_status": 200, "group": "evidence", "body": _eval_body([hit])},
    ]
    (tmp_path / "responses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    meta = {
        "batch_wall_s": 1.0,
        "health_before": {},
        "health_after": {},
        "start_device": {},
    }
    (tmp_path / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert eval_service.report(tmp_path) == 0
    group = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["groups"]
    ev = group["evidence"]
    assert ev["llm_calls_total"] == 3
    assert ev["llm_calls_cached"] == 2
    assert ev["stages"]["llm_per_call_ms"]["n"] == 1
    assert ev["stages"]["llm_per_call_ms"]["p50"] == 1800.0
