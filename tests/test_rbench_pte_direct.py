"""pte_direct runner: prefix reuse, decode loop, stops, timing; fake model, no executorch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "deploy" / "pi" / "bench"
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (BENCH, SERVICE, REPO / "src", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import rakshak_edge_service as svc  # noqa: E402
from rbench.errors import MissingDependency  # noqa: E402
from rbench.runners import make_runner  # noqa: E402
from rbench.runners.pte_direct import PteDirectRunner  # noqa: E402

SPLIT = 9
EOS = 0
# Token ids 1..26 decode to letters a..z, so the text of a call is easy to read.
PROMPTS = {
    "A": [1, 2, SPLIT, 3, 4],  # prefix [1,2,9], suffix [3,4]
    "B": [1, 2, SPLIT, 5, 6, 7],  # same prefix as A, suffix [5,6,7]
    "C": [7, 8, SPLIT, 3],  # another prefix [7,8,9], suffix [3]
    "D": [1, 2, 3, 4],  # no split id
    "E": [1, 2, SPLIT],  # split id is the last token
}


class FakeModel:
    """Records each forward call; returns queued token ids (EOS once the queue is empty)."""

    def __init__(self, returns=()):
        self.returns = list(returns)
        self.calls: list[tuple[list[int], int]] = []
        self.fail_next = False

    def forward(self, ids, pos):
        self.calls.append((list(ids), pos))
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("model blew up")
        return self.returns.pop(0) if self.returns else EOS


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.5
        return self.now


def _encode(prompt: str) -> list[int]:
    return list(PROMPTS[prompt])


def _decode(ids: list[int]) -> str:
    return "".join(chr(96 + i) for i in ids)


def _runner(model: FakeModel, reuse: bool = True, max_context: int = 2048, clock=None):
    return PteDirectRunner.from_parts(
        model.forward,
        _encode,
        _decode,
        eos_ids=[EOS, 128001],
        max_context=max_context,
        reuse_prefix=reuse,
        clock=clock or FakeClock(),
        split_id=SPLIT,
    )


def test_first_prompt_prefills_prefix_at_zero_then_suffix_at_split():
    model = FakeModel([0, 7])  # prefix result ignored, then one new token
    runner = _runner(model)
    text, timing = runner.generate("A", max_new_tokens=1)
    assert model.calls == [([1, 2, SPLIT], 0), ([3, 4], 3)]
    assert text == "g"
    assert timing["prefix_reused"] is False
    assert timing["prefix_tokens"] == 3
    assert timing["prefill_tokens"] == 5
    assert timing["prompt_tokens"] == 5


def test_same_prefix_forwards_only_the_suffix_at_split():
    model = FakeModel([0, 7, 8])
    runner = _runner(model)
    runner.generate("A", max_new_tokens=1)
    model.calls.clear()
    _, timing = runner.generate("B", max_new_tokens=1)
    assert model.calls == [([5, 6, 7], 3)]
    assert timing["prefix_reused"] is True
    assert timing["prefill_tokens"] == len(PROMPTS["B"]) - 3
    assert timing["prompt_tokens"] == len(PROMPTS["B"])
    assert timing["prefix_tokens"] == 3


def test_different_prefix_reprefills_at_zero_and_replaces_stored_prefix():
    model = FakeModel([0, 7, 0, 7, 0, 7])
    runner = _runner(model)
    runner.generate("A", max_new_tokens=1)
    model.calls.clear()
    _, timing = runner.generate("C", max_new_tokens=1)
    assert model.calls == [([7, 8, SPLIT], 0), ([3], 3)]
    assert timing["prefix_reused"] is False
    assert runner._prefix_ids == [7, 8, SPLIT]
    model.calls.clear()
    _, timing = runner.generate("A", max_new_tokens=1)
    assert model.calls == [([1, 2, SPLIT], 0), ([3, 4], 3)]
    assert timing["prefix_reused"] is False


def test_no_split_id_runs_whole_prompt_at_zero_and_clears_prefix():
    model = FakeModel([7, 0, 7, 0])
    runner = _runner(model)
    runner.generate("A", max_new_tokens=1)
    assert runner._prefix_ids == [1, 2, SPLIT]
    model.calls.clear()
    _, timing = runner.generate("D", max_new_tokens=1)
    assert model.calls == [([1, 2, 3, 4], 0)]
    assert timing["prefix_reused"] is False and timing["prefix_tokens"] == 0
    assert runner._prefix_ids is None
    model.calls.clear()
    runner.generate("A", max_new_tokens=1)
    assert model.calls[0] == ([1, 2, SPLIT], 0)


def test_split_id_as_last_token_runs_whole_prompt_and_clears_prefix():
    model = FakeModel([7, 0, 7, 0])
    runner = _runner(model)
    runner.generate("A", max_new_tokens=1)
    model.calls.clear()
    _, timing = runner.generate("E", max_new_tokens=1)
    assert model.calls == [([1, 2, SPLIT], 0)]
    assert timing["prefix_tokens"] == 0 and timing["prefix_reused"] is False
    assert runner._prefix_ids is None
    model.calls.clear()
    _, timing = runner.generate("B", max_new_tokens=1)
    assert model.calls[0] == ([1, 2, SPLIT], 0)
    assert timing["prefix_reused"] is False


def test_reuse_disabled_always_runs_whole_prompt_at_zero():
    model = FakeModel([7, 8, 7, 8])
    runner = _runner(model, reuse=False)
    _, first = runner.generate("A", max_new_tokens=1)
    _, second = runner.generate("A", max_new_tokens=1)
    assert model.calls == [([1, 2, SPLIT, 3, 4], 0), ([1, 2, SPLIT, 3, 4], 0)]
    for timing in (first, second):
        assert timing["prefix_reused"] is False
        assert timing["prefix_tokens"] == 0
        assert timing["prefill_tokens"] == 5


def test_decode_feeds_each_new_token_at_len_ids_then_onwards():
    model = FakeModel([5, 6, 7, EOS])
    runner = _runner(model, reuse=False)
    text, timing = runner.generate("D", max_new_tokens=10)
    assert model.calls == [([1, 2, 3, 4], 0), ([5], 4), ([6], 5), ([7], 6)]
    assert text == "efg"
    assert timing["new_tokens"] == 3


def test_eos_is_not_in_text_and_is_not_a_stop_hit():
    model = FakeModel([5, EOS])
    runner = _runner(model, reuse=False)
    text, timing = runner.generate("D", max_new_tokens=10)
    assert text == "e"
    assert timing["stop_hit"] is False
    assert timing["new_tokens"] == 1


def test_stop_when_includes_the_stopping_token_and_makes_no_further_call():
    model = FakeModel([5, 6, 7])
    runner = _runner(model, reuse=False)
    text, timing = runner.generate("D", max_new_tokens=10, stop_when=lambda t: t.endswith("f"))
    assert text == "ef"
    assert timing["stop_hit"] is True
    assert model.calls == [([1, 2, 3, 4], 0), ([5], 4)]


def test_max_new_tokens_reached_stops_without_extra_call():
    model = FakeModel([5, 6, 7, 8])
    runner = _runner(model, reuse=False)
    text, timing = runner.generate("D", max_new_tokens=2)
    assert text == "ef"
    assert timing["new_tokens"] == 2 and timing["stop_hit"] is False
    assert model.calls == [([1, 2, 3, 4], 0), ([5], 4)]


def test_zero_new_tokens_returns_empty_after_prefill():
    model = FakeModel([5])
    runner = _runner(model, reuse=False)
    text, timing = runner.generate("D", max_new_tokens=0)
    assert text == ""
    assert timing["new_tokens"] == 0
    assert model.calls == [([1, 2, 3, 4], 0)]


def test_same_prompt_gives_the_same_suffix_calls_whatever_came_before():
    fresh = FakeModel([7, 0])
    _runner(fresh).generate("A", max_new_tokens=1)

    used = FakeModel([0, 7, 9, 0, 7, 0])
    runner = _runner(used)
    runner.generate("C", max_new_tokens=1)
    runner.generate("D", max_new_tokens=1)
    used.calls.clear()
    runner.generate("A", max_new_tokens=1)
    assert used.calls == fresh.calls


def test_context_overflow_raises_before_any_forward_call():
    model = FakeModel()
    runner = _runner(model, max_context=10)
    with pytest.raises(ValueError, match="context"):
        runner.generate("A", max_new_tokens=5)  # 5 + 5 > 10 - 1
    assert model.calls == []
    runner.generate("A", max_new_tokens=4)  # 5 + 4 <= 9 is fine
    assert model.calls


def test_forward_exception_clears_stored_prefix_and_propagates():
    model = FakeModel([0, 7])
    runner = _runner(model)
    runner.generate("A", max_new_tokens=1)
    assert runner._prefix_ids == [1, 2, SPLIT]
    model.fail_next = True
    with pytest.raises(RuntimeError, match="blew up"):
        runner.generate("B", max_new_tokens=1)
    assert runner._prefix_ids is None


def test_reset_clears_stored_prefix_so_next_call_reprefills():
    model = FakeModel([0, 7, 8])
    runner = _runner(model)
    runner.generate("A", max_new_tokens=1)
    runner.reset()
    assert runner._prefix_ids is None
    model.calls.clear()
    _, timing = runner.generate("B", max_new_tokens=1)
    assert model.calls[0] == ([1, 2, SPLIT], 0)
    assert timing["prefix_reused"] is False


def test_timing_keys_and_ttft_bounded_by_wall():
    model = FakeModel([0, 7, 8])
    runner = _runner(model)
    _, timing = runner.generate("A", max_new_tokens=2)
    for key in (
        "wall_ms",
        "ttft_ms",
        "prompt_tokens",
        "prefill_tokens",
        "prefix_tokens",
        "prefix_reused",
        "new_tokens",
        "stop_hit",
        "prefill_tok_s",
        "decode_tok_s",
    ):
        assert key in timing, key
    assert timing["ttft_ms"] <= timing["wall_ms"]
    assert timing["prefill_tok_s"] == pytest.approx(5 / (timing["ttft_ms"] / 1000.0), rel=1e-3)


def test_make_runner_returns_direct_runner():
    assert isinstance(make_runner("pte_llm_direct"), PteDirectRunner)


def test_load_without_executorch_raises_missing_dependency(tmp_path):
    try:
        import executorch.extension.pybindings.portable_lib  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("executorch is importable here")
    with pytest.raises(MissingDependency, match="executorch"):
        PteDirectRunner().load(tmp_path / "m.pte", 2, {"tokenizer": tmp_path / "tok.model"})


def test_load_without_tokenizer_raises_missing_dependency(tmp_path):
    with pytest.raises(MissingDependency):
        PteDirectRunner().load(tmp_path / "m.pte", 2, {})


def test_env_default_rejects_unknown_runner(monkeypatch):
    monkeypatch.setenv("RAKSHAK_LLM_RUNNER", "bogus")
    with pytest.raises(SystemExit, match="RAKSHAK_LLM_RUNNER"):
        svc._env_llm_runner_default()


def test_cli_default_llm_runner_is_direct(monkeypatch):
    monkeypatch.delenv("RAKSHAK_LLM_RUNNER", raising=False)
    assert svc._env_llm_runner_default() == "direct"


def test_service_health_reports_llm_runner():
    def device():
        return {"temp_c": 50.0}

    service = svc.Service(
        object(),
        threads=2,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=device,
        llm_runner="direct",
    )
    assert service.health()["llm_runner"] == "direct"
    default = svc.Service(
        object(), threads=2, temp_limit=75.0, log_path=None, models={}, device_fn=device
    )
    assert default.health()["llm_runner"] == "runner"
