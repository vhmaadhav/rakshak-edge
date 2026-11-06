"""pte_llm runner tests with a mocked ExecuTorch API (no executorch on the laptop)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "bench"
sys.path.insert(0, str(BENCH))

from rbench.errors import MissingDependency  # noqa: E402
from rbench.runners import pte_llm  # noqa: E402


class _Stats:
    def __init__(self, **kw):
        self.model_load_start_ms = kw.get("load_start", 0)
        self.model_load_end_ms = kw.get("load_end", 0)
        self.inference_start_ms = kw.get("inf_start", 0)
        self.prompt_eval_end_ms = kw.get("prompt_end", 0)
        self.first_token_ms = kw.get("first", 0)
        self.inference_end_ms = kw.get("inf_end", 0)
        self.num_prompt_tokens = kw.get("n_prompt", 0)
        self.num_generated_tokens = kw.get("n_gen", 0)


class FakeRunner:
    """Mimics TextLLMRunner: the method loads on the first generate; reset clears the state."""

    instances: list[FakeRunner] = []

    def __init__(self, model_path, tokenizer_path, data_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.calls: list[str] = []
        self.loaded = False
        FakeRunner.instances.append(self)

    def reset(self):
        self.calls.append("reset")

    def generate(self, prompt, config, token_callback, stats_callback):
        self.calls.append(f"generate:{config.max_new_tokens}")
        if not self.loaded:
            self.loaded = True
            stats_callback(_Stats(load_start=1000, load_end=3500))
        for tok in ["A", "B", "C"][: config.max_new_tokens]:
            token_callback(tok)
        stats_callback(
            _Stats(
                load_start=1000,
                load_end=3500,
                inf_start=10_000,
                prompt_end=10_400,
                first=10_400,
                inf_end=11_400,
                n_prompt=32,
                n_gen=3,
            )
        )


class FakeConfig:
    def __init__(self, **kw):
        self.echo = kw.get("echo", True)
        self.max_new_tokens = kw.get("max_new_tokens", -1)
        self.temperature = kw.get("temperature", 0.8)
        self.num_bos = kw.get("num_bos", 0)
        self.num_eos = kw.get("num_eos", 0)


@pytest.fixture
def fake_executorch(monkeypatch):
    FakeRunner.instances.clear()
    runner_mod = types.ModuleType("executorch.extension.llm.runner")
    runner_mod.TextLLMRunner = FakeRunner
    runner_mod.GenerationConfig = FakeConfig
    portable = types.ModuleType("executorch.extension.pybindings.portable_lib")
    portable._unsafe_reset_threadpool = lambda n: setattr(portable, "_n", n)
    portable._threadpool_get_thread_count = lambda: getattr(portable, "_n", 0)
    for name, mod in {
        "executorch": types.ModuleType("executorch"),
        "executorch.extension": types.ModuleType("executorch.extension"),
        "executorch.extension.llm": types.ModuleType("executorch.extension.llm"),
        "executorch.extension.llm.runner": runner_mod,
        "executorch.extension.pybindings": types.ModuleType("executorch.extension.pybindings"),
        "executorch.extension.pybindings.portable_lib": portable,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(pte_llm, "_preload_kernels", lambda: ["stub.so"])
    return portable


def test_missing_executorch_raises_missing_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "executorch.extension.llm.runner", None)
    runner = pte_llm.PteLlmRunner()
    with pytest.raises(MissingDependency, match="executorch"):
        runner.load(tmp_path / "m.pte", 2, {"tokenizer": tmp_path / "tok.model"})


def test_load_needs_tokenizer_sidecar(fake_executorch, tmp_path):
    with pytest.raises(MissingDependency, match="tokenizer"):
        pte_llm.PteLlmRunner().load(tmp_path / "m.pte", 2, {})


def test_load_warms_up_outside_prompt_timing_and_sets_threads(fake_executorch, tmp_path):
    runner = pte_llm.PteLlmRunner()
    runner.load(tmp_path / "m.pte", 2, {"tokenizer": tmp_path / "tok.model"})
    fake = FakeRunner.instances[-1]
    # warmup: reset, then one token; the method load shows up as 2500 ms
    assert fake.calls == ["reset", "generate:1"]
    assert runner.load_ms == 2500.0
    assert runner.threads_set == 2 and runner.threads_reported == 2
    assert runner.kernel_libs == ["stub.so"]


def test_generate_resets_before_every_prompt_and_reports_timing(fake_executorch, tmp_path):
    runner = pte_llm.PteLlmRunner()
    runner.load(tmp_path / "m.pte", 4, {"tokenizer": tmp_path / "tok.model"})
    fake = FakeRunner.instances[-1]
    fake.calls.clear()
    text, timing = runner.generate("hi", max_new_tokens=3)
    text2, _ = runner.generate("hi again", max_new_tokens=3)
    assert text == "ABC" and text2 == "ABC"
    assert fake.calls == ["reset", "generate:3", "reset", "generate:3"]
    assert timing["ttft_ms"] == 400.0  # first token minus inference start
    assert timing["prefill_tok_s"] == pytest.approx(32 / 0.4, rel=1e-6)
    assert timing["decode_tok_s"] == pytest.approx(2 / 1.0, rel=1e-6)  # (n_gen - 1) / decode
    assert timing["new_tokens"] == 3 and timing["prompt_tokens"] == 32


def test_timing_handles_missing_stats():
    assert pte_llm._timing(None, 12.5) == {"wall_ms": 12.5, "new_tokens": None}
    assert pte_llm._model_load_ms(None) is None


class _StopRunner:
    """Streams tokens and honours stop(): tokens after a stop request are not emitted."""

    def __init__(self):
        self.stopped = False
        self.calls = []

    def reset(self):
        self.calls.append("reset")

    def stop(self):
        self.calls.append("stop")
        self.stopped = True

    def generate(self, prompt, config, token_callback, stats_callback):
        self.calls.append(f"generate:{config.max_new_tokens}")
        for tok in ["asks", " for", " OTP.", " more", " text"][: config.max_new_tokens]:
            if self.stopped:
                break
            token_callback(tok)
        stats_callback(_Stats(inf_start=0, prompt_end=10, first=5, inf_end=50, n_prompt=7, n_gen=3))


def test_generate_stop_predicate_calls_runner_stop_once_and_reports_it():
    runner = pte_llm.PteLlmRunner()
    runner.config_cls = FakeConfig
    runner.runner = _StopRunner()
    seen = []

    def stop_when(text):
        seen.append(text)
        return text.endswith(".")

    text, timing = runner.generate("hi", max_new_tokens=20, stop_when=stop_when)
    assert text == "asks for OTP."
    assert timing["stop_hit"] is True
    assert runner.runner.calls == ["reset", "generate:20", "stop"]
    assert seen[-1] == "asks for OTP."


def test_generate_without_predicate_reports_no_stop():
    runner = pte_llm.PteLlmRunner()
    runner.config_cls = FakeConfig
    runner.runner = _StopRunner()
    text, timing = runner.generate("hi", max_new_tokens=3)
    assert text == "asks for OTP."  # three tokens: the budget, not a stop
    assert timing["stop_hit"] is False
    assert "stop" not in runner.runner.calls
