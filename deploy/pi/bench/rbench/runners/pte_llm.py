"""ExecuTorch LLM runner for the Llama .pte models (Python bindings, no compilation).

Verified on the Pi on 2026-11-06 against executorch 1.5.1 (see docs/LLAMA_PI_RESULTS.md):

- ``executorch.extension.llm.runner.TextLLMRunner(model_path=, tokenizer_path=)`` builds the
  runner. The ``.pte`` method is loaded lazily, on the first ``generate`` call; that cost is
  reported by the Stats object (``model_load_end_ms - model_load_start_ms``), so ``load()``
  runs one short warmup generation to take it out of the per-prompt timing.
- ``generate(prompt, config, token_callback, stats_callback)`` streams the tokens and fills the
  stats. Python exposes no ``generate_from_pos`` and no raw logits; ``prefill()`` and
  ``reset()`` exist but are not used here.
- The pip wheel's runtime does not register the Llama custom kernels (``llama::custom_sdpa.out``,
  ``llama::fast_hadamard_transform.out``) or ``quantized_decomposed::embedding_byte.dtype_out``
  until their shipped shared libraries are loaded with RTLD_GLOBAL. ``_preload_kernels`` does
  that for the two libraries in the wheel. Nothing is compiled.
- Threads: ``portable_lib._unsafe_reset_threadpool(n)`` sets the ExecuTorch threadpool; the
  count it reports back is recorded.

Prompts are the ELA evidence packages built by the worker (src/rakshak/ela_eval.py).
"""

from __future__ import annotations

import ctypes
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rbench.errors import MissingDependency

INSTALL_HINT = (
    "the ExecuTorch Python LLM runner is not importable. Install executorch==1.5.1 and "
    "torch==2.14.0 (CPU) into ~/rakshak-bench/.venv (aarch64 Linux wheels) and rerun."
)
WARMUP_PROMPT = "Hello"


def _preload_kernels() -> list[str]:
    """Load the kernel libraries shipped in the wheel so their registrations reach the runtime."""
    import executorch.extension.llm.custom_ops as custom_ops

    base = os.path.dirname(os.path.abspath(custom_ops.__file__))
    et_root = os.path.dirname(os.path.dirname(os.path.dirname(base)))
    loaded: list[str] = []
    for path in (
        os.path.join(base, "libcustom_ops_aot_lib.so"),
        os.path.join(et_root, "lib", "libexecutorch_kernels_quantized.so"),
    ):
        if os.path.isfile(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded.append(path)
    return loaded


class PteLlmRunner:
    kind = "pte_llm"

    def __init__(self) -> None:
        self.runner: Any = None
        self.config_cls: Any = None
        self.load_ms: float | None = None
        self.threads_set: int | None = None
        self.threads_reported: int | None = None
        self.kernel_libs: list[str] = []

    def load(self, path: Path, threads: int, extra: dict[str, Path]) -> None:
        try:
            from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
            from executorch.extension.pybindings import portable_lib
        except ImportError as exc:
            raise MissingDependency(INSTALL_HINT) from exc
        tokenizer = extra.get("tokenizer")
        if tokenizer is None:
            raise MissingDependency("pte_llm model needs its tokenizer.model sidecar")
        self.kernel_libs = _preload_kernels()
        portable_lib._unsafe_reset_threadpool(threads)
        self.threads_set = threads
        self.threads_reported = int(portable_lib._threadpool_get_thread_count())
        self.config_cls = GenerationConfig
        self.runner = TextLLMRunner(model_path=str(path), tokenizer_path=str(tokenizer))
        # Warmup: the first generate loads the method. Its cost is kept out of the prompt timings.
        stats: dict[str, Any] = {}
        self._generate(WARMUP_PROMPT, 1, stats)
        self.load_ms = _model_load_ms(stats.get("raw"))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        stop_when: Callable[[str], bool] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Greedy generation. Returns the text and the timing block (ms, tok/s).

        ``stop_when`` gets the text generated so far after each token. When it returns True the
        runner is asked to stop (``stop()``). The caller must still clip the text itself: a stop
        request may land one token late. ``timing["stop_hit"]`` says whether the predicate fired.
        """
        stats: dict[str, Any] = {}
        text = self._generate(prompt, max_new_tokens, stats, stop_when)
        timing = _timing(stats.get("raw"), stats.get("wall_ms", 0.0))
        timing["stop_hit"] = bool(stats.get("stop_hit", False))
        return text, timing

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        stats: dict[str, Any],
        stop_when: Callable[[str], bool] | None = None,
    ) -> str:
        pieces: list[str] = []
        config = self.config_cls(
            echo=False, max_new_tokens=max_new_tokens, temperature=0.0, num_bos=0, num_eos=0
        )

        def on_token(piece: str) -> None:
            pieces.append(piece)
            if stop_when is not None and not stats.get("stop_hit") and stop_when("".join(pieces)):
                stats["stop_hit"] = True
                self.runner.stop()

        def on_stats(obj: Any) -> None:
            # The object is reused by the runner; copy the numbers we need now.
            stats["raw"] = _copy_stats(obj)

        # Each prompt is independent: clear the KV cache and position first. Without this the
        # runner keeps the previous prompts' positions and overflows max_context_len after ~6.
        self.runner.reset()
        started = time.perf_counter()
        self.runner.generate(prompt, config, on_token, on_stats)
        stats["wall_ms"] = (time.perf_counter() - started) * 1000.0
        return "".join(pieces)


class _StatsCopy:
    """Plain copy of the fields of one Stats object (the runner reuses its own)."""

    def __init__(self, obj: Any) -> None:
        for name in (
            "model_load_start_ms",
            "model_load_end_ms",
            "inference_start_ms",
            "prompt_eval_end_ms",
            "first_token_ms",
            "inference_end_ms",
            "num_prompt_tokens",
            "num_generated_tokens",
        ):
            setattr(self, name, getattr(obj, name, None))


def _copy_stats(obj: Any) -> _StatsCopy:
    return _StatsCopy(obj)


def _model_load_ms(raw: Any) -> float | None:
    if raw is None or raw.model_load_start_ms is None or raw.model_load_end_ms is None:
        return None
    value = raw.model_load_end_ms - raw.model_load_start_ms
    return round(float(value), 3) if value > 0 else None


def _timing(raw: Any, wall_ms: float) -> dict[str, Any]:
    """Per-prompt timing from one generate call. Times in the Stats object are milliseconds.

    ttft_ms: inference start to first token (includes the prompt prefill).
    prefill_tok_s: prompt tokens over the prefill window.
    decode_tok_s: tokens after the first, over the time from the first token to the end.
    """
    out: dict[str, Any] = {"wall_ms": round(wall_ms, 3), "new_tokens": None}
    if raw is None:
        return out
    inf_start = raw.inference_start_ms
    prompt_end = raw.prompt_eval_end_ms
    inf_end = raw.inference_end_ms
    first = raw.first_token_ms
    n_prompt = raw.num_prompt_tokens
    n_gen = raw.num_generated_tokens
    out["new_tokens"] = n_gen
    out["prompt_tokens"] = n_prompt
    if first and inf_start and first >= inf_start:
        out["ttft_ms"] = round(float(first - inf_start), 3)
    if n_prompt and prompt_end and inf_start and prompt_end > inf_start:
        out["prefill_tok_s"] = round(n_prompt / ((prompt_end - inf_start) / 1000.0), 3)
    if n_gen and n_gen > 1 and first and inf_end and inf_end > first:
        out["decode_tok_s"] = round((n_gen - 1) / ((inf_end - first) / 1000.0), 3)
    return out
