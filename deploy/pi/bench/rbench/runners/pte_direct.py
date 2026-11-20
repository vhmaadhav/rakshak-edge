"""Direct ExecuTorch runner for the Llama .pte, with optional prefix reuse (no TextLLMRunner).

Why it exists: ``pte_llm.PteLlmRunner`` must ``reset()`` before every prompt, so each call
re-reads the whole explainer prompt. On the Pi (12 real explainer prompts, 3 threads) the prompt
prefill is about 1,317 ms of the 1,724 ms mean per call. The first ~61-67 tokens (the system turn)
are the same for every call. The Python ``TextLLMRunner`` cannot rewind to a position, so the
prefix cannot be reused through it. This runner drives the ``.pte`` ``forward`` method directly:
``forward(tokens, start_pos)`` writes the tokens into the KV cache at ``start_pos..`` and returns
the logits of the last token.

Split rule: the prefix ends just after the FIRST end-of-turn token (``split_id``, 128009) in the
prompt. That point depends only on the prompt, never on earlier calls, so a given prompt always
runs the same forward sequence. The prefix is kept in the KV cache and reused while the next
prompt starts with the same prefix tokens. Stale KV entries beyond the current position are
overwritten as decoding proceeds, so no cache clearing is needed inside a call.

Outputs are not guaranteed identical to the TextLLMRunner path: splitting the prefill changes the
numerics slightly. Verified on the Pi on 2026-11-11, 12 prompts, 3 threads: the plain direct loop
without reuse gave the same sentences as TextLLMRunner on 12 of 12; with prefix reuse, 11 of 12
(the one that differs is unterminated in both). Prefix reuse is therefore opt-in and evaluated.

On the 40-row eval of 2026-11-11 the reuse path gave total p50 957 ms against 1,683 ms for
TextLLMRunner, the same 27 of 33 raw point pass, and 33 of 34 identical sentences
(docs/EDGE_PIPELINE.md, "Prefix reuse").

Greedy decoding (argmax). ``load()`` builds the parts from the ExecuTorch and tokenizer libraries;
``PteDirectRunner.from_parts`` builds a runner from injected parts (used by the tests).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from rbench.errors import MissingDependency
from rbench.runners.pte_llm import INSTALL_HINT, WARMUP_PROMPT, _preload_kernels

EOT_ID = 128009  # <|eot_id|> in the Llama 3 tokenizer: the end of the system turn
DEFAULT_MAX_CONTEXT = 2048


class PteDirectRunner:
    kind = "pte_llm_direct"

    def __init__(self, reuse_prefix: bool = True) -> None:
        self.reuse_prefix = reuse_prefix
        self.load_ms: float | None = None
        self.threads_set: int | None = None
        self.threads_reported: int | None = None
        self.kernel_libs: list[str] = []
        self._forward: Callable[[list[int], int], int] | None = None
        self._encode: Callable[[str], list[int]] | None = None
        self._decode: Callable[[list[int]], str] | None = None
        self._eos: frozenset[int] = frozenset()
        self._max_context = DEFAULT_MAX_CONTEXT
        self._split_id = EOT_ID
        self._clock: Callable[[], float] = time.perf_counter
        self._prefix_ids: list[int] | None = None

    @classmethod
    def from_parts(
        cls,
        forward: Callable[[list[int], int], int],
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        eos_ids: Iterable[int],
        max_context: int = DEFAULT_MAX_CONTEXT,
        reuse_prefix: bool = True,
        clock: Callable[[], float] = time.perf_counter,
        split_id: int = EOT_ID,
    ) -> PteDirectRunner:
        runner = cls(reuse_prefix=reuse_prefix)
        runner._bind(forward, encode, decode, eos_ids, max_context, clock, split_id)
        return runner

    def _bind(
        self,
        forward: Callable[[list[int], int], int],
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        eos_ids: Iterable[int],
        max_context: int,
        clock: Callable[[], float],
        split_id: int,
    ) -> None:
        self._forward = forward
        self._encode = encode
        self._decode = decode
        self._eos = frozenset(int(x) for x in eos_ids)
        self._max_context = int(max_context)
        self._clock = clock
        self._split_id = int(split_id)
        self._prefix_ids = None

    def load(self, path: Path, threads: int, extra: dict[str, Path]) -> None:
        try:
            import numpy as np
            import torch
            from executorch.extension.pybindings import portable_lib as pl
            from pytorch_tokenizers import get_tokenizer
        except ImportError as exc:
            raise MissingDependency(INSTALL_HINT) from exc
        tokenizer = extra.get("tokenizer")
        if tokenizer is None:
            raise MissingDependency("pte_llm_direct model needs its tokenizer.model sidecar")
        self.kernel_libs = _preload_kernels()
        pl._unsafe_reset_threadpool(threads)
        self.threads_set = threads
        self.threads_reported = int(pl._threadpool_get_thread_count())
        model = pl._load_for_executorch(str(path))
        tok = get_tokenizer(str(tokenizer))
        eos = [int(x) for x in model.run_method("get_eos_ids", ())]
        max_context = int(model.run_method("get_max_context_len", ())[0])

        def forward(ids: list[int], pos: int) -> int:
            out = model.forward(
                (
                    torch.tensor([ids], dtype=torch.long),
                    torch.tensor([pos], dtype=torch.long),
                )
            )
            return int(np.argmax(out[0].numpy().reshape(-1)))

        def encode(text: str) -> list[int]:
            return [int(x) for x in tok.encode(text, bos=False, eos=False, allowed_special="all")]

        def decode(ids: list[int]) -> str:
            return tok.decode(ids)

        self._bind(forward, encode, decode, eos, max_context, time.perf_counter, EOT_ID)
        # Warmup: the first forward loads the kernels and pages in the weights. Its cost is
        # reported as load_ms, outside the prompt timings.
        started = time.perf_counter()
        self.generate(WARMUP_PROMPT, 1)
        self.load_ms = round((time.perf_counter() - started) * 1000.0, 3)

    def reset(self) -> None:
        """Forget the stored prefix: the next call re-prefills from position 0."""
        self._prefix_ids = None

    def _split_point(self, ids: list[int]) -> int:
        if not self.reuse_prefix or self._split_id not in ids:
            return 0
        p = ids.index(self._split_id) + 1
        return p if p < len(ids) else 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        stop_when: Callable[[str], bool] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Greedy generation. Returns the text and the timing block (ms, tok/s)."""
        if self._forward is None or self._encode is None or self._decode is None:
            raise RuntimeError("PteDirectRunner is not loaded")
        clock = self._clock
        t0 = clock()
        ids = list(self._encode(prompt))
        if not ids:
            raise ValueError("prompt encodes to zero tokens")
        p = self._split_point(ids)
        if len(ids) + max_new_tokens > self._max_context - 1:
            raise ValueError(
                f"prompt of {len(ids)} tokens plus {max_new_tokens} new tokens exceeds the "
                f"context of {self._max_context}"
            )
        forward = self._forward
        prefix_reused = False
        gen: list[int] = []
        text = ""
        stop_hit = False
        # Accounting (hardware-independent): every forward call, and the single-token decode
        # forwards among them. stop_reason says what ended the loop.
        forwards = 0
        decode_forwards = 0
        prefix_forward = False
        stop_reason = "zero_budget"
        try:
            if p > 0:
                if self._prefix_ids == ids[:p]:
                    prefix_reused = True
                else:
                    self._prefix_ids = None
                    forward(ids[:p], 0)
                    forwards += 1
                    prefix_forward = True
                    self._prefix_ids = ids[:p]
            else:
                self._prefix_ids = None
            nxt = forward(ids[p:], p)
            forwards += 1
            t_first = clock()
            pos = len(ids)
            if max_new_tokens > 0:
                while True:
                    if nxt in self._eos:
                        stop_reason = "eos"
                        break
                    gen.append(nxt)
                    text = self._decode(gen)
                    if stop_when is not None and stop_when(text):
                        stop_hit = True
                        stop_reason = "stop_when"
                        break
                    if len(gen) >= max_new_tokens:
                        stop_reason = "max_new_tokens"
                        break
                    nxt = forward([nxt], pos)
                    forwards += 1
                    decode_forwards += 1
                    pos += 1
        except BaseException:
            self._prefix_ids = None
            raise
        t_end = clock()
        return text, _timing(
            prompt_tokens=len(ids),
            prefix_tokens=p,
            prefix_reused=prefix_reused,
            new_tokens=len(gen),
            stop_hit=stop_hit,
            t0=t0,
            t_first=t_first,
            t_end=t_end,
            forward_calls=forwards,
            decode_forwards=decode_forwards,
            prefix_forward=prefix_forward,
            stop_reason=stop_reason,
        )


def _timing(
    *,
    prompt_tokens: int,
    prefix_tokens: int,
    prefix_reused: bool,
    new_tokens: int,
    stop_hit: bool,
    t0: float,
    t_first: float,
    t_end: float,
    forward_calls: int = 0,
    decode_forwards: int = 0,
    prefix_forward: bool = False,
    stop_reason: str = "",
) -> dict[str, Any]:
    """Timing for one generate call. Clock values are seconds (perf_counter units)."""
    reused_len = prefix_tokens if prefix_reused else 0
    prefill_tokens = prompt_tokens - reused_len
    ttft = max(t_first - t0, 0.0)
    out: dict[str, Any] = {
        "wall_ms": round(max(t_end - t0, 0.0) * 1000.0, 3),
        "ttft_ms": round(ttft * 1000.0, 3),
        "prompt_tokens": prompt_tokens,
        "prefill_tokens": prefill_tokens,
        "prefix_tokens": prefix_tokens,
        "prefix_reused": prefix_reused,
        "new_tokens": new_tokens,
        "stop_hit": stop_hit,
        "forward_calls": forward_calls,
        "decode_forwards": decode_forwards,
        "prefix_forward": prefix_forward,
        "stop_reason": stop_reason,
    }
    if prefill_tokens > 0 and ttft > 0:
        out["prefill_tok_s"] = round(prefill_tokens / ttft, 3)
    decode = t_end - t_first
    if new_tokens > 1 and decode > 0:
        out["decode_tok_s"] = round((new_tokens - 1) / decode, 3)
    return out
