"""Explainer speed candidates and hardware-independent accounting (docs/EXPLAINER_SPEED_PLAN.md).

``run_explainer`` is the one entry point the edge pipeline calls for an English advisory. The
``baseline`` candidate is exactly the shipped call (rbench.slot.generate_slot_advisory with the
describe_kw variant, the default 20-token cap, one call per collapsed evidence type). The other
candidates change only what the plan names: the length rule, the cap, the trimmed user turn, or
one batched call per advisory. Every candidate ends in the same cleanup, verifier and template
fallback (slot._finish_points), so the output contract is the same for all of them.

``AccountingGenerator`` implements the Generator protocol (``generate(prompt, max_new_tokens,
stop_when) -> (text, timing)``) and logs one record per call. ``summarise_accounting`` turns a log
into per-advisory counts: prompt tokens, reused and newly prefilled tokens, decode tokens, forward
calls and stop reasons. These counts do not depend on the machine.

The steady-state counts assume the system-turn prefix is in the KV cache (what the service sees
after the first advisory). ``prefill_steady`` is ``prompt_tokens - prefix_tokens`` for each call;
``prefill_observed`` is what the runner actually processed in this run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rbench.slot import (
    _TERMINATOR,
    DESCRIBE_SYSTEM,
    SLOT_MAX_NEW_TOKENS,
    _chat,
    _finish_points,
    clean_sentence,
    fact_text,
    generate_slot_advisory,
)

from rakshak.evidence import EvidencePackage
from rakshak.verifier import skip_llm_for_risk

BASELINE = "baseline"
# Short-sentence rule, the same wording as slot.DESCRIBE_SHORT_SYSTEM, for the batched candidate.
BATCH_SYSTEM = (
    "You describe what a text message says or asks the reader to do. You are given numbered facts "
    "about the message. For each fact, write one short plain-English sentence that restates it. "
    "Keep each sentence under 15 words. Do not judge the message or its sender. Use only the "
    "facts. Output only the numbered sentences, one per line."
)
BATCH_PREFIX = "1. This message"
BATCH_USER_TAIL = "Write one sentence for each fact, numbered to match."
# Per-point token budget for the batched call (the batch cap is this times the number of points).
BATCH_CAP_PER_POINT = SLOT_MAX_NEW_TOKENS
_LINE_RE = re.compile(r"^\s*(\d+)[.)]\s*(.*)$")


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    kind: str  # "per_item" (generate_slot_advisory) or "batch" (one call per advisory)
    variant: str | None = None  # rbench.slot.PROMPT_VARIANTS name for per_item candidates
    # Per-item cap in tokens, or None for SLOT_MAX_NEW_TOKENS. Tight caps are passed explicitly
    # by the harness (run_explainer(cap=...)) because they come from the baseline distribution.
    cap: int | None = None


CANDIDATES: dict[str, Candidate] = {
    BASELINE: Candidate(
        BASELINE,
        "shipped: describe_kw prompt, one call per collapsed evidence type, cap 20",
        "per_item",
        variant="describe_kw",
    ),
    "c1_short": Candidate(
        "c1_short",
        "C1 length rule in the system turn (reused prefix), cap 20 (prompt effect only)",
        "per_item",
        variant="describe_kw_short",
    ),
    "c1_tight": Candidate(
        "c1_tight",
        "C1 with the cap set to the baseline p95 of terminated outputs",
        "per_item",
        variant="describe_kw_short",
    ),
    "c2_batch": Candidate(
        "c2_batch",
        "C2 one call per advisory, numbered facts and sentences, cap 20 per point",
        "batch",
    ),
    "c3_trim": Candidate(
        "c3_trim",
        "C3 = C1 tight cap, with the trailing 'Write the sentence' line removed from the user turn",
        "per_item",
        variant="describe_kw_trim",
    ),
}


def candidate_names() -> list[str]:
    return list(CANDIDATES)


def run_explainer(
    gen: Any,
    package: EvidencePackage,
    lang: str = "en",
    *,
    candidate: str = BASELINE,
    cap: int | None = None,
    guards: bool = True,
    cleanup: bool = True,
) -> dict[str, Any]:
    """One advisory through the named candidate. Returns the generate_slot_advisory dict shape.

    ``cap`` overrides the candidate's own cap (used for the tight-cap candidates). For the
    baseline with ``cap=None`` the call is exactly the shipped one.
    """
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown explainer candidate {candidate!r}; one of {candidate_names()}")
    spec = CANDIDATES[candidate]
    effective_cap = cap if cap is not None else (spec.cap or SLOT_MAX_NEW_TOKENS)
    if spec.kind == "per_item":
        return generate_slot_advisory(
            gen,
            package,
            lang=lang,
            max_new_tokens=effective_cap,
            guards=guards,
            variant=spec.variant,
            cleanup=cleanup,
        )
    if lang != "en":
        raise ValueError("batched candidates are English only")
    return generate_batch_advisory(
        gen,
        package,
        max_new_tokens_per_point=(cap if cap is not None else BATCH_CAP_PER_POINT),
        guards=guards,
        cleanup=cleanup,
    )


def batch_prompt(package: EvidencePackage) -> str:
    """The batched chat prompt for all evidence items of ``package`` (one fact per line)."""
    facts = "\n".join(f"{i}. {fact_text(e)}" for i, e in enumerate(package.evidence, 1))
    user = f"FACTS:\n{facts}\n\n{BATCH_USER_TAIL}"
    return _chat(BATCH_SYSTEM, user, BATCH_PREFIX)


def batch_stop(n_points: int) -> Callable[[str], bool]:
    """Stop once the continuation holds a sentence end for each point.

    The predicate is built per call and is not cached: rbench.explainer_cache keys on the
    function's module and qualname, so do not wrap this batch generator in CachedGenerator.
    """

    def stop(generated: str) -> bool:
        return len(_TERMINATOR.findall(generated)) >= n_points

    return stop


def parse_batch(generated: str, n_points: int) -> tuple[list[str], list[str]]:
    """(raw_points, statuses) for a batched continuation of BATCH_PREFIX.

    Each line "k. sentence" fills point k (first occurrence wins). A point is "ok" when its line
    has a sentence end, "unterminated" when the line has none, and "missing" when no line gives
    its number. Only "ok" points carry text; the rest are "" and fall back to the template.
    """
    full = f"{BATCH_PREFIX}{generated}"
    points = [""] * n_points
    statuses = ["missing"] * n_points
    for line in full.split("\n"):
        m = _LINE_RE.match(line)
        if m is None:
            continue
        k = int(m.group(1))
        if not 1 <= k <= n_points or statuses[k - 1] != "missing":
            continue
        body = m.group(2)
        sentence, status = clean_sentence(body, "")
        if status == "ok":
            points[k - 1] = " ".join(sentence.split())
        statuses[k - 1] = status
    return points, statuses


def generate_batch_advisory(
    gen: Any,
    package: EvidencePackage,
    *,
    max_new_tokens_per_point: int = BATCH_CAP_PER_POINT,
    guards: bool = True,
    cleanup: bool = True,
) -> dict[str, Any]:
    """One call for the whole advisory (candidate C2). Same return shape as generate_slot_advisory."""
    n = len(package.evidence)
    if guards and skip_llm_for_risk(package):
        return generate_slot_advisory_skip(gen, package, guards)
    risk = package.risk if guards else None
    prompt = batch_prompt(package)
    cap = max_new_tokens_per_point * n
    generated, timing = gen.generate(prompt, max_new_tokens=cap, stop_when=batch_stop(n))
    raw_points, statuses = parse_batch(generated, n)
    call = {
        "evidence_id": package.evidence[0].id if n else None,
        "evidence_ids": [e.id for e in package.evidence],
        "prefix": BATCH_PREFIX,
        "prompt": prompt,
        "prompt_tokens": timing.get("prompt_tokens"),
        "new_tokens": timing.get("new_tokens"),
        "ttft_ms": timing.get("ttft_ms"),
        "prefill_tok_s": timing.get("prefill_tok_s"),
        "prefill_tokens": timing.get("prefill_tokens"),
        "prefix_reused": timing.get("prefix_reused"),
        "decode_tok_s": timing.get("decode_tok_s"),
        "wall_ms": timing.get("wall_ms"),
        "stop_hit": timing.get("stop_hit"),
        "forward_calls": timing.get("forward_calls"),
        "decode_forwards": timing.get("decode_forwards"),
        "prefix_forward": timing.get("prefix_forward"),
        "prefix_tokens": timing.get("prefix_tokens"),
        "stop_reason": timing.get("stop_reason"),
        "generated": generated,
        "status": "ok" if all(s == "ok" for s in statuses) else "partial",
        "point_statuses": statuses,
        "cached": bool(timing.get("cached", False)),
    }
    return _finish_points(
        package,
        "en",
        risk,
        raw_points,
        list(statuses),
        [call],
        [BATCH_PREFIX] * n,
        mode="batch",
        cleanup=cleanup,
    )


def generate_slot_advisory_skip(gen: Any, package: EvidencePackage, guards: bool) -> dict[str, Any]:
    """The G1 skip result, the same dict generate_slot_advisory returns for a LOW-risk package.

    Calling generate_slot_advisory with guards=True does the skip without any generator call.
    """
    return generate_slot_advisory(gen, package, lang="en", guards=guards)


class AccountingGenerator:
    """Generator protocol wrapper that logs one record per call (the runner's timing and the text).

    The inner generator is called unchanged; the wrapper only records. ``take()`` returns the log
    since the last take and clears it, so the caller can take one log per advisory.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._log: list[dict[str, Any]] = []

    def generate(self, prompt: str, max_new_tokens: int = 64, stop_when=None):
        text, timing = self.inner.generate(prompt, max_new_tokens=max_new_tokens, stop_when=stop_when)
        self._log.append(
            {
                "prompt_tokens": timing.get("prompt_tokens"),
                "prefix_tokens": timing.get("prefix_tokens"),
                "prefix_reused": timing.get("prefix_reused"),
                "prefill_tokens": timing.get("prefill_tokens"),
                "new_tokens": timing.get("new_tokens"),
                "decode_forwards": timing.get("decode_forwards"),
                "forward_calls": timing.get("forward_calls"),
                "prefix_forward": timing.get("prefix_forward"),
                "stop_reason": timing.get("stop_reason"),
                "stop_hit": timing.get("stop_hit"),
                "max_new_tokens": int(max_new_tokens),
                "wall_ms": timing.get("wall_ms"),
                "cached": bool(timing.get("cached", False)),
            }
        )
        return text, timing

    def take(self) -> list[dict[str, Any]]:
        log, self._log = self._log, []
        return log

    def __getattr__(self, name: str) -> Any:
        if name in ("inner", "_log"):
            raise AttributeError(name)
        return getattr(self.inner, name)


def summarise_accounting(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-advisory counts from an AccountingGenerator log (one advisory = one take()).

    Missing fields fall back to what the protocol implies: decode forwards are new_tokens - 1
    (at least 0), forward calls are the suffix forward plus the decode forwards, and a prefix
    forward is counted only when the timing says it ran. Hardware-independent by construction.
    """
    prompt = prefix = prefill_obs = prefill_steady = decode = forwards = steady_forwards = 0
    output = reused = 0
    stops: dict[str, int] = {}
    for rec in log:
        p = int(rec.get("prompt_tokens") or 0)
        pre = int(rec.get("prefix_tokens") or 0)
        new = int(rec.get("new_tokens") or 0)
        prompt += p
        prefix += pre
        output += new
        obs = rec.get("prefill_tokens")
        prefill_obs += int(obs) if obs is not None else p
        prefill_steady += max(p - pre, 0)
        reused += pre if rec.get("prefix_reused") else 0
        dec = rec.get("decode_forwards")
        dec = int(dec) if dec is not None else max(new - 1, 0)
        decode += dec
        suffix_and_decode = 1 + dec
        fwd = rec.get("forward_calls")
        fwd = int(fwd) if fwd is not None else suffix_and_decode + (1 if pre and not rec.get("prefix_reused") else 0)
        forwards += fwd
        steady_forwards += fwd - (1 if rec.get("prefix_forward") else 0)
        reason = str(rec.get("stop_reason") or ("stop_when" if rec.get("stop_hit") else "unknown"))
        stops[reason] = stops.get(reason, 0) + 1
    return {
        "calls": len(log),
        "prompt_tokens": prompt,
        "prefix_tokens": prefix,
        "reused_prefix_tokens": reused,
        "prefill_observed": prefill_obs,
        "prefill_steady": prefill_steady,
        "decode_tokens": decode,
        "output_tokens": output,
        "tokens_processed_steady": prefill_steady + decode,
        "forward_calls": forwards,
        "forward_calls_steady": steady_forwards,
        "stop_reasons": stops,
    }


def accounting_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, sort_keys=True)


def point_status_counts(statuses: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in statuses:
        out[s] = out.get(s, 0) + 1
    return out


__all__ = [
    "BASELINE",
    "BATCH_PREFIX",
    "CANDIDATES",
    "AccountingGenerator",
    "Candidate",
    "batch_prompt",
    "batch_stop",
    "candidate_names",
    "generate_batch_advisory",
    "parse_batch",
    "point_status_counts",
    "run_explainer",
    "summarise_accounting",
]
