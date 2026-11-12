"""ELA "slot" mode for the on-device explainer (rbench ``--ela-mode slot``).

The code writes the whole advisory JSON: ``lang``, the template ``summary`` and one point per
evidence item. The explainer writes one short sentence per item, from that item's fact alone,
as a completion of the assistant prefix ``This message``. The model never writes JSON.

Each sentence is cleaned (cut at its first sentence end; empty, unterminated or newline/quote/
brace-bearing output is rejected) and then checked by the same verifier as points mode
(``check_points``, via ``score_advisory``). A point that is rejected at any step is replaced by
the deterministic template sentence (``render_advisory``), so the advisory a user sees always
passes the verifier. Raw verifier numbers describe the model's sentences before that fallback.

Mode ``slot_kw`` (``keyword_prefix=True``) is the same contract with one change: the assistant
prefix already holds the evidence type's verifier keyword, for example "This message uses time
pressure because it", and the model completes the clause. The prefix comes from the type's
``TYPE_PHRASE`` (its parenthetical examples dropped), so it states only the rule's own fact. The
stop rules, the terminator check, the verifier and the template fallback are unchanged.

Pure functions only: the runner is passed in, so the laptop tests use a mock.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any

from rakshak.ela_eval import score_advisory
from rakshak.evidence import EvidencePackage
from rakshak.prompts import BOS, EOT, EOT_HDR, SOT, fact_slot
from rakshak.verifier import (
    _SAFE_RE,
    TYPE_PHRASE,
    _positive_matches,
    _template_summary,
    check_points,
    render_advisory,
    skip_llm_for_risk,
    template_only_advisory,
)

SLOT_SYSTEM = (
    "You explain why a text message may be a scam. You are given one fact about the message. "
    "Write one short plain-language sentence in English about that fact. Use only the fact. "
    "Do not add numbers, links or instructions. Output only the sentence."
)
SLOT_PREFIX = "This message"  # assistant prefix; generation continues it
SLOT_MAX_NEW_TOKENS = 20  # per point; the stop predicate usually ends the call earlier
SLOT_LANG = "en"
# Fix A (edge pipeline only, docs/EDGE_PIPELINE.md "Fixes A and B"). The model describes what the
# message says or asks for and does not judge it. The default prompts above are unchanged; a
# variant is chosen by name (``generate_slot_advisory(..., variant=)``).
DESCRIBE_SYSTEM = (
    "You describe what a text message says or asks the reader to do. You are given one fact "
    "about the message. Write one short plain-English sentence that restates the fact. Do not "
    "judge the message or its sender. Use only the fact. Output only the sentence."
)
PLAIN_SYSTEM = (
    "Restate the fact you are given in one short plain-English sentence. Do not judge the "
    "message. Use only the fact. Output only the sentence."
)
HINGLISH_LANG = "hinglish"
# Hinglish explainer (edge service, --hinglish-mode llm; docs/EDGE_PIPELINE.md "Hinglish"). The
# instructions are English; the sentence is Hinglish in Roman letters. Describe, do not judge.
HINGLISH_SYSTEM = (
    "You describe what a text message says or asks the reader to do. You are given one fact "
    "about the message. Write one short sentence in Hinglish, which is Hindi written in Roman "
    "letters mixed with English words, that restates the fact. Use only Roman letters, not "
    "Devanagari. Do not judge the message or its sender. Use only the fact. Output only the "
    "sentence."
)
HINGLISH_PREFIX = "Yeh message"  # assistant prefix for the Hinglish explainer
# The lead of a clean point, by language. strip_echo and the echo tests use it.
POINT_LEAD = {"en": "This message ", "hinglish": "Yeh message "}
PROMPT_VARIANTS: dict[str, dict[str, Any]] = {
    # prefix "This message"; describe system text
    "describe": {"system": DESCRIBE_SYSTEM, "keyword": False},
    # prefix "This message <type clause> and" (the keyword is in the prefix); describe system text
    "describe_kw": {"system": DESCRIBE_SYSTEM, "keyword": True},
    # prefix "This message"; shorter restate-only system text
    "describe_plain": {"system": PLAIN_SYSTEM, "keyword": False},
}
# slot_kw: a shorter system text, because the keyword prefix adds tokens to the same prompt budget.
SLOT_KW_SYSTEM = (
    "You explain why a text message may be a scam, using only the fact you are given. "
    "Finish the sentence you are given in plain English. Do not add numbers, links or instructions."
)
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
# Fix C (edge pipeline, cleanup=True): the prompt's own words echoed at the end of a sentence,
# e.g. "... and restates the fact." or "... and has been matched." Only a clause at the end is
# matched, and only these phrases.
_ECHO_RE = re.compile(
    r"(?i)\s*,?\s*(?:and\s+|which\s+)?"
    r"(?:restates?\s+the\s+fact|(?:has|have|is|was)\s+been\s+matched|(?:is|was)\s+matched)"
    r"\s*([.!?])\s*$"
)
# A stripped remainder that ends on one of these words is not a complete sentence.
_DANGLING = frozenset(
    "and or but because to the a an of with that which is are was has have been by for in on "
    "at it this be from as".split()
)
_DUP_RATIO = 0.9  # normalised similarity at or above this counts as a duplicate point
_STOP_CHARS = ("\n", '"', "{", "}")
_TERMINATOR = re.compile(r"[.!?](?=\s|$)")


def fact_text(evidence, lang: str = SLOT_LANG) -> str:
    """One fact in the same wording as ``rakshak.prompts.facts_block``, for a single item."""
    phrase = TYPE_PHRASE[lang][evidence.type]
    phrase = phrase[:1].upper() + phrase[1:]
    slot = fact_slot(evidence)
    return f'{phrase}. Matched text: "{slot}"' if slot else f"{phrase}."


def _chat(system: str, user: str, prefix: str) -> str:
    return (
        f"{BOS}"
        f"{SOT}system{EOT_HDR}\n\n{system}{EOT}"
        f"{SOT}user{EOT_HDR}\n\n{user}{EOT}"
        f"{SOT}assistant{EOT_HDR}\n\n{prefix}"
    )


def slot_prompt(evidence, lang: str = SLOT_LANG) -> str:
    """Compact Llama 3 chat prompt for one item: minimal system text, one fact, and the assistant
    prefix. It ends inside the assistant turn, after SLOT_PREFIX."""
    if lang != SLOT_LANG:
        raise ValueError("slot mode is English only")
    user = f"FACT: {fact_text(evidence, lang)}\n\nWrite the sentence for this fact."
    return _chat(SLOT_SYSTEM, user, SLOT_PREFIX)


def keyword_clause(evidence, lang: str = SLOT_LANG) -> str:
    """The evidence type's TYPE_PHRASE without its parenthetical examples, e.g. "uses time
    pressure" for urgency. It keeps the verifier keyword and no number or link."""
    if lang != SLOT_LANG:
        raise ValueError("slot mode is English only")
    return " ".join(_PAREN_RE.sub("", TYPE_PHRASE[lang][evidence.type]).split())


def kw_prefix(evidence, lang: str = SLOT_LANG) -> str:
    """slot_kw assistant prefix for one item, e.g. "This message uses time pressure because it"."""
    return f"{SLOT_PREFIX} {keyword_clause(evidence, lang)} because it"


def variant_prefix(evidence, variant: str, lang: str = SLOT_LANG) -> str:
    """Assistant prefix for a PROMPT_VARIANTS entry (Fix A)."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    if not PROMPT_VARIANTS[variant]["keyword"]:
        return SLOT_PREFIX
    return f"{SLOT_PREFIX} {keyword_clause(evidence, lang)} and"


def variant_prompt(evidence, variant: str, lang: str = SLOT_LANG) -> tuple[str, str]:
    """(prefix, prompt) for one item under a PROMPT_VARIANTS entry. The prompt ends inside the
    assistant turn, after the prefix."""
    if lang != SLOT_LANG:
        raise ValueError("slot mode is English only")
    prefix = variant_prefix(evidence, variant, lang)
    user = f"FACT: {fact_text(evidence, lang)}\n\nWrite the sentence for this fact."
    return prefix, _chat(PROMPT_VARIANTS[variant]["system"], user, prefix)


def slot_kw_prompt(evidence, lang: str = SLOT_LANG) -> str:
    """slot_kw prompt: the short system text, one fact, and the keyword prefix. It ends inside the
    assistant turn, after kw_prefix(evidence)."""
    if lang != SLOT_LANG:
        raise ValueError("slot mode is English only")
    user = f"FACT: {fact_text(evidence, lang)}\n\nFinish the sentence."
    return _chat(SLOT_KW_SYSTEM, user, kw_prefix(evidence, lang))


def hinglish_prompt(evidence) -> tuple[str, str]:
    """(prefix, prompt) for one item in Hinglish: the English system text above, the fact in the
    Hinglish TYPE_PHRASE, and the assistant prefix "Yeh message". The prompt ends inside the
    assistant turn, after the prefix."""
    user = f"FACT: {fact_text(evidence, HINGLISH_LANG)}\n\nWrite the sentence for this fact."
    return HINGLISH_PREFIX, _chat(HINGLISH_SYSTEM, user, HINGLISH_PREFIX)


def slot_stop(generated: str) -> bool:
    """Stop once the continuation holds a sentence end or a character that cannot be in a point."""
    return bool(_TERMINATOR.search(generated)) or any(c in generated for c in _STOP_CHARS)


def clean_sentence(generated: str, prefix: str = SLOT_PREFIX) -> tuple[str, str]:
    """(point text, status) for one continuation of ``prefix`` (SLOT_PREFIX by default).

    status is "ok", "empty" (nothing generated), or "unterminated" (no sentence end before a
    newline, quote, brace, or the token budget ran out). Only "ok" gives text; the rest give "".
    """
    if not generated.strip():
        return "", "empty"
    term = _TERMINATOR.search(generated)
    cuts = [generated.find(c) for c in _STOP_CHARS if c in generated]
    cut = min(cuts) if cuts else None
    if term is None or (cut is not None and cut < term.end()):
        return "", "unterminated"
    sentence = " ".join(f"{prefix}{generated[: term.end()]}".split())
    return sentence, "ok"


def strip_echo(sentence: str, lead: str = POINT_LEAD["en"]) -> tuple[str, bool]:
    """Fix C: remove a trailing prompt echo from one point sentence.

    Returns (sentence, changed). The echo clause is removed only when what remains starts with
    ``lead`` ("This message " for English), has at least four words and does not end on a
    function word. Otherwise the result is "" (changed True), so the caller falls back to the
    template for that point.
    """
    m = _ECHO_RE.search(sentence)
    if m is None:
        return sentence, False
    remainder = sentence[: m.start()].rstrip()
    words = remainder.split()
    last = words[-1].lower().strip(".,;:!?\"'") if words else ""
    if not remainder.startswith(lead) or len(words) < 4 or last in _DANGLING:
        return "", True
    return f"{remainder}{m.group(1)}", True


def _norm_point(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def duplicate_points(points: list[str]) -> list[int | None]:
    """Fix C: for each point, the index of the earlier point it duplicates (None if it is kept).

    Two points are duplicates when their normalised text is equal or their similarity ratio is at
    least ``_DUP_RATIO``. The first occurrence is kept. Empty points are never duplicates.
    """
    kept: list[tuple[int, str]] = []
    out: list[int | None] = []
    for i, text in enumerate(points):
        norm = _norm_point(text)
        dup_of = None
        if norm:
            for j, prev in kept:
                if norm == prev or difflib.SequenceMatcher(None, norm, prev).ratio() >= _DUP_RATIO:
                    dup_of = j
                    break
            if dup_of is None:
                kept.append((i, norm))
        out.append(dup_of)
    return out


def generate_slot_advisory(
    runner: Any,
    package: EvidencePackage,
    lang: str = SLOT_LANG,
    max_new_tokens: int = SLOT_MAX_NEW_TOKENS,
    keyword_prefix: bool = False,
    guards: bool = False,
    variant: str | None = None,
    cleanup: bool = False,
) -> dict[str, Any]:
    """One advisory in slot mode (or slot_kw when ``keyword_prefix``): one generate call per
    evidence item, then assembly and checks.

    ``runner.generate(prompt, max_new_tokens=..., stop_when=...)`` must return ``(text, timing)``.
    Returns the raw and final JSON texts, the per-call records, per-point statuses and sources,
    the score dict (``score_advisory`` on the raw output, with ``safe_claim`` restricted to the
    model's own sentences), and whether the final output passes ``check_points``.

    ``guards=True`` turns on the edge guards (verifier.py): G1 skips every call for a LOW-risk
    package and returns the template advisory with ``skipped=True``; G2 passes the package risk
    to the verifier, and the sentences it rejects are listed in ``g2_rejected``.

    ``variant`` (a PROMPT_VARIANTS name, Fix A) replaces the default slot prompt and prefix. It
    cannot be combined with ``keyword_prefix``. None keeps the default slot prompt.

    ``cleanup=True`` (Fix C, edge pipeline) runs two deterministic steps on the model's sentences
    before assembly: ``strip_echo`` removes a trailing prompt echo, and a later point that
    duplicates an earlier one is blanked. A blanked or invalid point becomes "" and so falls back
    to the template in ``render_advisory``. The verifier still checks every final point. The
    per-point record is returned as ``cleanup`` (``original`` is the model's sentence).
    """
    if variant is not None and keyword_prefix:
        raise ValueError("variant and keyword_prefix are separate prompt modes")
    if lang == HINGLISH_LANG and (variant is not None or keyword_prefix):
        raise ValueError("hinglish has its own prompt; variant and keyword_prefix are English only")
    if lang not in POINT_LEAD:
        raise ValueError(f"lang must be 'en' or 'hinglish', got {lang!r}")
    n = len(package.evidence)
    if guards and skip_llm_for_risk(package):
        final_output = json.dumps(template_only_advisory(package, lang), ensure_ascii=False)
        return {
            "raw_output": "",
            "output": final_output,
            "adv": None,
            "calls": [],
            "mode": "template",
            "skipped": True,
            "prefixes": [],
            "point_checks": [],
            "point_problems": [[] for _ in range(n)],
            "point_status": ["skipped"] * n,
            "point_sources": ["template"] * n,
            "g2_rejected": [],
            "cleanup": [],
            "final_verifier_ok": check_points(final_output, package, lang).ok,
        }
    risk = package.risk if guards else None
    calls: list[dict[str, Any]] = []
    statuses: list[str] = []
    raw_points: list[str] = []
    prefixes: list[str] = []
    for evidence in package.evidence:
        if lang == HINGLISH_LANG:
            prefix, prompt = hinglish_prompt(evidence)
        elif variant is not None:
            prefix, prompt = variant_prompt(evidence, variant, lang)
        elif keyword_prefix:
            prefix, prompt = kw_prefix(evidence, lang), slot_kw_prompt(evidence, lang)
        else:
            prefix, prompt = SLOT_PREFIX, slot_prompt(evidence, lang)
        generated, timing = runner.generate(
            prompt, max_new_tokens=max_new_tokens, stop_when=slot_stop
        )
        sentence, status = clean_sentence(generated, prefix)
        statuses.append(status)
        raw_points.append(sentence)
        prefixes.append(prefix)
        calls.append(
            {
                "evidence_id": evidence.id,
                "prefix": prefix,
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
                "generated": generated,
                "status": status,
                "cached": bool(timing.get("cached", False)),
            }
        )
    cleanup_rec: list[dict[str, Any]] | None = None
    if cleanup:
        cleanup_rec = [
            {"original": s, "echo_stripped": False, "duplicate_of": None} for s in raw_points
        ]
        for i, sentence in enumerate(raw_points):
            if not sentence:
                continue
            new_sentence, changed = strip_echo(sentence, POINT_LEAD[lang])
            if changed:
                cleanup_rec[i]["echo_stripped"] = True
                if not new_sentence:
                    statuses[i] = "echo_invalid"
            raw_points[i] = new_sentence
        for i, j in enumerate(duplicate_points(raw_points)):
            if j is not None:
                cleanup_rec[i]["duplicate_of"] = j
                raw_points[i] = ""
                statuses[i] = "duplicate"
    raw_output = json.dumps(assemble(package, raw_points, lang), ensure_ascii=False)
    adv = score_advisory(raw_output, package, lang, risk)
    # The template summary can say "does not make the message safe" (LOW risk). The reassurance
    # metric counts only the model's own sentences, not the template.
    adv["safe_claim"] = any(bool(_positive_matches(_SAFE_RE, p)) for p in raw_points)
    adv["rendered_safe_claim"] = adv["safe_claim"]
    rendered = render_advisory(raw_output, package, lang, risk)
    final = {
        "lang": lang,
        "summary": rendered["summary"],
        "points": [p["text"] for p in rendered["points"]],
    }
    final_output = json.dumps(final, ensure_ascii=False)
    # Raw verdict per point (the model's sentence or "" before fallback), for per-type rates.
    raw_verdict = check_points(raw_output, package, lang, risk)
    point_checks = [[c.type, c.ok] for c in raw_verdict.points]
    point_problems = [list(c.problems) for c in raw_verdict.points]
    g2_rejected = [
        {"index": c.index, "type": c.type, "sentence": raw_points[c.index], "problem": p}
        for c in raw_verdict.points
        for p in c.problems
        if p.startswith("attribution of scam intent")
    ]
    return {
        "raw_output": raw_output,
        "output": final_output,
        "adv": adv,
        "calls": calls,
        "mode": (
            HINGLISH_LANG
            if lang == HINGLISH_LANG
            else variant
            if variant is not None
            else ("slot_kw" if keyword_prefix else "slot")
        ),
        "skipped": False,
        "prefixes": prefixes,
        "point_checks": point_checks,
        "point_problems": point_problems,
        "point_status": statuses,
        "point_sources": [p["source"] for p in rendered["points"]],
        "g2_rejected": g2_rejected,
        "cleanup": cleanup_rec or [],
        # The final check runs without G2: its summary is the template, which is not LLM text.
        "final_verifier_ok": check_points(final_output, package, lang).ok,
    }


def assemble(package: EvidencePackage, points: list[str], lang: str = SLOT_LANG) -> dict[str, Any]:
    """The advisory object in the points contract. The summary is always the template."""
    return {"lang": lang, "summary": _template_summary(package, lang), "points": list(points)}
