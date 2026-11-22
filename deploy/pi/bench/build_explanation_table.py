#!/usr/bin/env python3
"""Build the explanation table (verdict-first, step 2) by running the explainer offline.

The explainer conditions on each point's fact. The fact's type set is finite; its slot (the
matched text from the message) is not. So each sentence is generated once per (language, prompt
variant, type, has-slot) with the placeholder ``[[SLOT]]`` where the slot goes, using the same
runner, system text, prefix, stop rule and cleanup as the live explainer (rbench.slot). Llama
writes every sentence. Nothing is written by hand.

A sentence is kept for a risk only when its filled form passes ``check_points`` at that risk
(G2 included) for every sample message of its type. Samples are the rule-fixture messages
(tests/fixtures/rule_cases.json) that fire the type. A type with no sample, or a sentence that
fails any of them, gets no entry for that risk; at run time its point falls back to the template.

Usage, on the Pi (the Llama .pte is loaded here, not on the laptop):

    deploy/pi/bench/build_explanation_table.py --root ~/rakshak-bench --out PATH

Tests use a stub runner (tests/test_explanation_table.py), so the laptop never loads the model.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SERVICE = REPO / "deploy" / "pi" / "service"
sys.path[:0] = [str(REPO / "src"), str(HERE), str(SERVICE), str(REPO / "deploy" / "pi")]

from explanation_table import (  # noqa: E402
    PLACEHOLDER,
    TABLE_RISKS,
    TABLE_VERSION,
    fill,
    key,
)
from rbench.slot import (  # noqa: E402
    _PAREN_RE,
    HINGLISH_PREFIX,
    HINGLISH_SYSTEM,
    POINT_LEAD,
    PROMPT_VARIANTS,
    SLOT_PREFIX,
    _chat,
    clean_sentence,
    slot_stop,
    strip_echo,
)

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import Evidence, Risk, build_package  # noqa: E402
from rakshak.prompts import fact_slot  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402
from rakshak.url_model import URL_MODEL_TYPE, url_model_text  # noqa: E402
from rakshak.verifier import TYPE_PHRASE, _template_summary, check_points  # noqa: E402

LANGS = ("en", "hinglish")
EN_VARIANT = "describe_kw"  # the live explainer's variant (edge_pipeline.EDGE_PROMPT_VARIANT)
HINGLISH_VARIANT = "hinglish"
FIXTURE = REPO / "tests" / "fixtures" / "rule_cases.json"
DEFAULT_OUT = SERVICE / "explanation_table.json"
DEFAULT_MODEL_ID = "llama32_1b_spinquant"
PTE_PATH = "models/llama32_1b_spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
TOKENIZER_PATH = "models/llama32_1b_spinquant/tokenizer.model"


def variant_for(lang: str) -> str:
    return EN_VARIANT if lang == "en" else HINGLISH_VARIANT


def _phrase(etype: str, lang: str) -> str:
    phrase = TYPE_PHRASE[lang][etype]
    return phrase[:1].upper() + phrase[1:]


def _keyword_clause(etype: str, lang: str) -> str:
    return " ".join(_PAREN_RE.sub("", TYPE_PHRASE[lang][etype]).split())


# Offline only: long enough that the sentence finishes (20 left 15 keys unterminated). The online
# explainer keeps SLOT_MAX_NEW_TOKENS.
OFFLINE_MAX_NEW_TOKENS = 40
GENERIC_INSTRUCTION = (
    "Write the sentence for this fact. Refer to the evidence generically (for example 'this link', "
    "'this number' or 'this payment request') and do not quote any value from the message."
)


# A second instruction for the later trials: asks for one complete sentence explicitly.
GENERIC_INSTRUCTION_B = (
    "Write one complete sentence for this fact and end it with a full stop. Refer to the evidence "
    "generically, for example 'this link' or 'this message', and do not quote any value."
)
# The prompt trials, tried in order for each (language, type) until every slot shape with samples
# has an entry. Greedy decoding, so a trial is deterministic. t1 is the round-1 prompt, so a key
# it already covers keeps its round-1 sentence. A key that no trial covers is listed in
# ``rejected`` with every reason it failed.
PROMPT_TRIALS = (
    {"name": "t1", "variant": EN_VARIANT, "instruction": "generic", "max_new_tokens": 40},
    {"name": "t2", "variant": "describe_plain", "instruction": "generic", "max_new_tokens": 40},
    {"name": "t3", "variant": EN_VARIANT, "instruction": "generic_b", "max_new_tokens": 64},
    {"name": "t4", "variant": "describe_plain", "instruction": "generic_b", "max_new_tokens": 64},
)
INSTRUCTIONS = {"generic": GENERIC_INSTRUCTION, "generic_b": GENERIC_INSTRUCTION_B}
# Synthetic inputs for the evidence shapes the fixture does not have (no-slot keys, and the URL
# model with a slot). The runtime still checks every real message, so these only decide which
# sentences may enter the table.
NEUTRAL_TEXT = "Please review the notice below before you reply."
URL_MODEL_TEXT = "Check https://example-secure.com/login now."


def table_prompt(
    etype: str, lang: str, variant: str, instruction: str = GENERIC_INSTRUCTION
) -> tuple[str, str]:
    """(prefix, prompt) for one type. The prompt is the live prompt's shape (the same system text
    and prefix) with the fact stated without its matched text, and an instruction to refer to the
    evidence generically. The slot value is never in the prompt; the runtime appends it."""
    user = f"FACT: {_phrase(etype, lang)}.\n\n{instruction}"
    if lang == "hinglish":
        return HINGLISH_PREFIX, _chat(HINGLISH_SYSTEM, user, HINGLISH_PREFIX)
    spec = PROMPT_VARIANTS[variant]
    prefix = f"{SLOT_PREFIX} {_keyword_clause(etype, lang)} and" if spec["keyword"] else SLOT_PREFIX
    return prefix, _chat(spec["system"], user, prefix)


def load_samples(path: Path = FIXTURE) -> dict[str, list[tuple[str, Any]]]:
    """Per type, (message, rule evidence) pairs from the rule fixture, in fixture order."""
    cases = json.loads(path.read_text(encoding="utf-8"))
    samples: dict[str, list[tuple[str, Any]]] = {}
    for case in cases:
        text = case["text"]
        for ev in run_rules(canonicalize(text)):
            samples.setdefault(ev.type, []).append((text, ev))
    return samples


def _problem_free_point(sentence: str, package, lang: str, risk: Risk) -> bool:
    """True when the one-point package passes check_points for this sentence at this risk."""
    summary = _template_summary(package, lang)
    raw = json.dumps({"lang": lang, "summary": summary, "points": [sentence]}, ensure_ascii=False)
    verdict = check_points(raw, package, lang, risk)
    return bool(verdict.schema_ok and verdict.points[0].ok)


def synthetic_samples(
    etype: str, samples: dict[str, list[tuple[str, Any]]]
) -> list[tuple[str, Any]]:
    """Samples for the slot shapes the fixture lacks, built directly as evidence. A no-slot item
    has empty text (its fact slot is ""). The URL model's slot shape is a domain and a score. A
    type whose fixture samples already cover a shape gets nothing for it."""
    have = {bool(fact_slot(ev)) for _t, ev in samples.get(etype, [])}
    out: list[tuple[str, Any]] = []
    if False not in have:
        out.append((NEUTRAL_TEXT, Evidence(id="S0", type=etype, span=(0, 0), text="", weight=0.5)))
    if True not in have and etype == URL_MODEL_TYPE:
        text = url_model_text("example-secure.com", 0.91)
        evidence = Evidence(id="S1", type=etype, span=(0, 0), text=text, weight=0.5)
        out.append((URL_MODEL_TEXT, evidence))
    return out


def validate_entry(
    sentence: str, etype: str, lang: str, has_slot: bool, samples: Sequence[tuple[str, Any]]
) -> list[str]:
    """The risks at which the full runtime text passes the verifier on every sample. The text is
    the sentence plus the slot clause with each sample's own slot value. Empty when there is no
    sample, when the sentence is empty or holds a placeholder, or when any sample fails."""
    if not sentence or PLACEHOLDER in sentence:
        return []
    usable = [(text, ev) for text, ev in samples if bool(fact_slot(ev)) == has_slot]
    if not usable:
        return []
    kept: list[str] = []
    for risk in TABLE_RISKS:
        ok = True
        for text, ev in usable:
            canon = canonicalize(text)
            package = dataclasses.replace(
                build_package(canon, [ev], "uncertain"), risk=Risk(risk), evidence=[ev]
            )
            filled = fill(sentence, fact_slot(ev), etype)
            if not _problem_free_point(filled, package, lang, Risk(risk)):
                ok = False
                break
        if ok:
            kept.append(risk)
    return kept


def generate_sentence(
    runner,
    etype: str,
    lang: str,
    variant: str,
    max_new_tokens: int = OFFLINE_MAX_NEW_TOKENS,
    instruction: str = GENERIC_INSTRUCTION,
) -> tuple[str, str]:
    """(cleaned sentence, status) for one type, the way the live explainer cleans a point."""
    prefix, prompt = table_prompt(etype, lang, variant, instruction)
    generated, _timing = runner.generate(prompt, max_new_tokens=max_new_tokens, stop_when=slot_stop)
    sentence, status = clean_sentence(generated, prefix)
    if status != "ok":
        return "", status
    sentence, _changed = strip_echo(sentence, POINT_LEAD[lang])
    return (sentence, "ok") if sentence else ("", "echo_invalid")


def build_table(
    runner,
    samples: dict[str, list[tuple[str, Any]]],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    langs: Iterable[str] = LANGS,
    built: str | None = None,
    trials: Sequence[dict[str, Any]] = PROMPT_TRIALS,
) -> dict[str, Any]:
    """The table document. For each (language, type), the prompt trials run in order; a trial's
    sentence is kept for every slot shape (with samples) it passes at some risk, and later trials
    run only while a shape is still open. The entry records the trial that produced it."""
    entries: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    for lang in langs:
        variant = variant_for(lang)
        for etype in TYPE_PHRASE[lang]:
            pool = list(samples.get(etype, [])) + synthetic_samples(etype, samples)
            shapes = {bool(fact_slot(ev)) for _t, ev in pool}
            open_shapes = [s for s in (True, False) if s in shapes]
            for s in (False, True):
                if s not in open_shapes:
                    rejected.append(
                        {
                            "key": key(lang, variant, etype, s),
                            "reason": "no sample with this slot shape",
                        }
                    )
            tried: dict[bool, list[str]] = {s: [] for s in open_shapes}
            last: dict[bool, str] = {}
            for trial in trials:
                if not open_shapes:
                    break
                sentence, status = generate_sentence(
                    runner,
                    etype,
                    lang,
                    trial["variant"],
                    max_new_tokens=trial["max_new_tokens"],
                    instruction=INSTRUCTIONS[trial["instruction"]],
                )
                if status != "ok":
                    for s in open_shapes:
                        last[s] = f"explainer output {status}"
                        tried[s].append(f"{trial['name']}: {last[s]}")
                    continue
                for s in list(open_shapes):
                    risks = validate_entry(sentence, etype, lang, s, pool)
                    if not risks:
                        last[s] = "fails the verifier on the samples"
                        tried[s].append(f"{trial['name']}: {last[s]}")
                        continue
                    entries[key(lang, variant, etype, s)] = {
                        "lang": lang,
                        "variant": variant,
                        "type": etype,
                        "has_slot": s,
                        "sentence": sentence,
                        "risks": risks,
                        "samples": sum(1 for _t, ev in pool if bool(fact_slot(ev)) == s),
                        "trial": trial["name"],
                        "prompt_variant": trial["variant"] if lang == "en" else HINGLISH_VARIANT,
                        "instruction": trial["instruction"],
                        "max_new_tokens": trial["max_new_tokens"],
                    }
                    open_shapes.remove(s)
            for s in open_shapes:
                rejected.append(
                    {"key": key(lang, variant, etype, s), "reason": last[s], "tried": tried[s]}
                )
    return {
        "version": TABLE_VERSION,
        "placeholder": PLACEHOLDER,
        "model": model_id,
        "built": built or _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trials": [t["name"] for t in trials],
        "entries": entries,
        "rejected": rejected,
    }


def _load_llm(root: Path, threads: int):
    from rbench.runners import make_runner

    llm = make_runner("pte_llm_direct")
    llm.load(root / PTE_PATH, threads, {"tokenizer": root / TOKENIZER_PATH})
    return llm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=str(Path.home() / "rakshak-bench"))
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--fixture", default=str(FIXTURE))
    p.add_argument("--built", default=None, help="fixed build timestamp (default: now, UTC)")
    args = p.parse_args(argv)
    samples = load_samples(Path(args.fixture))
    runner = _load_llm(Path(args.root), args.threads)
    doc = build_table(runner, samples, built=args.built)
    out = Path(args.out)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "entries": len(doc["entries"]),
                "rejected": len(doc["rejected"]),
                "trials_used": {
                    name: sum(1 for e in doc["entries"].values() if e["trial"] == name)
                    for name in doc["trials"]
                },
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
