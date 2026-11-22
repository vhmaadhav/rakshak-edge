"""Explanation table (verdict-first, step 2): the offline builder with a stub runner, the slot-free
prompt shape, the runtime clause and lookup, and the runtime verifier guard. No model is loaded."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
BENCH = REPO / "deploy" / "pi" / "bench"
for p in (SERVICE, REPO / "src", BENCH, REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import build_explanation_table as bt  # noqa: E402
from explanation_table import (  # noqa: E402
    PLACEHOLDER,
    TABLE_RISKS,
    TABLE_VERSION,
    ExplanationTable,
    fill,
    key,
    slot_clause,
    slot_label,
)
from rbench.slot import variant_prefix, variant_prompt  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import Evidence, Risk, build_package  # noqa: E402
from rakshak.prompts import fact_slot  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402
from rakshak.verifier import check_points  # noqa: E402

# A slot-free continuation that keeps the type keyword in the prefix, so the lexicon check passes
# for every type. It refers to the evidence generically and quotes no value.
GOOD_TAIL = " asks for action on this link."


class StubRunner:
    """Stands in for the Llama runner: returns a fixed continuation and records the calls."""

    def __init__(self, tail: str = GOOD_TAIL):
        self.tail = tail
        self.prompts: list[str] = []
        self.max_new_tokens: list[int] = []

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.prompts.append(prompt)
        self.max_new_tokens.append(max_new_tokens)
        return self.tail, {"wall_ms": 0.0}


def _samples():
    return bt.load_samples(REPO / "tests" / "fixtures" / "rule_cases.json")


def _built(tail: str = GOOD_TAIL) -> dict:
    return bt.build_table(StubRunner(tail), _samples(), built="2026-11-22T00:00:00Z")


def _package(text: str, risk: Risk = Risk.MEDIUM):
    canon = canonicalize(text)
    return dataclasses.replace(build_package(canon, run_rules(canon), "scam"), risk=risk)


def test_table_prompt_matches_the_live_prompt_except_the_fact():
    ev = Evidence(id="e1", type="urgency", span=(0, 6), text="Act now", weight=0.5)
    live_prefix, live_prompt = variant_prompt(ev, "describe_kw")
    prefix, prompt = bt.table_prompt("urgency", "en", "describe_kw")
    assert prefix == live_prefix == variant_prefix(ev, "describe_kw")
    # The matched text never reaches the explainer, and the refer-generically instruction is there.
    assert "Act now" not in prompt and PLACEHOLDER not in prompt
    assert "do not quote any value" in prompt
    # Same chat frame: the only difference from the live prompt is the fact and instruction line.
    live_fact = f'FACT: {bt._phrase("urgency", "en")}. Matched text: "Act now"'
    assert live_fact in live_prompt
    assert prompt.split("<|start_header_id|>assistant")[0].split("system")[0] == (
        live_prompt.split("<|start_header_id|>assistant")[0].split("system")[0]
    )


def test_fill_appends_a_deterministic_slot_clause():
    assert fill("This message asks for an OTP on this link.", "sbi-kyc.in", "suspicious_url") == (
        "This message asks for an OTP on this link. (domain: sbi-kyc.in)"
    )
    assert fill("This message is urgent.", "", "urgency") == "This message is urgent."
    assert slot_clause("urgency", "Act now") == " (matched text: Act now)"
    assert slot_label("shortened_url") == "domain"
    assert slot_label("phishing_url_model") == "domain"


def test_stored_sentences_are_slot_free_and_placeholder_never_reaches_the_runtime():
    doc = _built()
    assert doc["entries"], "the stub's keyword-bearing sentence should validate for some types"
    for entry in doc["entries"].values():
        assert PLACEHOLDER not in entry["sentence"]
    table = ExplanationTable(doc)
    any_key = next(iter(doc["entries"].values()))
    risk = Risk(any_key["risks"][0])
    slot = "x" if any_key["has_slot"] else ""
    stored = table.sentence(any_key["type"], any_key["lang"], any_key["variant"], slot, risk)
    assert stored == any_key["sentence"]  # the stored sentence; the slot clause is added later
    poisoned = json.loads(json.dumps(doc))
    for entry in poisoned["entries"].values():
        entry["sentence"] = "This message has a [[SLOT]] left in it."
    assert ExplanationTable(poisoned).sentence(
        any_key["type"], any_key["lang"], any_key["variant"], slot, risk
    ) is None


def test_builder_keeps_only_full_runtime_texts_that_pass_the_verifier_at_every_kept_risk():
    doc = _built()
    samples = _samples()
    for k, entry in doc["entries"].items():
        assert entry["risks"] and set(entry["risks"]) <= set(TABLE_RISKS)
        assert key(entry["lang"], entry["variant"], entry["type"], entry["has_slot"]) == k
        pool = [
            (t, ev)
            for t, ev in samples.get(entry["type"], [])
            + bt.synthetic_samples(entry["type"], samples)
            if bool(fact_slot(ev)) == entry["has_slot"]
        ]
        for risk in entry["risks"]:
            for text, ev in pool:
                package = dataclasses.replace(
                    build_package(canonicalize(text), [ev], "uncertain"),
                    risk=Risk(risk),
                    evidence=[ev],
                )
                # The full runtime text: sentence plus the clause with this sample's own slot.
                full = fill(entry["sentence"], fact_slot(ev), entry["type"])
                raw = json.dumps(
                    {"lang": "en", "summary": bt._template_summary(package, "en"), "points": [full]}
                )
                assert check_points(raw, package, "en", Risk(risk)).points[0].ok


def test_builder_rejects_a_sentence_that_still_holds_the_placeholder():
    doc = bt.build_table(StubRunner(" asks for action on [[SLOT]]."), _samples(), built="x")
    assert doc["entries"] == {}
    reasons = {r["reason"] for r in doc["rejected"]}
    assert "fails the verifier on the samples" in reasons


def test_builder_tries_at_most_the_prompt_trials_per_type_and_records_the_trial():
    runner = StubRunner()
    doc = bt.build_table(runner, _samples(), langs=("en",), built="x")
    types = list(bt.TYPE_PHRASE["en"])
    assert len(runner.prompts) <= len(types) * len(bt.PROMPT_TRIALS)
    # The first trial is the round-1 prompt and budget: it is the only one a stub-ok type needs.
    assert set(runner.max_new_tokens) <= {t["max_new_tokens"] for t in bt.PROMPT_TRIALS}
    assert all(e["trial"] in doc["trials"] for e in doc["entries"].values())
    assert doc["trials"] == [t["name"] for t in bt.PROMPT_TRIALS]


def test_every_type_has_a_no_slot_input_and_the_url_model_has_a_slot_input():
    samples = _samples()
    for etype in bt.TYPE_PHRASE["en"]:
        shapes = {bool(fact_slot(ev)) for _t, ev in samples.get(etype, [])}
        synth = {bool(fact_slot(ev)) for _t, ev in bt.synthetic_samples(etype, samples)}
        assert False in (shapes | synth), etype  # no-slot input, from the fixture or built here
    url_samples = bt.synthetic_samples("phishing_url_model", samples)
    assert True in {bool(fact_slot(ev)) for _t, ev in url_samples}
    assert fact_slot(url_samples[-1][1]) == "example-secure.com score 0.91"


def test_unterminated_output_is_rejected_with_its_reason():
    doc = bt.build_table(StubRunner(" asks for action on this link"), _samples(), langs=("en",))
    reasons = {r["reason"] for r in doc["rejected"]}
    assert any(r.startswith("explainer output") for r in reasons)


def test_hinglish_prompt_uses_the_hinglish_system_text_and_prefix():
    prefix, prompt = bt.table_prompt("urgency", "hinglish", "hinglish")
    assert prefix == "Yeh message"
    assert "Yeh message" in prompt and PLACEHOLDER not in prompt


def test_version_other_than_the_table_version_is_refused():
    with pytest.raises(ValueError):
        ExplanationTable({"version": TABLE_VERSION + 1, "entries": {}})
