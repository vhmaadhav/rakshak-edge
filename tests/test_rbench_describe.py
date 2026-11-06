"""Fix A: the describe prompt variants for the edge explainer (rbench.slot.PROMPT_VARIANTS).

The variants change the system text and the assistant prefix only. The default slot prompt is
unchanged (tested in test_rbench_slot.py and test_rbench_slot_kw.py)."""

from __future__ import annotations

import pytest
from rbench.slot import (
    DESCRIBE_SYSTEM,
    PLAIN_SYSTEM,
    PROMPT_VARIANTS,
    SLOT_PREFIX,
    generate_slot_advisory,
    slot_prompt,
    variant_prefix,
    variant_prompt,
)

from rakshak.canonicalize import canonicalize
from rakshak.evidence import Risk, build_package
from rakshak.rules import Evidence
from rakshak.verifier import ATTRIBUTION_RE, has_lexicon_keyword


def _ev(etype: str, text: str = "x", weight: float = 0.9, i: int = 0) -> Evidence:
    return Evidence(id=f"ev{i}", type=etype, span=(0, len(text)), text=text, weight=weight)


def _package(evidence, risk=Risk.MEDIUM):
    canon = canonicalize("Dear customer, see the details below.")
    return build_package(canon, evidence, "scam" if risk is not Risk.LOW else "benign")


class _Runner:
    def __init__(self, sentences):
        self.sentences = list(sentences)
        self.prompts = []

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.prompts.append(prompt)
        text = self.sentences.pop(0)
        return text, {"wall_ms": 100.0, "new_tokens": 5, "prompt_tokens": 90, "stop_hit": True}


@pytest.mark.parametrize("variant", sorted(PROMPT_VARIANTS))
def test_variant_system_text_has_no_attribution_words(variant):
    """The point of Fix A: the instruction itself never asks for, or uses, a judgement word."""
    system = PROMPT_VARIANTS[variant]["system"]
    assert not ATTRIBUTION_RE.search(system), variant
    assert "scam" not in system.lower() and "fraud" not in system.lower()


def test_describe_system_constants_are_the_ones_used():
    assert PROMPT_VARIANTS["describe"]["system"] is DESCRIBE_SYSTEM
    assert PROMPT_VARIANTS["describe_plain"]["system"] is PLAIN_SYSTEM


def test_prefix_is_plain_for_describe_and_carries_the_keyword_for_describe_kw():
    ev = _ev("shortened_url", "http://bit.ly/abc")
    assert variant_prefix(ev, "describe") == SLOT_PREFIX
    assert variant_prefix(ev, "describe_plain") == SLOT_PREFIX
    kw = variant_prefix(ev, "describe_kw")
    assert kw == "This message contains a shortened link and"
    assert has_lexicon_keyword(kw, "en", "shortened_url")


@pytest.mark.parametrize("variant", ["describe", "describe_kw", "describe_plain"])
@pytest.mark.parametrize(
    "etype",
    ["credential_request", "electricity_disconnect", "apk_install", "shortened_url", "urgency"],
)
def test_prompt_ends_inside_the_assistant_turn_after_the_prefix(variant, etype):
    ev = _ev(etype, "Matched text", 0.5)
    prefix, prompt = variant_prompt(ev, variant)
    assert prompt.endswith(prefix)
    assert prompt.startswith("<|begin_of_text|>")
    assert "FACT:" in prompt


def test_describe_prompt_length_is_close_to_the_default_slot_prompt():
    ev = _ev("credential_request", "OTP", 0.9)
    default_len = len(slot_prompt(ev).split())
    for variant in ("describe", "describe_kw"):
        _, prompt = variant_prompt(ev, variant)
        assert abs(len(prompt.split()) - default_len) <= 0.25 * default_len, variant


def test_variant_and_keyword_prefix_are_separate_modes():
    pkg = _package([_ev("shortened_url", "http://bit.ly/x")])
    with pytest.raises(ValueError):
        generate_slot_advisory(_Runner(["x"]), pkg, variant="describe", keyword_prefix=True)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        variant_prefix(_ev("urgency"), "nope")


def test_default_mode_is_unchanged_when_no_variant_is_given():
    pkg = _package([_ev("shortened_url", "http://bit.ly/x")])
    runner = _Runner(["This message is a shortened link."])
    out = generate_slot_advisory(runner, pkg)
    assert runner.prompts[0] == slot_prompt(pkg.evidence[0])
    assert out["mode"] == "slot"


def test_variant_is_used_for_every_item_and_recorded_as_the_mode():
    pkg = _package([_ev("shortened_url", "http://bit.ly/x", i=0), _ev("urgency", "now", 0.4, i=1)])
    runner = _Runner(
        [
            "This message contains a shortened link and is a link.",
            "This message uses time pressure and is urgent.",
        ]
    )
    out = generate_slot_advisory(runner, pkg, variant="describe_kw", guards=True)
    assert out["mode"] == "describe_kw"
    assert runner.prompts[0] == variant_prompt(pkg.evidence[0], "describe_kw")[1]
    assert runner.prompts[1] == variant_prompt(pkg.evidence[1], "describe_kw")[1]
    assert [c["prefix"] for c in out["calls"]] == [
        "This message contains a shortened link and",
        "This message uses time pressure and",
    ]


def test_guards_still_apply_to_variant_sentences():
    """A variant sentence that attributes scam intent at MEDIUM is rejected by G2 (the template
    replaces it) and listed in g2_rejected."""
    pkg = _package([_ev("shortened_url", "http://bit.ly/x", weight=0.7)])
    assert pkg.risk is Risk.MEDIUM
    runner = _Runner(["This message is a scam because it contains a shortened link."])
    out = generate_slot_advisory(runner, pkg, variant="describe", guards=True)
    assert out["g2_rejected"] and out["g2_rejected"][0]["index"] == 0
    assert out["point_sources"] == ["template"]
    assert out["final_verifier_ok"] is True


def test_describe_sentence_passes_the_checks_with_its_keyword():
    pkg = _package([_ev("shortened_url", "http://bit.ly/x")])
    runner = _Runner(["This message contains a shortened link and is a link to a website."])
    out = generate_slot_advisory(runner, pkg, variant="describe_kw", guards=True)
    assert out["point_checks"] == [["shortened_url", True]]
    assert out["point_sources"] == ["llm"]
