"""Edge guards in the verifier: G1 (LOW risk gets no LLM text) and G2 (attribution check).

The benign examples come from docs/LLAMA_PI_RESULTS.md: the Twitter "urgent favour" message and
the Canara Bank PIN notice. Both produced attribution sentences from the explainer.
"""

from __future__ import annotations

import json

import pytest

from rakshak.ela_eval import package_for_sms
from rakshak.evidence import Risk
from rakshak.verifier import (
    attribution_problems,
    check_points,
    render_advisory,
    skip_llm_for_risk,
    template_only_advisory,
)

LOW_TEXT = "Meeting moved to 4 pm. Please bring the report."
TWITTER_LOW = "Hey it's Sam from twitter Could I ask an urgent favour please?"
HIGH_TEXT = "Dear customer share your OTP to avoid account suspension"
MEDIUM_PRIZE = (
    "Congrats! You have won the 1st Prize. For Registration please Contact PAYTM Helpline"
)
MEDIUM_PARCEL = "Please pay the parcel customs fee now to receive your delivery"

# Exact attribution sentences, as the explainer wrote them (LLAMA_PI_RESULTS.md, slot_kw section).
TWITTER_ATTRIBUTION = (
    "This message uses urgent language, which is a common tactic used by scammers to try to."
)
CANARA_ATTRIBUTION = (
    "This message warns about PIN sharing, and it is trying to rush you into making a decision."
)


def _pkg(text):
    return package_for_sms(text)[1]


def _advisory(points, summary="This message uses urgent language. Verify it yourself."):
    return json.dumps({"summary": summary, "points": points})


@pytest.fixture(scope="module")
def pkgs():
    out = {
        "low": _pkg(LOW_TEXT),
        "twitter": _pkg(TWITTER_LOW),
        "high": _pkg(HIGH_TEXT),
        "medium": _pkg(MEDIUM_PRIZE),
        "medium_parcel": _pkg(MEDIUM_PARCEL),
    }
    assert out["low"].risk is Risk.LOW
    assert out["twitter"].risk is Risk.LOW
    assert out["high"].risk is Risk.HIGH
    assert out["medium"].risk is Risk.MEDIUM
    assert out["medium_parcel"].risk is Risk.MEDIUM
    return out


# --- G2 rule: which strings are rejected ------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        TWITTER_ATTRIBUTION,
        CANARA_ATTRIBUTION,
        "This message is a scam.",
        "This is a fraud attempt.",
        "Scammers want your OTP.",
        "This message is a common tactic used by scammers.",
        "The sender is trying to steal your details.",
        "The sender wants to trick you into paying.",
        "They may be a scam, and scammers want your PIN.",  # unhedged term in the same point
        "This message is not a scam.",  # negation is not an exemption
        "It is a phishing message.",
        "Yeh message dhoka hai.",  # Hinglish
        "Yeh message thagi ka hai.",
        "Yeh message chori ka tareeka hai.",
    ],
)
@pytest.mark.parametrize("risk", [Risk.LOW, Risk.MEDIUM])
def test_attribution_is_rejected_at_low_and_medium(sentence, risk):
    problems = attribution_problems(sentence, risk)
    assert problems and problems[0].startswith("attribution of scam intent"), sentence


@pytest.mark.parametrize("sentence", [TWITTER_ATTRIBUTION, CANARA_ATTRIBUTION])
def test_benign_doc_examples_are_rejected_at_low_and_medium(sentence):
    for risk in (Risk.LOW, Risk.MEDIUM):
        assert attribution_problems(sentence, risk), (risk, sentence)


@pytest.mark.parametrize("sentence", [TWITTER_ATTRIBUTION, CANARA_ATTRIBUTION, "This is a scam."])
def test_high_risk_is_not_checked(sentence):
    assert attribution_problems(sentence, Risk.HIGH) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "This message may be a scam because it contains a shortened link.",
        "This message could be a scam because it asks for an OTP.",
        "This message might be a fraud.",
        "Yeh message dhoka ho sakta hai.",
        "Yeh message thagi ho sakti hai.",
        # Fix A (2026-11-06): "likely" is a hedge at MEDIUM and UNCERTAIN.
        "This message is likely a scam because it claims a prize.",
        "This message is likely a fraud because it asks for an OTP.",
        "This message is likely phishing.",
    ],
)
def test_hedged_forms_pass_at_medium_and_uncertain(sentence):
    assert attribution_problems(sentence, Risk.MEDIUM) == []
    assert attribution_problems(sentence, Risk.UNCERTAIN) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "This message is a scam because it claims a prize.",
        "This message is a scam that may be risky.",
        # the hedge covers only the first attribution term; the second is unhedged
        "This message is likely a scam, and scammers want your PIN.",
    ],
)
def test_unhedged_forms_fail_at_medium(sentence):
    assert attribution_problems(sentence, Risk.MEDIUM), sentence


@pytest.mark.parametrize(
    "sentence",
    [
        "This message is likely a scam because it claims a prize.",
        "This message is likely a scam because it asks for money.",
        "This message is likely a fraud.",
    ],
)
def test_likely_hedge_is_strict_at_low_and_unchecked_at_high(sentence):
    assert attribution_problems(sentence, Risk.LOW), sentence
    assert attribution_problems(sentence, Risk.HIGH) == []


def test_guard_off_by_default_adds_no_attribution_problems(pkgs):
    # risk=None is the old contract: no attribution problem is ever raised, at any risk level.
    for pkg in pkgs.values():
        points = [TWITTER_ATTRIBUTION] * len(pkg.evidence)
        verdict = check_points(_advisory(points), pkg)
        problems = [p for c in verdict.points for p in c.problems] + verdict.summary_problems
        assert not any(p.startswith("attribution") for p in problems)
    assert attribution_problems(TWITTER_ATTRIBUTION, None) == []


def test_check_points_rejects_the_doc_sentence_at_low_and_medium(pkgs):
    # "twitter" is LOW with one urgency item; "medium" is MEDIUM with two prize items.
    for key in ("twitter", "medium"):
        pkg = pkgs[key]
        points = [TWITTER_ATTRIBUTION] * len(pkg.evidence)
        verdict = check_points(_advisory(points), pkg, "en", pkg.risk)
        assert not verdict.ok
        assert all(not c.ok for c in verdict.points), key
        assert all(
            any(p.startswith("attribution of scam intent") for p in c.problems)
            for c in verdict.points
        ), key


def test_check_points_passes_the_same_sentence_at_high_when_it_is_otherwise_valid(pkgs):
    pkg = pkgs["high"]
    points = ["This message asks for an OTP to avoid account suspension."] * len(pkg.evidence)
    verdict = check_points(_advisory(points), pkg, "en", Risk.HIGH)
    assert verdict.ok, verdict.points


def test_risk_argument_must_match_the_package(pkgs):
    with pytest.raises(ValueError):
        check_points(_advisory(["x"]), pkgs["low"], "en", Risk.HIGH)


def test_render_advisory_falls_back_to_template_for_guarded_points(pkgs):
    pkg = pkgs["medium"]
    points = [CANARA_ATTRIBUTION] * len(pkg.evidence)
    rendered = render_advisory(_advisory(points), pkg, "en", pkg.risk)
    assert all(p["source"] == "template" for p in rendered["points"])
    assert rendered["mode"] == "partial"


def test_template_points_pass_the_guard_at_every_risk(pkgs):
    # A fallback point must never be rejected by the guard it falls back from.
    from rakshak.verifier import template_point

    for key, pkg in pkgs.items():
        for e in pkg.evidence:
            text = template_point(e.type, "en")
            assert attribution_problems(text, pkg.risk) == [], (key, text)


def test_template_advisory_passes_the_contract_without_the_guard(pkgs):
    # The LOW template summary says "no known scam pattern" (a negated mention, by design).
    # Templates are deterministic and are not run through G2, so the contract uses risk=None.
    for key, pkg in pkgs.items():
        advisory = json.dumps(template_only_advisory(pkg))
        verdict = check_points(advisory, pkg, "en")
        assert verdict.ok, (key, verdict.summary_problems, [c.problems for c in verdict.points])


# --- G1 ----------------------------------------------------------------------


def test_g1_skips_only_low_risk(pkgs):
    assert skip_llm_for_risk(pkgs["low"]) is True
    assert skip_llm_for_risk(pkgs["twitter"]) is True
    assert skip_llm_for_risk(pkgs["medium"]) is False
    assert skip_llm_for_risk(pkgs["high"]) is False


def test_template_only_advisory_has_one_point_per_item(pkgs):
    pkg = pkgs["twitter"]
    adv = template_only_advisory(pkg)
    assert len(adv["points"]) == len(pkg.evidence) == 1
    assert check_points(json.dumps(adv), pkg, "en").ok
