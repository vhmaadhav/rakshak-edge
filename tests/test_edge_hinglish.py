"""Hinglish on the edge service: per-request routing, the Hinglish slot prompt, and both
--hinglish-mode values through EdgeAnalyzer with a mocked LLM. No model is loaded."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (
    REPO / "deploy" / "pi" / "service",
    REPO / "src",
    REPO / "deploy" / "pi" / "bench",
    REPO / "deploy" / "pi",
):
    sys.path.insert(0, str(p))

from edge_pipeline import (  # noqa: E402
    DETECTOR_ONLY_QUIET_SUMMARY_HINGLISH,
    DETECTOR_ONLY_SUMMARY_HINGLISH,
    EdgeAnalyzer,
    route_lang,
)
from rbench.slot import (  # noqa: E402
    HINGLISH_PREFIX,
    HINGLISH_SYSTEM,
    POINT_LEAD,
    clean_sentence,
    fact_text,
    generate_slot_advisory,
    hinglish_prompt,
    slot_prompt,
    slot_stop,
    strip_echo,
)

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.datasets.lang import detect_lang  # noqa: E402
from rakshak.evidence import build_package  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402
from rakshak.verifier import (  # noqa: E402
    TYPE_PHRASE,
    check_points,
    has_lexicon_keyword,
    template_only_advisory,
)

# Rule-matched Hinglish SMS (rules fire on them; checked in the probe, see docs).
HINGLISH_SCAM = (
    "Aapka account turant band ho jayega. OTP bhejo abhi. Official link: https://bit.ly/abc"
)
HINGLISH_DIGITAL = (
    "National Cyber Cell se bol raha hoon. Investigation ke liye aapki identity verify karni hogi."
)
HINGLISH_BENIGN = "Aaj shaam ghar aa rahe ho? Dal chawal bana leta hoon."
ENGLISH_SCAM = "Dear customer share your OTP to avoid account suspension"
ENGLISH_LOW = "Hey it's Sam from twitter Could I ask an urgent favour please?"
HI_DEVA = "आपका खाता बंद हो जाएगा, कृपया OTP भेजें।"
THRESHOLD = 0.35

# Sentences a model might write. GOOD passes the Hinglish lexicon for urgency and G2.
GOOD_URGENCY = " jaldi karne ka dabav dalta hai, aur turant action maangta hai."
ATTRIBUTION_LOW_HEDGE = " inaam ya lottery ka dawa ho sakta hai, jo dhoka ho sakta hai."
UNTERMINATED = " jaldi karne ka dabav dalta hai aur"


class FakeDetector:
    def __init__(self, p: float):
        self.p = p

    def score(self, text: str) -> float:
        return self.p


class FakeLLM:
    def __init__(self, sentence: str):
        self.sentence = sentence
        self.prompts: list[str] = []

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.prompts.append(prompt)
        return self.sentence, {
            "wall_ms": 120.0,
            "new_tokens": 6,
            "prompt_tokens": 90,
            "ttft_ms": 40.0,
            "stop_hit": True,
        }


class ByTypeLLM(FakeLLM):
    """Answers each call with the continuation for the fact in its prompt (matched on the
    Hinglish type phrase), so every evidence type gets its own sentence."""

    def __init__(self, answers: dict[str, str]):
        super().__init__("")
        self.answers = answers

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.prompts.append(prompt)
        low = prompt.lower()
        text = next((sent for phrase, sent in self.answers.items() if phrase in low), "")
        return text, {
            "wall_ms": 120.0,
            "new_tokens": 6,
            "prompt_tokens": 90,
            "ttft_ms": 40.0,
            "stop_hit": True,
        }


def _by_type(**extra):
    phrases = TYPE_PHRASE["hinglish"]
    answers = {
        phrases["credential_request"].lower(): " OTP ya PIN maangta hai.",
        phrases["account_suspension"].lower(): " account ya card band hone ka dawa karta hai.",
        phrases["urgency"].lower(): GOOD_URGENCY,
        phrases["shortened_url"].lower(): " chhota link deta hai.",
    }
    answers.update(extra)
    return ByTypeLLM(answers)


def _analyzer(llm, mode, p=0.9):
    return EdgeAnalyzer(FakeDetector(p), llm, THRESHOLD, hinglish_mode=mode)


# --- routing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        (HINGLISH_SCAM, "hinglish"),
        (HINGLISH_DIGITAL, "hinglish"),
        (ENGLISH_SCAM, "en"),
        (ENGLISH_LOW, "en"),
        (HI_DEVA, "en"),  # hi_deva is not routed yet
        ("I uploaded mine to Facebook", "en"),  # detect_lang "other": routed en
    ],
)
def test_auto_routes_on_detect_lang(text, expected):
    lang = route_lang(detect_lang(text), "auto")
    assert lang == expected


def test_english_rows_route_en_and_hinglish_rows_route_hinglish_under_auto():
    assert detect_lang(HINGLISH_SCAM) == "hinglish"
    assert route_lang("en", "auto") == "en"
    assert route_lang("other", "auto") == "en"
    assert route_lang("hi_deva", "auto") == "en"


def test_request_override_wins_over_detection_both_ways():
    assert route_lang("hinglish", "en") == "en"
    assert route_lang("en", "hinglish") == "hinglish"
    assert route_lang("en", "auto") == "en"


def test_unknown_request_lang_is_refused():
    with pytest.raises(ValueError):
        route_lang("en", "fr")


def test_unknown_hinglish_mode_is_refused():
    with pytest.raises(ValueError):
        EdgeAnalyzer(FakeDetector(0.9), FakeLLM(""), THRESHOLD, hinglish_mode="both")


# --- prompt construction ----------------------------------------------------------------------


def _package(text: str):
    return build_package(canonicalize(text), run_rules(canonicalize(text)), "scam")


def test_hinglish_prompt_is_english_instructions_hinglish_fact_and_yeh_prefix():
    pkg = _package(HINGLISH_SCAM)
    ev = pkg.evidence[0]
    prefix, prompt = hinglish_prompt(ev)
    assert prefix == HINGLISH_PREFIX == "Yeh message"
    assert prompt.endswith(HINGLISH_PREFIX)
    assert HINGLISH_SYSTEM in prompt
    assert "Hinglish" in HINGLISH_SYSTEM and "Roman" in HINGLISH_SYSTEM
    # The fact carries the Hinglish phrase for the type, not the English one.
    assert TYPE_PHRASE["hinglish"][ev.type] in prompt
    assert TYPE_PHRASE["en"][ev.type] not in prompt
    assert fact_text(ev, "hinglish").startswith(TYPE_PHRASE["hinglish"][ev.type][:1].upper())


def test_hinglish_system_text_has_no_attribution_words():
    # Describe, do not judge (Fix A): the instruction must not invite the attribution G2 rejects.
    for word in ("scam", "fraud", "phishing", "dhoka", "thag", "fake"):
        assert word not in HINGLISH_SYSTEM.lower()


def test_english_prompt_is_unchanged_by_the_hinglish_addition():
    ev = _package(ENGLISH_SCAM).evidence[0]
    prompt = slot_prompt(ev, "en")
    assert prompt.endswith("This message")
    assert HINGLISH_SYSTEM not in prompt
    with pytest.raises(ValueError):
        slot_prompt(ev, "hinglish")


def test_hinglish_stop_predicate_and_cleanup_use_the_yeh_lead():
    assert POINT_LEAD["hinglish"] == "Yeh message "
    assert slot_stop(" jaldi karne ka dabav dalta hai.")
    assert not slot_stop(" jaldi karne ka")
    text, status = clean_sentence(" jaldi karne ka dabav dalta hai.", HINGLISH_PREFIX)
    assert status == "ok" and text == "Yeh message jaldi karne ka dabav dalta hai."
    _, status = clean_sentence(" jaldi karne ka", HINGLISH_PREFIX)
    assert status == "unterminated"
    # strip_echo keeps a clean Hinglish point and removes an English echo at the end
    kept, changed = strip_echo(
        "Yeh message jaldi karne ka dabav dalta hai.", POINT_LEAD["hinglish"]
    )
    assert kept == "Yeh message jaldi karne ka dabav dalta hai." and not changed
    stripped, changed = strip_echo(
        "Yeh message jaldi karne ka dabav dalta hai and restates the fact.",
        POINT_LEAD["hinglish"],
    )
    assert changed and stripped == "Yeh message jaldi karne ka dabav dalta hai."


def test_hinglish_generate_refuses_variant_and_keyword_prefix():
    pkg = _package(HINGLISH_SCAM)
    with pytest.raises(ValueError):
        generate_slot_advisory(FakeLLM(""), pkg, lang="hinglish", variant="describe_kw")
    with pytest.raises(ValueError):
        generate_slot_advisory(FakeLLM(""), pkg, lang="hinglish", keyword_prefix=True)


def test_hinglish_calls_use_the_hinglish_prompt():
    llm = FakeLLM(GOOD_URGENCY)
    out = generate_slot_advisory(llm, _package(HINGLISH_SCAM), lang="hinglish", guards=True)
    assert out["mode"] == "hinglish"
    assert llm.prompts and all(p.endswith("Yeh message") for p in llm.prompts)
    assert all(HINGLISH_SYSTEM in p for p in llm.prompts)
    assert all(c["prefix"] == "Yeh message" for c in out["calls"])


def test_hinglish_templates_pass_the_verifier_at_every_risk():
    pkg = _package(HINGLISH_SCAM)

    adv = template_only_advisory(pkg, "hinglish")
    for risk in (pkg.risk,):
        verdict = check_points(json.dumps(adv, ensure_ascii=False), pkg, "hinglish", risk)
        assert verdict.ok, verdict.points


def test_good_urgency_sentence_has_a_hinglish_keyword():
    assert has_lexicon_keyword(GOOD_URGENCY, "hinglish", "urgency")


# --- both modes through EdgeAnalyzer -----------------------------------------------------------


def test_template_mode_makes_no_llm_call_and_uses_hinglish_templates():
    llm = FakeLLM(GOOD_URGENCY)
    out = _analyzer(llm, "template").analyse(HINGLISH_SCAM)
    assert llm.prompts == []
    assert out["llm_calls"] == []
    assert out["advisory"]["lang"] == "hinglish"
    assert out["advisory"]["path"] == "template_hinglish"
    assert out["advisory"]["mode"] == "template"
    assert out["decision"]["lang_detected"] == "hinglish"
    assert all(p["source"] == "template" for p in out["advisory"]["points"])
    assert out["verifier"]["final_contract_ok"] is True
    assert out["advisory"]["summary"]["text"].startswith("Is message mein")


def test_llm_mode_passing_sentence_is_kept_after_the_hinglish_checks():
    llm = _by_type()
    out = _analyzer(llm, "llm").analyse(HINGLISH_SCAM)
    assert out["advisory"]["lang"] == "hinglish"
    assert out["advisory"]["path"] == "explainer"
    assert out["advisory"]["mode"] == "llm", out["advisory"]["points"]
    assert len(llm.prompts) == 4
    assert llm.prompts and all(HINGLISH_SYSTEM in p for p in llm.prompts)
    llm_points = [p for p in out["advisory"]["points"] if p["source"] == "llm"]
    assert llm_points, out["advisory"]["points"]
    assert all(p["text"].startswith("Yeh message") for p in llm_points)
    assert all(p["check"]["ok"] for p in llm_points)
    assert out["verifier"]["final_contract_ok"] is True
    assert out["advisory"]["summary"]["source"] == "template"


def test_llm_mode_g2_rejects_unhedged_attribution_and_falls_back_to_template():
    # MEDIUM (prize plus shortened link): the attribution words sit outside the hedge.
    text = (
        "Aapne inaam jeeta hai! Turant claim karne ke liye link par click karein https://bit.ly/abc"
    )
    llm = FakeLLM(" dhokebaaz ka tareeka hai, aur inaam ka dawa karta hai.")
    out = _analyzer(llm, "llm").analyse(text)
    assert out["decision"]["risk"] == "MEDIUM"
    assert out["verifier"]["g2_rejected"], out["verifier"]
    assert all(x["sentence"].startswith("Yeh message") for x in out["verifier"]["g2_rejected"])
    assert all(p["source"] == "template" for p in out["advisory"]["points"])
    assert out["advisory"]["mode"] == "template"
    assert out["verifier"]["final_contract_ok"] is True


def test_llm_mode_hedged_attribution_is_kept_at_medium():
    text = (
        "Aapne inaam jeeta hai! Turant claim karne ke liye link par click karein https://bit.ly/abc"
    )
    llm = FakeLLM(ATTRIBUTION_LOW_HEDGE)
    out = _analyzer(llm, "llm").analyse(text)
    assert out["decision"]["risk"] == "MEDIUM"
    assert out["verifier"]["g2_rejected"] == []


def test_llm_mode_unterminated_call_falls_back_to_the_hinglish_template():
    llm = FakeLLM(UNTERMINATED)
    out = _analyzer(llm, "llm").analyse(HINGLISH_SCAM)
    assert out["llm_calls"] and out["llm_calls"][0]["status"] == "unterminated"
    assert all(p["source"] == "template" for p in out["advisory"]["points"])
    assert out["advisory"]["lang"] == "hinglish"
    assert out["verifier"]["final_contract_ok"] is True


def test_llm_mode_english_sentence_fails_the_hinglish_lexicon_and_is_templated():
    llm = FakeLLM(" claims a prize or lottery.")
    out = _analyzer(llm, "llm").analyse(HINGLISH_SCAM)
    assert all(p["source"] == "template" for p in out["advisory"]["points"])
    assert out["advisory"]["lang"] == "hinglish"


def test_low_risk_hinglish_in_llm_mode_is_template_only_with_g1():
    llm = FakeLLM(GOOD_URGENCY)
    out = _analyzer(llm, "llm", p=0.05).analyse(HINGLISH_BENIGN)
    assert out["decision"]["risk"] == "LOW"
    assert llm.prompts == []
    assert out["advisory"]["path"] == "low_template"
    assert out["advisory"]["lang"] == "hinglish"


def test_request_override_forces_english_and_hinglish():
    llm = FakeLLM(GOOD_URGENCY)
    forced_en = _analyzer(llm, "template").analyse(HINGLISH_SCAM, lang="en")
    assert forced_en["advisory"]["lang"] == "en"
    assert forced_en["decision"]["lang_detected"] == "hinglish"
    assert forced_en["advisory"]["path"] == "explainer"  # English path, English prompt
    forced_hi = _analyzer(FakeLLM(GOOD_URGENCY), "template").analyse(ENGLISH_SCAM, lang="hinglish")
    assert forced_hi["advisory"]["lang"] == "hinglish"
    assert forced_hi["advisory"]["path"] == "template_hinglish"
    assert forced_hi["decision"]["lang_detected"] == "en"


def test_english_row_under_auto_is_the_english_path_and_records_detection():
    llm = FakeLLM("This message uses time pressure because it asks you to act now.")
    out = _analyzer(llm, "template").analyse(ENGLISH_SCAM)
    assert out["advisory"]["lang"] == "en"
    assert out["advisory"]["path"] == "explainer"
    assert out["decision"]["lang_detected"] == "en"
    assert llm.prompts and all("Yeh message" not in p for p in llm.prompts)


@pytest.mark.parametrize("mode", ["template", "llm"])
def test_hinglish_detector_only_summary_is_hinglish_and_passes_the_contract(mode):
    out = _analyzer(FakeLLM(""), mode, p=0.9).analyse(HI_DEVA + " ")
    if out["decision"]["detector_only"]:
        assert out["advisory"]["summary"]["text"] in (
            DETECTOR_ONLY_QUIET_SUMMARY_HINGLISH,
            DETECTOR_ONLY_SUMMARY_HINGLISH,
        )
        assert out["advisory"]["lang"] == "en"  # hi_deva routes en under auto
    hinglish_only = _analyzer(FakeLLM(""), mode, p=0.9).analyse(
        "Aaj shaam ghar aa rahe ho? Dal chawal bana leta hoon.", lang="hinglish"
    )
    assert hinglish_only["verifier"]["final_contract_ok"] is True
    if hinglish_only["decision"]["detector_only"]:
        assert hinglish_only["advisory"]["lang"] == "hinglish"
        assert hinglish_only["advisory"]["summary"]["text"] in (
            DETECTOR_ONLY_QUIET_SUMMARY_HINGLISH,
            DETECTOR_ONLY_SUMMARY_HINGLISH,
        )
