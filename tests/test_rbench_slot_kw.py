"""rbench ELA slot_kw mode: keyword prefix per evidence type, assembly, fallback (mocked)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "bench"
sys.path.insert(0, str(BENCH))

from rbench import slot, worker  # noqa: E402

from rakshak.evidence import EvidencePackage, Risk, _allowed_actions  # noqa: E402
from rakshak.rules import Evidence  # noqa: E402
from rakshak.verifier import (  # noqa: E402
    _ACTION_VERB_RE,
    _SAFE_RE,
    TYPE_PHRASE,
    check_points,
    has_lexicon_keyword,
)

ALL_TYPES = sorted(TYPE_PHRASE["en"])  # the 16 evidence types the rules can emit
GOOD_COMPLETION = " is the reason it was flagged."


class ScriptedRunner:
    """Returns scripted continuations in call order; records prompts and stop predicates."""

    def __init__(self, outputs, prompt_tokens=95):
        self.outputs = list(outputs)
        self.prompts = []
        self.stop_fns = []
        self.prompt_tokens = prompt_tokens

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.prompts.append(prompt)
        self.stop_fns.append(stop_when)
        text = self.outputs.pop(0)
        timing = {
            "wall_ms": 1500.0,
            "new_tokens": len(text.split()),
            "prompt_tokens": self.prompt_tokens,
            "ttft_ms": 900.0,
            "prefill_tok_s": 100.0,
            "decode_tok_s": 12.0,
            "stop_hit": False,
        }
        return text, timing


def _package(etype: str, risk: Risk = Risk.MEDIUM, text: str = "x") -> EvidencePackage:
    ev = Evidence(id="E1", type=etype, span=(0, len(text)), text=text, weight=0.7)
    return EvidencePackage(
        risk=risk,
        confidence_state="scam",
        evidence=[ev],
        allowed_actions=_allowed_actions([ev], risk),
    )


def _two_items():
    from rakshak.ela_eval import package_for_sms

    return package_for_sms("Please share your OTP to avoid account suspension")[
        1
    ]  # credential_request, account_suspension


@pytest.mark.parametrize("etype", ALL_TYPES)
def test_prefix_for_every_type_carries_a_keyword_the_verifier_accepts(etype):
    pkg = _package(etype)
    prefix = slot.kw_prefix(pkg.evidence[0])
    assert prefix.startswith("This message ") and prefix.endswith(" because it")
    # The verifier itself decides: the keyword check is the same one check_points runs.
    assert has_lexicon_keyword(prefix, "en", etype), prefix
    # The prefix states only the rule's fact: no digits, links, safety words or instructions.
    assert not re.search(r"\d", prefix)
    assert "/" not in prefix and "." not in prefix
    assert not _SAFE_RE.search(prefix)
    assert not _ACTION_VERB_RE.search(prefix)


@pytest.mark.parametrize("etype", ALL_TYPES)
def test_prefix_plus_neutral_completion_passes_check_points_for_every_type(etype):
    pkg = _package(etype)
    prefix = slot.kw_prefix(pkg.evidence[0])
    output = json.dumps(slot.assemble(pkg, [prefix + GOOD_COMPLETION]))
    verdict = check_points(output, pkg)
    assert verdict.ok, verdict.points[0].problems


def test_keyword_clause_drops_parenthetical_examples():
    pkg = _package("urgency")
    assert slot.keyword_clause(pkg.evidence[0]) == "uses time pressure"
    pkg = _package("shortened_url", text="bit.ly/x")
    assert slot.kw_prefix(pkg.evidence[0]) == "This message contains a shortened link because it"


def test_kw_prompt_is_short_and_ends_inside_the_assistant_turn_after_the_prefix():
    pkg = _two_items()
    prompt = slot.slot_kw_prompt(pkg.evidence[0])
    prefix = slot.kw_prefix(pkg.evidence[0])
    assert prompt.endswith("<|end_header_id|>\n\n" + prefix)
    assert prompt.count("FACT:") == 1
    assert "OTP" in prompt and "account suspension" not in prompt
    assert len(prompt) < 800
    assert len(slot.SLOT_KW_SYSTEM) < len(slot.SLOT_SYSTEM)


def test_prompt_rejects_non_english():
    with pytest.raises(ValueError):
        slot.kw_prefix(_package("urgency").evidence[0], lang="hinglish")
    with pytest.raises(ValueError):
        slot.slot_kw_prompt(_package("urgency").evidence[0], lang="hinglish")


def test_clean_sentence_joins_the_prefix_and_keeps_the_stop_rules():
    prefix = "This message uses time pressure because it"
    assert slot.clean_sentence(" is urgent.", prefix) == (
        "This message uses time pressure because it is urgent.",
        "ok",
    )
    assert slot.clean_sentence(" is urgent. Also call", prefix)[1] == "ok"
    assert slot.clean_sentence("   ", prefix) == ("", "empty")
    assert slot.clean_sentence(" is urgent and", prefix) == ("", "unterminated")
    assert slot.clean_sentence(' is "urgent".', prefix) == ("", "unterminated")
    # Default prefix keeps the slot mode behaviour.
    assert slot.clean_sentence(" asks for an OTP.")[0] == "This message asks for an OTP."


def test_all_points_from_the_model_pass_and_assemble_with_the_keyword_prefix():
    pkg = _two_items()
    runner = ScriptedRunner([" is trying to get your code.", " is about your account."])
    res = slot.generate_slot_advisory(runner, pkg, keyword_prefix=True)
    assert res["mode"] == "slot_kw"
    assert res["prefixes"] == [
        "This message asks for an OTP, PIN or password because it",
        "This message claims an account or card is suspended because it",
    ]
    assert res["point_status"] == ["ok", "ok"]
    assert res["point_sources"] == ["llm", "llm"]
    assert res["adv"]["verifier_ok"] is True
    assert res["final_verifier_ok"] is True
    assert res["point_checks"] == [["credential_request", True], ["account_suspension", True]]
    assert all(fn is slot.slot_stop for fn in runner.stop_fns)
    assert all(c["prefix"] in c["prompt"] for c in res["calls"])
    assert res["prefixes"] == [c["prefix"] for c in res["calls"]]


def test_keyword_is_in_the_point_even_when_the_model_adds_nothing_useful():
    pkg = _package("prize_lottery")
    res = slot.generate_slot_advisory(ScriptedRunner([" is a claim."]), pkg, keyword_prefix=True)
    assert res["point_status"] == ["ok"]
    assert res["point_checks"] == [["prize_lottery", True]]


def test_point_with_positive_instruction_falls_back_and_output_still_passes():
    pkg = _two_items()
    bad = " asks you to click the link now."
    res = slot.generate_slot_advisory(
        ScriptedRunner([bad, " is about your account."]), pkg, keyword_prefix=True
    )
    assert res["point_sources"] == ["template", "llm"]
    assert res["point_checks"][0][1] is False
    assert res["final_verifier_ok"] is True
    assert check_points(res["output"], pkg).ok is True
    assert "click" not in json.loads(res["output"])["points"][0]


def test_empty_and_unterminated_points_fall_back_to_the_template():
    pkg = _two_items()
    res = slot.generate_slot_advisory(
        ScriptedRunner(["  ", " is trying and"]), pkg, keyword_prefix=True
    )
    assert res["point_status"] == ["empty", "unterminated"]
    assert res["point_sources"] == ["template", "template"]
    assert res["final_verifier_ok"] is True
    assert res["adv"]["points_pass_n"] == 0


def test_slot_mode_is_unchanged_by_default():
    pkg = _two_items()
    res = slot.generate_slot_advisory(
        ScriptedRunner([" asks for an OTP.", " claims the account or card is suspended."]), pkg
    )
    assert res["mode"] == "slot"
    assert res["prefixes"] == [slot.SLOT_PREFIX, slot.SLOT_PREFIX]
    assert res["calls"][0]["prompt"] == slot.slot_prompt(pkg.evidence[0])
    assert res["point_sources"] == ["llm", "llm"]


def test_worker_and_cli_accept_slot_kw():
    args = worker.build_parser().parse_args(
        ["--root", "r", "--model", "m", "--dataset", "d", "--out", "o", "--ela-mode", "slot_kw"]
    )
    assert args.ela_mode == "slot_kw"
    assert "slot_kw" in worker.SLOT_MODES
