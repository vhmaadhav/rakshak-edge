"""Fix C: deterministic cleanup of the edge explainer's sentences (rbench.slot, cleanup=True).

The bad strings are the real outputs of the last 40-row eval (responses.jsonl, Fix A and B run):
row 0 ("and has been matched"), row 14 ("and restates the fact"), row 4 (one sentence written six
times). The cleanup changes no prompt and no verifier rule: an echo is removed only when a valid
sentence remains, a duplicate becomes "" and falls back to the template, and the verifier still
checks every final point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "service"
REPO = Path(__file__).resolve().parents[1]
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

from edge_pipeline import EdgeAnalyzer  # noqa: E402
from rbench.slot import duplicate_points, generate_slot_advisory, strip_echo  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import Risk, build_package  # noqa: E402
from rakshak.rules import Evidence  # noqa: E402
from rakshak.verifier import template_point  # noqa: E402

ROW0_ECHO = "This message claims a prize or lottery and has been matched."
ROW14_ECHO = "This message threatens electricity disconnection and restates the fact."
ROW4_REPEAT = "This message claims a prize or lottery and is asking the reader to do something."
ROW3_CLEAN = "This message uses time pressure and asks the reader to respond within 24 hours."


def _ev(etype: str, i: int, text: str = "x") -> Evidence:
    return Evidence(id=f"ev{i}", type=etype, span=(0, len(text)), text=text, weight=0.9)


def _package(evidence, risk=Risk.MEDIUM):
    canon = canonicalize("Dear customer, see the details below.")
    return build_package(canon, evidence, "scam")


class _Runner:
    """Returns one scripted continuation per call (the text after the assistant prefix)."""

    def __init__(self, continuations):
        self.continuations = list(continuations)

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        text = self.continuations.pop(0)
        return text, {"wall_ms": 100.0, "new_tokens": 5, "prompt_tokens": 90, "stop_hit": True}


# --- strip_echo ---------------------------------------------------------------


def test_row0_has_been_matched_is_stripped_to_the_fact():
    assert strip_echo(ROW0_ECHO) == ("This message claims a prize or lottery.", True)


def test_row14_restates_the_fact_is_stripped_to_the_fact():
    assert strip_echo(ROW14_ECHO) == ("This message threatens electricity disconnection.", True)


@pytest.mark.parametrize("sentence", [ROW3_CLEAN, ROW4_REPEAT])
def test_a_sentence_without_an_echo_is_unchanged(sentence):
    assert strip_echo(sentence) == (sentence, False)


def test_echo_that_leaves_a_fragment_falls_back_to_the_template():
    # "because it" is a dangling clause after removal
    assert strip_echo("This message claims a prize because it and restates the fact.") == ("", True)


def test_echo_that_leaves_too_few_words_falls_back():
    assert strip_echo("This message restates the fact.") == ("", True)


def test_echo_that_does_not_start_with_this_message_falls_back():
    assert strip_echo("It claims a prize and has been matched.") == ("", True)


def test_echo_in_the_middle_of_a_sentence_is_not_touched():
    s = "This message has been matched to a prize and asks you to reply."
    assert strip_echo(s) == (s, False)


# --- duplicate_points ---------------------------------------------------------


def test_row4_six_identical_sentences_keep_the_first_only():
    assert duplicate_points([ROW4_REPEAT] * 6) == [None, 0, 0, 0, 0, 0]


def test_near_identical_text_counts_as_a_duplicate():
    a = "This message uses time pressure and is urgent."
    b = "this message uses time pressure and is urgent"
    assert duplicate_points([a, b]) == [None, 0]


def test_different_points_are_both_kept():
    assert duplicate_points([ROW3_CLEAN, ROW4_REPEAT]) == [None, None]


def test_empty_points_are_never_duplicates():
    assert duplicate_points(["", "", ROW3_CLEAN]) == [None, None, None]


# --- generate_slot_advisory with cleanup ---------------------------------------


def test_cleanup_strips_the_echo_and_blanks_the_duplicate_before_assembly():
    package = _package([_ev("prize_lottery", 0), _ev("prize_lottery", 1)])
    runner = _Runner(
        [
            " claims a prize or lottery and has been matched.",
            " claims a prize or lottery and has been matched.",
        ]
    )
    out = generate_slot_advisory(runner, package, cleanup=True)
    assert out["cleanup"][0]["echo_stripped"] is True
    assert out["cleanup"][0]["original"] == ROW0_ECHO
    assert out["cleanup"][1]["duplicate_of"] == 0
    assert out["point_status"][1] == "duplicate"
    assert out["point_sources"] == ["llm", "template"]
    final = json.loads(out["output"])
    assert final["points"][0] == "This message claims a prize or lottery."
    assert final["points"][1] == template_point("prize_lottery")
    assert out["final_verifier_ok"] is True


def test_cleanup_off_keeps_the_model_sentences_as_before():
    package = _package([_ev("prize_lottery", 0), _ev("prize_lottery", 1)])
    runner = _Runner(
        [
            " claims a prize or lottery and has been matched.",
            " claims a prize or lottery and has been matched.",
        ]
    )
    out = generate_slot_advisory(runner, package)
    assert out["cleanup"] == []
    raw = json.loads(out["raw_output"])["points"]
    assert raw == [ROW0_ECHO, ROW0_ECHO]


class _FixedLLM:
    """Every call returns the same continuation (the row-4 failure: one sentence, many items)."""

    def __init__(self, continuation: str):
        self.continuation = continuation
        self.calls = 0

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.calls += 1
        return self.continuation, {
            "wall_ms": 120.0,
            "new_tokens": 6,
            "prompt_tokens": 90,
            "stop_hit": True,
        }


class _Detector:
    def score(self, text):
        return 0.9


def test_end_to_end_row4_style_repeat_is_one_point_and_contract_passes():
    # Fix D: two items of one type give one explainer call and one point (count 2). Fix C's
    # dedupe is not reached here; it still applies to points of different types (slot level).
    text = "Congrats! You have won the 1st Prize. For Registration please Contact PAYTM Helpline"
    llm = _FixedLLM(" is asking the reader to do something.")
    out = EdgeAnalyzer(_Detector(), llm, 0.35).analyse(text)
    points = out["advisory"]["points"]
    assert len(points) == 1 and points[0]["count"] == len(out["evidence"]) == 2
    assert len(out["llm_calls"]) == llm.calls == 1
    assert out["verifier"]["final_contract_ok"] is True
    assert points[0]["source"] == "llm"
    assert points[0]["cleanup"] == {"echo_stripped": False, "duplicate_of": None}
    assert points[0]["check"]["ok"] is True


def test_end_to_end_echo_is_stripped_in_the_final_text_and_kept_in_llm_text():
    text = "Congrats! You have won the 1st Prize. For Registration please Contact PAYTM Helpline"
    llm = _FixedLLM(" has been matched.")
    out = EdgeAnalyzer(_Detector(), llm, 0.35).analyse(text)
    first = out["advisory"]["points"][0]
    assert first["text"] == "This message claims a prize or lottery."
    assert first["llm_text"] == ROW0_ECHO
    assert first["cleanup"]["echo_stripped"] is True
    assert out["verifier"]["final_contract_ok"] is True
