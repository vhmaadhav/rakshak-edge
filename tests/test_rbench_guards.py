"""rbench slot path with the edge guards on (mocked runner): G1 skips the explainer, G2 rejects
attribution sentences at LOW and MEDIUM and reports them."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "bench"
sys.path.insert(0, str(BENCH))

from rbench import slot  # noqa: E402

from rakshak.ela_eval import package_for_sms  # noqa: E402
from rakshak.evidence import Risk  # noqa: E402
from rakshak.verifier import template_point  # noqa: E402

LOW_TEXT = "Hey it's Sam from twitter Could I ask an urgent favour please?"
MEDIUM_TEXT = "Congrats! You have won the 1st Prize. For Registration please Contact PAYTM Helpline"
HIGH_TEXT = "Dear customer share your OTP to avoid account suspension"


class CountingRunner:
    """Returns scripted continuations in call order and counts the calls."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.calls += 1
        text = self.outputs.pop(0)
        return text, {
            "wall_ms": 100.0,
            "new_tokens": 5,
            "prompt_tokens": 90,
            "ttft_ms": 50.0,
            "stop_hit": True,
        }


def _pkg(text):
    return package_for_sms(text)[1]


def test_g1_low_risk_makes_no_generate_call_and_returns_the_template():
    pkg = _pkg(LOW_TEXT)
    assert pkg.risk is Risk.LOW
    runner = CountingRunner([])
    res = slot.generate_slot_advisory(runner, pkg, guards=True)
    assert runner.calls == 0
    assert res["skipped"] is True
    assert res["calls"] == []
    assert res["mode"] == "template"
    assert res["final_verifier_ok"] is True
    final = json.loads(res["output"])
    assert final["points"] == [template_point(e.type) for e in pkg.evidence]
    assert res["point_sources"] == ["template"] * len(pkg.evidence)


def test_g1_is_off_without_guards_so_low_risk_still_calls_the_runner():
    pkg = _pkg(LOW_TEXT)
    n = len(pkg.evidence)
    runner = CountingRunner(
        ["asks for an OTP."] * n
    )  # any continuation; the call count is the test
    res = slot.generate_slot_advisory(runner, pkg)
    assert runner.calls == n
    assert res["skipped"] is False


def test_g2_rejects_attribution_at_medium_and_lists_it():
    pkg = _pkg(MEDIUM_TEXT)
    assert pkg.risk is Risk.MEDIUM
    n = len(pkg.evidence)
    runner = CountingRunner(
        [" is a common tactic used by scammers to try to get details."] + [" is a scam."] * (n - 1)
    )
    res = slot.generate_slot_advisory(runner, pkg, guards=True)
    assert res["skipped"] is False
    assert len(res["g2_rejected"]) == n
    assert all(r["problem"].startswith("attribution of scam intent") for r in res["g2_rejected"])
    assert res["point_sources"] == ["template"] * n
    assert res["final_verifier_ok"] is True


def test_g2_passes_a_hedged_sentence_at_medium():
    pkg = _pkg(MEDIUM_TEXT)
    n = len(pkg.evidence)
    runner = CountingRunner(
        [" may be a scam because it claims a prize."]
        + [" may be a scam because it claims a prize."] * (n - 1)
    )
    res = slot.generate_slot_advisory(runner, pkg, guards=True)
    assert res["g2_rejected"] == []
    assert res["point_sources"] == ["llm"] * n


def test_g2_is_not_applied_at_high_risk():
    pkg = _pkg(HIGH_TEXT)
    assert pkg.risk is Risk.HIGH
    n = len(pkg.evidence)
    runner = CountingRunner([" is a scam because it asks for your OTP."] * n)
    res = slot.generate_slot_advisory(runner, pkg, guards=True)
    assert res["g2_rejected"] == []


def test_guards_off_reproduces_the_old_behaviour_on_medium():
    pkg = _pkg(MEDIUM_TEXT)
    n = len(pkg.evidence)
    runner = CountingRunner(
        [" is a common tactic used by scammers to try to get details."] + [" is a scam."] * (n - 1)
    )
    res = slot.generate_slot_advisory(runner, pkg)
    assert res["g2_rejected"] == []
    assert res["skipped"] is False
