"""rbench ELA slot mode: prompt, sentence cleaning, assembly and fallback (mocked runner)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "bench"
sys.path.insert(0, str(BENCH))

from rbench import slot, worker  # noqa: E402

from rakshak.ela_eval import package_for_sms  # noqa: E402
from rakshak.verifier import check_points  # noqa: E402

TWO_ITEMS = (
    "Please share your OTP to avoid account suspension"  # credential_request, account_suspension
)
GOOD_1 = " asks for an OTP, PIN or password."
GOOD_2 = " claims the account or card is suspended."


class ScriptedRunner:
    """Returns scripted continuations in call order; records the prompts and stop predicates."""

    def __init__(self, outputs, prompt_tokens=90, stop_hit=False):
        self.outputs = list(outputs)
        self.prompts = []
        self.stop_fns = []
        self.prompt_tokens = prompt_tokens
        self.stop_hit = stop_hit

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
            "stop_hit": self.stop_hit,
        }
        return text, timing


def _pkg(text=TWO_ITEMS):
    return package_for_sms(text)[1]


def test_prompt_carries_one_fact_and_ends_inside_the_assistant_turn():
    pkg = _pkg()
    assert len(pkg.evidence) == 2
    prompt = slot.slot_prompt(pkg.evidence[0])
    assert prompt.count("FACT:") == 1
    assert "OTP" in prompt and "account suspension" not in prompt
    assert prompt.endswith("<|end_header_id|>\n\nThis message")
    assert len(prompt) < 800  # about 120 tokens of Llama text; the real count is on the Pi


def test_prompt_rejects_non_english():
    with pytest.raises(ValueError):
        slot.slot_prompt(_pkg().evidence[0], lang="hinglish")


def test_clean_sentence_keeps_first_sentence_and_rejects_unfinished_output():
    assert slot.clean_sentence(GOOD_1) == ("This message asks for an OTP, PIN or password.", "ok")
    assert slot.clean_sentence(" asks for OTP. Also tell them to call") == (
        "This message asks for OTP.",
        "ok",
    )
    assert slot.clean_sentence("   \n") == ("", "empty")
    assert slot.clean_sentence(" asks for an OTP and") == ("", "unterminated")
    assert slot.clean_sentence(' asks for "OTP". Yes') == ("", "unterminated")
    assert slot.clean_sentence(' asks {"x": 1}.') == ("", "unterminated")


def test_stop_predicate_fires_on_sentence_end_or_forbidden_character():
    assert slot.slot_stop(" asks for an OTP") is False
    assert slot.slot_stop(" asks for an OTP.") is True
    assert slot.slot_stop(" asks for an OTP\n") is True
    assert slot.slot_stop(' asks for "OTP') is True


def test_all_points_from_the_model_pass_the_verifier():
    pkg = _pkg()
    runner = ScriptedRunner([GOOD_1, GOOD_2])
    res = slot.generate_slot_advisory(runner, pkg)
    assert res["point_status"] == ["ok", "ok"]
    assert res["point_sources"] == ["llm", "llm"]
    assert res["adv"]["verifier_ok"] is True
    assert res["final_verifier_ok"] is True
    assert res["adv"]["mode"] == "llm"
    assert res["adv"]["points_pass_n"] == 2
    assert [c["stop_hit"] for c in res["calls"]] == [False, False]
    assert all(fn is slot.slot_stop for fn in runner.stop_fns)


def test_empty_point_falls_back_to_template_and_is_counted():
    pkg = _pkg()
    res = slot.generate_slot_advisory(ScriptedRunner([GOOD_1, "  "]), pkg)
    assert res["point_status"] == ["ok", "empty"]
    assert res["point_sources"] == ["llm", "template"]
    # Raw verdict: the empty point fails, the model's sentence passes.
    assert res["adv"]["verifier_ok"] is False
    assert res["adv"]["points_pass_n"] == 1
    assert res["adv"]["mode"] == "partial"
    assert res["final_verifier_ok"] is True


def test_point_rejected_by_the_verifier_falls_back_and_the_output_still_passes():
    pkg = _pkg()
    # "click" is a positive instruction: the verifier rejects the point after cleaning.
    bad = " asks you to click the link now."
    res = slot.generate_slot_advisory(ScriptedRunner([bad, GOOD_2]), pkg)
    assert res["point_status"] == ["ok", "ok"]
    assert res["point_sources"] == ["template", "llm"]
    assert res["adv"]["points_pass_n"] == 1
    assert res["final_verifier_ok"] is True
    final = __import__("json").loads(res["output"])
    assert (
        final["points"][0] == slot.assemble(pkg, [""])["points"][0]
        or "click" not in final["points"][0]
    )
    assert check_points(res["output"], pkg).ok is True


def test_unterminated_point_is_not_sent_as_text():
    pkg = _pkg()
    res = slot.generate_slot_advisory(ScriptedRunner([" asks for an OTP and", GOOD_2]), pkg)
    assert res["point_status"] == ["unterminated", "ok"]
    assert res["point_sources"] == ["template", "llm"]
    assert res["final_verifier_ok"] is True


def test_raw_output_is_the_points_contract_with_template_summary():
    pkg = _pkg()
    res = slot.generate_slot_advisory(ScriptedRunner([GOOD_1, GOOD_2]), pkg)
    raw = json.loads(res["raw_output"])
    assert set(raw) == {"lang", "summary", "points"}
    assert raw["lang"] == "en" and len(raw["points"]) == len(pkg.evidence)
    assert raw["summary"] == slot.assemble(pkg, [])["summary"]


def test_safe_claim_counts_only_model_sentences_not_the_template():
    pkg = package_for_sms("Meeting at 4 pm.")[1]  # no evidence: template only
    res = slot.generate_slot_advisory(ScriptedRunner([]), pkg)
    assert res["calls"] == []
    assert res["adv"]["safe_claim"] is False  # the template's own "not safe" wording is not counted
    assert res["final_verifier_ok"] is True


def test_calls_match_evidence_count_and_record_prompts_and_timing():
    pkg = _pkg("You won a prize! Send Rs 500 parcel customs fee to claim, visit http://bit.ly/x1")
    n = len(pkg.evidence)
    runner = ScriptedRunner(["asks for a prize."] * n, prompt_tokens=88, stop_hit=True)
    res = slot.generate_slot_advisory(runner, pkg, max_new_tokens=20)
    assert len(res["calls"]) == n and len(runner.prompts) == n
    assert all(c["prompt_tokens"] == 88 and c["stop_hit"] is True for c in res["calls"])
    assert all("FACT:" in c["prompt"] for c in res["calls"])
    agg = worker._aggregate_calls(res["calls"])
    assert agg["calls"] == n
    assert agg["prompt_tokens"] == 88 * n
    assert agg["ttft_ms"] == 900.0
    assert agg["wall_ms"] == pytest.approx(1500.0 * n)


def test_worker_accepts_ela_mode_flag_and_defaults_to_points():
    args = worker.build_parser().parse_args(
        ["--root", "r", "--model", "m", "--dataset", "d", "--out", "o"]
    )
    assert args.ela_mode == "points"
    args = worker.build_parser().parse_args(
        ["--root", "r", "--model", "m", "--dataset", "d", "--out", "o", "--ela-mode", "slot"]
    )
    assert args.ela_mode == "slot"
    with pytest.raises(SystemExit):
        worker.build_parser().parse_args(
            ["--root", "r", "--model", "m", "--dataset", "d", "--out", "o", "--ela-mode", "x"]
        )


def test_run_spec_defaults_to_points_mode():
    from rbench.guard import RunSpec

    spec = RunSpec(model="m", dataset="d", limit=30, threads=2, warmup=0, seed=0, name="")
    assert spec.ela_mode == "points"
