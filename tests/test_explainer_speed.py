"""Explainer speed candidates, accounting and the gate rules (fake generator, no executorch)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "deploy" / "pi" / "bench"
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (BENCH, SERVICE, REPO / "src", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import explainer_speed as bench  # noqa: E402
from rbench import slot  # noqa: E402
from rbench.explainer_candidates import (  # noqa: E402
    BASELINE,
    CANDIDATES,
    AccountingGenerator,
    batch_prompt,
    generate_batch_advisory,
    parse_batch,
    run_explainer,
    summarise_accounting,
)
from rbench.runners.pte_direct import PteDirectRunner  # noqa: E402

from rakshak.ela_eval import package_for_sms  # noqa: E402
from rakshak.verifier import check_points  # noqa: E402

OTP_SMS = "Please share your OTP to avoid account suspension"
PRIZE_SMS = (
    "Congrats! You have won the 1st Prize Maruti ALTO. For Registration please Contact PAYTM "
    "Helpline and send your bank details"
)


def _package(text: str):
    from edge_pipeline import collapse_by_type

    _canon, package = package_for_sms(text)
    return collapse_by_type(package)


class FakeGen:
    """Returns scripted continuations in call order and a timing block like the direct runner."""

    def __init__(self, outputs, prompt_tokens=100, prefix_tokens=60, reused=True):
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.caps: list[int] = []
        self.prompt_tokens = prompt_tokens
        self.prefix_tokens = prefix_tokens
        self.reused = reused

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.prompts.append(prompt)
        self.caps.append(max_new_tokens)
        text = self.outputs.pop(0) if self.outputs else ""
        new = max(len(text.split()), 1)
        timing = {
            "wall_ms": 100.0,
            "prompt_tokens": self.prompt_tokens,
            "prefix_tokens": self.prefix_tokens,
            "prefix_reused": self.reused,
            "prefill_tokens": self.prompt_tokens - (self.prefix_tokens if self.reused else 0),
            "new_tokens": new,
            "decode_forwards": new - 1,
            "forward_calls": 1 + (new - 1) + (0 if self.reused else 1),
            "prefix_forward": not self.reused,
            "stop_reason": "stop_when" if stop_when is not None else "max_new_tokens",
            "stop_hit": stop_when is not None,
        }
        return text, timing


def _sentence(evidence_type_phrase: str) -> str:
    return f" {evidence_type_phrase} and is a scam attempt."


def test_baseline_is_the_shipped_call_on_the_same_generator():
    package = _package(OTP_SMS)
    n = len(package.evidence)
    outputs = [_sentence("asks for an OTP")] * n
    gen_a = FakeGen(list(outputs))
    gen_b = FakeGen(list(outputs))
    via_candidate = run_explainer(gen_a, package, "en", candidate=BASELINE)
    direct = slot.generate_slot_advisory(
        gen_b,
        package,
        lang="en",
        max_new_tokens=slot.SLOT_MAX_NEW_TOKENS,
        guards=True,
        variant="describe_kw",
        cleanup=True,
    )
    assert via_candidate["output"] == direct["output"]
    assert gen_a.prompts == gen_b.prompts
    assert gen_a.caps == gen_b.caps == [slot.SLOT_MAX_NEW_TOKENS] * n


def test_candidate_table_and_unknown_name():
    assert set(CANDIDATES) == {BASELINE, "c1_short", "c1_tight", "c2_batch", "c3_trim"}
    with pytest.raises(ValueError):
        run_explainer(FakeGen([]), _package(OTP_SMS), "en", candidate="nope")


def test_length_rule_sits_in_the_system_turn_so_the_reused_prefix_is_unchanged_in_size():
    package = _package(OTP_SMS)
    ev = package.evidence[0]
    base_prefix, base_prompt = slot.variant_prompt(ev, "describe_kw")
    short_prefix, short_prompt = slot.variant_prompt(ev, "describe_kw_short")
    assert base_prefix == short_prefix
    assert "under 15 words" in short_prompt
    system_base = base_prompt.split("<|eot_id|>")[0]
    system_short = short_prompt.split("<|eot_id|>")[0]
    assert "under 15 words" not in system_base and "under 15 words" in system_short


def test_trim_candidate_drops_the_trailing_instruction_only():
    package = _package(OTP_SMS)
    ev = package.evidence[0]
    _, trim_prompt = slot.variant_prompt(ev, "describe_kw_trim")
    _, short_prompt = slot.variant_prompt(ev, "describe_kw_short")
    assert "Write the sentence for this fact." not in trim_prompt
    assert "Write the sentence for this fact." in short_prompt
    assert trim_prompt.count("FACT: ") == 1


def test_parse_batch_numbered_lines_first_occurrence_wins_and_flags_missing():
    text = (
        " uses time pressure and is a scam.\n"
        "2. This message asks for a PIN without a sentence end\n"
        "2. This message repeats point two.\n"
        "5. This message is out of range."
    )
    points, statuses = parse_batch(text, 3)
    assert statuses == ["ok", "unterminated", "missing"]
    assert points[0] == "This message uses time pressure and is a scam."
    assert points[1] == "" and points[2] == ""


def test_batch_advisory_one_call_passes_the_contract_and_falls_back_per_point():
    package = _package(OTP_SMS)  # two evidence types after collapse_by_type
    n = len(package.evidence)
    assert n == 2
    # Point 1 is a good sentence; point 2 has no line, so it falls back to the template.
    # The runner returns the continuation of the prefix "1. This message" only.
    gen = FakeGen([" asks for a password or verification code.\n"], prompt_tokens=180)
    res = generate_batch_advisory(gen, package, max_new_tokens_per_point=20)
    assert len(gen.prompts) == 1
    assert res["calls"][0]["prompt"] == batch_prompt(package)
    assert gen.caps == [20 * n]
    assert res["final_verifier_ok"] is True
    assert check_points(res["output"], package, "en").ok
    sources = res["point_sources"]
    assert sources[0] == "llm" and all(s == "template" for s in sources[1:])


def test_guard_skips_low_risk_with_no_generator_call_for_every_candidate():
    from rakshak.verifier import skip_llm_for_risk

    low = None
    for text in ("Hi, are we still meeting for lunch on Friday?", "Thanks for the update."):
        pkg = _package(text)
        if skip_llm_for_risk(pkg):
            low = pkg
            break
    if low is None:
        pytest.skip("no LOW sample in the probes")
    for name in CANDIDATES:
        gen = FakeGen([])
        res = run_explainer(gen, low, "en", candidate=name)
        assert res["skipped"] is True and gen.prompts == [], name


def test_accounting_counts_steady_state_and_observed_separately():
    log = [
        {
            "prompt_tokens": 100,
            "prefix_tokens": 60,
            "prefix_reused": True,
            "prefill_tokens": 40,
            "new_tokens": 6,
            "decode_forwards": 5,
            "forward_calls": 6,
            "prefix_forward": False,
            "stop_reason": "stop_when",
        },
        {
            "prompt_tokens": 100,
            "prefix_tokens": 60,
            "prefix_reused": False,
            "prefill_tokens": 100,
            "new_tokens": 20,
            "decode_forwards": 19,
            "forward_calls": 22,
            "prefix_forward": True,
            "stop_reason": "max_new_tokens",
        },
    ]
    s = summarise_accounting(log)
    assert s["calls"] == 2
    assert s["prompt_tokens"] == 200
    assert s["reused_prefix_tokens"] == 60  # only the reused call counts as reused
    assert s["prefill_observed"] == 140
    assert s["prefill_steady"] == 80  # 40 + 40: the prefix is reused in steady state
    assert s["decode_tokens"] == 24
    assert s["tokens_processed_steady"] == 104
    assert s["forward_calls"] == 28
    assert s["forward_calls_steady"] == 28 - 1  # one prefix forward ran
    assert s["stop_reasons"] == {"stop_when": 1, "max_new_tokens": 1}


def test_accounting_generator_logs_and_passes_through():
    gen = FakeGen(["This message is short."])
    wrap = AccountingGenerator(gen)
    text, timing = wrap.generate("p", max_new_tokens=9, stop_when=lambda s: False)
    assert text == "This message is short."
    assert timing["prompt_tokens"] == 100
    log = wrap.take()
    assert len(log) == 1 and log[0]["max_new_tokens"] == 9
    assert wrap.take() == []


def test_tight_cap_is_ceil_p95_of_terminated_baseline_outputs():
    recs = [
        {"calls": [{"status": "ok", "new_tokens": n} for n in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]},
        {"calls": [{"status": "unterminated", "new_tokens": 20}]},
    ]
    cap = bench.tight_cap(recs)
    assert cap["n_terminated_calls"] == 10
    assert cap["max_terminated"] == 13
    # linear p95 over 4..13: index 8.55 -> 12 + 0.55 = 12.55 -> ceil 13
    assert cap["p95"] == pytest.approx(12.55)
    assert cap["cap"] == 13


def _rec(cand, rep, row, walls, checks, unterminated_status="ok", final=True, n_points=1):
    return {
        "candidate": cand,
        "rep": rep,
        "row_id": row,
        "n_points": n_points,
        "skipped": False,
        "point_checks": checks,
        "term_status": [unterminated_status] * n_points,
        "point_status": ["ok"] * n_points,
        "final_verifier_ok": final,
        "llm_wall_ms": walls,
        "advisory_wall_ms": walls,
        "accounting": {
            "calls": 1,
            "prompt_tokens": 100,
            "reused_prefix_tokens": 60,
            "prefill_steady": 40,
            "prefill_observed": 40,
            "decode_tokens": 6,
            "tokens_processed_steady": 46,
            "forward_calls": 7,
            "forward_calls_steady": 7,
        },
        "calls": [{"status": unterminated_status, "stop_reason": "stop_when", "generated": ""}],
        "output": "o",
        "raw_output": "r",
    }


def test_gate_verdicts_follow_the_preregistered_rules():
    base = [_rec(BASELINE, r, f"row{i}", 1000.0 + i, [True]) for r in range(2) for i in range(5)]
    good = [
        _rec("c3_trim", r, f"row{i}", 500.0 + i, [True], n_points=1)
        for r in range(2)
        for i in range(5)
    ]
    records = base + good
    summary = bench.summarise(records, {"cap": 13})
    v = summary["arms"]["c3_trim"]
    assert v["gates"]["E1_unterminated_zero"] is True
    assert v["gates"]["E2_raw_pass_not_lower"] is True
    assert v["gates"]["E3_final_contract_all"] is True
    assert v["gates"]["E4_tokens_or_calls_down_30pct"] is False  # identical tokens here
    assert v["llm_wall_ratio_vs_baseline_p50"]["point"] == pytest.approx(0.5, rel=0.01)
    assert v["passed_all"] is False


def test_gate_e1_fails_on_any_unterminated_point_and_e3_on_contract():
    base = [_rec(BASELINE, 0, f"row{i}", 1000.0, [True]) for i in range(4)]
    arm = [_rec("c1_short", 0, f"row{i}", 400.0, [True], unterminated_status="unterminated")
           for i in range(3)]
    arm.append(_rec("c1_short", 0, "row3", 400.0, [True], final=False))
    summary = bench.summarise(base + arm, {"cap": 13})
    gates = summary["arms"]["c1_short"]["gates"]
    assert gates["E1_unterminated_zero"] is False
    assert gates["E3_final_contract_all"] is False


def test_e4_passes_on_a_large_enough_token_or_call_reduction():
    base = [_rec(BASELINE, 0, f"row{i}", 1000.0, [True]) for i in range(4)]
    arm = []
    for i in range(4):
        r = _rec("c2_batch", 0, f"row{i}", 900.0, [True])
        r["accounting"]["tokens_processed_steady"] = 30  # 30 / 46 = 0.65 <= 0.70
        r["accounting"]["forward_calls_steady"] = 7
        arm.append(r)
    summary = bench.summarise(base + arm, {"cap": 13})
    assert summary["arms"]["c2_batch"]["gates"]["E4_tokens_or_calls_down_30pct"] is True


class _Tick:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.5
        return self.now


def test_pte_direct_reports_forward_calls_and_stop_reason():
    # The direct runner reports the stop reason and the forward count on each call.
    class Model:
        def __init__(self, returns):
            self.returns = list(returns)
            self.calls = []

        def forward(self, ids, pos):
            self.calls.append((list(ids), pos))
            return self.returns.pop(0) if self.returns else 0

    eos = [0]
    model = Model([7, 8, 0])  # prefix forward, suffix forward, then decode forwards
    runner = PteDirectRunner.from_parts(
        model.forward,
        lambda text: [1, 2, 9, 3, 4],
        lambda ids: "".join(chr(96 + i) for i in ids),
        eos_ids=eos,
        split_id=9,
        clock=_Tick(),
    )
    _, timing = runner.generate("x", max_new_tokens=20)
    assert timing["prefix_forward"] is True
    assert timing["forward_calls"] == 3  # prefix, suffix, one decode step that returns EOS
    assert timing["stop_reason"] == "eos"
    assert timing["decode_forwards"] == 1
