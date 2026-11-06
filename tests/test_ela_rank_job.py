"""Rank telemetry and injection cohorts, without loading or running a model."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rakshak.ela_eval import aggregate, injection_comparisons, package_for_sms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle" / "jobs"))
import llama_ela_eval as job  # noqa: E402


def test_rank_token_ids_include_space_and_count_tokens_not_characters():
    calls = []

    def tokenizer(text, *, add_special_tokens):
        assert add_special_tokens is False
        calls.append(text)
        return {"input_ids": {"prefix": [10, 11], " OTP": [12], " जल्दी": [13, 14]}[text]}

    prefix, candidates = job._rank_token_ids(tokenizer, "prefix", [" OTP", " जल्दी"])
    assert prefix == [10, 11]
    assert candidates == [[12], [13, 14]]
    assert sum(map(len, candidates)) == 3
    assert calls == ["prefix", " OTP", " जल्दी"]


def test_rank_empty_never_imports_torch_or_scores(monkeypatch):
    _, pkg = package_for_sms("Meeting at 4 pm.")
    monkeypatch.setitem(sys.modules, "torch", None)

    def forbidden(*args):
        pytest.fail("empty evidence must never score candidates")

    monkeypatch.setattr(job, "_mean_logprobs", forbidden)
    _, latency, info = job._rank(None, None, pkg, "en")
    assert latency == 0.0
    assert info["rank_empty"]
    assert info["rank_targets"] == info["rank_candidates"] == 0


def test_rank_counts_actual_scored_targets_and_preserves_choices(monkeypatch):
    _, pkg = package_for_sms("Please share your OTP to avoid account suspension")
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None)
    ))

    def score(tok, model, prefix, candidates):
        assert all(c.startswith(" ") for c in candidates)
        calls.append(candidates)
        return [-4, -3, -1, -2], 9

    monkeypatch.setattr(job, "_mean_logprobs", score)
    _, latency, info = job._rank(None, None, pkg, "en")
    assert info["choice"] == [2] * len(pkg.evidence)
    assert info["rank_targets"] == 9 * len(calls)
    assert info["rank_candidates"] == 4 * len(calls)
    assert info["rank_items"] == len(pkg.evidence)
    assert not info["rank_empty"] and latency >= 0


def row(sid, safe=False, items=1):
    return {"sample_id": sid, "safe_claim": safe, "rank_items": items, "family": "encoded"}


def test_injection_excludes_empty_pairs_and_matches_freeform_cohort():
    clean = {a: [row("yes"), row("empty", items=0), row("lost")] for a in job.INJECT_ARMS}
    injected = {a: [row("yes"), row("empty", items=0), row("lost", items=0)]
                for a in job.INJECT_ARMS}
    injected["freeform"][0]["safe_claim"] = True
    res = injection_comparisons(clean, injected)
    rank = res["harmful"]["rank"]
    assert rank["pairs"] == 1 and rank["excluded_no_candidates"] == 2
    assert rank["per_family"]["encoded"]["n"] == 1
    assert res["harmful"]["freeform"]["pairs"] == 3
    paired = res["paired_vs_freeform"]["rank"]
    assert paired["freeform"]["pairs"] == paired["arm"]["pairs"] == 1
    assert paired["freeform_only_harmful"] == 1
    assert paired["mcnemar_exact_p"] == 1.0


def test_no_eligible_rank_pairs_reports_no_safety_estimate():
    clean = {a: [row("empty", items=0)] for a in job.INJECT_ARMS}
    res = injection_comparisons(clean, clean)
    assert res["harmful"]["rank"]["pairs"] == 0
    assert res["harmful"]["rank"]["harmful_rate"] is None
    assert res["fisher_vs_freeform_p"]["rank"] is None
    assert res["paired_vs_freeform"]["rank"]["mcnemar_exact_p"] is None


def test_rank_zero_generated_tokens_can_still_do_work():
    base = {"json_ok": True, "verifier_ok": True, "unsupported_n": 0,
            "action_ok": True, "safe_claim": False, "new_tokens": 0, "rank_candidates": 4}
    res = aggregate([
        {**base, "rank_targets": 17, "latency_s": 2.0},
        {**base, "rank_targets": 0, "rank_candidates": 0, "latency_s": 0.0},
    ])
    assert res["mean_new_tokens"] == 0
    assert res["rank_target_tokens_total"] == 17
    assert res["zero_work_rate"] == 0.5
    assert res["latency_p50_s_worked"] == 2.0
