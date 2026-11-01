"""Synthetic fixtures only: frozen baselines, campaign audits and paired rate reporting."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rakshak.eval import _rules_pre_widening
from rakshak.eval.audit import clustered_rate, paired_rate_test
from rakshak.eval.rule_audit import audit_fold, cohort_report, score_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle" / "jobs"))
import rule_review_audit as job  # noqa: E402


def test_baseline_is_byte_identical_to_pre_dc223d3_rules():
    assert hashlib.sha256(Path(_rules_pre_widening.__file__).read_bytes()).hexdigest() == (
        "c7c3e3185c982b9879f9f45800d6d1ccca06408a646bb584ae62ce7ccd2ce9a3"
    )


def test_urgency_changes_risk_but_keeps_evidence():
    pred = score_text("URGENT: Your card payment is due tomorrow.")
    assert pred["pre_urgency"]["warn"]
    assert not pred["pre_widening"]["warn"]
    assert pred["pre_widening"]["any_evidence"]
    assert pred["pre_widening"]["urgency_only"]


def test_fold_is_campaign_stable_and_has_both_sides():
    groups = [f"camp_{i}" for i in range(100)]
    assert set(map(audit_fold, groups)) == {"audit", "remainder"}
    assert list(map(audit_fold, groups)) == list(reversed(list(map(audit_fold, groups[::-1]))))


def test_cluster_rate_resamples_campaigns_and_is_reproducible():
    values, groups = [1, 1, 0], ["a", "a", "b"]
    a = clustered_rate(values, groups, n_boot=100)
    assert a == clustered_rate(values, groups, n_boot=100)
    assert a["rate"] == pytest.approx(2 / 3)
    assert a["n_clusters"] == 2 and a["ci95"] == [0, 1]
    with pytest.raises(ValueError):
        clustered_rate(values, ["a"])


def test_paired_identical_outcomes_have_zero_delta_and_no_gain():
    res = paired_rate_test([0, 1, 1], [0, 1, 1], ["a", "b", "b"], n_boot=20)
    assert res["rate"] == 0 and res["ci95"] == [0, 0] and res["p_value"] == 1


def test_report_fpr_keeps_abstaining_negatives_in_denominator():
    frame = pd.DataFrame({"campaign_id": ["a", "b", "c"],
                          "label_binary": [0, 0, 1], "label_scam": [0, 0, None]})
    predictions = {i: score_text(t) for i, t in enumerate([
        "Please share your OTP", "See you tomorrow", "Free prize claim now",
    ])}
    res = cohort_report(frame, predictions, n_boot=20)
    neg = res["labels"]["label_binary"]["slices"]["negative"]
    high = neg["variants"]["frozen_current"]["high"]
    assert neg["n"] == high["n"] == 2
    assert high["events"] == 1 and high["rate"] == 0.5
    assert res["labels"]["label_scam"]["unknown"] == 1


def test_zero_event_ci_is_flagged_and_wilson_remains_nonzero():
    frame = pd.DataFrame({"campaign_id": ["a", "b"], "label_binary": [0, 0]})
    preds = {i: score_text("Meeting tomorrow") for i in range(2)}
    res = cohort_report(frame, preds, n_boot=10)
    row = res["labels"]["label_binary"]["slices"]["negative"]["variants"]["frozen_current"]["high"]
    assert row["ci95"] == [0, 0] and row["degenerate"]
    assert row["wilson95"][1] > 0


def test_job_preserves_split_ids_and_reports_independent_external(tmp_path, monkeypatch):
    sms = pd.DataFrame({"id": ["s1", "s2"], "text": ["Please share your OTP", "See you"],
                        "campaign_id": ["c1", "c2"], "source": ["scamshield"] * 2,
                        "label_binary": [1, 0], "eval_ok": [True] * 2})
    ground = pd.DataFrame({"id": ["g1", "g2"], "text": ["hello", None],
                           "policy_negative": [True] * 2, "text_available": [True, False],
                           "eval_ok": [True, False], "label_binary": [0, 0]})
    smish = ground.iloc[:1].copy()
    smish["id"], smish["policy_negative"] = "x1", None
    ext = pd.concat([ground.iloc[:1], smish], ignore_index=True)
    ext["campaign_id"] = ["external:1", "external:2"]
    monkeypatch.setattr(job.data, "resolve_data_root", lambda: tmp_path)
    monkeypatch.setattr(job.data, "load_sms", lambda _: sms)
    monkeypatch.setattr(job.data, "load_runs", lambda *a, **kw: [
        SimpleNamespace(name="campaign", test=[0, 1])
    ])
    monkeypatch.setattr(job.pd, "read_parquet", lambda p: ground if "groundlm" in str(p) else smish)
    monkeypatch.setattr(job, "external_campaigns", lambda _: (ext, {}))
    monkeypatch.setattr(job, "sha256", lambda _: "frozen-hash")
    monkeypatch.setattr(job, "env_report", lambda: {})
    monkeypatch.setattr(job, "cohort_report", lambda df, _: {"n": len(df)})
    monkeypatch.setenv("RAKSHAK_WORK", str(tmp_path))
    ctx = SimpleNamespace(work=tmp_path, results={}, log=lambda _: None)
    assert job.run(ctx)
    import json

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["reports"]["external_smishx/eval_eligible"]["scope"] == job.INDEPENDENT
    assert metrics["reports"]["groundlm/all_scoreable"]["scope"] == job.IN_SAMPLE
    assert metrics["groundlm_count_audit"]["policy_negative_missing_text"] == 1
    assert metrics["reports"]["split/campaign/test"]["n"] == 2
    preds = pd.read_parquet  # use the file size; the loader above is intentionally mocked
    assert preds and (tmp_path / "rule_predictions.parquet").stat().st_size > 0
