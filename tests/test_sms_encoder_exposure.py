"""Tests for kaggle/jobs/sms_encoder_exposure.py (plan docs/ENCODER_EXPOSURE_PLAN.md) and
Amendment 2 helpers of kaggle/jobs/sms_teacher_suite.py. Synthetic inputs only; no torch, no
data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_encoder_exposure as job  # noqa: E402
import sms_encoder_fold as fold  # noqa: E402
import sms_retrain as base  # noqa: E402
import sms_teacher_suite as suite  # noqa: E402


def _extra(n_groups: int, rows_per_group: int, source: str, offset: int = 0) -> pd.DataFrame:
    rows = []
    for g in range(n_groups):
        for r in range(rows_per_group):
            rows.append(
                {
                    "id": f"{source}-{offset + g}-{r}",
                    "source": source,
                    "cluster": offset + g,
                    "label": r % 2,
                }
            )
    return pd.DataFrame(rows)


# ---- calibration hold-out ----


def test_holdout_is_disjoint_by_split_group():
    extra = pd.concat(
        [_extra(50, 3, "cloveai_india_spam_sms"), _extra(40, 3, "finee_synth", offset=1000)],
        ignore_index=True,
    )
    mask, info = job.calibration_holdout(extra)
    held = set(extra.loc[mask, "cluster"])
    train = set(extra.loc[~mask, "cluster"])
    assert held and train
    assert held.isdisjoint(train)


def test_holdout_takes_twenty_percent_of_groups_per_source():
    extra = pd.concat(
        [_extra(50, 2, "cloveai_india_spam_sms"), _extra(40, 2, "finee_synth", offset=1000)],
        ignore_index=True,
    )
    _mask, info = job.calibration_holdout(extra)
    assert info["by_source"]["cloveai_india_spam_sms"] == {"split_groups": 50, "held_groups": 10}
    assert info["by_source"]["finee_synth"] == {"split_groups": 40, "held_groups": 8}


def test_holdout_is_deterministic_for_a_seed():
    extra = _extra(60, 2, "cloveai_india_spam_sms")
    a, _ = job.calibration_holdout(extra, seed=4)
    b, _ = job.calibration_holdout(extra, seed=4)
    c, _ = job.calibration_holdout(extra, seed=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_holdout_keeps_whole_groups():
    extra = _extra(30, 4, "cloveai_india_spam_sms")
    mask, _ = job.calibration_holdout(extra)
    for g in extra["cluster"].unique():
        vals = mask[extra["cluster"].to_numpy() == g]
        assert vals.all() or (~vals).all()


# ---- logit threshold rule ----


def test_logit_threshold_is_smallest_with_at_most_one_percent_fp():
    neg = np.arange(1000, dtype=np.float64)  # margins 0..999
    tau = job.logit_threshold(neg)
    # 1 % of 1000 = 10 false positives allowed: negatives >= tau must be at most 10
    assert int(np.sum(neg >= tau)) <= 10
    assert int(np.sum(neg >= np.nextafter(tau, -np.inf))) > 10


def test_logit_threshold_one_negative_allows_no_false_positive():
    # floor(0.01 x 1) = 0 false positives allowed, so tau sits just above the single negative
    assert job.logit_threshold([0.5]) > 0.5


def test_fpr_threshold_is_minus_infinity_when_every_negative_may_flag():
    assert suite.fpr_threshold([0.1, 0.2], max_fpr=1.0) == float("-inf")


def test_logit_threshold_uses_margins_not_probabilities():
    rng = np.random.default_rng(0)
    m = rng.normal(0.0, 5.0, 500)  # margins beyond float32 saturation for probabilities
    tau = job.logit_threshold(m)
    assert np.isfinite(tau)
    p = job.expit(m)
    tau_p = job.logit_threshold(p)
    # the probability form has ties at 1.0 and cannot separate beyond float64 resolution; the margin
    # rule keeps the cut strictly ordered
    assert int(np.sum(m >= tau)) <= 5
    assert np.isfinite(tau_p) or tau_p == float("-inf")


def test_margin_threshold_is_monotone_with_probability():
    m = np.linspace(-6, 6, 201)
    tau = 1.25
    p = job.expit(m)
    assert np.array_equal(m >= tau, p >= job.expit(tau))


def test_margin_matches_logit_difference():
    logits = np.array([[1.0, 3.0], [2.0, -1.0]])
    assert np.allclose(job.margins(logits), [2.0, -3.0])


def test_suite_threshold_rule_is_the_same_count_rule():
    rng = np.random.default_rng(1)
    neg = rng.normal(0, 1, 700)
    assert job.logit_threshold(neg) == suite.fpr_threshold(neg)


def test_fold_protocol_uses_fold_thresholds_and_pools_both_halves():
    neg = np.concatenate([np.linspace(0, 0.4, 100), np.linspace(0.6, 1.0, 100)])
    folds = np.array([0] * 100 + [1] * 100)
    out = suite.fold_protocol(neg, folds, {"ext": np.array([0.9, 0.1])})
    assert out["pooled_negatives"] == 200
    assert out["by_fold"]["0"]["heldout_negatives"] == 100
    assert out["by_fold"]["0"]["fp"] == int(np.sum(neg[folds == 1] >= out["tau"]["0"]))
    assert out["pooled_negative_fp"] == out["by_fold"]["0"]["fp"] + out["by_fold"]["1"]["fp"]
    assert out["external"]["ext"]["n"] == 2


# ---- gates across seeds ----


def _seed(fp=5, tp=160, rec=0.95, f1=0.98):
    return {"fp_neg": fp, "tp_smish": tp, "recall_pos": rec, "f1_clean": f1}


def test_gates_pass_when_median_and_two_seeds_pass():
    rows = [_seed(fp=5), _seed(fp=12), _seed(fp=9)]
    gates = job.aggregate_gates(rows)
    assert gates["G1"]["median"] == 9
    assert gates["G1"]["seeds_pass"] == 2
    assert gates["G1"]["pass"] is True
    assert job.verdict(rows) == "pass"


def test_single_bad_seed_does_not_fail_when_median_and_two_pass():
    rows = [_seed(fp=5), _seed(fp=50), _seed(fp=7)]
    assert job.aggregate_gates(rows)["G1"]["pass"] is True


def test_gate_fails_when_median_fails():
    rows = [_seed(fp=12), _seed(fp=20), _seed(fp=5)]
    gates = job.aggregate_gates(rows)
    assert gates["G1"]["pass"] is False
    assert gates["arm_pass"] is False
    assert job.verdict(rows) == "fail"


def test_g3_uses_ninety_percent_recall():
    ok = [_seed(rec=82 / 91), _seed(rec=83 / 91), _seed(rec=82 / 91)]
    bad = [_seed(rec=81 / 91), _seed(rec=81 / 91), _seed(rec=82 / 91)]
    assert job.aggregate_gates(ok)["G3"]["pass"] is True
    assert job.aggregate_gates(bad)["G3"]["pass"] is False


def test_g4_threshold_is_the_campaign_gate():
    assert job.aggregate_gates([_seed(f1=0.9713)] * 3)["G4"]["pass"] is True
    assert job.aggregate_gates([_seed(f1=0.9712)] * 3)["G4"]["pass"] is False


def test_verdict_is_incomplete_without_three_seeds():
    assert job.verdict([_seed(), _seed()]).startswith("incomplete")


# ---- ordering: the threshold is written before any test set is scored ----


def test_threshold_precedes_every_test_set():
    calls: list[str] = []
    rng = np.random.default_rng(2)

    def score(_model, name):
        calls.append(f"score:{name}")
        return rng.normal(0, 1, 20)

    def rule(s_legit, s_cal, y_cal):
        return float(np.quantile(s_legit, 0.99)), False

    pipe = base.run_seed_pipeline(
        fit=lambda: None,
        score=score,
        rule=rule,
        y_cal=np.array([0, 1] * 10),
        post_sets=[job.SET_CH_SCAM, *suite.TEST_SETS],
        on_threshold=lambda tau, fb: calls.append("threshold"),
    )
    events = pipe["events"]
    first_test = min(
        i for i, e in enumerate(events) if e[6:] in suite.TEST_SETS and e.startswith("score:")
    )
    assert events.index("threshold") < first_test
    assert calls.index("threshold") < calls.index(f"score:{suite.TEST_SETS[0]}")


# ---- arguments and recipe ----


def test_default_seeds_and_recipe_match_the_fold_job():
    args = job.parse_args([])
    assert args.seeds == [0, 1, 2]
    assert fold.MODEL_REVISIONS[job.MODEL] == "c5955035435e2bf121cde7f3c8863ef52ff35d82"
    assert job.F.DEFAULTS["lr"] == 3e-5
    assert job.F.DEFAULTS["epochs"] == 4


def test_seeds_must_be_distinct():
    with pytest.raises(SystemExit):
        job.parse_args(["--seeds", "1", "1"])


# ---- suite: checkpoint argument stays on the base teacher's pin by default ----


def test_suite_default_checkpoint_is_the_base_teacher():
    args = suite.parse_args([])
    assert args.checkpoint_sha256 == suite.X.TEACHER_SHA


def test_suite_rejects_a_malformed_checkpoint_digest():
    with pytest.raises(SystemExit):
        suite.parse_args(["--checkpoint-sha256", "XYZ"])
