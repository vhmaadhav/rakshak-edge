"""Tests for kaggle/jobs/sms_crossfit_confirm.py (docs/CROSSFIT_CONFIRM_PLAN.md, frozen 2026-11-12).

Synthetic inputs only: no torch, no model, no data, no inbox rows. The fold construction is the
sms_groupneg one (rakshak.eval.groupneg.cross_fit_folds), run on synthetic template keys.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_crossfit_confirm as job  # noqa: E402
import sms_teacher_suite as S  # noqa: E402

from rakshak.eval.groupneg import assign_folds, cross_fit_folds, negative_clusters  # noqa: E402

SENTINEL = "ZQXSENTINEL"


def _hex_keys(n: int) -> list[str]:
    """Distinct random-looking keys: char-5-gram Jaccard between them is near zero."""
    return [hashlib.sha1(str(i).encode()).hexdigest() for i in range(n)]


def _plan_keys() -> list[str]:
    """1133 keys in 653 clusters: 480 pairs of identical keys and 173 singletons. Sizes come out
    567 and 566 under the balanced assignment."""
    keys = _hex_keys(653)
    out: list[str] = []
    for i, k in enumerate(keys):
        out.extend([k, k] if i < 480 else [k])
    assert len(out) == 1133
    return out


# ---- fold construction ----


def test_fold_plan_matches_the_pre_registered_counts():
    folds, info = job.fold_plan(_plan_keys())
    assert info["clusters"] == 653
    assert sorted(info["fold_sizes"]) == [566, 567]
    assert folds.shape == (1133,)


def test_fold_plan_fails_when_the_cluster_count_differs():
    keys = _hex_keys(1133)  # every key its own cluster
    with pytest.raises(ValueError, match="do not match the plan"):
        job.fold_plan(keys)


def test_fold_plan_is_the_sms_groupneg_construction():
    keys = _plan_keys()
    clusters, folds = cross_fit_folds(keys, n_folds=2)
    assert clusters == negative_clusters(keys)
    assert folds == assign_folds(clusters, n_folds=2)
    got, _info = job.fold_plan(keys)
    assert np.array_equal(got, np.asarray(folds))


def test_no_cluster_spans_both_folds():
    keys = _plan_keys()
    clusters, folds = cross_fit_folds(keys, n_folds=2)
    for c in set(clusters):
        assert len({f for f, cc in zip(folds, clusters, strict=True) if cc == c}) == 1


def test_sms_groupneg_calls_the_shared_construction():
    text = (JOBS / "sms_groupneg.py").read_text()
    assert "cross_fit_folds(keys, n_folds=N_FOLDS)" in text
    assert job.cross_fit_folds is cross_fit_folds


def test_groundlm_negative_mask_and_keys_follow_build_sets_rule():
    gl = pd.DataFrame({
        "policy_negative": [True, True, False, True, True],
        "text_available": [True, False, True, True, True],
        "eval_ok": [True, True, True, True, False],
        "template_key": ["a", "b", "c", "d", "e"],
    })
    mask = job.groundlm_negative_mask(gl)
    assert mask.tolist() == [True, False, False, True, False]
    assert job.template_keys(gl, np.flatnonzero(mask)) == ["a", "d"]


# ---- thresholds ----


def test_tau_flags_at_most_floor_one_percent_of_the_fold():
    margins = np.arange(567, dtype=np.float64)  # 567 negatives, floor(5.67) = 5 allowed
    tau = job.tau_for_fold(margins)
    rec = job.tau_record(margins, tau)
    assert rec["k_allowed"] == 5
    assert rec["flagged_own_fold"] == 5
    assert tau > 561 and tau <= 562 + 1e-9


def test_held_out_flags_use_the_other_folds_threshold():
    margins = np.array([0.1, 0.9, 0.9, 0.7])
    folds = np.array([0, 0, 1, 1])
    taus = {"0": 0.8, "1": 0.4}
    held = job.held_out_flags(margins, folds, taus)
    # fold 0 negatives are judged at tau_1 (0.4), fold 1 negatives at tau_0 (0.8)
    assert held.tolist() == [False, True, True, False]


def test_thresholds_written_before_any_positive_is_scored():
    good = ["score:neg_f0", "threshold:0", "score:neg_f1", "threshold:1",
            "score:groundlm_pos", "score:smishx_smishing"]
    assert job.cross_order_ok(good)


def test_tau_0_after_fold_1_is_scored_fails():
    bad = ["score:neg_f0", "score:neg_f1", "threshold:0", "threshold:1", "score:groundlm_pos"]
    assert not job.cross_order_ok(bad)


def test_external_set_before_tau_1_fails():
    bad = ["score:neg_f0", "threshold:0", "score:neg_f1", "score:groundlm_pos", "threshold:1"]
    assert not job.cross_order_ok(bad)


# ---- gates ----


def _per_tau(smish=157, pos=82, f1=0.972):
    return {t: {"smish_tp": smish, "pos_tp": pos, "n_pos": 91, "f1_clean": f1} for t in ("0", "1")}


def test_all_gates_pass_at_the_limits():
    out = job.evaluate_confirm(11, 1133, _per_tau())
    assert out["all_pass"] is True
    assert out["verdict"] == "pass"
    assert out["failed"] == []


def test_c1_fails_at_twelve_false_positives():
    out = job.evaluate_confirm(12, 1133, _per_tau())
    assert out["gates"]["C1_pooled_heldout_groundlm_fp"]["pass"] is False
    assert out["verdict"].startswith("fail")


def test_c2_needs_157_at_both_thresholds():
    per = _per_tau()
    per["1"]["smish_tp"] = 156
    out = job.evaluate_confirm(11, 1133, per)
    assert out["gates"]["C2_smishx_smishing_tp"]["pass"] is False


def test_c3_is_82_of_91_not_81():
    assert job.evaluate_confirm(11, 1133, _per_tau(pos=82))["gates"][
        "C3_groundlm_pos_recall"]["pass"] is True
    assert job.evaluate_confirm(11, 1133, _per_tau(pos=81))["gates"][
        "C3_groundlm_pos_recall"]["pass"] is False


def test_c4_failure_alone_is_the_indian_sms_stage_outcome():
    out = job.evaluate_confirm(11, 1133, _per_tau(f1=0.9700))
    assert out["failed"] == ["C4_campaign_clean_f1"]
    assert "Indian-SMS stage" in out["verdict"]


def test_c4_uses_the_pre_registered_floor():
    assert job.C4_MIN_F1 == S.G4_MIN_F1 == 0.9713
    assert job.C1_MAX_FP == S.G1_MAX_FP == 11
    assert job.C2_MIN_TP == S.G2_MIN_TP == 157


# ---- privacy ----


def _fake_sets(n_neg=12, n_pos=5, n_smish=4, n_ham=3, n_camp=8):
    rng = np.random.default_rng(0)

    def one(n, label, cat=""):
        return {
            "ids": np.array([f"row_{label}_{i}" for i in range(n)]),
            "labels": np.full(n, label, dtype=np.int64),
            "texts": [SENTINEL] * n,
            "codes": np.arange(n),
            "clean": np.ones(n, dtype=bool),
            "category": np.array([cat or "c1"] * n),
            "xlmr": rng.random(n) > 0.5,
            "ids_": None,
        }

    camp = one(n_camp, 1)
    camp["labels"] = (np.arange(n_camp) % 2).astype(np.int64)
    return {
        job.SET_NEG: one(n_neg, 0),
        job.SET_POS: one(n_pos, 1),
        job.SET_SMISH: one(n_smish, 1),
        job.SET_HAM: one(n_ham, 0),
        job.SET_CAMP: camp,
        job.SET_LANG: {**one(6, 0), "labels": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)},
        job.SET_TEMP: {**one(6, 0), "labels": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)},
    }


def test_privacy_scan_of_every_output_without_text(tmp_path):
    sets = _fake_sets()
    rng = np.random.default_rng(1)
    scores = {n: rng.normal(size=len(s["labels"])) for n, s in sets.items()}
    folds = np.array([0, 1] * 6)
    p1_labels = np.array([0, 1, 0, 1])
    p1_ids = [f"phone1_{i}" for i in range(4)]
    p1_scores = rng.normal(size=4)
    scores[job.SET_P1] = p1_scores
    clean = np.ones(len(sets[job.SET_CAMP]["labels"]), dtype=bool)
    clean[::3] = False
    taus = {"0": 0.1, "1": -0.2}
    held = job.held_out_flags(scores[job.SET_NEG], folds, taus)
    blocks = {
        t: job.block_at(sets, scores, taus[t], clean, p1_labels, p1_scores) for t in ("0", "1")
    }
    per_tau = {t: {"smish_tp": blocks[t]["smish_tp"], "pos_tp": blocks[t]["pos_tp"],
                   "n_pos": blocks[t]["n_pos"], "f1_clean": blocks[t]["f1_clean"]}
               for t in ("0", "1")}
    heldout = job.heldout_block(sets, held, folds)
    gate = job.evaluate_confirm(heldout["pooled_fp"], heldout["pooled_n"], per_tau)
    dep = job.dep_block(sets, scores, 0.0, scores[job.SET_NEG], clean, p1_labels, p1_scores)
    payload = {"job": job.JOB, "gates": gate["gates"], "verdict": gate["verdict"],
               "thresholds": {"tau": {"0": taus["0"], "1": taus["1"]}},
               "heldout_groundlm": heldout, "at_tau": blocks, "at_tau_dep": dep,
               "auc": job.auc_block(sets, scores)}
    thr = job.threshold_payload({"clusters": 653, "fold_sizes": [567, 566]},
                                {"0": {"tau": 0.1, "n": 567}}, None, 1133)
    (tmp_path / "metrics.json").write_text(json.dumps(payload, default=str))
    (tmp_path / "thresholds.json").write_text(json.dumps(thr))
    (tmp_path / "report.md").write_text(job.render_report(payload))
    frame = job.probs_frame(3, sets, {**scores, job.SET_NEG: scores[job.SET_NEG]}, folds,
                            p1_ids, p1_labels, clean)
    frame.to_parquet(tmp_path / "probs.parquet", index=False)

    scanned = 0
    for path in tmp_path.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            blob = " ".join([str(v) for v in df.to_numpy().ravel().tolist()] +
                            [str(c) for c in df.columns])
        else:
            blob = path.read_text(encoding="utf-8")
        assert SENTINEL not in blob, path.name
    assert scanned == 4
    assert set(frame.columns) == {"seed", "set", "id", "label", "fold", "margin", "clean",
                                  "category", "xlmr_pred", "p_scam"}
    assert not any("message" in c for c in frame.columns)
    assert "phone1_0" in set(frame["id"])


def test_report_names_no_text_and_states_the_label():
    payload = {"job": job.JOB, "plan": job.PLAN, "verdict": "pass", "gates": {},
               "thresholds": {"tau": {}}}
    gates = job.evaluate_confirm(11, 1133, _per_tau())["gates"]
    payload["gates"] = gates
    text = job.render_report(payload)
    assert "GroundLM-calibrated (threshold only, cross-fitted, template-disjoint)" in text


def test_errors_are_redacted_and_frames_only(monkeypatch):
    class Ctx:
        argv: list[str] = []
        results: dict = {}
        logged: list[str] = []

        def log(self, msg):
            self.logged.append(msg)

    def boom(ctx):
        raise ValueError(f"message carries {SENTINEL}")

    monkeypatch.setattr(job, "_run", boom)
    ctx = Ctx()
    with pytest.raises(job.RedactedError) as info:
        job.run(ctx)
    assert SENTINEL not in str(info.value)
    assert "ValueError (message redacted)" in str(info.value)
    assert all(SENTINEL not in line for line in ctx.logged)
    assert job.REDACT_ERRORS is True


# ---- build wiring ----


def test_job_dependencies_are_registered():
    text = (ROOT / "kaggle" / "build_kernel.py").read_text()
    assert "'sms_crossfit_confirm': (" in text
    assert "'kaggle/jobs/sms_encoder_exposure.py'" in text


def test_pinned_plan_constants():
    assert job.SEED == 3
    assert job.GL_NEG_N == 1133 and job.GL_POS_N == 91
    assert job.SX_SMISH_N == 226 and job.SX_HAM_N == 606
    assert job.FOLD_SIZES == (566, 567)
    assert job.N_CLUSTERS == 653


# ---- W1h: ensemble, default unchanged, ordering (fake scorers, no torch) ----


def test_default_arguments_are_the_single_seed_3_run():
    args = job.parse_args([])
    assert args.seeds == [3]
    assert args.ensemble is None
    fields = job.seed_fields(args.seeds, args.ensemble)
    assert fields == {"seed": 3, "seeds": [3], "ensemble": None}


def test_w1h_flags_parse_to_three_seeds_with_mean():
    args = job.parse_args(["--seeds", "4", "5", "6", "--ensemble", "mean"])
    assert args.seeds == [4, 5, 6]
    assert args.ensemble == "mean"
    assert job.seed_fields(args.seeds, args.ensemble)["seed"] is None


@pytest.mark.parametrize("argv", [
    ["--seeds", "4", "5"],                      # several seeds need a rule
    ["--seeds", "4", "--ensemble", "mean"],     # a rule needs two seeds
    ["--seeds", "4", "4", "--ensemble", "mean"],  # seeds must be distinct
    ["--ensemble", "median", "--seeds", "4", "5"],  # only mean is pre-registered
])
def test_bad_seed_flags_are_refused(argv):
    with pytest.raises(SystemExit):
        job.parse_args(argv)


def test_ensemble_margin_is_the_equal_weight_mean():
    a = np.array([1.0, -2.0, 0.5])
    b = np.array([3.0, 0.0, 0.5])
    c = np.array([-1.0, 1.0, 2.0])
    got = job.ensemble_margins([a, b, c], "mean")
    assert got.dtype == np.float64
    assert np.allclose(got, (a + b + c) / 3.0)


def test_single_seed_margin_is_its_own_margin_and_a_copy():
    a = np.array([0.25, -0.5])
    got = job.ensemble_margins([a], None)
    assert np.array_equal(got, a)
    got[0] = 9.0
    assert a[0] == 0.25


def test_several_seeds_without_a_rule_and_non_finite_margins_are_refused():
    with pytest.raises(ValueError):
        job.ensemble_margins([np.zeros(2), np.zeros(2)], None)
    with pytest.raises(ValueError):
        job.ensemble_margins([np.array([0.0, np.nan])], None)


def test_threshold_payload_default_keys_and_ensemble_keys():
    base = job.threshold_payload({"clusters": 653}, {}, None, 1133)
    assert base["seed"] == 3 and base["seeds"] == [3] and base["ensemble"] is None
    assert "ensemble" not in base["rule"]
    ens = job.threshold_payload({"clusters": 653}, {}, None, 1133, seeds=[4, 5, 6],
                                ensemble="mean")
    assert ens["seed"] is None and ens["seeds"] == [4, 5, 6] and ens["ensemble"] == "mean"
    assert "mean logit margin" in ens["rule"]


OFFSETS = {4: 0.0, 5: 0.5, 6: -0.3}


def _fake_run(seeds, ensemble, n_neg=20):
    """A fake scorer: margin of negative I under seed s is 0.1 I + offset(s); external text j is
    10 + s + 0.01 j. Every call is traced as S<seed>:<kind>; the threshold writes as W<folds>."""
    trace: list[str] = []
    writes: list[tuple[str, dict, float | None]] = []

    def score_seed(seed, texts):
        kind = "neg" if texts[0].startswith("neg") else texts[0]
        trace.append(f"S{seed}:{kind}")
        if kind == "neg":
            off = OFFSETS.get(seed, 0.0)
            return np.array([0.1 * int(t[3:]) + off for t in texts])
        return np.array([10.0 + seed + 0.01 * j for j in range(len(texts))])

    def write(recs, tau_dep):
        label = "W" + ",".join(sorted(recs)) + ("+dep" if tau_dep is not None else "")
        trace.append(label)
        writes.append((label, {k: dict(v) for k, v in recs.items()}, tau_dep))

    folds = np.array([0, 1] * (n_neg // 2))
    neg_texts = [f"neg{i}" for i in range(n_neg)]
    external = [(job.SET_POS, [f"ext:pos{j}" for j in range(3)]),
                (job.SET_SMISH, [f"ext:smish{j}" for j in range(2)])]
    timings: dict = {}
    out = job.ordered_protocol(seeds, ensemble, score_seed, neg_texts, folds, external,
                               write, timings)
    return out, trace, writes, folds


def test_single_seed_protocol_order_and_values_are_unchanged():
    out, trace, writes, folds = _fake_run([3], None)
    assert trace == ["S3:neg", "W0", "S3:neg", "W0,1", "W0,1+dep",
                     "S3:ext:pos0", "S3:ext:smish0"]
    assert out["events"] == ["score:neg_f0", "threshold:0", "score:neg_f1", "threshold:1",
                             f"score:{job.SET_POS}", f"score:{job.SET_SMISH}"]
    # one seed: the score is the seed's own margin
    assert np.array_equal(out["neg"], out["neg_by_seed"][3])
    assert out["taus"]["0"] == pytest.approx(job.tau_for_fold(out["neg"][folds == 0]))


def test_ensemble_scores_every_seed_before_each_threshold_and_before_any_external():
    out, trace, writes, folds = _fake_run([4, 5, 6], "mean")
    # fold 0: all three seeds, then tau_0 is written, before fold 1 is scored
    assert trace[:4] == ["S4:neg", "S5:neg", "S6:neg", "W0"]
    # fold 1: all three seeds, then tau_1, then tau_dep
    assert trace[4:9] == ["S4:neg", "S5:neg", "S6:neg", "W0,1", "W0,1+dep"]
    first_ext = min(i for i, e in enumerate(trace) if ":ext:" in e)
    assert first_ext > trace.index("W0,1+dep") > trace.index("W0,1")
    # every external set is scored by every seed, in the given order
    assert trace[9:] == (
        [f"S{s}:ext:pos0" for s in (4, 5, 6)] + [f"S{s}:ext:smish0" for s in (4, 5, 6)])


def test_ensemble_thresholds_come_from_the_mean_margin_not_one_seed():
    out, trace, writes, folds = _fake_run([4, 5, 6], "mean")
    idx0 = np.flatnonzero(folds == 0)
    mean_f0 = np.array([np.mean([0.1 * i + OFFSETS[s] for s in (4, 5, 6)]) for i in idx0])
    assert np.allclose(out["neg"][idx0], mean_f0)
    assert out["taus"]["0"] == pytest.approx(job.tau_for_fold(mean_f0))
    # the first threshold written already holds the ensemble tau_0
    first_label, first_recs, first_dep = writes[0]
    assert first_label == "W0" and first_dep is None
    assert first_recs["0"]["tau"] == pytest.approx(out["taus"]["0"])
    assert writes[-1][0] == "W0,1+dep" and writes[-1][2] == pytest.approx(out["tau_dep"])


def test_ensemble_external_scores_are_the_mean_over_seeds():
    out, trace, writes, folds = _fake_run([4, 5, 6], "mean")
    expected = np.array([10.0 + np.mean([4, 5, 6]) + 0.01 * j for j in range(3)])
    assert np.allclose(out["scores"][job.SET_POS], expected)
    assert set(out["by_seed"][5]) == {job.SET_NEG, job.SET_POS, job.SET_SMISH}


def test_per_seed_block_reports_each_seed_at_its_own_threshold():
    sets = _fake_sets()
    rng = np.random.default_rng(2)
    scores = {n: rng.normal(size=len(s["labels"])) for n, s in sets.items()}
    scores[job.SET_P1] = rng.normal(size=4)
    folds = np.array([0, 1] * 6)
    clean = np.ones(len(sets[job.SET_CAMP]["labels"]), dtype=bool)
    p1_labels = np.array([0, 1, 0, 1])
    block = job.per_seed_block(sets, scores, scores[job.SET_NEG], folds, clean, p1_labels)
    assert set(block["tau"]) == {"0", "1"}
    assert block["tau"]["0"]["tau"] == pytest.approx(
        job.tau_for_fold(scores[job.SET_NEG][folds == 0]))
    assert block["heldout_pooled_n"] == 12
    assert "C1_pooled_heldout_groundlm_fp" in block["gates_reported_not_gated"]
