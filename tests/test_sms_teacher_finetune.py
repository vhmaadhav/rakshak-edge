"""Tests for kaggle/jobs/sms_teacher_finetune.py (plan docs/TEACHER_FINETUNE_PLAN.md).
Synthetic inputs only; no torch, no model, no inbox data. Sentinels stand in for texts and ids, and
the leak check asserts none of them reaches a payload or a report."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_teacher_finetune as job  # noqa: E402

TEXT_SENTINEL = "ZQTEXTSENTINEL"
ID_SENTINEL = "ZQROWIDSENTINEL"
SENDER_SENTINEL = "ZQSENDERSENTINEL"


def _good_row(**over) -> dict:
    row = {"groundlm_fp": 3, "smishx_tp": 200, "campaign_f1": 0.98, "donor_fpr": 0.004}
    row.update(over)
    return row


# ---- argument parsing and hyperparameters ----


def test_arms_are_required_and_limited():
    assert job.parse_args(["--arm", "T0"]).arm == "T0"
    assert job.parse_args(["--arm", "T1"]).arm == "T1"
    with pytest.raises(SystemExit):
        job.parse_args([])
    with pytest.raises(SystemExit):
        job.parse_args(["--arm", "S1"])


def test_defaults_equal_the_plan_hyperparameters():
    args = job.parse_args(["--arm", "T1"])
    assert args.model == "jhu-clsp/mmBERT-base"
    assert args.seed == 0
    assert args.lr == 3e-5
    assert args.epochs == 4
    assert args.weight_decay == 0.01
    assert args.warmup_ratio == 0.1
    assert args.max_len == 128
    assert args.batch_size == 32
    assert args.eval_batch_size == 128
    assert args.patience == 2
    assert args.smoke is False
    assert job.effective_hparams(args) == {
        "lr": 3e-5,
        "epochs": 4,
        "warmup": 0.1,
        "weight_decay": 0.01,
    }


def test_smoke_flag_and_one_epoch():
    args = job.parse_args(["--arm", "T1", "--smoke"])
    assert args.smoke is True
    assert job.effective_hparams(args)["epochs"] == 1
    assert job.smoke_row_cap(32) == 1920  # 60 steps of batch 32, below the 2,000 row cap
    assert job.smoke_row_cap(64) == 2000  # 60 steps of batch 64 would exceed 2,000 rows


@pytest.mark.parametrize("argv", [["--arm", "T0", "--lr", "0"], ["--arm", "T0", "--epochs", "0"]])
def test_invalid_hyperparameters_are_refused(argv):
    with pytest.raises(SystemExit):
        job.parse_args(argv)


# ---- training-set assembly ----


def _pool_frame(n: int, prefix: str) -> tuple[list[str], np.ndarray]:
    texts = [f"{prefix} row {i}" for i in range(n)]
    return texts, np.array([i % 2 for i in range(n)], dtype=np.int64)


def test_t0_has_no_inbox_rows():
    base_t = ["c0", "c1", "c2", "c3"]
    base_y = [0, 1, 0, 1]
    extra_t, extra_y = ["i0", "i1"], [0, 1]
    texts, y, origin = job.training_rows("T0", base_t, base_y, extra_t, extra_y)
    assert texts == base_t
    assert list(y) == base_y
    assert set(origin) == {"campaign"}
    counts = job.origin_counts(y, origin)
    assert counts["by_origin"]["inbox"] == {"total": 0, "label0": 0, "label1": 0}


def test_t1_is_t0_plus_pool_rows_with_counts():
    base_t = ["c0", "c1", "c2", "c3"]
    base_y = [0, 1, 0, 1]
    extra_t, extra_y = ["i0", "i1", "i2"], [0, 0, 1]
    texts, y, origin = job.training_rows("T1", base_t, base_y, extra_t, extra_y)
    assert texts == base_t + extra_t
    assert list(y) == base_y + extra_y
    counts = job.origin_counts(y, origin)
    assert counts["total"] == 7
    assert counts["label0"] == 4 and counts["label1"] == 3
    assert counts["by_origin"]["campaign"] == {"total": 4, "label0": 2, "label1": 2}
    assert counts["by_origin"]["inbox"] == {"total": 3, "label0": 2, "label1": 1}


def test_t1_refuses_an_inbox_row_that_matches_a_test_only_text():
    with pytest.raises(ValueError, match="test-only"):
        job.training_rows(
            "T1", ["c0"], [0], ["Your   OTP is 1234"], [0], test_only=["your otp is 1234"]
        )


def test_training_rows_refuses_length_mismatch():
    with pytest.raises(ValueError):
        job.training_rows("T0", ["a", "b"], [0], [], [])


# ---- smoke subsampler ----


def test_smoke_subsample_respects_cap_and_is_stratified():
    labels = np.array([0] * 9000 + [1] * 3000, dtype=np.int64)
    idx = job.smoke_indices(labels, 1920, seed=0)
    assert idx.size <= 1920
    assert idx.size >= 1900
    y = labels[idx]
    share1 = (y == 1).mean()
    assert abs(share1 - 0.25) < 0.02  # the 3:1 ratio of the full set


def test_smoke_subsample_is_deterministic_and_seeded():
    labels = np.array([0] * 500 + [1] * 500, dtype=np.int64)
    a = job.smoke_indices(labels, 200, seed=0)
    b = job.smoke_indices(labels, 200, seed=0)
    c = job.smoke_indices(labels, 200, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(np.diff(a) > 0)  # sorted, no repeats


def test_smoke_subsample_keeps_everything_when_small():
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    assert list(job.smoke_indices(labels, 2000, seed=0)) == [0, 1, 2, 3]


def test_smoke_subsample_steps_within_limit():
    labels = np.array([0] * 5000 + [1] * 5000, dtype=np.int64)
    idx = job.smoke_indices(labels, job.smoke_row_cap(32), seed=0)
    steps = -(-idx.size // 32)
    assert steps <= job.SMOKE_STEPS


# ---- gates and hypotheses ----


def test_gates_use_the_s_arm_definitions_and_no_onnx_gate():
    out = job.gate_eval_teacher(_good_row())
    assert "G4_onnx_bytes" not in out
    assert set(out) >= {
        "G1_groundlm_fp",
        "G2_smishx_smishing_tp",
        "G3_campaign_f1",
        "G5_test_donor_legit_fpr",
        "provisional_pass",
    }
    assert out["provisional_pass"] is True


def test_gate_boundaries_are_inclusive_as_in_s_arms():
    row = _good_row(groundlm_fp=11, smishx_tp=157, campaign_f1=0.9713, donor_fpr=0.01)
    assert job.gate_eval_teacher(row)["provisional_pass"] is True
    assert job.gate_eval_teacher(_good_row(groundlm_fp=12))["provisional_pass"] is False
    assert job.gate_eval_teacher(_good_row(smishx_tp=156))["provisional_pass"] is False
    assert job.gate_eval_teacher(_good_row(campaign_f1=0.9712))["provisional_pass"] is False
    assert job.gate_eval_teacher(_good_row(donor_fpr=0.0101))["provisional_pass"] is False


def test_missing_value_fails_its_gate():
    out = job.gate_eval_teacher(_good_row(donor_fpr=None))
    assert out["G5_test_donor_legit_fpr"]["passes"] is False
    assert out["provisional_pass"] is False


@pytest.mark.parametrize(
    ("row", "h7", "h8", "h9", "rule"),
    [
        (_good_row(), True, True, True, 1),  # everything holds
        (_good_row(donor_fpr=0.02), False, True, False, 3),  # G5 fails: H7 false
        (_good_row(groundlm_fp=12), False, True, False, 3),  # G1 fails: H7 false
        (_good_row(smishx_tp=150), True, False, False, 2),  # H7 holds, H8 fails
        (_good_row(smishx_tp=150, donor_fpr=0.02), False, False, False, 3),  # both fail
        (_good_row(campaign_f1=0.96), True, True, False, None),  # only G3 fails
    ],
)
def test_hypothesis_and_stop_rule_branches(row, h7, h8, h9, rule):
    out = job.evaluate_hypotheses(row)
    assert out["H7"]["holds"] is h7
    assert out["H8"]["holds"] is h8
    assert out["H9"]["holds"] is h9
    assert out["provisional_pass"] is h9
    assert out["stop_rule"] == rule
    assert out["stop_rule_text"] == job.STOP_TEXT[rule]


def test_stop_rule_one_implies_h7_and_h8():
    for smishx in (157, 226):
        for fp in (0, 11):
            out = job.evaluate_hypotheses(_good_row(smishx_tp=smishx, groundlm_fp=fp))
            if out["stop_rule"] == 1:
                assert out["H7"]["holds"] and out["H8"]["holds"]


def test_stop_rule_texts_name_the_plan_rules():
    assert "provisionally" in job.STOP_TEXT[1]
    assert "H7 but not H8" in job.STOP_TEXT[2]
    assert "fails H7" in job.STOP_TEXT[3]


# ---- per-category FPR, saturation, donor and pool blocks ----


def test_category_fpr_counts_legit_rows_per_policy_category():
    labels = [0, 0, 0, 0, 1, 0]
    cats = [
        "benign_transactional",
        "promotion",
        "promotion",
        "benign_transactional",
        "scam",
        "promotion",
    ]
    scores = [0.9, 0.2, 0.7, 0.1, 0.99, 0.5]
    out = job.category_fpr(labels, cats, scores, tau=0.5)
    bt = out["benign_transactional"]
    assert bt["n"] == 2
    assert (bt["legit_fpr"]["k"], bt["legit_fpr"]["n"]) == (1, 2)
    pr = out["promotion"]
    assert pr["n"] == 3
    assert (pr["legit_fpr"]["k"], pr["legit_fpr"]["n"]) == (2, 3)
    assert out["legit_rows_by_category"] == {"benign_transactional": 2, "promotion": 3}


def test_category_fpr_empty_category_has_no_rate():
    out = job.category_fpr([0, 1], ["benign_transactional", "scam"], [0.1, 0.9], tau=0.5)
    assert out["promotion"]["n"] == 0
    assert out["promotion"]["legit_fpr"]["rate"] is None


def test_saturation_share_counts_exact_ones():
    out = job.saturation_share([1.0, 0.99999, 1.0, 0.5])
    assert (out["k"], out["n"]) == (2, 4)
    assert out["share"] == pytest.approx(0.5)
    assert job.saturation_share([])["share"] is None


def test_donor_block_fields_and_scam_recall_intervals():
    labels = np.array([0, 0, 1, 1, 1])
    cats = np.array(["benign_transactional", "promotion", "scam", "scam", "scam"])
    scores = np.array([0.1, 0.9, 0.95, 0.2, 1.0])
    novel = np.array([True, True, True, False, True])
    out = job.donor_block(labels, cats, scores, 0.5, novel)
    assert out["legit_fpr"]["k"] == 1 and out["legit_fpr"]["n"] == 2
    assert out["scam_recall"]["k"] == 2 and out["scam_recall"]["n"] == 3
    assert out["scam_recall"]["clopper_pearson95"] is not None
    assert out["saturated"]["k"] == 1
    assert set(out["by_category"]) >= set(job.CATEGORIES)


def test_pool_block_legit_and_scam():
    legit = job.pool_block(np.zeros(4), [0.1, 0.6, 1.0, 0.2], 0.5, "legit")
    assert legit["fpr"]["k"] == 2 and legit["fpr"]["n"] == 4
    assert legit["saturated"]["k"] == 1
    scam = job.pool_block(np.ones(2), [0.9, 0.4], 0.5, "scam")
    assert scam["recall"]["k"] == 1 and scam["recall"]["n"] == 2
    with pytest.raises(ValueError):
        job.pool_block(np.ones(1), [0.9], 0.5, "other")


# ---- section 4 measures ----


def test_d1_block_perfect_separation_and_no_bootstrap_when_zero():
    parts = {
        "sx_pos": np.array([0.9, 0.95]),
        "sx_ham": np.array([0.1]),
        "pos226": np.array([0.95]),
        "gl": np.array([0.2, 0.3]),
        "donor_legit": np.array([0.1, 0.2]),
        "camp_y": np.array([0, 1]),
        "camp_s": np.array([0.1, 0.9]),
    }
    out = job.d1_block(parts, n_boot=0)
    assert out["A1_smishx_vs_groundlm"] == 1.0
    assert out["A2_smishx_vs_donor_filled"] == 1.0
    assert out["auc_smishx_vs_smishx_ham"] == 1.0
    assert out["campaign_test_filled"] == 1.0
    assert out["A1_smishing_only"] == 1.0
    assert "bootstrap" not in out


def test_d1_block_bootstrap_intervals_when_asked():
    rng = np.random.default_rng(3)
    parts = {
        "sx_pos": rng.uniform(0.5, 1.0, 40),
        "sx_ham": rng.uniform(0.0, 0.5, 40),
        "pos226": rng.uniform(0.5, 1.0, 20),
        "gl": rng.uniform(0.0, 0.5, 30),
        "donor_legit": rng.uniform(0.0, 0.5, 30),
        "camp_y": np.array([0, 1, 0, 1]),
        "camp_s": np.array([0.1, 0.9, 0.2, 0.8]),
    }
    out = job.d1_block(parts, n_boot=50)
    for key in ("A1_smishx_vs_groundlm", "A2_smishx_vs_donor_filled"):
        lo, hi = out["bootstrap"][key]["ci95"]
        assert lo <= out["bootstrap"][key]["point"] <= hi


# ---- filled and scrubbed mapping and alignment ----


def _donor_frames():
    filled = pd.DataFrame(
        {
            "id": ["r1", "r2", "r3"],
            "text": ["FILLED ONE", "FILLED TWO", "FILLED THREE"],
            "sender": ["s1", "s2", "s1"],
            "label": [0, 1, 0],
        }
    )
    scrubbed = pd.DataFrame(
        {
            "id": ["r1", "r2", "r3"],
            "text": ["SCRUBBED <NAME> ONE", "SCRUBBED TWO", "SCRUBBED <AMT> THREE"],
            "sender": ["s1", "s2", "s1"],
            "label": [0, 1, 0],
        }
    )
    return filled, scrubbed


def test_scrubbed_by_attach_returns_the_scrubbed_text_of_the_same_row():
    filled, scrubbed = _donor_frames()
    sub = filled.iloc[[2, 0]][["text", "sender"]].reset_index(drop=True)
    assert job.scrubbed_by_attach(sub, filled, scrubbed) == [
        "SCRUBBED <AMT> THREE",
        "SCRUBBED <NAME> ONE",
    ]


def test_check_aligned_accepts_matching_frames_and_refuses_others():
    filled, scrubbed = _donor_frames()
    job.check_aligned(filled, scrubbed, "donor")
    broken = scrubbed.copy()
    broken["label"] = [0, 0, 0]
    with pytest.raises(ValueError, match="do not line up"):
        job.check_aligned(filled, broken, "donor")


# ---- report and leak check ----


def _scored_payload(*, smoke: bool) -> dict:
    """A scored payload built from synthetic arrays and frames, through the job's own blocks."""
    labels = np.array([0, 0, 0, 1, 1, 0], dtype=np.int64)
    cats = np.array(
        ["benign_transactional", "promotion", "benign_transactional", "scam", "scam", "promotion"]
    )
    scores = np.array([0.1, 0.9, 0.2, 0.95, 0.4, 1.0])
    novel = np.ones(6, dtype=bool)
    donor_df = pd.DataFrame(
        {
            "id": [ID_SENTINEL] * 6,
            "text": [TEXT_SENTINEL] * 6,
            "sender": [SENDER_SENTINEL] * 6,
            "label": labels,
            "label_policy": cats,
        }
    )
    tau = 0.5
    donor = job.donor_block(donor_df["label"], donor_df["label_policy"], scores, tau, novel)
    gate_row = _good_row()
    gates = job.gate_eval_teacher(gate_row)
    hyp = job.evaluate_hypotheses(gate_row)
    parts = {
        "sx_pos": np.array([0.9, 0.95]),
        "sx_ham": np.array([0.1]),
        "pos226": np.array([0.95]),
        "gl": np.array([0.2, 0.3]),
        "donor_legit": np.array([0.1, 0.9]),
        "camp_y": np.array([0, 1]),
        "camp_s": np.array([0.1, 0.9]),
    }
    block = job.d1_block(parts, n_boot=0)
    block_pool = job.pool_block(np.zeros(6), scores, tau, "legit")
    scam_pool = job.pool_block(np.ones(2), [0.9, 0.4], tau, "scam")
    full = job.origin_counts(labels, np.array(["campaign"] * 3 + ["inbox"] * 3, dtype=object))
    sat = {"donor": job.saturation_share(scores)}
    return {
        "job": job.JOB,
        "arm": "T1",
        "status": "scored",
        "smoke": smoke,
        "model": "jhu-clsp/mmBERT-base",
        "seed": 0,
        "training": {
            "full": full,
            "used": full,
            "epochs_run": 1,
            "best_epoch": 0,
            "cal_pr_auc": 0.99,
            "seconds": 1.0,
        },
        "threshold": {"tau": tau, "tau_repr": repr(tau), "fallback_flag": False},
        "checkpoint": {"bytes": 123, "weights_bytes": 100},
        "gate_row": gate_row,
        "gates": gates,
        "hypotheses": hyp,
        "campaign": {"shipped_threshold": {"f1": 0.98}},
        "groundlm": {},
        "smishx": {},
        "donor": donor,
        "donor_scrubbed": donor,
        "legit_val": block_pool,
        "legit_val_scrubbed": block_pool,
        "val_scam": scam_pool,
        "val_scam_scrubbed": scam_pool,
        "measures": {"all": block, "clean": block},
        "contamination": {"test_donor_filled": {"n": 6, "contaminated": 0}},
        "saturated": sat,
        "timings": {"score_s": 2.0},
    }


def test_leak_check_no_text_or_id_reaches_payload_or_report():
    payload = _scored_payload(smoke=False)
    dumped = json.dumps(payload, default=str)
    report = job.render_report(payload)
    for blob in (dumped, report):
        assert TEXT_SENTINEL not in blob
        assert ID_SENTINEL not in blob
        assert SENDER_SENTINEL not in blob


def test_scored_report_names_the_gates_threshold_and_stop_rule():
    report = job.render_report(_scored_payload(smoke=False))
    assert "## Gates (one seed)" in report
    assert "Provisional pass: True" in report
    assert "Threshold tau = 0.5" in report
    assert "Stop rule: **1**" in report
    assert "SMOKE RUN" not in report


def test_smoke_report_says_the_numbers_are_not_results():
    report = job.render_report(_scored_payload(smoke=True))
    assert "SMOKE RUN: the numbers below are not results." in report


def test_stopped_report_has_no_scored_sections():
    payload = {
        "arm": "T1",
        "status": job.inbox.STOP_STATUS,
        "reason": "counts only",
        "smoke": False,
    }
    report = job.render_report(payload)
    assert "stopped: insufficient data" in report
    assert "## Gates" not in report


def test_redacted_errors_keep_type_and_frames_only():
    exc = ValueError(f"message with {TEXT_SENTINEL} inside")
    assert job.redacted_error(exc) == "ValueError (message redacted)"
    assert TEXT_SENTINEL not in job.redacted_error(exc)


def test_job_side_imports_are_in_the_kernel_bundle():
    """Every kaggle/jobs module the job imports, directly or through a sibling, is in its
    JOB_DEPENDENCIES entry (the bundle built by kaggle/build_kernel.py)."""
    spec = importlib.util.spec_from_file_location(
        "build_kernel", REPO / "kaggle" / "build_kernel.py"
    )
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    bundled = set(bk.JOB_DEPENDENCIES["sms_teacher_finetune"])
    job_modules = {p.stem for p in JOBS.glob("*.py")}
    seen: set[str] = set()
    todo = ["sms_teacher_finetune"]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse((JOBS / f"{name}.py").read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            todo += [n for n in names if n in job_modules]
    for name in seen - {"sms_teacher_finetune"}:
        if name == "_common":
            continue  # bundled for every job
        assert f"kaggle/jobs/{name}.py" in bundled, name
