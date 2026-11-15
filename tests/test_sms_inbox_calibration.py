"""Tests for kaggle/jobs/sms_inbox_calibration.py (docs/INBOX_CALIBRATION_PLAN.md, W1f).

Synthetic inputs only: no torch, no model, no data, no inbox rows. The privacy test runs the pure
path (calibration selection, thresholds, arm blocks, probs frame, outputs) end to end on sentinel
text and sender strings and scans every written file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_inbox_calibration as job  # noqa: E402
import sms_inbox_retrain as IN  # noqa: E402
import sms_teacher_suite as S  # noqa: E402
import sms_teacher_xsource as X  # noqa: E402

SENTINEL = "ZQXSENTINEL"
SENDER_SENTINEL = "ZQXBANK"
W1E_METRICS = ROOT / "kaggle" / "out" / "sms_encoder_exposure-w1e" / "output" / "metrics.json"


# ---- fixtures ----


def _donor_rows(prefix: str, policy: str, n: int, tagged: bool = False) -> list[dict]:
    rows = []
    for k in range(n):
        tag = " <NAME> paid <AMOUNT>" if tagged else ""
        rows.append(
            {
                "id": f"{prefix}-{k}",
                "text": f"{SENTINEL} {prefix} message {k} with reference {k * 13}{tag}",
                "sender": f"AX-{SENDER_SENTINEL}{prefix}{k % 4}-S",
                "sender_style": "alphanumeric",
                "label_policy": policy,
                "synthetic": False,
                "human_reviewed": False,
                "label_source": "codex",
                "scrubber_version": "t",
            }
        )
    return rows


def _donor_df(tag_every: int = 2) -> pd.DataFrame:
    rows = _donor_rows("legit-t", "benign_transactional", 30)
    rows += _donor_rows("legit-p", "promotion", 10)
    rows += _donor_rows("scam", "scam", 6)
    rows += _donor_rows("personal", "benign_personal", 4)
    rows += _donor_rows("unclear", "unclear", 3)
    for i, r in enumerate(rows):
        if i % tag_every == 0:
            r["text"] += " <NAME> paid <AMOUNT>"
    return pd.DataFrame(rows)


def _probs(seed_rows: dict[int, list[bool]], ids: list[str]) -> pd.DataFrame:
    frames = []
    for seed, clean in seed_rows.items():
        frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "set": S.SET_CAMP,
                    "id": ids,
                    "label": 0,
                    "margin": 0.0,
                    "clean": clean,
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "set": S.SET_GL_NEG,
                    "id": ids,
                    "label": 0,
                    "margin": 0.0,
                    "clean": True,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _fake_set(
    name: str, n: int, rng: np.random.Generator, pos_rate: float = 0.3, with_xlmr: bool = False
) -> dict:
    labels = (rng.random(n) < pos_rate).astype(np.int64)
    return {
        "ids": np.array([f"{name}-{i}" for i in range(n)]),
        "labels": labels,
        "texts": [f"{SENTINEL} {name} {i}" for i in range(n)],
        "codes": np.arange(n) % 7,
        "clean": np.ones(n, dtype=bool),
        "category": np.array(["bank" if i % 2 else "promo" for i in range(n)]),
        "row_key": np.full(n, np.nan),
        "xlmr": (rng.random(n) < 0.1) if with_xlmr else None,
        "label_fine": np.array([""] * n),
        "name": name,
    }


def _fake_sets(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    sets = {
        S.SET_CAMP: _fake_set(S.SET_CAMP, 120, rng),
        S.SET_LANG: _fake_set(S.SET_LANG, 60, rng),
        S.SET_TEMP: _fake_set(S.SET_TEMP, 60, rng),
        S.SET_GL_NEG: _fake_set(S.SET_GL_NEG, 80, rng, pos_rate=0.0, with_xlmr=True),
        S.SET_GL_POS: _fake_set(S.SET_GL_POS, 20, rng, pos_rate=1.0, with_xlmr=True),
        S.SET_SX_SMISH: _fake_set(S.SET_SX_SMISH, 30, rng, pos_rate=1.0),
        S.SET_SX_HAM: _fake_set(S.SET_SX_HAM, 40, rng, pos_rate=0.0),
    }
    return sets


def _gate_row(fp=5, tp=200, rec=0.95, f1=0.98, fpr=0.005) -> dict:
    return {"fp_neg": fp, "tp_smish": tp, "recall_pos": rec, "f1_clean": f1, "donor_fpr": fpr}


# ---- W1e pins and model names ----


def test_w1e_pins_match_the_kernel_metrics():
    if not W1E_METRICS.is_file():
        pytest.skip("local W1e metrics.json not present")
    per_seed = json.loads(W1E_METRICS.read_text())["per_seed"]
    for seed, sha in job.W1E_SHAS.items():
        assert per_seed[str(seed)]["checkpoint_sha256"] == sha


def test_model_names_are_w1e_seeds_then_teacher():
    names = job.model_names((0, 1, 2))
    assert list(names) == ["w1e-s0", "w1e-s1", "w1e-s2", job.TEACHER]
    assert (
        names[job.TEACHER]
        == X.TEACHER_SHA
        == ("673b970e9f4746dbbbb5a00aa84eb056d6750cbc2897ba712ab1aa1b234abc7e")
    )
    assert job.model_names((1,)) == {"w1e-s1": job.W1E_SHAS[1], job.TEACHER: X.TEACHER_SHA}


def test_pinned_limits_match_the_plan():
    assert job.G1_MAX_FP == 11
    assert job.G2_MIN_TP == 157
    assert job.G3_MIN_RECALL == 0.90
    assert job.G4_MIN_F1 == 0.9713
    assert job.G5_MAX_FPR == 0.01
    assert job.MIN_CAL_LEGIT == 500
    assert job.REDACT_ERRORS is True


def test_parse_args_defaults_and_rejects_unknown_seed():
    args = job.parse_args([])
    assert args.seeds == [0, 1, 2]
    with pytest.raises(SystemExit):
        job.parse_args(["--seeds", "5"])
    with pytest.raises(SystemExit):
        job.parse_args(["--seeds", "0", "0"])


# ---- threshold ----


def test_calibration_tau_flags_at_most_one_percent():
    m = np.arange(100, dtype=np.float64)  # 0..99; floor(0.01 x 100) = 1 allowed
    out = job.calibration_tau(m)
    assert out["k_allowed"] == 1
    assert out["flagged"] <= 1
    assert out["flagged"] == 1
    # one step lower flags two rows, so the threshold is the smallest one
    assert int(np.sum(m >= np.nextafter(out["tau"], -np.inf))) == 2


def test_calibration_tau_with_fewer_than_100_rows_allows_none():
    out = job.calibration_tau(np.linspace(-3, 3, 50))
    assert out["k_allowed"] == 0
    assert out["flagged"] == 0


def test_calibration_tau_is_minus_inf_only_when_every_row_may_flag():
    out = job.calibration_tau(np.array([0.1, 0.2, 0.3]), max_fpr=1.0)
    assert out["tau"] == float("-inf")
    assert out["flagged"] == 3


# ---- clean mask from the W1e probs ----


def test_clean_mask_is_read_for_matching_ids_in_order():
    ids = ["c0", "c1", "c2"]
    probs = _probs({0: [True, False, True], 1: [True, False, True]}, ids)
    mask = job.clean_from_probs(probs, ids)
    assert mask.tolist() == [True, False, True]


def test_clean_mask_rejects_mismatched_ids():
    probs = _probs({0: [True, False]}, ["c0", "c1"])
    with pytest.raises(ValueError, match="ids differ"):
        job.clean_from_probs(probs, ["c1", "c0"])


def test_clean_mask_rejects_seeds_that_disagree():
    probs = _probs({0: [True, False], 1: [False, False]}, ["c0", "c1"])
    with pytest.raises(ValueError, match="disagree"):
        job.clean_from_probs(probs, ["c0", "c1"])


# ---- calibration selection (labels and fill) ----


def test_calibration_keeps_legit_policies_and_drops_personal_and_unclear():
    raw = _donor_df(tag_every=10**9)
    filled, raw_lab, counts, _tags = job.calib_frame_pair(raw)
    assert set(filled["label"].unique()) == {0, 1}
    assert counts["filled"]["by_policy"]["benign_personal"] == 4
    assert counts["filled"]["dropped_personal_or_unclear"] == 7
    legit = filled[filled["label"] == 0]
    assert len(legit) == 40
    assert filled["id"].tolist() == raw_lab["id"].tolist()


def test_filled_calibration_has_no_tags_and_raw_keeps_them():
    raw = _donor_df(tag_every=2)
    filled, raw_lab, _counts, tags = job.calib_frame_pair(raw)
    assert not any(IN.S.TAG_RE.search(t) for t in filled["text"])
    assert any(IN.S.TAG_RE.search(t) for t in raw_lab["text"])
    assert tags["with_tag"] > 0


def test_codex_labelled_rows_are_used_for_calibration():
    raw = _donor_df(tag_every=10**9)
    raw["human_reviewed"] = False
    filled, _raw_lab, counts, _tags = job.calib_frame_pair(raw)
    assert counts["filled"]["codex_labelled"] == len(raw)
    assert (filled["label"] == 0).sum() == 40


# ---- gates ----


def test_gates_pass_when_every_seed_passes():
    rows = [_gate_row(), _gate_row(fp=9, tp=190, rec=0.92), _gate_row(fp=3, tp=170, rec=0.99)]
    assert job.verdict(rows) == "pass"


def test_one_bad_seed_does_not_fail_a_gate_when_median_and_two_seeds_pass():
    rows = [_gate_row(fp=5), _gate_row(fp=9), _gate_row(fp=60)]
    agg = job.aggregate_gates(rows)
    assert agg["G1_groundlm_neg_fp"]["pass"]
    assert agg["G1_groundlm_neg_fp"]["seeds_pass"] == 2
    assert job.verdict(rows) == "pass"


def test_two_bad_seeds_fail_the_gate():
    rows = [_gate_row(fp=5), _gate_row(fp=60), _gate_row(fp=70)]
    agg = job.aggregate_gates(rows)
    assert not agg["G1_groundlm_neg_fp"]["pass"]
    assert job.verdict(rows) == "fail"


def test_g5_uses_the_one_percent_limit_on_phone_one_legit_fpr():
    ok = [_gate_row(fpr=0.009)] * 3
    bad = [_gate_row(fpr=0.011), _gate_row(fpr=0.011), _gate_row(fpr=0.009)]
    assert job.verdict(ok) == "pass"
    assert job.aggregate_gates(bad)["G5_phone1_legit_fpr"]["median"] == pytest.approx(0.011)
    assert job.verdict(bad) == "fail"


def test_incomplete_seed_list_is_reported_as_incomplete():
    assert job.verdict([_gate_row(), _gate_row()]) == "incomplete (2 of 3 seeds)"


# ---- two-pass order ----


def test_two_pass_writes_thresholds_before_any_test_set():
    rng = np.random.default_rng(1)
    log: list[str] = []

    def make(model: str):
        def score(name: str) -> np.ndarray:
            log.append(f"{model}:{name}")
            return rng.normal(size=60 if name.startswith("calib") else 30)

        return score

    written: list[dict] = []

    def write(taus: dict) -> None:
        log.append("WRITE")
        written.append(taus)

    out = job.two_pass({"w1e-s0": make("w1e-s0"), "teacher": make("teacher")}, write)
    write_at = log.index("WRITE")
    first_test = min(
        i for i, e in enumerate(log) if e != "WRITE" and e.split(":")[1] in job.TEST_NAMES
    )
    assert all(e.split(":")[1] in job.CAL_NAMES for e in log[:write_at])
    assert write_at < first_test
    assert out["events"].index("threshold") < min(
        i for i, e in enumerate(out["events"]) if e.startswith("test:")
    )
    assert set(written[0]) == {"w1e-s0", "teacher"}
    assert set(written[0]["teacher"]) == {"filled", "raw"}


# ---- phone-1 block, novelty, gate inputs ----


def test_novel_full_is_aligned_to_all_rows():
    legit = np.array([True, False, True, True])
    out = job.novel_full(legit, np.array([True, False, True]))
    assert out.tolist() == [True, False, False, True]


def test_phone_block_counts_legit_and_scam_at_tau():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 2.0, 0.2, 3.0, 5.0, -1.0])
    novel = np.array([False, True, False, True, True, True])
    out = job.phone_block(labels, scores, tau=1.0, novel=novel)
    assert out["legit_fpr"]["k"] == 2 and out["legit_fpr"]["n"] == 4
    assert out["legit_fpr_template_novel"]["k"] == 2 and out["legit_fpr_template_novel"]["n"] == 2
    assert out["scam_recall"]["k"] == 1 and out["scam_recall"]["n"] == 2


def test_gate_inputs_read_the_block_and_phone_counts():
    rng = np.random.default_rng(2)
    sets = _fake_sets()
    flags = {n: rng.random(len(sets[n]["labels"])) > 0.5 for n in S.TEST_SETS}
    block = S.arm_block(sets, flags, min_recall=job.G3_MIN_RECALL)
    phone = job.phone_block(
        np.array([0] * 9 + [1]), rng.normal(size=10), 0.0, np.zeros(10, dtype=bool)
    )
    n_pos = len(flags[S.SET_GL_POS])
    gi = job.gate_inputs(block, phone, n_pos)
    assert gi["fp_neg"] == block["groundlm_neg_fp"]["k"]
    assert gi["n_pos"] == n_pos
    assert gi["donor_fpr"] == phone["legit_fpr"]["rate"]
    assert gi["f1_clean"] == block["campaign_clean"]["f1"]


# ---- checkpoint index ----


def test_index_checkpoints_rejects_a_missing_pin(tmp_path):
    with pytest.raises(FileNotFoundError, match="found 0"):
        job.index_checkpoints(tmp_path, {"w1e-s0": "0" * 64})


# ---- privacy: a synthetic run writes every output; no sentinel text or sender may appear ----


def test_privacy_scan_of_every_output(tmp_path):
    raw = _donor_df(tag_every=2)
    filled, raw_lab, counts, _ = job.calib_frame_pair(raw)
    cal_f = filled[filled["label"] == 0]
    cal_r = raw_lab[raw_lab["label"] == 0]
    cal_f_texts = list(cal_f["text"])
    cal_r_texts = list(cal_r["text"])
    p1 = _donor_df(tag_every=3)
    p1_f, p1_r, _c, _t = job.calib_frame_pair(p1)
    p1_labels = {
        job.P1_FILLED: p1_f["label"].to_numpy().astype(np.int64),
        job.P1_RAW: p1_r["label"].to_numpy().astype(np.int64),
    }
    n_p1 = len(p1_f)
    rng = np.random.default_rng(4)
    sets = _fake_sets()
    clean = np.ones(len(sets[S.SET_CAMP]["labels"]), dtype=bool)
    clean[::3] = False
    results = {}
    frames = []
    for name in ("w1e-s0", job.TEACHER):
        scores = {n: rng.normal(size=len(sets[n]["labels"])) for n in S.TEST_SETS}
        scores[job.CAMP_FILLED] = rng.normal(size=len(sets[S.SET_CAMP]["labels"]))
        scores[job.P1_FILLED] = rng.normal(size=n_p1)
        scores[job.P1_RAW] = rng.normal(size=n_p1)
        cal = {
            job.CAL_FILLED: rng.normal(size=len(cal_f_texts)),
            job.CAL_RAW: rng.normal(size=len(cal_r_texts)),
        }
        taus = {
            "filled": job.calibration_tau(cal[job.CAL_FILLED]),
            "raw": job.calibration_tau(cal[job.CAL_RAW]),
        }
        novel_f = np.zeros(n_p1, dtype=bool)
        novel_f[np.asarray(p1_labels[job.P1_FILLED] == 0)] = True
        primary = job.arm_result(
            sets,
            scores,
            taus["filled"]["tau"],
            job.P1_FILLED,
            p1_labels[job.P1_FILLED],
            novel_f,
            clean,
        )
        secondary = job.arm_result(
            sets, scores, taus["raw"]["tau"], job.P1_RAW, p1_labels[job.P1_RAW], novel_f, clean
        )
        results[name] = {"sha256": "x", "tau": taus, "primary": primary, "secondary": secondary}
        frames.append(
            job.probs_frame(
                name,
                sets,
                {**cal, **scores},
                {job.P1_FILLED: p1_labels[job.P1_FILLED], job.P1_RAW: p1_labels[job.P1_RAW]},
                clean,
            )
        )
    probs = pd.concat(frames, ignore_index=True)
    assert set(probs.columns) == {
        "model",
        "set",
        "id",
        "label",
        "margin",
        "category",
        "clean",
        "xlmr_pred",
        "p_scam",
    }
    rows = [results["w1e-s0"]["primary"]["gate_inputs"]]
    payload = {
        "job": job.JOB,
        "status": "scored",
        "verdict": job.verdict(rows * 3),
        "gates": job.aggregate_gates(rows * 3),
        "models": results,
        "prep": {**counts, "cal_legit": len(cal_f)},
    }
    prep = payload["prep"]
    job.write_outputs(tmp_path, payload, probs, prep)
    # a stopped run writes the same kinds of files
    stop = tmp_path / "stopped"
    job.write_outputs(stop, {"job": job.JOB, "status": job.STOP_STATUS, "counts": prep}, None, prep)

    secrets = [SENTINEL, SENDER_SENTINEL]
    texts = cal_f_texts[:5] + cal_r_texts[:5] + list(p1["text"])[:5]
    scanned = 0
    for path in list(tmp_path.rglob("*")) + list(stop.rglob("*")):
        if not path.is_file():
            continue
        scanned += 1
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            cells = [str(v) for v in df.to_numpy().ravel().tolist()]
            blob = " ".join(cells + [str(c) for c in df.columns])
        else:
            blob = path.read_text(encoding="utf-8")
        for s in secrets:
            assert s not in blob, path.name
        for text in texts:
            assert text not in blob, path.name
    assert scanned >= 5
    assert "stopped: insufficient data" in (stop / "metrics.json").read_text()


def test_render_report_names_no_text():
    payload = {
        "verdict": "fail",
        "models": {
            "w1e-s0": {
                "primary": {
                    "tau": -1.0,
                    "gate_inputs": {
                        "fp_neg": 5,
                        "tp_smish": 200,
                        "tp_pos": 90,
                        "n_pos": 91,
                        "recall_pos": 0.989,
                        "f1_clean": 0.98,
                        "donor_fpr": 0.004,
                        "donor_k": 1,
                        "donor_n": 250,
                    },
                },
                "secondary": {
                    "tau": -2.0,
                    "gate_inputs": {
                        "fp_neg": 6,
                        "tp_smish": 190,
                        "tp_pos": 80,
                        "n_pos": 91,
                        "recall_pos": 0.879,
                        "f1_clean": 0.97,
                        "donor_fpr": 0.01,
                        "donor_k": 2,
                        "donor_n": 250,
                    },
                },
            }
        },
    }
    text = job.render_report(payload)
    assert SENTINEL not in text and SENDER_SENTINEL not in text
    assert "Teacher row is a reference only" in text


def test_error_is_redacted_when_a_run_raises_text(monkeypatch):
    def leaky(ctx):
        raise ValueError(f"could not read {SENTINEL} from {SENDER_SENTINEL}")

    monkeypatch.setattr(job, "_run", leaky)

    class Ctx:
        def __init__(self):
            self.lines: list[str] = []

        def log(self, msg):
            self.lines.append(str(msg))

    ctx = Ctx()
    with pytest.raises(job.RedactedError) as info:
        job.run(ctx)
    assert SENTINEL not in str(info.value) and SENDER_SENTINEL not in str(info.value)
    blob = "\n".join(ctx.lines)
    assert SENTINEL not in blob and SENDER_SENTINEL not in blob
    assert "ValueError (message redacted)" in str(info.value)


# ---- build wiring ----


def test_job_dependencies_include_the_encoder_and_suite_modules():
    text = (ROOT / "kaggle" / "build_kernel.py").read_text()
    assert "'sms_inbox_calibration': (" in text
    assert "'kaggle/jobs/sms_encoder_exposure.py'" in text
    assert "'kaggle/jobs/sms_teacher_suite.py'" in text


def test_teacher_suite_import_is_the_same_module_the_job_uses():
    assert job.S is S
    assert X.TEACHER_DIR == "best_checkpoint"
