"""Teacher suite W1a (docs/TEACHER_SUITE_PLAN.md, frozen 2026-11-12). Scoring only, no training.

Scores the mmBERT-base fold teacher (source_holdout:holdout_scamshield, seed 0, sha256 673b970e...)
and the shipped v0 student (ONNX sha 5341e14c...) on the full SMS suite, applies the
calibration-only
thresholds from the plan, and reports the arms T, S-and-T, S-or-T under both threshold rules.

Order of operations (enforced):
1. Teacher checkpoint and provenance; v0 ONNX located by sha256. No test data read yet.
2. Teacher training partition (24,391 rows) and calibration partition (2,754 rows) rebuilt with
   rakshak.eval.data.load_runs; both sizes asserted.
3. Teacher scores the calibration split only. tau_maxF1 and tau_cal1% are computed, and
   thresholds.json is written, by sms_retrain.run_seed_pipeline, before any test set is scored.
4. Teacher then scores the test sets; the v0 student scores the same rows at its shipped threshold.
5. Arms, gates, paired McNemar against the released GroundLM XLM-R decision, intervals, outputs.

Input form: rakshak.eval.data.canon_text(stored text), the form sms_teacher trained on. The text is
the
stored scrubbed form for every set. No inbox rows and no surrogate fill are used here.

PRIVACY (binding). No message text, sender string or inbox content is written anywhere.
thresholds.json,
metrics.json, report.md and probs.parquet carry ids, labels, categories, scores, thresholds, counts
and
intervals only. Errors are logged as exception type and frames (X.run redaction).

Launch (T4, internet on; the data, the teacher kernel and the student kernel):
  kaggle/launch.sh sms_teacher_suite --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data \
      --kernels maadhavvh/rakshak-sms-student maadhavvh/rakshak-sms-teacher \
      --job-args="--teacher-dir /kaggle/input --v0-onnx /kaggle/input"

Flags (--job-args="FLAGS"):
  --teacher-dir PATH        folder searched for best_checkpoint/model.safetensors with the pinned
  sha256
  --v0-onnx PATH            folder searched for best_student.onnx with the pinned sha256
  --max-len INT             teacher token limit (default 128)
  --eval-batch-size INT     teacher inference batch (default 128)
  --n-boot INT              bootstrap resamples (default 2000)
  --checkpoint-sha256 HEX   sha256 of the checkpoint's model.safetensors to score (default: the base
                            teacher's pin). Any encoder fold checkpoint works: its best_checkpoint
                            folder
                            under --teacher-dir is located by this digest, and its manifest.json
                            (sms_encoder_fold output) gives the provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sms_retrain as base
import sms_surrogate_d0 as d0
import sms_teacher_xsource as X
from _common import env_report, seed_everything, timed
from baseline_v0 import SMS_THRESHOLD, _onnx_session, rate_row, score_sms_texts

from rakshak.eval import data as D
from rakshak.eval import metrics as M

JOB = "sms_teacher_suite"
PLAN = "docs/TEACHER_SUITE_PLAN.md"
DEFAULT_ROOT = "/kaggle/input"
MAX_LEN = 128
EVAL_BATCH = 128
N_BOOT = 2000
BOOT_SEED = 0

RUN_NAME = "source_holdout:holdout_scamshield"
TRAIN_ROWS = 24391
CAL_ROWS = 2754
GL_NEG_ROWS = 1133
GL_POS_ROWS = 91
SX_SMISH_ROWS = 226

STUDENT_TAU = 0.35127517580986023  # plan section 3; must equal the shipped threshold
FPR_CAL = 0.01
RECALL_LOW_PCT = 95  # tau_low: largest threshold with cal recall >= 0.95 (amendment 1)
G1_MAX_FP = 11
G2_MIN_TP = 157
G3_MIN_RECALL = 0.95
G4_MIN_F1 = 0.9713

SET_GL_NEG = "groundlm_neg"
SET_GL_POS = "groundlm_pos"
SET_SX_SMISH = "smishx_smishing"
SET_SX_HAM = "smishx_ham"
SET_CAMP = "campaign_test"
SET_LANG = "lang_shift"
SET_TEMP = "temporal"
TEST_SETS = (SET_CAMP, SET_LANG, SET_TEMP, SET_GL_NEG, SET_GL_POS, SET_SX_SMISH, SET_SX_HAM)
ARMS = ("T", "SandT", "SorT")
RULES = ("maxf1", "cal1pct")

assert STUDENT_TAU == SMS_THRESHOLD, "student threshold must be the shipped one"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--teacher-dir", default=DEFAULT_ROOT)
    p.add_argument("--v0-onnx", default=DEFAULT_ROOT)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--checkpoint-sha256", default=X.TEACHER_SHA)
    args = p.parse_args(list(argv or []))
    sha = args.checkpoint_sha256
    if len(sha) != 64 or set(sha) - set("0123456789abcdef"):
        p.error("--checkpoint-sha256 must be 64 lowercase hex characters")
    if args.max_len < 1 or args.eval_batch_size < 1 or args.n_boot < 1:
        p.error("--max-len, --eval-batch-size and --n-boot must be positive")
    return args


# ---- pure helpers (unit-tested in tests/test_sms_teacher_suite.py) ----


def threshold_pair(s_cal: np.ndarray, y_cal: np.ndarray) -> dict:
    """The two teacher thresholds, from the calibration split only (plan section 3)."""
    y = np.asarray(y_cal).astype(np.int64)
    s = np.asarray(s_cal, dtype=np.float64)
    tau_f1, best_f1 = M.threshold_max_f1(y, s)
    cal1 = M.threshold_for_fpr(y, s, FPR_CAL)
    if not np.isfinite(tau_f1) or not np.isfinite(cal1["threshold"]):
        raise ValueError("a teacher threshold is not finite on the calibration split")
    pos_scores = np.sort(s[y == 1])[::-1]
    k = -(-len(pos_scores) * RECALL_LOW_PCT // 100)  # ceil(n x 0.95), integer arithmetic
    tau_low = float(pos_scores[k - 1])
    return {
        "maxf1": float(tau_f1),
        "maxf1_cal_f1": float(best_f1),
        "cal1pct": float(cal1["threshold"]),
        "cal1pct_cal_fpr": float(cal1["fpr"]),
        "cal1pct_cal_tpr": float(cal1["tpr"]),
        "low95": tau_low,
        "low95_cal_recall": float(np.mean(s[y == 1] >= tau_low)),
    }


def arm_flags(p_s: np.ndarray, p_t: np.ndarray, tau_t: float) -> dict[str, np.ndarray]:
    """Flags per arm. The student alone (S) is a reference, not an arm. ``>=`` is a flag."""
    p_s = np.asarray(p_s, dtype=np.float64)
    p_t = np.asarray(p_t, dtype=np.float64)
    s_hit = p_s >= STUDENT_TAU
    t_hit = p_t >= tau_t
    return {"T": t_hit, "SandT": s_hit & t_hit, "SorT": s_hit | t_hit, "S": s_hit}


def evaluate_gates(
    fp_neg: int, n_neg: int, tp_smish: int, n_smish: int, tp_pos: int, n_pos: int, f1_clean: float,
    min_recall: float = G3_MIN_RECALL,
) -> dict:
    """G1 to G4 of plan section 6. An arm passes only when all four hold."""
    g1 = fp_neg <= G1_MAX_FP
    g2 = tp_smish >= G2_MIN_TP
    g3 = n_pos > 0 and (tp_pos / n_pos) >= min_recall
    g4 = bool(np.isfinite(f1_clean)) and f1_clean >= G4_MIN_F1
    return {
        "G1_groundlm_neg_fp": {"k": int(fp_neg), "n": int(n_neg), "max": G1_MAX_FP,
                               "pass": bool(g1)},
        "G2_smishx_smishing_tp": {"k": int(tp_smish), "n": int(n_smish), "min": G2_MIN_TP,
                                  "pass": bool(g2)},
        "G3_groundlm_pos_recall": {"k": int(tp_pos), "n": int(n_pos), "min_rate": min_recall,
                                   "pass": bool(g3)},
        "G4_campaign_clean_f1": {"value": float(f1_clean), "min": G4_MIN_F1, "pass": bool(g4)},
        "arm_pass": bool(g1 and g2 and g3 and g4),
    }


def _f1(tp: float, fp: float, fn: float) -> float:
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else float("nan")


def _precision(tp: float, fp: float, fn: float) -> float:
    den = tp + fp
    return tp / den if den > 0 else float("nan")


def cluster_ci(
    y: np.ndarray, pred: np.ndarray, codes: np.ndarray, metric: str,
    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> list[float]:
    """Campaign-clustered bootstrap percentile 95 % interval for F1 or precision.
    Each resample draws clusters with replacement; a row's weight is its cluster's draw count."""
    if metric not in ("f1", "precision"):
        raise ValueError(f"unknown metric {metric!r}")
    pos = np.asarray(y).astype(np.int64) == 1
    hit = np.asarray(pred, dtype=bool)
    codes = np.asarray(codes, dtype=np.int64)
    if codes.size == 0:
        return [float("nan"), float("nan")]
    n_clusters = int(codes.max()) + 1
    rng = np.random.default_rng(seed)
    fn_of = _f1 if metric == "f1" else _precision
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
        w = counts[codes].astype(np.float64)
        tp = float(np.sum(w * (hit & pos)))
        fp = float(np.sum(w * (hit & ~pos)))
        fn = float(np.sum(w * (~hit & pos)))
        draws[b] = fn_of(tp, fp, fn)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def clean_mask(ids: np.ndarray, texts_hit: np.ndarray, train_ids: set[str]) -> np.ndarray:
    """A row is clean when neither its id nor its canonical text is in the training partition."""
    id_hit = np.fromiter((str(i) in train_ids for i in ids), dtype=bool, count=len(ids))
    return ~(id_hit | np.asarray(texts_hit, dtype=bool))


def arm_block(sets: dict[str, dict], flags: dict[str, np.ndarray],
              min_recall: float = G3_MIN_RECALL) -> dict:
    """Counts, rates, F1 and precision with intervals, gates, paired and per-category tables for one
    arm (or the student reference) over every set."""
    neg, pos = flags[SET_GL_NEG], flags[SET_GL_POS]
    smish, ham = flags[SET_SX_SMISH], flags[SET_SX_HAM]
    camp = sets[SET_CAMP]
    clean = camp["clean"]
    f_camp = flags[SET_CAMP]
    all_f1 = M.metrics_from_counts(*M.confusion_counts(camp["labels"], f_camp))
    clean_f1 = M.metrics_from_counts(*M.confusion_counts(camp["labels"][clean], f_camp[clean]))
    lang, temp = sets[SET_LANG], sets[SET_TEMP]
    f_lang, f_temp = flags[SET_LANG], flags[SET_TEMP]
    lang_f1 = M.metrics_from_counts(*M.confusion_counts(lang["labels"], f_lang))
    temp_prec = M.metrics_from_counts(*M.confusion_counts(temp["labels"], f_temp))
    ref_neg, ref_pos = sets[SET_GL_NEG]["xlmr"], sets[SET_GL_POS]["xlmr"]
    f1_clean = clean_f1["f1"]
    gates = evaluate_gates(
        int(neg.sum()), len(neg), int(smish.sum()), len(smish),
        int(pos.sum()), len(pos), f1_clean, min_recall=min_recall,
    )
    return {
        "groundlm_neg_fp": rate_row(int(neg.sum()), len(neg)),
        "groundlm_pos_recall": rate_row(int(pos.sum()), len(pos)),
        "smishx_smishing_recall": rate_row(int(smish.sum()), len(smish)),
        "smishx_ham_fpr": rate_row(int(ham.sum()), len(ham)),
        "campaign_all": {**all_f1, "f1_ci95": cluster_ci(
            camp["labels"], f_camp, camp["codes"], "f1")},
        "campaign_clean": {**clean_f1, "n_clean": int(clean.sum()), "f1_ci95": cluster_ci(
            camp["labels"][clean], f_camp[clean], camp["codes"][clean], "f1")},
        "lang_shift": {**lang_f1, "f1_ci95": cluster_ci(
            lang["labels"], f_lang, lang["codes"], "f1")},
        "temporal": {**temp_prec, "precision_ci95": cluster_ci(
            temp["labels"], f_temp, temp["codes"], "precision")},
        "gates": gates,
        "paired_groundlm_negatives": base.paired_table(neg, ref_neg),
        "paired_groundlm_positives": base.paired_table(pos, ref_pos),
        "groundlm_neg_by_category": base.bucket_counts(sets[SET_GL_NEG]["category"], neg, ref_neg),
    }


def render_report(payload: dict) -> str:
    lines = [f"# {JOB} (plan {PLAN})", "", "| arm | G1 GL-neg FP | G2 SmishX TP | G3 GL-pos recall "
             "| G4 clean F1 | pass |", "|---|---|---|---|---|---|"]
    for key, block in payload["arms"].items():
        g = block["gates"]
        lines.append(
            f"| {key} | {g['G1_groundlm_neg_fp']['k']}/{g['G1_groundlm_neg_fp']['n']} "
            f"| {g['G2_smishx_smishing_tp']['k']}/{g['G2_smishx_smishing_tp']['n']} "
            f"| {g['G3_groundlm_pos_recall']['k']}/{g['G3_groundlm_pos_recall']['n']} "
            f"| {g['G4_campaign_clean_f1']['value']:.4f} | {g['arm_pass']} |"
        )
    lines += ["", "Thresholds: " + json.dumps(payload["thresholds"], sort_keys=True), ""]
    return "\n".join(lines)


def _write(work: Path, payload: dict, thresholds: dict, probs: pd.DataFrame) -> None:
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(work / "metrics.json")
    (work / "report.md").write_text(render_report(payload))
    probs.to_parquet(work / "probs.parquet", index=False)


def band_flags(p: np.ndarray, tau_high: float, tau_low: float) -> dict[str, np.ndarray]:
    """Two-band output (amendment 1): SCAM at or above tau_high, VERIFY from tau_low up to tau_high,
    BENIGN below tau_low. When tau_low > tau_high the VERIFY band is empty."""
    p = np.asarray(p, dtype=np.float64)
    scam = p >= tau_high
    verify = (p >= tau_low) & ~scam
    return {"scam": scam, "verify": verify, "benign": ~(scam | verify)}


def band_for_arm(p_s: np.ndarray, p_t: np.ndarray, tau_high: float, tau_low: float,
                 arm: str) -> dict[str, np.ndarray]:
    """Bands for the teacher alone (T) or the teacher gated by the student (SandT)."""
    bands = band_flags(p_t, tau_high, tau_low)
    if arm == "SandT":
        gate = np.asarray(p_s, dtype=np.float64) >= STUDENT_TAU
        bands = {"scam": bands["scam"] & gate, "verify": bands["verify"] & gate}
        bands["benign"] = ~(bands["scam"] | bands["verify"])
    elif arm != "T":
        raise ValueError(f"unknown two-band arm {arm!r}")
    return bands


def two_band_block(sets: dict[str, dict], bands: dict[str, dict[str, np.ndarray]]) -> dict:
    """Counts per band for each set, with Wilson intervals on the visible and verify rates. On
    GroundLM negatives, 'visible' is the visible false-positive count and 'verify' the verify-band
    count, reported separately."""
    out: dict[str, Any] = {}
    for name, b in bands.items():
        n = len(b["scam"])
        row: dict[str, Any] = {
            "n": n,
            "visible": rate_row(int(b["scam"].sum()), n),
            "verify": rate_row(int(b["verify"].sum()), n),
            "benign": int(b["benign"].sum()),
        }
        if name == SET_CAMP:
            clean = sets[name]["clean"]
            n_clean = int(clean.sum())
            row["clean"] = {
                "n": n_clean,
                "visible": rate_row(int(b["scam"][clean].sum()), n_clean),
                "verify": rate_row(int(b["verify"][clean].sum()), n_clean),
                "benign": int(b["benign"][clean].sum()),
            }
        out[name] = row
    return out


def build_sets(root, sms, labels, texts, ids, camp_run, lang_run, temp_run, train_ids,
               train_keys) -> tuple[dict, dict]:
    """The scored sets of plan section 4 (campaign test, lang_shift, temporal, and the external
    GroundLM and SmishX sets), each with its clean mask, and the GroundLM XLM-R join statistics.
    Takes the teacher training partition as ``train_ids`` and ``train_keys``. No model is used."""
    # 3. external sets (scrubbed text only, eval_ok, as the other SMS jobs)
    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    gl_avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    gl_ok = gl["eval_ok"].fillna(False).astype(bool).to_numpy()
    gl_pol = gl["policy_negative"].fillna(False).astype(bool).to_numpy()
    gl_label = gl["label_binary"].to_numpy().astype(np.int64)
    neg_idx = np.flatnonzero(gl_pol & gl_avail & gl_ok)
    pos_idx = np.flatnonzero((gl_label == 1) & gl_avail & gl_ok)
    if len(neg_idx) != GL_NEG_ROWS or len(pos_idx) != GL_POS_ROWS:
        raise ValueError(f"GroundLM sets {len(neg_idx)}/{len(pos_idx)}, expected "
                         f"{GL_NEG_ROWS}/{GL_POS_ROWS}")
    gl_text = base.str_texts(gl["text"].to_numpy())
    gl_id = gl["id"].astype(str).to_numpy()
    gl_key = pd.to_numeric(gl["row_key"], errors="coerce").to_numpy(dtype=np.float64)
    gl_cat = gl["category"].astype(str).to_numpy()

    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    sx_ok = sx["eval_ok"].fillna(False).astype(bool).to_numpy()
    sx_fine = sx["label_fine"].astype(str).to_numpy()
    smish_idx = np.flatnonzero(sx_ok & sx_avail & (sx_fine == "smishing"))
    ham_idx = np.flatnonzero(sx_ok & sx_avail & (sx_fine == "legitimate"))
    if len(smish_idx) != SX_SMISH_ROWS:
        raise ValueError(f"SmishX smishing {len(smish_idx)}, expected {SX_SMISH_ROWS}")
    sx_text = base.str_texts(sx["text"].to_numpy())
    sx_id = sx["id"].astype(str).to_numpy()
    sx_label = sx["label_binary"].to_numpy().astype(np.int64)

    # GroundLM XLM-R decisions joined on row_key (released model, its own threshold 0.5)
    from rakshak.data import retrain_prep as RP

    xl = RP.load_indian_sources(root).get("groundlm_xlmr_probs")
    if xl is None:
        raise FileNotFoundError("groundlm_xlmr_probs is not in the data root")
    needed = np.zeros(len(gl), dtype=bool)
    needed[neg_idx] = True
    needed[pos_idx] = True
    xlmr_all, join = base.xlmr_reference(gl, xl, needed)

    # 4. the sets (no test score yet). Clean = not in the teacher's training partition.
    def campaign_style(run_, label: str) -> dict:
        pos = run_.test
        t = [texts[i] for i in pos]
        hit = X.contamination_mask(t, train_keys)
        codes = pd.factorize(pd.Series(D.cluster_values(sms, pos, D.CLUSTER_COLUMNS["sms"])))[0]
        return {
            "ids": ids[pos], "labels": labels[pos], "texts": t, "codes": codes,
            "clean": clean_mask(ids[pos], hit, train_ids), "category": np.array([""] * len(pos)),
            "row_key": np.full(len(pos), np.nan), "xlmr": None,
            "label_fine": np.array([""] * len(pos)),
            "name": label,
        }

    def external_set(texts_: list[str], ids_: np.ndarray, labels_: np.ndarray, cats: np.ndarray,
                     keys: np.ndarray, xlmr: np.ndarray | None, name: str) -> dict:
        hit = X.contamination_mask(texts_, train_keys)
        return {
            "ids": ids_, "labels": labels_, "texts": texts_, "codes": np.arange(len(texts_)),
            "clean": ~hit, "category": cats, "row_key": keys, "xlmr": xlmr,
            "label_fine": np.array([""] * len(texts_)), "name": name,
        }

    sets: dict[str, dict] = {
        SET_CAMP: campaign_style(camp_run, SET_CAMP),
        SET_LANG: campaign_style(lang_run, SET_LANG),
        SET_TEMP: campaign_style(temp_run, SET_TEMP),
        SET_GL_NEG: external_set(
            [gl_text[i] for i in neg_idx], gl_id[neg_idx], np.zeros(len(neg_idx), dtype=np.int64),
            gl_cat[neg_idx], gl_key[neg_idx], xlmr_all[neg_idx], SET_GL_NEG),
        SET_GL_POS: external_set(
            [gl_text[i] for i in pos_idx], gl_id[pos_idx], np.ones(len(pos_idx), dtype=np.int64),
            gl_cat[pos_idx], gl_key[pos_idx], xlmr_all[pos_idx], SET_GL_POS),
        SET_SX_SMISH: external_set(
            [sx_text[i] for i in smish_idx], sx_id[smish_idx], sx_label[smish_idx],
            np.array([""] * len(smish_idx)), np.full(len(smish_idx), np.nan), None, SET_SX_SMISH),
        SET_SX_HAM: external_set(
            [sx_text[i] for i in ham_idx], sx_id[ham_idx], sx_label[ham_idx],
            np.array([""] * len(ham_idx)), np.full(len(ham_idx), np.nan), None, SET_SX_HAM),
    }
    for name in TEST_SETS:
        if len(sets[name]["texts"]) == 0:
            raise ValueError(f"set {name} is empty")
    return sets, join


def fpr_threshold(neg_scores: Sequence[float], max_fpr: float = FPR_CAL) -> float:
    """Smallest threshold with at most floor(max_fpr x n) negatives at score >= tau. Works on
    probabilities or on logit margins (flag when score >= tau). -inf when every negative may
    flag."""
    s = np.sort(np.asarray(neg_scores, dtype=np.float64))[::-1]
    n = int(s.size)
    if n == 0:
        raise ValueError("no negatives to set a threshold on")
    k = int(math.floor(max_fpr * n))
    if k >= n:
        return float("-inf")
    return float(np.nextafter(s[k], np.inf))


def negative_folds(texts: Sequence[str]) -> tuple[np.ndarray, dict]:
    """Two template-disjoint folds of the GroundLM negatives (Amendment 2): the existing clusterer
    with its default threshold, then whole clusters assigned to the lighter fold. Deterministic."""
    from rakshak.eval import groupneg as GN

    clusters = GN.negative_clusters(list(texts))
    folds = np.asarray(GN.assign_folds(clusters, 2), dtype=np.int64)
    sizes = sorted(int(np.sum(folds == f)) for f in (0, 1))
    n_clusters = int(len(set(clusters)))
    info = {
        "clusters": n_clusters,
        "fold_sizes": sizes,
        "matches_amendment2": bool(sizes == [566, 567] and n_clusters == 653),
    }
    return folds, info


def fold_protocol(
    neg: Sequence[float], folds: np.ndarray, external: dict[str, Sequence[float]]
) -> dict:
    """Amendment 2 calibration-only cross-fit (docs/TEACHER_SUITE_PLAN.md). For fold f, tau_f is the
    smallest threshold with at most floor(1 % x |f|) false positives on fold f's negatives (no
    positives). tau_f is applied to the other fold's negatives; the two held-out counts are pooled.
    External sets are reported at tau_0 and tau_1 separately. Works on probabilities or logits."""
    neg = np.asarray(neg, dtype=np.float64)
    folds = np.asarray(folds)
    taus = {f: fpr_threshold(neg[folds == f]) for f in (0, 1)}
    by_fold: dict[str, Any] = {}
    pooled_fp = 0
    pooled_n = 0
    for f in (0, 1):
        held = neg[folds != f]
        fp = int(np.sum(held >= taus[f]))
        by_fold[str(f)] = {"tau": taus[f], "heldout_negatives": int(held.size), "fp": fp}
        pooled_fp += fp
        pooled_n += int(held.size)
    ext = {}
    for name, arr in external.items():
        arr = np.asarray(arr, dtype=np.float64)
        ext[name] = {"n": int(arr.size), "at_tau_0": int(np.sum(arr >= taus[0])),
                     "at_tau_1": int(np.sum(arr >= taus[1]))}
    return {
        "tau": {"0": taus[0], "1": taus[1]},
        "pooled_negative_fp": pooled_fp,
        "pooled_negatives": pooled_n,
        "pooled_negative_fpr": (pooled_fp / pooled_n) if pooled_n else None,
        "by_fold": by_fold,
        "external": ext,
    }


def teacher_margins(model, tokenizer, texts: Sequence[str], *, device, max_len: int,
                    eval_batch_size: int) -> np.ndarray:
    """Logit margin logit_1 - logit_0 per text (canon input, fp16 autocast, as teacher_probs). The
    probability is the sigmoid of the margin; the margin does not saturate at float32 1.0."""
    import sms_teacher as T

    cfg = T.Config(fp16=True, max_len=max_len, batch_size=eval_batch_size,
                   eval_batch_size=eval_batch_size, patience=0, adv_aug=0.0, train_canon=True,
                   log=lambda _msg: None)
    logits = T.predict_logits(model, tokenizer, [D.canon_text(t) for t in texts], cfg, device)
    return logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)


def locate_checkpoint(root: Path, sha: str) -> Path:
    """The best_checkpoint folder under ``root`` whose model.safetensors has sha256 ``sha``. The
    base
    teacher's pin uses its own locator (X.locate_teacher), so the default path is unchanged."""
    if sha == X.TEACHER_SHA:
        return X.locate_teacher(root)
    if not root.exists():
        raise FileNotFoundError(f"--teacher-dir {root} does not exist")
    folders = sorted({
        Path(d) for d, _dirs, files in os.walk(root)
        if Path(d).name == X.TEACHER_DIR and X.TEACHER_WEIGHTS in files
    })
    if not folders:
        raise FileNotFoundError(f"no {X.TEACHER_DIR}/{X.TEACHER_WEIGHTS} under {root}")
    found = {str(f): X.file_sha256(f / X.TEACHER_WEIGHTS) for f in folders}
    good = [f for f in folders if found[str(f)] == sha]
    if not good:
        shas = sorted(v[:12] for v in found.values())
        raise ValueError(f"checkpoint sha256 mismatch: no checkpoint under {root} has the digest "
                         f"{sha[:12]} (found {shas})")
    if len(good) > 1:
        raise FileNotFoundError(f"{len(good)} checkpoints match the digest {sha[:12]}")
    return good[0]


def checkpoint_provenance(teacher_dir: Path, sha: str) -> dict:
    """Run, model, seed and input form of a checkpoint. An sms_encoder_fold manifest.json next to
    best_checkpoint is used for any digest other than the base teacher's; otherwise the teacher
    metrics.json (the unchanged default path)."""
    manifest = teacher_dir.parent / "manifest.json"
    if sha != X.TEACHER_SHA and manifest.is_file():
        m = json.loads(manifest.read_text())
        ck = m.get("checkpoint") or {}
        if ck.get("sha256") != sha:
            raise ValueError("manifest.json checkpoint.sha256 differs from --checkpoint-sha256")
        rows = m.get("rows") or {}
        if int(rows.get("train", -1)) != TRAIN_ROWS or int(rows.get("cal", -1)) != CAL_ROWS:
            raise ValueError(f"manifest rows train {rows.get('train')} "
                             f"cal {rows.get('cal')} differ "
                             f"from the plan's {TRAIN_ROWS} and {CAL_ROWS}")
        return {"run": str(m["split"]), "model": str(m["model"]), "seed": int(m["seed"]),
                "train_input": str(m.get("train_input")),
                "param_count": int(ck.get("param_count", -1)),
                "source": "manifest.json"}
    metrics_path = teacher_dir.parent / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            "checkpoint metrics.json or manifest.json must sit next to best_checkpoint"
        )
    return {**X.teacher_meta(json.loads(metrics_path.read_text())), "source": "metrics.json"}


# ---- the job ----


def run(ctx) -> bool:
    """Runs _run; any exception leaves as the redacted error of the Xsource job."""
    failure = None
    try:
        return _run(ctx)
    except Exception as exc:
        failure = exc
        for line in X.redacted_frames(exc):
            ctx.log(f"  at {line}")
        ctx.log(f"ERROR {X.redacted_error(exc)}")
    raise X.RedactedError(X.redacted_error(failure)) from None


def _run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seed_everything(0)
    env = env_report()
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})
    work = ctx.work

    # 1. teacher checkpoint and provenance (pinned sha, run, input form); v0 ONNX by sha
    sha = args.checkpoint_sha256
    teacher_dir = locate_checkpoint(Path(args.teacher_dir), sha)
    prov = checkpoint_provenance(teacher_dir, sha)
    if prov["run"] != RUN_NAME:
        raise ValueError(f"teacher run {prov['run']} differs from the plan's {RUN_NAME}")
    X.check_train_input(prov["train_input"], "canon")
    v0_path = d0.locate_v0(Path(args.v0_onnx))
    ctx.log(f"checkpoint found, sha {sha[:12]}, run {RUN_NAME} ({prov['source']}); v0 found")

    # 2. data and partitions; sizes asserted against the teacher's own logits counts
    root = D.resolve_data_root(None)
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    texts = base.str_texts(sms["text"])
    ids = sms["id"].astype(str).to_numpy()
    mask = D.eval_mask(sms)
    runs = D.load_runs(
        root, sms, patterns=["campaign", "lang_shift", "temporal", "source_holdout"],
        labels=labels, mask=mask,
    )

    def one(stem: str) -> Any:
        found = [r for r in runs if r.name.split(":")[0] == stem]
        if len(found) != 1:
            raise ValueError(f"expected one {stem} run, found {len(found)}")
        return found[0]

    teacher_run = [r for r in runs if r.name == RUN_NAME]
    if len(teacher_run) != 1:
        raise ValueError(f"teacher run {RUN_NAME} is not in the split files")
    teacher_run = teacher_run[0]
    if len(teacher_run.train) != TRAIN_ROWS or len(teacher_run.cal) != CAL_ROWS:
        raise ValueError(
            f"teacher partitions train {len(teacher_run.train)} cal {len(teacher_run.cal)}, "
            f"expected {TRAIN_ROWS} and {CAL_ROWS}"
        )
    camp_run, lang_run, temp_run = one("campaign"), one("lang_shift"), one("temporal")
    train_ids = set(ids[teacher_run.train].tolist())
    train_keys = X.canon_keys([texts[i] for i in teacher_run.train])

    sets, join = build_sets(root, sms, labels, texts, ids, camp_run, lang_run, temp_run,
                            train_ids, train_keys)
    ctx.log(f"XLM-R join: {join}")
    ctx.log("sets: " + ", ".join(f"{n} {len(sets[n]['texts'])}" for n in TEST_SETS))
    contamination = {
        n: {"n": len(s["texts"]), "contaminated": int((~s["clean"]).sum())} for n, s in sets.items()
    }

    # 5. teacher: calibration scores, thresholds, thresholds.json, then the test sets
    model, tokenizer, device = X.load_teacher(teacher_dir)
    cal_texts = [texts[i] for i in teacher_run.cal]
    y_cal = labels[teacher_run.cal]
    thr: dict[str, Any] = {}

    def rule(_legit: np.ndarray, s_cal: np.ndarray, y: np.ndarray) -> tuple[float, bool]:
        pair = threshold_pair(s_cal, y)
        thr.update(pair)
        return pair["maxf1"], False

    def score(_model, name: str) -> np.ndarray:
        if name == base.SET_LEGIT_VAL:  # this suite has no legit validation pool; empty by design
            return np.zeros(0)
        chunk = cal_texts if name == base.SET_CAL else sets[name]["texts"]
        return X.teacher_probs(
            model, tokenizer, chunk, device=device, canon=True,
            max_len=args.max_len, eval_batch_size=args.eval_batch_size,
        )

    def on_threshold(tau: float, fallback: bool) -> None:
        payload = {
            "plan": PLAN, "teacher_sha256": sha, "run": RUN_NAME,
            "train_rows": TRAIN_ROWS, "cal_rows": CAL_ROWS, "cal_positives": int(y_cal.sum()),
            "tau": {"maxf1": thr["maxf1"], "cal1pct": thr["cal1pct"], "low95": thr["low95"]},
            "low95_cal_recall": thr["low95_cal_recall"],
            "tau_maxf1_cal_f1": thr["maxf1_cal_f1"],
            "cal1pct_cal_fpr": thr["cal1pct_cal_fpr"], "cal1pct_cal_tpr": thr["cal1pct_cal_tpr"],
            "student_tau": STUDENT_TAU, "rule_fallback": bool(fallback),
        }
        (work / "thresholds.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        ctx.log("thresholds.json written before any test set is scored: "
                f"maxF1 {thr['maxf1']:.6f}, cal1% {thr['cal1pct']:.6f}")

    with timed("teacher_score_s", timings):
        pipe = base.run_seed_pipeline(
            fit=lambda: model,
            score=score,
            rule=rule,
            y_cal=y_cal,
            post_sets=list(TEST_SETS),
            on_threshold=on_threshold,
        )
    events = pipe["events"]
    first_test = min(i for i, e in enumerate(events) if e.startswith("score:") and
                     e[len("score:"):] in TEST_SETS)
    if events.index("threshold") > first_test:
        raise RuntimeError("a test set was scored before the threshold")
    p_t = {n: pipe["scores"][n] for n in TEST_SETS}

    # 6. v0 student on the same rows (shipped threshold; no tuning)
    session = _onnx_session(v0_path)
    with timed("student_score_s", timings):
        p_s = {n: score_sms_texts(session, sets[n]["texts"]) for n in TEST_SETS}

    # 7. arms under both rules, plus the student reference
    arms: dict[str, dict] = {}
    flags_by_key: dict[str, dict[str, np.ndarray]] = {}
    for rule_name in RULES:
        tau = thr["maxf1"] if rule_name == "maxf1" else thr["cal1pct"]
        per_set = {n: arm_flags(p_s[n], p_t[n], tau) for n in TEST_SETS}
        for arm in ARMS:
            key = f"{rule_name}/{arm}"
            flags = {n: per_set[n][arm] for n in TEST_SETS}
            flags_by_key[key] = flags
            with timed("metrics_s", timings):
                arms[key] = arm_block(sets, flags)
    student_flags = {n: arm_flags(p_s[n], p_t[n], 0.0)["S"] for n in TEST_SETS}
    arms["student_only"] = arm_block(sets, student_flags)

    # 8. row-level output: test sets only, ids and scores, never text
    frames = []
    for name in TEST_SETS:
        s = sets[name]
        frames.append(pd.DataFrame({
            "set": name, "id": s["ids"].astype(str), "row_key": s["row_key"],
            "label": s["labels"], "label_fine": s["label_fine"], "category": s["category"],
            "clean": s["clean"], "xlmr_pred": (np.full(len(s["texts"]), np.nan)
                                              if s["xlmr"] is None else s["xlmr"].astype(float)),
            "p_teacher": p_t[name], "p_student": p_s[name],
        }))
    probs = pd.concat(frames, ignore_index=True)

    thresholds = {"maxf1": thr["maxf1"], "cal1pct": thr["cal1pct"], "low95": thr["low95"]}
    tau_high, tau_low = thr["cal1pct"], thr["low95"]
    bands = {arm: {n: band_for_arm(p_s[n], p_t[n], tau_high, tau_low, arm) for n in TEST_SETS}
             for arm in ("T", "SandT")}
    two_band = {
        "tau_high": tau_high, "tau_low": tau_low, "ordered": bool(tau_low <= tau_high),
        "T": two_band_block(sets, bands["T"]), "SandT": two_band_block(sets, bands["SandT"]),
    }

    # 8b. Amendment 2 (post-hoc, calibration-only cross-fit), on probabilities and on logit margins.
    # Negatives only set thresholds; external positives and ham are reported at both fold
    # thresholds.
    folds, fold_info = negative_folds(sets[SET_GL_NEG]["texts"])
    ext_sets = {
        "smishx_smishing": SET_SX_SMISH,
        "smishx_ham": SET_SX_HAM,
        "groundlm_pos": SET_GL_POS,
    }
    margin_of = {
        n: teacher_margins(model, tokenizer, sets[n]["texts"], device=device,
                           max_len=args.max_len, eval_batch_size=args.eval_batch_size)
        for n in (SET_GL_NEG, *ext_sets.values())
    }
    amendment2 = {
        "plan": "docs/TEACHER_SUITE_PLAN.md Amendment 2",
        "folds": fold_info,
        "probs": fold_protocol(p_t[SET_GL_NEG], folds, {k: p_t[v] for k, v in ext_sets.items()}),
        "logits": fold_protocol(margin_of[SET_GL_NEG], folds,
                                {k: margin_of[v] for k, v in ext_sets.items()}),
    }
    ctx.log(f"amendment 2 folds {fold_info}; pooled negative FP probs "
            f"{amendment2['probs']['pooled_negative_fp']}, logits "
            f"{amendment2['logits']['pooled_negative_fp']}")

    payload = {
        "job": JOB, "status": "scored", "plan": PLAN, "checkpoint_sha256": sha,
        "amendment2": amendment2,
        "teacher": {"sha256": sha, "run": RUN_NAME, "train_input": "canon",
                    "model": prov["model"], "seed": prov["seed"]},
        "v0": {"sha256": d0.V0_SHA},
        "thresholds": thresholds, "student_tau": STUDENT_TAU,
        "events": events,
        "contamination": contamination,
        "xlmr_join": join,
        "env": env,
        "timings": timings,
        "arms": arms,
        "two_band": two_band,
    }
    _write(work, payload, thresholds, probs)
    passing = [k for k, v in arms.items() if v["gates"]["arm_pass"]]
    ctx.results["summary"] = (
        f"{JOB}: maxF1 tau {thr['maxf1']:.4f}, cal1% tau {thr['cal1pct']:.4f}; "
        f"arms passing all gates: {passing or 'none'}"
    )
    return True
