"""Cross-fitted GroundLM calibration, confirmation on a fresh model (docs/CROSSFIT_CONFIRM_PLAN.md,
frozen 2026-11-12).

mmBERT-base, the W1e recipe of sms_encoder_exposure (fold train partition plus build_pools arm
C extra rows, minus the 20 % CloveAI/FinEE calibration hold-out), trained with SEED 3 only. No
GroundLM row is used for training. The GroundLM policy negatives are split into two template-
disjoint folds with the sms_groupneg construction (rakshak.eval.groupneg.cross_fit_folds on the
template keys; asserted 653 clusters and fold sizes 566 / 567).

Thresholds, cross-fitted. For each fold F, tau_F is the smallest logit margin with at most
floor(0.01 x |F|) false positives on F's negatives (no positives). Order, enforced in code:
  1. fold 0 negatives are scored; tau_0 is computed and thresholds.json is written;
  2. fold 1 negatives are scored (the other fold); tau_1 is computed and thresholds.json is
     rewritten;
  3. only then are the positive and external sets scored (SmishX, GroundLM positives, campaign,
     lang_shift, temporal, phone 1 filled).
tau_0 is applied to fold 1's negatives and tau_1 to fold 0's; the two held-out counts are
pooled
(gate C1). The deployment threshold tau_dep is the same rule on all 1 133 negatives; it is reported
and stored, and its in-sample FP is not a result.

Gates C1 to C4 (all must hold): C1 pooled held-out GroundLM FP <= 11 / 1133; C2 SmishX smishing TP
>= 157 / 226 at both tau_0 and tau_1; C3 GroundLM positive recall >= 0.90 at both; C4 campaign clean
F1 >= 0.9713 at both.

Label: "GroundLM-calibrated (threshold only, cross-fitted, template-disjoint); training data
shares a source family (CloveAI) with GroundLM; no GroundLM row in training."

Output, in the kernel output folder: prep_report.json, thresholds.json (written before any positive
or external set is scored), metrics.json, report.md, probs.parquet (margins, probabilities, ids,
labels, fold, set; no text), seed3/best_checkpoint/ (safetensors) and seed3/manifest.json
(sha256).

Ensemble (W1h, docs/ENSEMBLE_CONFIRM_PLAN.md, frozen): --seeds 4 5 6 --ensemble mean trains each
seed with the same recipe (one kernel; each checkpoint is written as its seed finishes). The score
of every set is the mean logit margin of the seeds. The thresholds, order, gates C1 to C4 and the
reported items are the same, applied to the ensemble margins; per-seed results are reported (not
gated). Output adds seed<S>/ per seed, ensemble_manifest.json, and in metrics.json a per_seed block.
probs.parquet carries the ensemble rows with seed -1 and each member's rows with its own seed.
Without the flags the job is the single seed 3 run.

PRIVACY (binding). No message text, sender string or inbox row id is written anywhere. Inbox
rows are indexed phone1_<i>. Errors are logged as exception type and frames only (REDACT_ERRORS).

Launch (T4, internet on; the data and the inbox donor):
  kaggle/launch.sh sms_crossfit_confirm --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox

Flags (--job-args="FLAGS"):
  --max-len INT             token limit (default 128)
  --eval-batch-size INT     inference batch (default 128)
  --seeds INT...            training seeds (default 3); several seeds need --ensemble
  --ensemble mean           score = mean logit margin over the seeds (needs two or more seeds)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sms_encoder_exposure as EX
import sms_encoder_fold as F
import sms_inbox_retrain as IN
import sms_retrain as base
import sms_teacher_suite as S
import sms_teacher_xsource as X
from _common import env_report, seed_everything, timed
from baseline_v0 import rate_row
from scipy.special import expit

from rakshak.data import retrain_prep as RP
from rakshak.eval import data as D
from rakshak.eval import metrics as M
from rakshak.eval.groupneg import cross_fit_folds

JOB = "sms_crossfit_confirm"
PLAN = "docs/CROSSFIT_CONFIRM_PLAN.md"
SEED = 3
N_FOLDS = 2
FPR_CROSS = 0.01
N_CLUSTERS = 653
FOLD_SIZES = (566, 567)
GL_NEG_N = 1133
GL_POS_N = 91
SX_SMISH_N = 226
SX_HAM_N = 606
C1_MAX_FP = S.G1_MAX_FP  # 11 = floor(0.01 x 1133)
C2_MIN_TP = S.G2_MIN_TP  # 157 of 226
C3_MIN_RECALL = 0.90  # 82 of 91
C4_MIN_F1 = S.G4_MIN_F1  # 0.9713
DEFAULT_ROOT = Path("/kaggle/input")
REDACT_ERRORS = True

SET_NEG = S.SET_GL_NEG
SET_POS = S.SET_GL_POS
SET_SMISH = S.SET_SX_SMISH
SET_HAM = S.SET_SX_HAM
SET_CAMP = S.SET_CAMP
SET_LANG = S.SET_LANG
SET_TEMP = S.SET_TEMP
SET_P1 = "phone1_filled"
EXTERNAL_SETS = (SET_POS, SET_SMISH, SET_HAM, SET_CAMP, SET_LANG, SET_TEMP, SET_P1)
TAUS = ("0", "1")
ENSEMBLE_MODES = ("mean",)
ENSEMBLE_SEED = -1  # seed column of the ensemble rows in probs.parquet


class RedactedError(RuntimeError):
    """The job failed; the cause is in the log as type and frames, not as a message."""


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--max-len", type=int, default=F.DEFAULTS["max_len"])
    p.add_argument("--eval-batch-size", type=int, default=F.DEFAULTS["eval_batch_size"])
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"training seeds (default {SEED})")
    p.add_argument("--ensemble", choices=ENSEMBLE_MODES, default=None,
                   help="mean: the score is the mean logit margin over the seeds")
    args = p.parse_args(list(argv or []))
    if args.max_len < 1 or args.eval_batch_size < 1:
        p.error("--max-len and --eval-batch-size must be positive")
    seeds = [SEED] if args.seeds is None else list(args.seeds)
    if any(s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        p.error("--seeds must be distinct non-negative integers")
    if len(seeds) > 1 and args.ensemble is None:
        p.error("several seeds need --ensemble mean")
    if args.ensemble is not None and len(seeds) < 2:
        p.error("--ensemble needs two or more seeds")
    args.seeds = seeds
    return args


# ---- pure helpers (unit-tested in tests/test_sms_crossfit_confirm.py) ----


def groundlm_negative_mask(gl: pd.DataFrame) -> np.ndarray:
    """Scoreable policy negatives: policy_negative & text_available & eval_ok (build_sets' rule)."""
    avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    ok = gl["eval_ok"].fillna(False).astype(bool).to_numpy()
    pol = gl["policy_negative"].fillna(False).astype(bool).to_numpy()
    return pol & avail & ok


def template_keys(gl: pd.DataFrame, neg_idx: np.ndarray) -> list[str]:
    """The template key of each scoreable negative, in build_sets' negative order."""
    return gl["template_key"].fillna("").astype(str).to_numpy()[neg_idx].tolist()


def fold_plan(keys: Sequence[str]) -> tuple[np.ndarray, dict]:
    """sms_groupneg's fold construction on the template keys. Asserts 653 clusters and fold sizes
    566 and 567 (either fold may hold the larger). Returns (fold per negative, info)."""
    clusters, folds = cross_fit_folds(list(keys), n_folds=N_FOLDS)
    folds = np.asarray(folds, dtype=np.int64)
    n_clusters = int(len(set(clusters)))
    sizes = sorted(int(np.sum(folds == f)) for f in range(N_FOLDS))
    if n_clusters != N_CLUSTERS or tuple(sizes) != FOLD_SIZES:
        raise ValueError(
            f"cross-fit folds do not match the plan: clusters {n_clusters} (need {N_CLUSTERS}), "
            f"fold sizes {sizes} (need {list(FOLD_SIZES)})"
        )
    info = {
        "clusters": n_clusters,
        "fold_sizes": [int(np.sum(folds == f)) for f in range(N_FOLDS)],
        "rule": "sms_groupneg: negative_clusters(template_key) + assign_folds(., 2)",
    }
    return folds, info


def tau_for_fold(fold_margins: Sequence[float]) -> float:
    """Smallest margin with at most floor(0.01 x n) false positives on the fold's negatives."""
    return S.fpr_threshold(fold_margins, FPR_CROSS)


def tau_record(fold_margins: Sequence[float], tau: float) -> dict:
    m = np.asarray(fold_margins, dtype=np.float64)
    return {
        "tau": float(tau),
        "n": int(m.size),
        "k_allowed": int(math.floor(FPR_CROSS * m.size)),
        "flagged_own_fold": int(np.sum(m >= tau)),
    }


def held_out_flags(
    neg_margins: np.ndarray, folds: np.ndarray, taus: dict[str, float]
) -> np.ndarray:
    """Each negative flagged at the threshold of the other fold (cross-fit held-out decision)."""
    m = np.asarray(neg_margins, dtype=np.float64)
    f = np.asarray(folds)
    return np.where(f == 0, m >= taus["1"], m >= taus["0"])


def evaluate_confirm(
    pooled_fp: int, pooled_n: int, per_tau: dict[str, dict]
) -> dict:
    """Gates C1 to C4. ``per_tau`` maps "0" and "1" to smish_tp, pos_tp, n_pos, f1_clean."""
    c1 = pooled_fp <= C1_MAX_FP
    smish = {t: int(per_tau[t]["smish_tp"]) for t in TAUS}
    pos_rec = {t: per_tau[t]["pos_tp"] / per_tau[t]["n_pos"] for t in TAUS}
    f1 = {t: float(per_tau[t]["f1_clean"]) for t in TAUS}
    c2 = all(smish[t] >= C2_MIN_TP for t in TAUS)
    c3 = all(pos_rec[t] >= C3_MIN_RECALL for t in TAUS)
    c4 = all(np.isfinite(f1[t]) and f1[t] >= C4_MIN_F1 for t in TAUS)
    gates = {
        "C1_pooled_heldout_groundlm_fp": {
            "k": int(pooled_fp), "n": int(pooled_n), "max": C1_MAX_FP, "pass": bool(c1)},
        "C2_smishx_smishing_tp": {"k_by_tau": smish, "n": SX_SMISH_N, "min": C2_MIN_TP,
                                  "pass": bool(c2)},
        "C3_groundlm_pos_recall": {"rate_by_tau": pos_rec, "min": C3_MIN_RECALL,
                                   "pass": bool(c3)},
        "C4_campaign_clean_f1": {"value_by_tau": f1, "min": C4_MIN_F1, "pass": bool(c4)},
    }
    all_pass = bool(c1 and c2 and c3 and c4)
    failed = [g for g, v in gates.items() if not v["pass"]]
    if all_pass:
        verdict = "pass"
    elif c1 and c2 and c3 and not c4:
        verdict = (
            "C1-C3 pass, C4 fails: ships as an Indian-SMS stage beside the byte student; that "
            "combination needs its own pre-registered system-level check"
        )
    else:
        verdict = "fail (" + ", ".join(failed) + ")"
    return {"gates": gates, "all_pass": all_pass, "failed": failed, "verdict": verdict}


def cross_order_ok(events: Sequence[str]) -> bool:
    """True when tau_0 is written before fold 1 is scored, tau_1 before any external set, and no
    external set is scored before both thresholds exist."""
    def at(e: str) -> int:
        return events.index(e)

    if at("threshold:0") > at("score:neg_f1"):
        return False
    if at("threshold:1") > min(
        (i for i, e in enumerate(events) if e.startswith("score:") and
         e[len("score:"):] in EXTERNAL_SETS),
        default=len(events),
    ):
        return False
    return True


def probs_frame(
    seed: int,
    sets: dict[str, dict],
    scores: dict[str, np.ndarray],
    folds: np.ndarray,
    p1_ids: Sequence[str],
    p1_labels: np.ndarray,
    clean: np.ndarray,
) -> pd.DataFrame:
    """Margins and probabilities per row for every set. Inbox rows carry index ids only."""
    parts: list[pd.DataFrame] = []

    def add(name, ids, labels, margins, fold=None, clean_col=None, category=None, xlmr=None):
        n = len(margins)
        parts.append(pd.DataFrame({
            "seed": seed,
            "set": name,
            "id": np.asarray(ids).astype(str),
            "label": np.asarray(labels, dtype=np.float64),
            "fold": np.full(n, -1) if fold is None else np.asarray(fold, dtype=np.int64),
            "margin": np.asarray(margins, dtype=np.float64),
            "clean": (np.full(n, np.nan) if clean_col is None
                      else np.asarray(clean_col, dtype=np.float64)),
            "category": np.asarray(category if category is not None else [""] * n).astype(str),
            "xlmr_pred": (np.full(n, np.nan) if xlmr is None
                          else np.asarray(xlmr, dtype=np.float64)),
        }))

    neg = sets[SET_NEG]
    add(SET_NEG, neg["ids"], neg["labels"], scores[SET_NEG], fold=folds,
        category=neg["category"], xlmr=neg["xlmr"])
    for name in (SET_POS, SET_SMISH, SET_HAM, SET_LANG, SET_TEMP):
        s = sets[name]
        add(name, s["ids"], s["labels"], scores[name], category=s["category"], xlmr=s["xlmr"])
    camp = sets[SET_CAMP]
    add(SET_CAMP, camp["ids"], camp["labels"], scores[SET_CAMP], clean_col=clean)
    add(SET_P1, list(p1_ids), p1_labels, scores[SET_P1])
    out = pd.concat(parts, ignore_index=True)
    out["p_scam"] = expit(out["margin"].to_numpy(dtype=np.float64))
    return out


def block_at(
    sets: dict[str, dict],
    scores: dict[str, np.ndarray],
    tau: float,
    clean: np.ndarray,
    p1_labels: np.ndarray,
    p1_scores: np.ndarray,
) -> dict:
    """Every reported count at one threshold (gates use the same counts)."""
    pos = scores[SET_POS] >= tau
    smish = scores[SET_SMISH] >= tau
    ham = scores[SET_HAM] >= tau
    camp = sets[SET_CAMP]
    cflags = scores[SET_CAMP] >= tau
    y = camp["labels"].astype(np.int64)
    clean_f1 = M.metrics_from_counts(*M.confusion_counts(y[clean], cflags[clean]))
    all_f1 = M.metrics_from_counts(*M.confusion_counts(y, cflags))
    p1_legit = p1_labels == 0
    p1_scam = p1_labels == 1
    return {
        "smish_tp": int(smish.sum()),
        "smish_rate": rate_row(int(smish.sum()), len(smish)),
        "ham_fp": int(ham.sum()),
        "ham_fpr": rate_row(int(ham.sum()), len(ham)),
        "pos_tp": int(pos.sum()),
        "n_pos": int(len(pos)),
        "pos_recall": rate_row(int(pos.sum()), len(pos)),
        "f1_clean": float(clean_f1["f1"]),
        "campaign_clean": {**clean_f1, "n_clean": int(clean.sum()),
                           "f1_ci95": S.cluster_ci(y[clean], cflags[clean],
                                                   camp["codes"][clean], "f1")},
        "campaign_all": {**all_f1},
        "phone1_legit_fpr": rate_row(int((p1_scores[p1_legit] >= tau).sum()), int(p1_legit.sum())),
        "phone1_scam_recall": rate_row(int((p1_scores[p1_scam] >= tau).sum()),
                                       int(p1_scam.sum())),
        "paired_groundlm_positives": base.paired_table(pos, sets[SET_POS]["xlmr"]),
    }


def heldout_block(sets: dict[str, dict], held: np.ndarray, folds: np.ndarray) -> dict:
    neg = sets[SET_NEG]
    cats = np.asarray([str(c) for c in neg["category"]])
    by_cat = {}
    for cat in sorted(set(cats.tolist())):
        m = cats == cat
        by_cat[cat] = {"n": int(m.sum()), "fp": int(held[m].sum())}
    return {
        "pooled_fp": int(held.sum()),
        "pooled_n": int(len(held)),
        "pooled_fpr": rate_row(int(held.sum()), len(held)),
        "by_fold": {str(f): {"n": int(np.sum(folds == f)), "fp": int(held[folds == f].sum())}
                    for f in range(N_FOLDS)},
        "by_category": by_cat,
        "paired_vs_xlmr": base.paired_table(held, neg["xlmr"]),
    }


def auc_block(sets: dict[str, dict], scores: dict[str, np.ndarray]) -> dict:
    neg_s, pos_s = scores[SET_NEG], scores[SET_POS]
    gl_y = np.concatenate([np.zeros(len(neg_s)), np.ones(len(pos_s))]).astype(np.int64)
    gl_s = np.concatenate([neg_s, pos_s])
    sx_y = np.concatenate([np.ones(len(scores[SET_SMISH])), np.zeros(len(scores[SET_HAM]))])
    sx_s = np.concatenate([scores[SET_SMISH], scores[SET_HAM]])
    camp_y = sets[SET_CAMP]["labels"].astype(np.int64)
    return {
        "groundlm": X.auc_binary(gl_y, gl_s),
        "smishx": X.auc_binary(sx_y.astype(np.int64), sx_s),
        "campaign_all": X.auc_binary(camp_y, scores[SET_CAMP]),
    }


def dep_block(sets, scores, tau_dep, neg_margins, clean, p1_labels, p1_scores) -> dict:
    """Reported at tau_dep: lang_shift, temporal, and every count, on the same margins."""
    lang = sets[SET_LANG]
    temp = sets[SET_TEMP]
    lf = scores[SET_LANG] >= tau_dep
    tf = scores[SET_TEMP] >= tau_dep
    base_block = block_at(sets, scores, tau_dep, clean, p1_labels, p1_scores)
    base_block["lang_shift_f1"] = M.metrics_from_counts(*M.confusion_counts(lang["labels"], lf))
    base_block["temporal_precision"] = M.metrics_from_counts(
        *M.confusion_counts(temp["labels"], tf))
    base_block["groundlm_neg_fp_in_sample"] = int((neg_margins >= tau_dep).sum())
    return base_block


def threshold_payload(folds_info: dict, taus: dict, tau_dep: float | None, n_neg: int,
                      seeds: Sequence[int] = (SEED,), ensemble: str | None = None) -> dict:
    return {
        "plan": PLAN,
        **seed_fields(seeds, ensemble),
        "rule": "alarm if logit margin >= tau; tau_F = smallest margin with at most "
                "floor(0.01 x |F|) FP on fold F's negatives (no positives)"
                + ("; margin = mean logit margin over the seeds (ensemble)" if ensemble else ""),
        "order": "tau_0 written before fold 1 is scored; tau_1 before any positive or "
                 "external set; tau_dep is the same rule on all negatives",
        "folds": folds_info,
        "negatives": n_neg,
        "tau": {k: v for k, v in taus.items()},
        "tau_dep": tau_dep,
    }


def render_report(payload: dict) -> str:
    if "gates" not in payload:
        return f"# {JOB} (plan {PLAN})\n\nStatus: {payload.get('status')}.\n"
    lines = [
        f"# {JOB} (plan {PLAN})",
        "",
        "Label: GroundLM-calibrated (threshold only, cross-fitted, template-disjoint); training "
        "data shares a source family (CloveAI) with GroundLM; no GroundLM row in training.",
        "",
        f"Verdict: {payload['verdict']}.",
        "",
        "| gate | value | threshold | pass |",
        "|---|---|---|---|",
    ]
    g = payload["gates"]
    c1 = g["C1_pooled_heldout_groundlm_fp"]
    lines.append(f"| C1 pooled held-out GroundLM FP | {c1['k']}/{c1['n']} | <= {c1['max']} | "
                 f"{c1['pass']} |")
    for key, label in (("C2_smishx_smishing_tp", "C2 SmishX TP"),
                       ("C3_groundlm_pos_recall", "C3 GroundLM pos recall"),
                       ("C4_campaign_clean_f1", "C4 campaign clean F1")):
        v = g[key]
        vals = v.get("k_by_tau") or v.get("rate_by_tau") or v.get("value_by_tau")
        shown = ", ".join(f"tau_{t}: {x if not isinstance(x, float) else round(x, 4)}"
                          for t, x in vals.items())
        lines.append(f"| {label} | {shown} | {v['min']} | {v['pass']} |")
    lines += ["", "Reported, not gated: SmishX ham FPR, phone-1 legit FPR and scam recall (filled "
              "text, expected to be poor), per-category held-out FP, paired McNemar vs GroundLM "
              "XLM-R, AUCs, lang_shift and temporal at tau_dep.", ""]
    lines.append("Thresholds: " + json.dumps(payload["thresholds"]["tau"], sort_keys=True))
    if "per_seed" in payload:
        lines += ["", "Per seed, not gated (each seed at its own tau_F on its own negatives):", "",
                  "| seed | held-out GroundLM FP | SmishX TP tau_0, tau_1 | GroundLM pos TP "
                  "tau_0, tau_1 | campaign clean F1 tau_0, tau_1 |", "|---|---|---|---|---|"]
        for s, v in payload["per_seed"].items():
            lines.append(
                f"| {s} | {v['heldout_pooled_fp']}/{v['heldout_pooled_n']} | "
                f"{v['smish_tp']['0']}, {v['smish_tp']['1']} | "
                f"{v['pos_tp']['0']}, {v['pos_tp']['1']} of {v['n_pos']} | "
                f"{round(v['f1_clean']['0'], 4)}, {round(v['f1_clean']['1'], 4)} |")
    lines.append("")
    return "\n".join(lines)


# ---- ensemble and the ordered scoring (unit-tested with fake scorers) ----


def seed_fields(seeds: Sequence[int], ensemble: str | None) -> dict:
    """The seed keys of an output. One seed: "seed" holds it (the single-seed output). An ensemble
    lists its members in "seeds" and has no single "seed"."""
    return {
        "seed": int(seeds[0]) if len(seeds) == 1 else None,
        "seeds": [int(s) for s in seeds],
        "ensemble": ensemble,
    }


def ensemble_margins(per_seed: Sequence[np.ndarray], ensemble: str | None) -> np.ndarray:
    """One set's score. One seed (ensemble None) is its own margin. "mean" is the equal-weight mean
    of the seeds' logit margins; no seed is dropped or reweighted."""
    if ensemble not in (None, *ENSEMBLE_MODES):
        raise ValueError(f"unknown ensemble rule {ensemble!r}")
    if ensemble is None and len(per_seed) != 1:
        raise ValueError("several seeds need an ensemble rule")
    stack = np.stack([np.asarray(m, dtype=np.float64) for m in per_seed])
    if not np.all(np.isfinite(stack)):
        raise ValueError("non-finite logit margin")
    if ensemble is None:
        return stack[0].copy()
    return stack.mean(axis=0)


def gate_inputs(blocks: dict[str, dict]) -> dict[str, dict]:
    return {t: {"smish_tp": b["smish_tp"], "pos_tp": b["pos_tp"], "n_pos": b["n_pos"],
                "f1_clean": b["f1_clean"]} for t, b in blocks.items()}


def per_seed_block(
    sets: dict[str, dict],
    scores_s: dict[str, np.ndarray],
    neg_s: np.ndarray,
    folds: np.ndarray,
    clean: np.ndarray,
    p1_labels: np.ndarray,
) -> dict:
    """One seed on its own: its own tau_F on its own negatives and the same counts. Reported, not
    gated, and computed after every threshold and score exists (it decides nothing)."""
    taus = {str(f): tau_for_fold(neg_s[folds == f]) for f in range(N_FOLDS)}
    recs = {t: tau_record(neg_s[folds == int(t)], taus[t]) for t in TAUS}
    held = held_out_flags(neg_s, folds, taus)
    blocks = {t: block_at(sets, scores_s, taus[t], clean, p1_labels, scores_s[SET_P1])
              for t in TAUS}
    heldout = heldout_block(sets, held, folds)
    per_tau = gate_inputs(blocks)
    gate = evaluate_confirm(heldout["pooled_fp"], heldout["pooled_n"], per_tau)
    return {
        "tau": recs,
        "heldout_pooled_fp": heldout["pooled_fp"],
        "heldout_pooled_n": heldout["pooled_n"],
        "smish_tp": {t: per_tau[t]["smish_tp"] for t in TAUS},
        "pos_tp": {t: per_tau[t]["pos_tp"] for t in TAUS},
        "n_pos": per_tau["0"]["n_pos"],
        "f1_clean": {t: per_tau[t]["f1_clean"] for t in TAUS},
        "gates_reported_not_gated": gate["gates"],
        "verdict_reported_not_gated": gate["verdict"],
        "auc": auc_block(sets, scores_s),
    }


def ordered_protocol(
    seeds: Sequence[int],
    ensemble: str | None,
    score_seed: Callable[[int, Sequence[str]], np.ndarray],
    neg_texts: Sequence[str],
    folds: np.ndarray,
    external: Sequence[tuple[str, Sequence[str]]],
    write_thresholds: Callable[[dict[str, dict], float | None], None],
    timings: dict,
) -> dict:
    """The scoring order of docs/CROSSFIT_CONFIRM_PLAN.md, for one seed or for an ensemble.

    1. fold 0 negatives, scored by every seed; the set's margin; tau_0; thresholds written.
    2. fold 1 negatives likewise; tau_1; thresholds rewritten.
    3. tau_dep on all negatives; thresholds rewritten.
    4. the external sets in the given order (positives first), each scored by every seed.
    cross_order_ok checks the event order before anything is returned.
    """
    n = len(neg_texts)
    neg = {s: np.full(n, np.nan) for s in seeds}
    neg_ens = np.full(n, np.nan)
    events: list[str] = []
    taus: dict[str, float] = {}
    recs: dict[str, dict] = {}
    for f in range(N_FOLDS):
        idx = np.flatnonzero(folds == f)
        texts_f = [neg_texts[i] for i in idx]
        with timed(f"score_neg_f{f}", timings):
            for s in seeds:
                neg[s][idx] = score_seed(s, texts_f)
        events.append(f"score:neg_f{f}")
        neg_ens[idx] = ensemble_margins([neg[s][idx] for s in seeds], ensemble)
        taus[str(f)] = tau_for_fold(neg_ens[idx])
        recs[str(f)] = tau_record(neg_ens[idx], taus[str(f)])
        events.append(f"threshold:{f}")
        write_thresholds(dict(recs), None)
    tau_dep = float(S.fpr_threshold(neg_ens, FPR_CROSS))
    write_thresholds(dict(recs), tau_dep)
    scores: dict[str, np.ndarray] = {SET_NEG: neg_ens}
    by_seed: dict[int, dict[str, np.ndarray]] = {s: {SET_NEG: neg[s]} for s in seeds}
    for name, texts_ in external:
        with timed(f"score_{name}", timings):
            per = [score_seed(s, texts_) for s in seeds]
        for s, m in zip(seeds, per, strict=True):
            by_seed[s][name] = m
        events.append(f"score:{name}")
        scores[name] = ensemble_margins(per, ensemble)
    if not cross_order_ok(events):
        raise RuntimeError("a positive or external set was scored before the thresholds")
    return {"neg": neg_ens, "neg_by_seed": neg, "scores": scores, "by_seed": by_seed,
            "taus": taus, "recs": recs, "tau_dep": tau_dep, "events": events}


# ---- the job ----


def run(ctx) -> bool:
    """Runs _run; any exception leaves here as RedactedError (type and frames logged)."""
    failure = None
    try:
        return _run(ctx)
    except Exception as exc:
        failure = exc
        for line in X.redacted_frames(exc):
            ctx.log(f"  at {line}")
        ctx.log(f"ERROR {X.redacted_error(exc)}")
    raise RedactedError(X.redacted_error(failure)) from None


def _run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seeds, ensemble = args.seeds, args.ensemble
    multi = len(seeds) > 1
    import sms_teacher as T
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    seed_everything(EX.DATA_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{JOB} needs a CUDA GPU; launch with --gpu t4")
    device = torch.device("cuda")
    revision = F.MODEL_REVISIONS[EX.MODEL]
    env = env_report()
    env["device"] = str(device)
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})
    work = ctx.work
    cfg = T.Config(
        fp16=True,
        max_len=args.max_len,
        batch_size=F.DEFAULTS["batch_size"],
        eval_batch_size=args.eval_batch_size,
        patience=F.DEFAULTS["patience"],
        adv_aug=0.0,
        train_canon=True,
        log=ctx.log,
    )
    hp = {
        "lr": F.DEFAULTS["lr"],
        "epochs": F.DEFAULTS["epochs"],
        "warmup": F.DEFAULTS["warmup_ratio"],
        "weight_decay": F.DEFAULTS["weight_decay"],
    }

    # 1. data and the fold teacher partitions (sizes asserted as in the encoder exposure)
    root = D.resolve_data_root(None)
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    texts = np.asarray(base.str_texts(sms["text"]), dtype=object)
    ids = sms["id"].astype(str).to_numpy()
    runs = D.load_runs(
        root, sms, patterns=["campaign", "lang_shift", "temporal", "source_holdout"],
        labels=labels, mask=D.eval_mask(sms),
    )

    def one(stem: str) -> Any:
        found = [r for r in runs if r.name.split(":")[0] == stem]
        if len(found) != 1:
            raise ValueError(f"expected one {stem} run, found {len(found)}")
        return found[0]

    fold_run = [r for r in runs if r.name == S.RUN_NAME]
    if len(fold_run) != 1:
        raise ValueError(f"fold run {S.RUN_NAME} is not in the split files")
    fold_run = fold_run[0]
    if len(fold_run.train) != S.TRAIN_ROWS or len(fold_run.cal) != S.CAL_ROWS:
        raise ValueError(f"fold partitions train {len(fold_run.train)} cal {len(fold_run.cal)}")
    camp, lang, temp = one("campaign"), one("lang_shift"), one("temporal")

    # 2. build_pools arm C, then the calibration hold-out (excluded from training, as W1e)
    sources = RP.load_indian_sources(root)
    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    gl_avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    gl_texts = base.str_texts(gl["text"].to_numpy()[gl_avail])
    test_only = (
        [texts[i] for i in camp.test] + [texts[i] for i in lang.test]
        + [texts[i] for i in temp.test]
    )
    test_only += gl_texts + base.str_texts(sx["text"].to_numpy()[sx_avail])
    for name in (base.SET_SMSSHIELD, base.SET_IEEE_HAM):
        if name in sources:
            test_only += base.str_texts(sources[name]["text"])
    test_only = [t for t in test_only if t]
    n_neg = int((labels[camp.train] == 0).sum())
    with timed("prep_s", timings):
        prep = base.build_pools_once("C", sources, test_only, n_neg, gl_texts, RP.build_pools)
    extra = prep.train_extra
    ch_mask, ch_info = EX.calibration_holdout(extra)
    ch = extra[ch_mask]
    tr_extra = extra[~ch_mask]
    report = {
        "arm": "C",
        **seed_fields(seeds, ensemble),
        "data_seed": EX.DATA_SEED,
        "cal_seed": EX.CAL_SEED,
        "exposure_holdout": ch_info,
        "ch_legit": int((ch["label"].astype(int) == 0).sum()),
        "ch_scam": int((ch["label"].astype(int) == 1).sum()),
        "train_extra_used": int(len(tr_extra)),
        **prep.report,
    }
    (work / "prep_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    ctx.log(f"prep: train_extra {len(extra)}, held-out {len(ch)}, training extra {len(tr_extra)}")

    # 3. training data (fold train + extra rows); early stopping on the fold's own cal split
    cal_idx, tr_idx = fold_run.cal, fold_run.train
    train_raw = [str(t) for t in texts[tr_idx]] + base.str_texts(tr_extra["text"])
    train_y = np.concatenate([labels[tr_idx], tr_extra["label"].astype(int).to_numpy()])
    data = {
        "train_raw": train_raw,
        "train_y": train_y.astype(np.int64),
        "cal_y": labels[cal_idx],
        "cal_prepped": [D.canon_text(t) for t in texts[cal_idx]],
    }
    train_ids = set(ids[tr_idx].tolist()) | set(tr_extra["id"].astype(str).tolist())
    train_keys = X.canon_keys(train_raw)

    # 4. the scored sets (no model yet) and the cross-fit folds of the GroundLM negatives
    sets, join = S.build_sets(root, sms, labels, texts, ids, camp, lang, temp, train_ids,
                              train_keys)
    neg_idx = np.flatnonzero(groundlm_negative_mask(gl))
    folds, fold_info = fold_plan(template_keys(gl, neg_idx))
    if len(folds) != len(sets[SET_NEG]["texts"]) or len(folds) != GL_NEG_N:
        raise ValueError("fold assignment does not align with the GroundLM negative set")
    counts = {
        "groundlm_neg": len(sets[SET_NEG]["texts"]),
        "groundlm_pos": len(sets[SET_POS]["texts"]),
        "smishx_smishing": len(sets[SET_SMISH]["texts"]),
        "smishx_ham": len(sets[SET_HAM]["texts"]),
    }
    if (counts["groundlm_neg"], counts["groundlm_pos"], counts["smishx_smishing"],
            counts["smishx_ham"]) != (GL_NEG_N, GL_POS_N, SX_SMISH_N, SX_HAM_N):
        raise ValueError(f"scored set sizes {counts} do not match the plan")
    ctx.log(f"sets {counts}; folds {fold_info}; XLM-R join {join}")

    p1_raw = pd.read_parquet(IN.find_one(IN.TEST_FILE, [DEFAULT_ROOT]))
    p1_filled_raw, p1_tags = IN.fill_inbox_frame(p1_raw)
    IN.assert_filled(list(p1_filled_raw["text"]), "phone 1 (filled)")
    p1, p1_counts = IN.label_inbox(p1_filled_raw)
    del p1_raw, p1_filled_raw
    p1_ids = [f"phone1_{i}" for i in range(len(p1))]
    p1_texts = [str(t) for t in p1["text"]]
    p1_labels = p1["label"].to_numpy().astype(np.int64)
    report["phone1"] = {"counts": p1_counts, "tags_before_fill": p1_tags,
                        "legit": int((p1_labels == 0).sum()), "scam": int((p1_labels == 1).sum())}

    # 5. train each seed (the W1e recipe, the same for every seed). Each checkpoint is written as
    # soon as its seed finishes, so a timeout keeps the seeds already trained.
    with timed("download", timings):
        local = snapshot_download(repo_id=EX.MODEL, revision=revision,
                                  allow_patterns=F.SNAPSHOT_PATTERNS)
    tokenizer = AutoTokenizer.from_pretrained(local)
    models: dict[int, Any] = {}
    info: dict[int, dict] = {}
    ckpt_sha: dict[int, str] = {}
    for seed in seeds:
        t0 = time.time()
        with timed(f"train_s{seed}" if multi else "train_s", timings):
            out = T.train_once(cfg, hp, local, seed, tokenizer, device, data, math.inf)
        model = out["model"].to("cpu").eval()
        train_s = round(time.time() - t0, 1)
        ckpt = work / f"seed{seed}" / "best_checkpoint"
        model.save_pretrained(ckpt, safe_serialization=True)
        tokenizer.save_pretrained(ckpt)
        ckpt_sha[seed] = X.file_sha256(ckpt / "model.safetensors")
        models[seed] = model
        info[seed] = {
            "best_epoch": int(out["best_epoch"]),
            "cal_pr_auc": float(out["cal_pr_auc"]),
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "seconds": train_s,
        }
        ctx.log(f"seed {seed}: best epoch {out['best_epoch']} cal PR-AUC "
                f"{out['cal_pr_auc']:.4f}; checkpoint sha256 {ckpt_sha[seed][:12]}")
        del out

    def score_seed(seed: int, texts_: Sequence[str]) -> np.ndarray:
        # no training runs after the last seed, so a scored model may stay on the GPU
        model = models[seed].to(device)
        return S.teacher_margins(model, tokenizer, list(texts_), device=device,
                                 max_len=args.max_len, eval_batch_size=args.eval_batch_size)

    # 6. ordered scoring (ordered_protocol): fold 0 negatives, tau_0, then fold 1 negatives,
    # tau_1, then tau_dep, then the other sets. Each threshold file is written before the next step.
    thr_path = work / "thresholds.json"

    def write_thresholds(recs_now: dict[str, dict], tau_dep_now: float | None) -> None:
        payload = threshold_payload(
            fold_info,
            {k: {**recs_now[k], "fold": int(k)} for k in recs_now},
            tau_dep_now,
            GL_NEG_N,
            seeds=seeds,
            ensemble=ensemble,
        )
        thr_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        ctx.log(f"thresholds.json written: folds {sorted(recs_now)}, tau_dep "
                f"{'pending' if tau_dep_now is None else round(tau_dep_now, 6)}")

    external = [
        (SET_POS, sets[SET_POS]["texts"]),
        (SET_SMISH, sets[SET_SMISH]["texts"]),
        (SET_HAM, sets[SET_HAM]["texts"]),
        (SET_CAMP, sets[SET_CAMP]["texts"]),
        (SET_LANG, sets[SET_LANG]["texts"]),
        (SET_TEMP, sets[SET_TEMP]["texts"]),
        (SET_P1, p1_texts),
    ]
    proto = ordered_protocol(seeds, ensemble, score_seed, sets[SET_NEG]["texts"], folds,
                             external, write_thresholds, timings)
    neg_margins, scores, events = proto["neg"], proto["scores"], proto["events"]
    taus, recs, tau_dep = proto["taus"], proto["recs"], proto["tau_dep"]
    ctx.log(f"tau_0 = {taus['0']:.6f}, tau_1 = {taus['1']:.6f}, tau_dep = {tau_dep:.6f} "
            f"(all {GL_NEG_N} negatives)")

    # 7. cross-fit decisions, blocks, gates (on the ensemble margins when --ensemble is given)
    held = held_out_flags(neg_margins, folds, taus)
    clean = sets[SET_CAMP]["clean"]
    blocks: dict[str, dict] = {
        t: block_at(sets, scores, taus[t], clean, p1_labels, scores[SET_P1]) for t in TAUS
    }
    heldout = heldout_block(sets, held, folds)
    gate = evaluate_confirm(heldout["pooled_fp"], heldout["pooled_n"], gate_inputs(blocks))
    dep = dep_block(sets, scores, tau_dep, neg_margins, clean, p1_labels, scores[SET_P1])
    aucs = auc_block(sets, scores)
    per_seed: dict[str, dict] = {}
    if ensemble is not None:
        for s in seeds:
            per_seed[str(s)] = {
                **per_seed_block(sets, proto["by_seed"][s], proto["neg_by_seed"][s], folds,
                                 clean, p1_labels),
                "train": info[s],
                "checkpoint_sha256": ckpt_sha[s],
            }

    # 8. manifests (sha256 of each checkpoint and of its manifest) and the ensemble index
    thresholds = {"tau_0": taus["0"], "tau_1": taus["1"], "tau_dep": tau_dep}
    scope = "single seed" if ensemble is None else f"ensemble {ensemble} of seeds {seeds}"
    man_sha: dict[int, str] = {}
    for seed in seeds:
        manifest = F.build_manifest(
            model=EX.MODEL, revision=revision, seed=seed, split=S.RUN_NAME,
            checkpoint_sha256=ckpt_sha[seed],
            param_count=info[seed]["param_count"],
            train_rows=len(train_raw), train_pos=int(train_y.sum()),
            cal_rows=len(cal_idx), cal_pos=int(labels[cal_idx].sum()),
            best_epoch=info[seed]["best_epoch"], cal_pr_auc=info[seed]["cal_pr_auc"],
            history=[], hyperparameters=hp, thresholds=thresholds,
            precision="fp16 autocast", env=env, seconds=info[seed]["seconds"],
        )
        manifest["thresholds_scope"] = scope
        path = work / f"seed{seed}" / F.MANIFEST
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        man_sha[seed] = X.file_sha256(path)
    if ensemble is not None:
        index = {
            "ensemble": ensemble,
            "seeds": seeds,
            "rule": "mean logit margin over the seeds, equal weights",
            "thresholds": thresholds,
            "members": {str(s): {"checkpoint": f"seed{s}/best_checkpoint",
                                 "checkpoint_sha256": ckpt_sha[s],
                                 "manifest_sha256": man_sha[s]} for s in seeds},
        }
        (work / "ensemble_manifest.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n")

    payload = {
        "job": JOB,
        "status": "scored",
        "plan": PLAN,
        "model": EX.MODEL,
        "revision": revision,
        **seed_fields(seeds, ensemble),
        "label": "GroundLM-calibrated (threshold only, cross-fitted, template-disjoint); training "
                 "data shares a source family (CloveAI) with GroundLM; no GroundLM row in training",
        "verdict": gate["verdict"],
        "gates": gate["gates"],
        "all_pass": gate["all_pass"],
        "thresholds": {"tau": {t: recs[t] for t in TAUS}, "tau_dep": tau_dep,
                       "folds": fold_info, "order_ok": True},
        "heldout_groundlm": heldout,
        "at_tau": blocks,
        "at_tau_dep": dep,
        "auc": aucs,
        "events": events,
        "prep": report,
        "xlmr_join": join,
        "counts": counts,
        "checkpoint_sha256": None if multi else ckpt_sha[seeds[0]],
        "checkpoint_sha256_by_seed": {str(s): ckpt_sha[s] for s in seeds},
        "manifest_sha256_by_seed": {str(s): man_sha[s] for s in seeds},
        "train": None if multi else {k: info[seeds[0]][k] for k in ("best_epoch", "cal_pr_auc")},
        "train_by_seed": {str(s): info[s] for s in seeds},
        "env": env,
        "timings": timings,
    }
    if ensemble is not None:
        payload["per_seed"] = per_seed
    (work / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (work / "report.md").write_text(render_report(payload))
    if ensemble is None:
        frame = probs_frame(seeds[0], sets, scores, folds, p1_ids, p1_labels, clean)
    else:
        frame = pd.concat(
            [probs_frame(ENSEMBLE_SEED, sets, scores, folds, p1_ids, p1_labels, clean)]
            + [probs_frame(s, sets, proto["by_seed"][s], folds, p1_ids, p1_labels, clean)
               for s in seeds],
            ignore_index=True,
        )
    frame.to_parquet(work / "probs.parquet", index=False)
    torch_cache(torch)
    who = f"seed {seeds[0]}" if not multi else f"seeds {seeds} ensemble {ensemble}"
    ctx.results["summary"] = f"{JOB}: {who}; verdict {gate['verdict']}"
    return True

def torch_cache(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
