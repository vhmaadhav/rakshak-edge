"""Teacher fine-tune on filled data and inbox negatives (docs/TEACHER_FINETUNE_PLAN.md, frozen).

Two arms, one kernel per arm, one seed per run (seed 0 by default). Model jhu-clsp/mmBERT-base from
the hub, num_labels 2, the fixed hyperparameters of section 2 (sms_teacher defaults: lr 3e-5, up to
4 epochs, weight decay 0.01, warmup 0.1, max_len 128, batch 32, fp16 autocast, class-balanced
cross-entropy, early stopping on campaign cal PR-AUC with patience 2). No HPO, no adversarial
augmentation. The training loop is sms_teacher.train_once, called with custom data.

- T0: the campaign train split of sms.parquet, surrogate-filled. The control.
- T1: T0 plus the inbox train donor rows that S1 adds (same pools, caps and test-only exclusion as
  sms_inbox_retrain.build_inbox_pools).

All text is surrogate-filled as in the S arms (always on here), and every filled text that is scored
or trained on is checked with no <TAG> left. GroundLM, SmishX and the scrubbed inbox versions are
scored scrubbed (plan rule 4), and the inbox sets are also scored scrubbed for the tag-shift check.

PRIVACY (binding). No inbox text, no sender string and no inbox row id leaves memory. Logs,
metrics.json, thresholds.json and report.md carry only counts, labels, rates, Wilson and
Clopper-Pearson intervals, AUCs, bootstrap intervals, thresholds, saturation shares and hashes. No
row-level file is written. The only file besides those is the fine-tuned checkpoint folder
best_checkpoint/ (model weights and tokenizer files, as sms_teacher.py saves it). Errors are
logged as exception type and frames only (REDACT_ERRORS).

Pipeline:
1. Campaign data as sms_inbox_retrain loads it. sms.parquet is filled (fill_sms_frame) and checked.
2. GroundLM (policy negatives, G1) and SmishX loaded as sms_retrain does, scrubbed.
3. Inbox donors: the train and test donor are filled (fill_inbox_frame) and labelled (label_inbox),
   and the scrubbed frames are labelled row for row as well. The train donor pools use
   build_inbox_pools on the filled text with the S1 test-only texts, so the rows are those of S1.
4. Stop rule as S1, before any training: status "stopped: insufficient data".
5. Training rows: T0 = campaign train; T1 = T0 + pool train rows. Logged by label and origin. The
   smoke run takes a seeded stratified subset (below).
6. base.run_seed_pipeline: fit (sms_teacher.train_once returns the best-epoch model, which is saved
   to best_checkpoint), then campaign cal and legit_val scored, then rakshak.data.retrain_prep
   threshold_rule(legit_val, campaign cal, campaign cal labels), then thresholds.json is written.
   Only after that: campaign test, GroundLM, SmishX, val_scam, test donor, and the scrubbed versions
   of legit_val, val_scam and the test donor.
7. Gates G1, G2, G3 and G5 exactly as sms_retrain and sms_inbox_retrain define them, on one seed.
   provisional_pass = all four hold. G4 is not a gate for a teacher: the checkpoint size is
   reported.
8. Section 4 measures: A1, A2, AUC against SmishX ham and campaign AUC (xsource helpers), with
   bootstrap intervals on the clean rows; A1 and A2 again with only the smishing rows of SmishX
   (label_fine smishing, eval_ok) as positives; test-donor legit FPR overall and by Codex policy
   category (label_policy) with Wilson intervals; test-donor scam recall with Clopper-Pearson; the
   scrubbed inbox sets; and the share of rows at exactly 1.0 for every scored set.
9. Hypotheses H7, H8, H9 (section 5) and the stop rule of section 6 that applies (T1 only).
10. metrics.json, report.md, thresholds.json, best_checkpoint/ in the work directory.

Smoke (--smoke): training uses at most 2,000 rows (at most 60 steps of batch 32, stratified by
label, seeded) for 1 epoch. Everything else is the same code path. The payload is marked
"smoke": true and the report says the numbers are not results.

Launch (T4 GPU, internet on, the two datasets; no kernel attachments: v0 is not scored here):
  kaggle/launch.sh sms_teacher_finetune --run-name smoke --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox --job-args="--arm T1 --smoke"
  kaggle/launch.sh sms_teacher_finetune --run-name t0 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox --job-args="--arm T0"
  kaggle/launch.sh sms_teacher_finetune --run-name t1 --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox --job-args="--arm T1"

Flags (``--job-args="FLAGS"``):
  --arm {T0,T1}          required
  --smoke                the smoke subset and 1 epoch (see above)
  --model NAME           hub model (default jhu-clsp/mmBERT-base)
  --seed INT             (default 0)
  --lr FLOAT             (default 3e-5)
  --epochs INT           max epochs (default 4; 1 with --smoke)
  --weight-decay FLOAT   (default 0.01)
  --warmup-ratio FLOAT   (default 0.1)
  --max-len INT          (default 128)
  --batch-size INT       (default 32)
  --eval-batch-size INT  (default 128)
  --patience INT         early-stop patience on cal PR-AUC (default 2)
  --n-boot INT           bootstrap resamples (default 1000)
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sms_inbox_retrain as inbox
import sms_retrain as base
import sms_surrogate_d0 as d0
import sms_teacher_xsource as xs
from _common import env_report, seed_everything, timed
from baseline_v0 import groundlm_table, rules_any, smishx_table, sms_run_table

from rakshak.data import retrain_prep as RP
from rakshak.eval import data as D

JOB = "sms_teacher_finetune"
PLAN = "docs/TEACHER_FINETUNE_PLAN.md"
ARMS = ("T0", "T1")
DEFAULT_MODEL = "jhu-clsp/mmBERT-base"
DEFAULT_SEED = 0
LR = 3e-5
EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_LEN = 128
BATCH_SIZE = 32
EVAL_BATCH = 128
PATIENCE = 2
N_BOOT = 1000
BOOT_SEED = 0
SMOKE_ROWS = 2000
SMOKE_STEPS = 60
SMOKE_EPOCHS = 1
DEFAULT_ROOT = "/kaggle/input"
BEST_DIR = "best_checkpoint"
THRESHOLD_RULE = "rakshak.data.retrain_prep.threshold_rule"
CATEGORIES = ("benign_transactional", "promotion")
TRAIN_FP16 = True
SUFFIX_SCRUB = "_scrubbed"
# Read by kaggle/build_kernel.py: errors are logged as type and frames only, never the message.
REDACT_ERRORS = True

SET_CAL = base.SET_CAL
SET_LEGIT_VAL = base.SET_LEGIT_VAL
SET_CAMP = base.SET_CAMPAIGN_TEST
SET_GL = base.SET_GROUNDLM
SET_SX = base.SET_SMISHX
SET_VS = base.SET_VAL_SCAM
SET_DONOR = "test_donor"

# G1, G2, G3 and G5 (G4 is not a gate for a teacher). base.GATES and inbox.INBOX_GATES hold the
# S-arm definitions, so the rules are the same objects.
TEACHER_GATES: tuple = tuple(g for g in inbox.INBOX_GATES if g[0] != "G4_onnx_bytes")

STOP_TEXT = {
    1: (
        "T1 passes provisionally: run seeds 1 and 2 of T1. If the 3-seed rule holds, the next plan "
        "is deployment: either the encoder itself on the Pi (INT8, latency and memory measured "
        "beside the Llama) or a smaller encoder distilled from it, decided by measurement."
    ),
    2: (
        "T1 meets H7 but not H8 (false positives fixed, recall lost): the threshold rule is the "
        "binding constraint. Report the ranking measures; the next plan would revisit the rule "
        "openly, not this one."
    ),
    3: (
        "T1 fails H7: the inbox negatives are not learnable from one donor even by the encoder. "
        "Report and stop; more donors is then the only route."
    ),
    None: (
        "No stop rule of section 6 applies: H7 and H8 hold but the provisional pass fails, so "
        "G3 (campaign F1) is the only failed gate. Section 6 item 4 adds no rule; report only."
    ),
}


# ---- pure helpers (unit-tested in tests/test_sms_teacher_finetune.py) ----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args(list(argv or []))
    if args.lr <= 0:
        p.error("--lr must be positive")
    if (
        min(
            args.epochs,
            args.max_len,
            args.batch_size,
            args.eval_batch_size,
            args.patience,
            args.n_boot,
        )
        < 1
    ):
        p.error(
            "--epochs, --max-len, --batch-size, --eval-batch-size, --patience and --n-boot "
            "must be positive"
        )
    return args


def effective_hparams(args: argparse.Namespace) -> dict:
    """The fixed hyperparameters passed to sms_teacher.train_once. Smoke runs one epoch."""
    return {
        "lr": float(args.lr),
        "epochs": SMOKE_EPOCHS if args.smoke else int(args.epochs),
        "warmup": float(args.warmup_ratio),
        "weight_decay": float(args.weight_decay),
    }


def smoke_row_cap(batch_size: int) -> int:
    """At most SMOKE_ROWS rows and at most SMOKE_STEPS optimisation steps of one epoch."""
    return min(SMOKE_ROWS, SMOKE_STEPS * int(batch_size))


def training_rows(
    arm: str,
    base_texts: Sequence[str],
    base_labels: Sequence[int],
    extra_texts: Sequence[str],
    extra_labels: Sequence[int],
    test_only: Sequence[str] = (),
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """T0: the campaign train rows only. T1: those plus the inbox pool rows. Returns the texts, the
    labels and the origin of each row ("campaign" or "inbox"). An inbox row whose normalised text
    equals a test-only text raises (the message names no text)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    base_texts = list(base_texts)
    base_y = np.asarray(base_labels, dtype=np.int64)
    if arm == "T0":
        extra_texts, extra_y = [], np.zeros(0, dtype=np.int64)
    else:
        extra_texts = list(extra_texts)
        extra_y = np.asarray(extra_labels, dtype=np.int64)
    if len(base_texts) != base_y.size or len(extra_texts) != extra_y.size:
        raise ValueError("training_rows: texts and labels differ in length")
    if extra_texts and test_only:
        banned = {RP.normalize(t) for t in test_only if t}
        if any(RP.normalize(t) in banned for t in extra_texts):
            raise ValueError("an inbox training row matches a test-only text")
    texts = base_texts + extra_texts
    labels = np.concatenate([base_y, extra_y])
    origin = np.array(["campaign"] * len(base_texts) + ["inbox"] * len(extra_texts), dtype=object)
    return texts, labels, origin


def origin_counts(labels: Sequence[int], origin: Sequence[str]) -> dict:
    """Row counts by label and by origin. Counts only."""
    y = np.asarray(labels, dtype=np.int64)
    o = np.asarray(origin, dtype=object)
    out: dict[str, Any] = {
        "total": int(y.size),
        "label0": int((y == 0).sum()),
        "label1": int((y == 1).sum()),
        "by_origin": {},
    }
    for name in ("campaign", "inbox"):
        m = o == name
        out["by_origin"][name] = {
            "total": int(m.sum()),
            "label0": int((m & (y == 0)).sum()),
            "label1": int((m & (y == 1)).sum()),
        }
    return out


def smoke_indices(labels: Sequence[int], n_max: int, seed: int) -> np.ndarray:
    """Indices of a seeded subset of at most ``n_max`` rows, stratified by label: each label keeps
    floor(n_max * its share) rows, so the total never exceeds n_max. Returns all rows when there
    are no more than n_max. Deterministic for a given seed."""
    y = np.asarray(labels, dtype=np.int64)
    n = int(y.size)
    if n_max < 1:
        raise ValueError("smoke_indices needs a positive cap")
    if n <= n_max:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    picks = []
    for cls in np.unique(y).tolist():
        idx = np.flatnonzero(y == cls)
        k = (n_max * int(idx.size)) // n
        if k > 0:
            picks.append(rng.choice(idx, size=int(k), replace=False))
    if not picks:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(picks)).astype(np.int64)


def gate_eval_teacher(row: dict) -> dict:
    """G1, G2, G3 and G5 on one seed, with sms_retrain's gate rule (base._gate). provisional_pass
    is True only when all four pass. Row keys: groundlm_fp, smishx_tp, campaign_f1, donor_fpr."""
    out: dict[str, Any] = {}
    for gate, key, ok in TEACHER_GATES:
        out[gate] = base._gate([row.get(key)], ok)
    out["provisional_pass"] = bool(all(out[gate]["passes"] for gate, _, _ in TEACHER_GATES))
    return out


def evaluate_hypotheses(row: dict) -> dict:
    """H7, H8, H9 (plan section 5) and the stop rule of section 6 that applies.
    H7: G5 and G1 hold. H8: G2 holds. H9: provisional pass (G1, G2, G3, G5).
    Stop rule 1 when H9 holds; rule 2 when H7 holds and H8 does not; rule 3 when H7 fails; None when
    H7 and H8 hold but H9 fails, which can only mean G3 failed. Section 6 item 4 adds no rule."""
    gates = gate_eval_teacher(row)
    g1 = gates["G1_groundlm_fp"]["passes"]
    g5 = gates["G5_test_donor_legit_fpr"]["passes"]
    g2 = gates["G2_smishx_smishing_tp"]["passes"]
    h7 = bool(g1 and g5)
    h8 = bool(g2)
    h9 = bool(gates["provisional_pass"])
    if h9:
        rule = 1
    elif h7 and not h8:
        rule = 2
    elif not h7:
        rule = 3
    else:
        rule = None
    return {
        "H7": {
            "text": "T1 meets G5 (test-donor legit FPR <= 1 %) and G1 (GroundLM FP <= 11)",
            "holds": h7,
        },
        "H8": {"text": "T1 meets G2 (SmishX TP >= 157 of 226) at the same threshold", "holds": h8},
        "H9": {"text": "T1 passes provisionally (G1, G2, G3, G5)", "holds": h9},
        "provisional_pass": h9,
        "stop_rule": rule,
        "stop_rule_text": STOP_TEXT[rule],
    }


def category_fpr(
    labels: Sequence[int], categories: Sequence[str], scores: Sequence[float], tau: float
) -> dict:
    """Legit FPR at tau per policy category (the categories named in CATEGORIES), Wilson 95 %.
    Also the legit row count per category. Category names are policy labels, not text."""
    y = np.asarray(labels, dtype=np.int64)
    c = np.asarray([str(x) for x in categories], dtype=object)
    s = np.asarray(scores, dtype=np.float64)
    legit = y == 0
    counts = {str(k): int(v) for k, v in zip(*np.unique(c[legit], return_counts=True), strict=True)}
    out: dict[str, Any] = {"legit_rows_by_category": counts}
    for cat in CATEGORIES:
        m = legit & (c == cat)
        out[cat] = {
            "n": int(m.sum()),
            "legit_fpr": xs.rate_of(int((s[m] >= tau).sum()), int(m.sum())),
        }
    return out


def saturation_share(scores: Sequence[float]) -> dict:
    """Rows scored exactly 1.0 (the saturated score). Counts and share."""
    s = np.asarray(scores, dtype=np.float64)
    k = int((s == 1.0).sum())
    n = int(s.size)
    return {"k": k, "n": n, "share": (k / n) if n else None}


def donor_block(
    labels: Sequence[int],
    categories: Sequence[str],
    scores: Sequence[float],
    tau: float,
    novel: Sequence[bool],
) -> dict:
    """Test-donor measures at tau: legit FPR (Wilson), scam recall (Wilson and Clopper-Pearson),
    template-novel subsets (inbox.donor_rates), the per-category FPR, and the saturation share.
    The labels, categories and novelty flags are arrays: no text enters this block."""
    out = inbox.donor_rates(np.asarray(labels), np.asarray(scores), tau, np.asarray(novel))
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    scam = y == 1
    out["scam_recall"] = xs.scam_recall_of(int((s[scam] >= tau).sum()), int(scam.sum()))
    out["by_category"] = category_fpr(y, categories, s, tau)
    out["saturated"] = saturation_share(s)
    return out


def pool_block(labels: Sequence[int], scores: Sequence[float], tau: float, kind: str) -> dict:
    """A legit pool (kind "legit": FPR) or a scam pool (kind "scam": recall), with saturation."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if kind == "legit":
        n = int(s.size)
        return {
            "n": n,
            "fpr": xs.rate_of(int((s >= tau).sum()), n),
            "saturated": saturation_share(s),
        }
    if kind == "scam":
        n = int(s.size)
        return {
            "n": n,
            "recall": xs.scam_recall_of(int((s >= tau).sum()), n),
            "saturated": saturation_share(s),
            "label1_rows": int((y == 1).sum()),
        }
    raise ValueError(f"unknown pool kind {kind!r}")


def d1_block(parts: dict, n_boot: int) -> dict:
    """The section 4 measures on one view. ``parts`` holds score arrays: sx_pos and sx_ham (SmishX
    label 1 and 0), pos226 (SmishX smishing rows only), gl (GroundLM policy negatives), donor_legit
    (test-donor legit, filled), camp_y and camp_s (campaign test, filled). Bootstrap intervals only
    when n_boot > 0. Only counts, AUCs and intervals leave this function."""
    pos, ham, pos226 = parts["sx_pos"], parts["sx_ham"], parts["pos226"]
    gl, leg = parts["gl"], parts["donor_legit"]
    out: dict[str, Any] = {
        "rows": {
            "smishx_positives": int(pos.size),
            "smishx_smishing_positives": int(pos226.size),
            "smishx_ham": int(ham.size),
            "groundlm_policy_negatives": int(gl.size),
            "donor_legit": int(leg.size),
            "campaign_test": int(parts["camp_y"].size),
        },
        "A1_smishx_vs_groundlm": xs.auc_of(pos, gl),
        "A2_smishx_vs_donor_filled": xs.auc_of(pos, leg),
        "auc_smishx_vs_smishx_ham": xs.auc_of(pos, ham),
        "campaign_test_filled": xs.auc_binary(parts["camp_y"], parts["camp_s"]),
        "A1_smishing_only": xs.auc_of(pos226, gl),
        "A2_smishing_only": xs.auc_of(pos226, leg),
    }
    if n_boot > 0:
        out["bootstrap"] = {
            "A1_smishx_vs_groundlm": xs.bootstrap_auc(pos, gl, n_boot, BOOT_SEED),
            "A2_smishx_vs_donor_filled": xs.bootstrap_auc(pos, leg, n_boot, BOOT_SEED),
        }
    return out


def scrubbed_by_attach(
    sub: pd.DataFrame, filled_source: pd.DataFrame, scrubbed_source: pd.DataFrame
) -> list[str]:
    """The scrubbed text of each row of ``sub`` (rows of the filled labelled donor). The row is
    matched to its id by d0.attach_row_ids on the filled (text, sender), then its scrubbed text is
    read by that id. The scrubbed frame must hold the same rows in the same order."""
    if not scrubbed_source["id"].astype(str).is_unique:
        raise ValueError("the scrubbed donor frame holds repeated row ids")
    ids = d0.attach_row_ids(sub, filled_source)
    lookup = dict(zip(scrubbed_source["id"].astype(str), scrubbed_source["text"], strict=True))
    return [str(lookup[i]) for i in ids]


def check_aligned(filled: pd.DataFrame, scrubbed: pd.DataFrame, name: str) -> None:
    """The filled and scrubbed labelled frames hold the same rows in the same order."""
    same_len = len(filled) == len(scrubbed)
    same_ids = same_len and bool(
        (filled["id"].astype(str).to_numpy() == scrubbed["id"].astype(str).to_numpy()).all()
    )
    same_labels = same_len and bool(
        (filled["label"].to_numpy() == scrubbed["label"].to_numpy()).all()
    )
    if not (same_len and same_ids and same_labels):
        raise ValueError(f"{name}: filled and scrubbed donor rows do not line up")


def fmt_rate(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.4f} [{lo:.4f}, {hi:.4f}]"


def fmt_auc(v: Any) -> str:
    return "n/a" if v is None else f"{float(v):.4f}"


# ---- report ----


def render_report(payload: dict) -> str:
    """Markdown report (report.md). Counts, rates, AUCs, thresholds and gates only."""
    lines = [f"# sms_teacher_finetune arm {payload.get('arm')}", ""]
    if payload.get("smoke"):
        lines += [
            "**SMOKE RUN: the numbers below are not results.** At most "
            f"{SMOKE_ROWS} training rows, {SMOKE_EPOCHS} epoch, at most {SMOKE_STEPS} "
            "steps. The pipeline is what is being checked.",
            "",
        ]
    if payload.get("status") != "scored":
        lines += [f"Status: **{payload.get('status')}**", "", payload.get("reason", ""), ""]
        return "\n".join(lines)
    tr = payload["training"]
    ck = payload["checkpoint"]
    th = payload["threshold"]
    lines += [
        f"Model {payload['model']}, seed {payload['seed']}. Training rows "
        f"{tr['used']['total']} (campaign {tr['used']['by_origin']['campaign']['total']}, "
        f"inbox {tr['used']['by_origin']['inbox']['total']}); label 1 {tr['used']['label1']}. "
        f"Epochs run {tr['epochs_run']}, best epoch {tr['best_epoch']}, campaign cal PR-AUC "
        f"{tr['cal_pr_auc']:.6f}. Training {tr['seconds']} s, "
        f"scoring {payload['timings']['score_s']} s.",
        "",
        f"Threshold tau = {th['tau_repr']} (fallback flag {th['fallback_flag']}), written before "
        "any test set was scored.",
        "",
        f"Checkpoint {ck['bytes']} bytes in total, weights file {ck['weights_bytes']} bytes "
        "(G4 is not a gate for a teacher).",
        "",
        "## Gates (one seed)",
        "",
        "| Gate | Value | Pass |",
        "|---|---|---|",
    ]
    g = payload["gates"]
    gr = payload["gate_row"]
    lines += [
        f"| G1 GroundLM FP / 1133 | {gr['groundlm_fp']} | {g['G1_groundlm_fp']['passes']} |",
        f"| G2 SmishX TP / 226 | {gr['smishx_tp']} | {g['G2_smishx_smishing_tp']['passes']} |",
        f"| G3 campaign F1 (filled) | {gr['campaign_f1']:.6f} | {g['G3_campaign_f1']['passes']} |",
        f"| G5 test-donor legit FPR | {gr['donor_fpr']:.6f} | "
        f"{g['G5_test_donor_legit_fpr']['passes']} |",
        "",
        f"**Provisional pass: {payload['hypotheses']['provisional_pass']}.** G4 not applicable.",
        "",
        "## Hypotheses and stop rule (plan sections 5 and 6)",
        "",
    ]
    h = payload["hypotheses"]
    for key in ("H7", "H8", "H9"):
        lines.append(f"- {key}: {h[key]['holds']}. {h[key]['text']}.")
    lines += ["", f"Stop rule: **{h['stop_rule']}**. {h['stop_rule_text']}", ""]
    for view in ("all", "clean"):
        m = payload["measures"][view]
        lines += [f"## Section 4 measures, {view} rows", "", "| Measure | Value |", "|---|---|"]
        for key in (
            "A1_smishx_vs_groundlm",
            "A2_smishx_vs_donor_filled",
            "auc_smishx_vs_smishx_ham",
            "campaign_test_filled",
            "A1_smishing_only",
            "A2_smishing_only",
        ):
            lines.append(f"| {key} | {fmt_auc(m[key])} |")
        for key, b in (m.get("bootstrap") or {}).items():
            ci = b["ci95"]
            txt = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            lines.append(f"| bootstrap 95 % {key} | {txt} |")
        lines.append("")
    cl = payload["contamination"]
    lines += [
        "## Rows in the training partition (exact canonical match)",
        "",
        "| Set | Rows | In training partition |",
        "|---|---|---|",
    ]
    for name, c in cl.items():
        lines.append(f"| {name} | {c['n']} | {c['contaminated']} |")
    dn = payload["donor"]
    lines += [
        "",
        "## Test donor (filled)",
        "",
        f"Legit FPR {fmt_rate(dn['legit_fpr'])}.",
        f"Template-novel legit FPR {fmt_rate(dn['legit_fpr_template_novel'])}.",
        f"Scam recall {fmt_rate(dn['scam_recall'])}; Clopper-Pearson 95 % "
        f"{dn['scam_recall']['clopper_pearson95']}.",
        "",
    ]
    for cat in CATEGORIES:
        c = dn["by_category"][cat]
        lines.append(f"- {cat}: n {c['n']}, legit FPR {fmt_rate(c['legit_fpr'])}")
    lines += [
        "",
        "## Test donor (scrubbed, tag-shift comparison)",
        "",
        f"Legit FPR {fmt_rate(payload['donor_scrubbed']['legit_fpr'])}; scam recall "
        f"{fmt_rate(payload['donor_scrubbed']['scam_recall'])}.",
        "",
        "## Held-out legit and val scam (filled and scrubbed)",
        "",
        f"Held-out legit FPR filled {fmt_rate(payload['legit_val']['fpr'])}; scrubbed "
        f"{fmt_rate(payload['legit_val_scrubbed']['fpr'])}.",
        f"Val scam recall filled {fmt_rate(payload['val_scam']['recall'])}; scrubbed "
        f"{fmt_rate(payload['val_scam_scrubbed']['recall'])}.",
        "",
        "## Saturation (share of rows scoring exactly 1.0)",
        "",
        "| Set | k | n | share |",
        "|---|---|---|---|",
    ]
    for name, s in payload["saturated"].items():
        share = "n/a" if s["share"] is None else f"{100 * s['share']:.2f} %"
        lines.append(f"| {name} | {s['k']} | {s['n']} | {share} |")
    lines += ["", "One seed. Codex labels. Test-donor scam rows are few: read the intervals.", ""]
    return "\n".join(lines)


# ---- the job ----


def redacted_error(exc: BaseException) -> str:
    return f"{type(exc).__name__} (message redacted)"


def redacted_frames(exc: BaseException) -> list[str]:
    """file:line in function for each frame. Source text and the message are left out."""
    import traceback

    return [f"{f.filename}:{f.lineno} in {f.name}" for f in traceback.extract_tb(exc.__traceback__)]


class RedactedError(RuntimeError):
    """The job failed; the cause is in the log as type and frames, not as a message."""


def run(ctx) -> bool:
    """Runs _run; any exception leaves here as RedactedError (type and frames logged)."""
    failure = None
    try:
        return _run(ctx)
    except Exception as exc:
        failure = exc
        for line in redacted_frames(exc):
            ctx.log(f"  at {line}")
        ctx.log(f"ERROR {redacted_error(exc)}")
    # Raised outside the except block so the original message is not chained onto the error.
    raise RedactedError(redacted_error(failure)) from None


def _write(work: Path, payload: dict) -> None:
    """metrics.json and report.md in ``work``. No row-level file is written."""
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(work / "metrics.json")
    (work / "report.md").write_text(render_report(payload))


def _write_thresholds(work: Path, thresholds: dict) -> None:
    (work / "thresholds.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")


def _run(ctx) -> bool:
    args = parse_args(ctx.argv)
    arm = args.arm
    hp = effective_hparams(args)
    seed_everything(args.seed)
    # Imported here: torch is needed only on the kernel, not for the pure helpers.
    import sms_teacher as T
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("sms_teacher_finetune needs a CUDA GPU; launch with --gpu t4")
    device = torch.device("cuda:0")
    env = env_report()
    env["device"] = str(device)
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})
    work = ctx.work

    # 1. campaign data, filled (rule 4 applies to the external sets only)
    root = D.resolve_data_root(None)
    ctx.log(f"data root: {root}")
    sms = D.load_sms(root)
    sms, sms_fill_stats, sms_scrubbed = inbox.fill_sms_frame(sms)
    inbox.assert_filled(base.str_texts(sms["text"]), "sms.parquet")
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    sms_texts = base.str_texts(sms["text"])
    mask = D.eval_mask(sms)
    runs = D.load_runs(
        root, sms, patterns=["campaign", *base.SHIFT_PREFIXES], labels=labels, mask=mask
    )
    campaign_runs = [r for r in runs if r.name.split(":")[0] == "campaign"]
    if not campaign_runs:
        raise RuntimeError("no campaign split run found")
    run_ = campaign_runs[0]
    shift_runs = [r for r in runs if r.name.split(":")[0] in base.SHIFT_PREFIXES]
    y_cal = labels[run_.cal]
    y_test = labels[run_.test]
    clusters_test = D.cluster_values(sms, run_.test, base.SMS_CLUSTER_COLUMN)

    # 2. GroundLM and SmishX, scrubbed, as sms_retrain loads them
    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    gl_avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    gl_ok = gl["eval_ok"].fillna(False).astype(bool).to_numpy()
    gl_pol = gl["policy_negative"].fillna(False).astype(bool).to_numpy()
    neg_eval = gl_pol & gl_avail & gl_ok
    if int(neg_eval.sum()) != base.GROUNDLM_SCOREABLE:
        raise RuntimeError(
            f"expected {base.GROUNDLM_SCOREABLE} scoreable negatives, got {int(neg_eval.sum())}"
        )
    gl_avail_idx = np.flatnonzero(gl_avail)
    gl_texts = base.str_texts(gl["text"].to_numpy()[gl_avail_idx])
    gl_g1 = neg_eval[gl_avail_idx]
    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    sx_avail_idx = np.flatnonzero(sx_avail)
    sx_texts = base.str_texts(sx["text"].to_numpy()[sx_avail_idx])
    sx_label = sx["label_binary"].to_numpy().astype(np.int64)[sx_avail_idx]
    sx_fine = sx["label_fine"].astype(str).to_numpy()[sx_avail_idx]
    sx_ok = sx["eval_ok"].fillna(False).astype(bool).to_numpy()[sx_avail_idx]
    sx_226 = (sx_fine == "smishing") & sx_ok

    # 3. inbox donors: filled and scrubbed frames, labelled row for row
    roots = [Path(DEFAULT_ROOT)]
    train_scrub_raw = pd.read_parquet(inbox.find_one(inbox.TRAIN_FILE, roots))
    test_scrub_raw = pd.read_parquet(inbox.find_one(inbox.TEST_FILE, roots))
    train_raw, train_fill_stats = inbox.fill_inbox_frame(train_scrub_raw)
    test_raw, test_fill_stats = inbox.fill_inbox_frame(test_scrub_raw)
    inbox.assert_filled(list(train_raw["text"]), "train donor")
    inbox.assert_filled(list(test_raw["text"]), "test donor")
    train_df, train_counts = inbox.label_inbox(train_raw)
    test_df, test_counts = inbox.label_inbox(test_raw)
    train_scrub_df, _ = inbox.label_inbox(train_scrub_raw)
    test_scrub_df, _ = inbox.label_inbox(test_scrub_raw)
    del train_raw, test_raw, train_scrub_raw, test_scrub_raw
    check_aligned(train_df, train_scrub_df, "train donor")
    check_aligned(test_df, test_scrub_df, "test donor")
    ctx.log(f"train donor counts: {train_counts}")
    ctx.log(f"test donor counts: {test_counts}")

    # 4. pools, with the S1 test-only exclusion (filled texts, as S1)
    test_only = [sms_texts[i] for i in run_.test]
    for r in shift_runs:
        test_only += [sms_texts[i] for i in r.test]
    test_only += list(gl_texts) + list(sx_texts) + [t for t in test_df["text"] if t]
    test_only = [t for t in test_only if t]
    n_campaign_neg = int((labels[run_.train] == 0).sum())
    with timed("prep_s", timings):
        pools = inbox.build_inbox_pools(train_df, test_only, n_campaign_neg)
    prep: dict[str, Any] = {
        "train_donor": train_counts,
        "test_donor": test_counts,
        "test_only_texts": len(test_only),
        "fill": {
            "sms_by_source_before_fill": sms_fill_stats,
            "train_donor_before_fill": train_fill_stats,
            "test_donor_before_fill": test_fill_stats,
        },
        "stages": pools["report"]["stages"],
    }
    reason = inbox.stop_reason(pools["report"])
    if reason is not None:
        payload = {
            "job": JOB,
            "arm": arm,
            "status": inbox.STOP_STATUS,
            "reason": reason,
            "smoke": bool(args.smoke),
            "prep": prep,
        }
        _write(work, payload)
        ctx.results["summary"] = f"{JOB} arm {arm}: {inbox.STOP_STATUS}: {reason}"
        ctx.log(inbox.STOP_STATUS + ": " + reason)
        return False
    ctx.log(
        f"prep: train_extra {pools['report']['stages']['train_extra_total']}, "
        f"legit_val {pools['report']['stages']['val_legit']}, "
        f"val_scam {pools['report']['stages']['val_scam']}"
    )

    # 5. training rows (full partition first; the smoke subset comes after the contamination keys)
    extra = pools["train_extra"] if arm == "T1" else pools["train_extra"].iloc[0:0]
    base_idx = run_.train
    train_texts_full, train_y_full, origin_full = training_rows(
        arm,
        [sms_texts[i] for i in base_idx],
        labels[base_idx],
        list(extra["text"]),
        extra["label"].to_numpy(),
        test_only,
    )
    scrub_extra = scrubbed_by_attach(extra, train_df, train_scrub_df) if len(extra) else []
    train_scrub_full = [sms_scrubbed[i] for i in base_idx] + scrub_extra
    inbox.assert_filled(train_texts_full, "training rows")
    keys_filled = xs.canon_keys(train_texts_full)
    keys_scrub = xs.canon_keys(train_scrub_full)
    full_counts = origin_counts(train_y_full, origin_full)
    if args.smoke:
        keep = smoke_indices(train_y_full, smoke_row_cap(args.batch_size), args.seed)
        train_texts = [train_texts_full[i] for i in keep]
        train_y = train_y_full[keep]
        origin = origin_full[keep]
    else:
        train_texts, train_y, origin = train_texts_full, train_y_full, origin_full
    used_counts = origin_counts(train_y, origin)
    ctx.log(
        f"training rows (full) {full_counts['total']}: label0 {full_counts['label0']}, "
        f"label1 {full_counts['label1']}, origins {full_counts['by_origin']}"
    )
    ctx.log(
        f"training rows used {used_counts['total']} (steps at batch {args.batch_size}: "
        f"{-(-used_counts['total'] // args.batch_size)} per epoch), epochs {hp['epochs']}"
    )

    # 6. sets: filled (campaign cal and test, held-out legit, val scam, test donor) and scrubbed
    legit_texts = list(pools["val_legit"]["text"])
    scam_texts = list(pools["val_scam"]["text"])
    legit_scrub = scrubbed_by_attach(pools["val_legit"], train_df, train_scrub_df)
    scam_scrub = scrubbed_by_attach(pools["val_scam"], train_df, train_scrub_df)
    test_texts = [str(t) for t in test_df["text"]]
    test_scrub = [str(t) for t in test_scrub_df["text"]]
    y_donor = test_df["label"].to_numpy().astype(np.int64)
    donor_cat = test_df["label_policy"].astype(str).to_numpy()
    novel = inbox.novel_mask(test_texts, list(pools["train_extra"]["text"]))
    cal_texts = [sms_texts[i] for i in run_.cal]
    camp_texts = [sms_texts[i] for i in run_.test]
    inbox.assert_filled(legit_texts, "held-out legit")
    inbox.assert_filled(scam_texts, "val scam")
    inbox.assert_filled(test_texts, "test donor")
    inbox.assert_filled(cal_texts, "campaign cal")
    inbox.assert_filled(camp_texts, "campaign test")
    set_texts = {
        SET_CAL: cal_texts,
        SET_LEGIT_VAL: legit_texts,
        SET_CAMP: camp_texts,
        SET_GL: list(gl_texts),
        SET_SX: list(sx_texts),
        SET_VS: scam_texts,
        SET_DONOR: test_texts,
        SET_LEGIT_VAL + SUFFIX_SCRUB: legit_scrub,
        SET_VS + SUFFIX_SCRUB: scam_scrub,
        SET_DONOR + SUFFIX_SCRUB: test_scrub,
    }
    for name in (
        SET_GL,
        SET_SX,
        SET_LEGIT_VAL + SUFFIX_SCRUB,
        SET_VS + SUFFIX_SCRUB,
        SET_DONOR + SUFFIX_SCRUB,
    ):
        if not set_texts[name]:
            raise ValueError(f"set {name} is empty")

    # 7. fine-tune (sms_teacher.train_once, the fixed section 2 hyperparameters)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cfg = T.Config(
        fp16=TRAIN_FP16,
        max_len=args.max_len,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        patience=args.patience,
        adv_aug=0.0,
        train_canon=True,
        log=ctx.log,
    )
    data = {
        "train_raw": train_texts,
        "train_y": train_y,
        "cal_y": y_cal,
        "cal_prepped": [D.canon_text(t) for t in cal_texts],
    }
    with timed("train_s", timings):
        out = T.train_once(cfg, hp, args.model, args.seed, tokenizer, device, data, float("inf"))
    model = out["model"].to(device)
    ctx.log(
        f"best epoch {out['best_epoch']} cal PR-AUC {out['cal_pr_auc']:.6f} after "
        f"{len(out['history'])} epoch(s) ({out['seconds']} s)"
    )

    # Checkpoint: weights and tokenizer files, as sms_teacher saves best_checkpoint
    ckpt = work / BEST_DIR
    shutil.rmtree(ckpt, ignore_errors=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    ckpt_bytes = int(sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file()))
    weights_bytes = int((ckpt / "model.safetensors").stat().st_size)
    ctx.log(f"checkpoint: {ckpt_bytes} bytes, weights {weights_bytes} bytes")

    # 8. threshold first (run_seed_pipeline), then the post-threshold sets
    thresholds: dict[str, Any] = {
        "arm": arm,
        "rule": THRESHOLD_RULE,
        "seed": args.seed,
        "smoke": bool(args.smoke),
    }

    def score(model_, name: str) -> np.ndarray:
        return xs.teacher_probs(
            model_,
            tokenizer,
            set_texts[name],
            device=device,
            canon=True,
            max_len=args.max_len,
            eval_batch_size=args.eval_batch_size,
        )

    def on_threshold(tau: float, fallback: bool) -> None:
        thresholds.update(
            {
                "tau": tau,
                "tau_repr": repr(tau),
                "fallback_flag": bool(fallback),
                "n_legit_val": len(legit_texts),
                "n_campaign_cal": int(len(y_cal)),
            }
        )
        _write_thresholds(work, thresholds)
        ctx.log(f"tau {repr(tau)} fallback {fallback} written before any test set")

    post = [
        SET_CAMP,
        SET_GL,
        SET_SX,
        SET_VS,
        SET_DONOR,
        SET_LEGIT_VAL + SUFFIX_SCRUB,
        SET_VS + SUFFIX_SCRUB,
        SET_DONOR + SUFFIX_SCRUB,
    ]
    with timed("score_s", timings):
        pipe = base.run_seed_pipeline(
            fit=lambda: model,
            score=score,
            rule=RP.threshold_rule,
            y_cal=y_cal,
            post_sets=post,
            on_threshold=on_threshold,
        )
    tau = float(pipe["tau"])
    s = pipe["scores"]
    ctx.log(f"scored: {', '.join(sorted(s))}; events {pipe['events']}")

    # 9. gates and the teacher tables (the S-arm calls, same rows and masks)
    gl_rules = base.expand(len(gl), gl_avail_idx, rules_any(gl_texts), False)
    sx_rules = base.expand(len(sx), sx_avail_idx, rules_any(sx_texts), False)
    gl_full = base.expand(len(gl), gl_avail_idx, s[SET_GL], -1.0)
    sx_full = base.expand(len(sx), sx_avail_idx, s[SET_SX], -1.0)
    campaign = sms_run_table(
        y_cal, s[SET_CAL], y_test, s[SET_CAMP], clusters_test, threshold=tau, n_boot=args.n_boot
    )
    gl_block = groundlm_table(gl, gl_full, gl_rules, tau)
    sx_block = smishx_table(sx, sx_full, sx_rules, tau)
    donor = donor_block(y_donor, donor_cat, s[SET_DONOR], tau, novel)
    donor_scrub = donor_block(y_donor, donor_cat, s[SET_DONOR + SUFFIX_SCRUB], tau, novel)
    gate_row = {
        "groundlm_fp": int(gl_block["student_at_threshold"]["k"]),
        "smishx_tp": int(sx_block["smishing_recall"]["student_at_threshold"]["k"]),
        "campaign_f1": float(campaign["shipped_threshold"]["f1"]),
        "donor_fpr": float(donor["legit_fpr"]["rate"]),
    }
    gates = gate_eval_teacher(gate_row)
    hyp = evaluate_hypotheses(gate_row)
    if arm != "T1":
        hyp["stop_rule"] = None
        hyp["stop_rule_text"] = "Not applicable: section 6 applies to T1. Reported for T0 only."

    # 10. section 4 measures, all rows and clean rows (not in this arm's training partition)
    gl_g1_texts = [t for t, keep in zip(gl_texts, gl_g1, strict=True) if keep]
    s_gl_g1 = s[SET_GL][gl_g1]
    # Contamination per set, against the training partition in the same form (filled or scrubbed)
    contam_sets = {
        "smishx_scrubbed": (list(sx_texts), keys_scrub),
        "groundlm_g1_scrubbed": (gl_g1_texts, keys_scrub),
        "campaign_test_filled": (camp_texts, keys_filled),
        "test_donor_filled": (test_texts, keys_filled),
        "test_donor_scrubbed": (test_scrub, keys_scrub),
        "heldout_legit_filled": (legit_texts, keys_filled),
        "heldout_legit_scrubbed": (legit_scrub, keys_scrub),
        "val_scam_filled": (scam_texts, keys_filled),
        "val_scam_scrubbed": (scam_scrub, keys_scrub),
    }
    contamination: dict[str, Any] = {}
    clean: dict[str, np.ndarray] = {}
    for name, (texts_, keys_) in contam_sets.items():
        clean_mask = ~xs.contamination_mask(texts_, keys_)
        contamination[name] = {"n": int(len(texts_)), "contaminated": int((~clean_mask).sum())}
        clean[name] = clean_mask

    def measures(clean_only: bool) -> dict:
        def keep(name: str) -> np.ndarray:
            if clean_only:
                return clean[name]
            return np.ones(len(contam_sets[name][0]), dtype=bool)

        k_sx, k_gl = keep("smishx_scrubbed"), keep("groundlm_g1_scrubbed")
        k_donor, k_camp = keep("test_donor_filled"), keep("campaign_test_filled")
        sx_y = sx_label[k_sx]
        sx_s = s[SET_SX][k_sx]
        sx_p226 = sx_226[k_sx]
        don_y = y_donor[k_donor]
        don_s = s[SET_DONOR][k_donor]
        parts = {
            "sx_pos": sx_s[sx_y == 1],
            "sx_ham": sx_s[sx_y == 0],
            "pos226": sx_s[(sx_y == 1) & sx_p226],
            "gl": s_gl_g1[k_gl],
            "donor_legit": don_s[don_y == 0],
            "camp_y": y_test[k_camp],
            "camp_s": s[SET_CAMP][k_camp],
        }
        return d1_block(parts, args.n_boot if clean_only else 0)

    measures_block = {"all": measures(False), "clean": measures(True)}

    payload = {
        "job": JOB,
        "arm": arm,
        "status": "scored",
        "plan": PLAN,
        "smoke": bool(args.smoke),
        "model": args.model,
        "seed": args.seed,
        "hyperparameters": {
            **hp,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "patience": args.patience,
            "precision": "fp16 autocast over fp32 weights",
            "adv_aug": 0.0,
            "train_input": "canon",
            "smoke_row_cap": smoke_row_cap(args.batch_size) if args.smoke else None,
        },
        "args": {k: v for k, v in vars(args).items()},
        "env": env,
        "prep": prep,
        "training": {
            "full": full_counts,
            "used": used_counts,
            "epochs_run": len(out["history"]),
            "best_epoch": int(out["best_epoch"]),
            "cal_pr_auc": float(out["cal_pr_auc"]),
            "history": [
                {"epoch": int(h["epoch"]), "cal_pr_auc": float(h["cal_pr_auc"])}
                for h in out["history"]
            ],
            "seconds": out["seconds"],
            "stopped_by_budget": bool(out["stopped_by_budget"]),
        },
        "threshold": {
            "tau": tau,
            "tau_repr": repr(tau),
            "fallback_flag": bool(pipe["fallback"]),
            "rule": THRESHOLD_RULE,
        },
        "checkpoint": {"folder": BEST_DIR, "bytes": ckpt_bytes, "weights_bytes": weights_bytes},
        "gate_row": gate_row,
        "gates": gates,
        "hypotheses": hyp,
        "campaign": campaign,
        "groundlm": gl_block,
        "smishx": sx_block,
        "donor": donor,
        "donor_scrubbed": donor_scrub,
        "legit_val": pool_block(np.zeros(len(legit_texts)), s[SET_LEGIT_VAL], tau, "legit"),
        "legit_val_scrubbed": pool_block(
            np.zeros(len(legit_scrub)), s[SET_LEGIT_VAL + SUFFIX_SCRUB], tau, "legit"
        ),
        "val_scam": pool_block(np.ones(len(scam_texts)), s[SET_VS], tau, "scam"),
        "val_scam_scrubbed": pool_block(
            np.ones(len(scam_scrub)), s[SET_VS + SUFFIX_SCRUB], tau, "scam"
        ),
        "measures": measures_block,
        "contamination": contamination,
        "saturated": {name: saturation_share(s[name]) for name in sorted(s)},
        "pipeline_events": pipe["events"],
        "timings": timings,
    }
    _write(work, payload)
    smoke_tag = " SMOKE" if args.smoke else ""
    ctx.results["summary"] = (
        f"{JOB} arm {arm}{smoke_tag}: provisional_pass {hyp['provisional_pass']}, "
        f"H7 {hyp['H7']['holds']}, H8 {hyp['H8']['holds']}, stop rule {hyp['stop_rule']}, "
        f"tau {repr(tau)}, cal PR-AUC {out['cal_pr_auc']:.6f} at epoch {out['best_epoch']}"
    )
    return True
