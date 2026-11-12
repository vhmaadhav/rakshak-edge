"""Teacher cross-source ranking check D1 (docs/TEACHER_XSOURCE_PLAN.md, frozen). Scoring only.

Scores the SMS teacher and the shipped v0 SMS student on the same rows, then applies the frozen
decision rule (plan section 4) to the teacher only. v0 is a same-row reference.

Teacher (plan section 2). The checkpoint is the ``best_checkpoint`` folder of kernel
maadhavvh/rakshak-sms-teacher, found under --teacher-dir by the sha256 of its model.safetensors
(673b970e...). Its provenance is read from the ``metrics.json`` that sits next to the folder
(kaggle/out/sms_teacher/output/metrics.json locally), field ``meta``:
- ``best_checkpoint``: model jhu-clsp/mmBERT-base, run ``source_holdout:holdout_scamshield``,
  seed 0, cal_pr_auc 0.99988. That is the best cal PR-AUC over every run of the teacher job, and it
  comes from a SOURCE-HOLDOUT fold (the scamshield source is held out of its training partition),
  not from the campaign split. The plan does not name the run; this job uses the run in meta.
- ``train_input``: "canon". The job scores canon(text) unless --train-input raw is given, and fails
  if it disagrees with metrics.json.
- ``precision``: fp16 autocast over fp32 weights (fp16=True in predict_logits).
- ``max_len`` and ``eval_batch_size`` are NOT recorded in metrics.json. The job uses the sms_teacher
  defaults (128, 128), which is what a launch without those flags would have used.
If metrics.json is not attached, --teacher-run must name the run (it is checked against meta when
both are present). The training partition of that run is resolved the way sms_teacher resolves
splits (rakshak.eval.data.load_runs on the split files named by the run's stem). Contamination is
measured against that partition only.

PRIVACY (binding). No inbox text, no sender string and no inbox row id leaves memory. Logs,
metrics.json and report.md carry only counts, rates, Clopper-Pearson and Wilson intervals, AUCs,
bootstrap intervals, thresholds, quantiles and sha256 digests. Row ids are used in memory only (to
seed the surrogate fill and to match held-out rows). No row-level file is written. Errors are
logged as exception type and frames only (REDACT_ERRORS).

Sets (plan section 2; same builders as sms_inbox_retrain and sms_surrogate_d0):
- SmishX (external_smishx, text available): label-1 rows are the scam side of A1 and A2; label-0
  rows are "SmishX ham". Scrubbed only (plan rule 4).
- GroundLM policy negatives (G1, 1,133 rows): legit. Scrubbed only.
- Test donor (phone 1, labelled rows): scrubbed and filled (rakshak.data.surrogates, seed 20261107).
- Held-out legit of phones 2+3 (sender-disjoint 20 %): scrubbed and filled. Reported, not a measure.
- Campaign test split (all rows of the campaign test partition): scrubbed and filled.

Pipeline:
1. Teacher folder and sha check, then metrics.json provenance (before any data is read).
2. v0 ONNX located by sha (sms_surrogate_d0.locate_v0).
3. Campaign, lang_shift and temporal runs, GroundLM, SmishX exactly as sms_surrogate_d0 loads them.
4. Inbox donors, train-donor pools (inbox.build_inbox_pools, same caps and seed), held-out
   legit ids.
5. Teacher training partition (the run's train rows), canonicalised; exact-match contamination
   per set on the canonical scrubbed text.
6. Scoring. Teacher: sms_teacher.predict_logits, probability of class 1 = softmax of the two logits
   (sms_teacher.probs_from_logits). v0: baseline_v0.score_sms_texts. Filled texts pass
   surrogates.assert_no_tags first.
7. Measures per model, in two views: "all" rows, and "clean" rows (not in the training partition).
   The decision uses the teacher's clean view.
8. Outputs: metrics.json, report.md, ctx.results["summary"].

Quantile convention for the recall thresholds: for a keep percentage p and n SmishX label-1 rows,
k = ceil(n * p / 100) (integer arithmetic). The threshold is the k-th largest positive score, and a
row is flagged when score >= threshold (the same >= as the FPR). At least k positives are kept; ties
can keep more, and the achieved recall is reported.

AUC: sklearn.metrics.roc_auc_score. Bootstrap: 2,000 resamples within class (positives and negatives
drawn separately with replacement), numpy default_rng(0), percentile 95 % interval.

Decision (plan section 4), on the teacher's clean view, A1 = SmishX label-1 vs GroundLM legit and
A2 = SmishX label-1 vs filled test-donor legit:
- transfers: A1 >= 0.98 and A2 >= 0.98
- does_not_transfer: A1 < 0.93 or A2 < 0.85
- partial: otherwise

Launch (T4, internet on, the two datasets, the student and teacher kernels; the teacher kernel
output supplies best_checkpoint and metrics.json):
  kaggle/launch.sh sms_teacher_xsource --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student maadhavvh/rakshak-sms-teacher \
      --job-args="--teacher-dir /kaggle/input --teacher-run source_holdout:holdout_scamshield \
      --train-input canon --v0-onnx /kaggle/input"

Flags (``--job-args="FLAGS"``):
  --teacher-dir PATH         folder searched for best_checkpoint/model.safetensors with the pinned
                             sha256 (default /kaggle/input)
  --teacher-run NAME         split run of the teacher's training partition, e.g.
                             source_holdout:holdout_scamshield (default: from metrics.json)
  --train-input {canon,raw}  form the teacher was trained on (default canon, as metrics.json says)
  --max-len INT              teacher token limit (default 128)
  --eval-batch-size INT      teacher inference batch (default 128)
  --v0-onnx PATH             folder searched for best_student.onnx with the pinned sha256
                             (default /kaggle/input)
  --n-boot INT               bootstrap resamples (default 2000)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sms_inbox_retrain as inbox
import sms_retrain as base
import sms_surrogate_d0 as d0
from _common import env_report, seed_everything, timed
from baseline_v0 import ARTIFACT_PINS, _onnx_session, score_sms_texts
from sklearn.metrics import roc_auc_score

from rakshak.data import surrogates as S
from rakshak.eval import data as D

JOB = "sms_teacher_xsource"
PLAN = "docs/TEACHER_XSOURCE_PLAN.md"
TEACHER_DIR = "best_checkpoint"
TEACHER_WEIGHTS = "model.safetensors"
TEACHER_SHA = "673b970e9f4746dbbbb5a00aa84eb056d6750cbc2897ba712ab1aa1b234abc7e"
DEFAULT_ROOT = "/kaggle/input"
V0_SHA = ARTIFACT_PINS["best_student.onnx"][0]
N_BOOT = 2000
BOOT_SEED = 0
KEEP_PCTS = (70, 90)
TRANSFER_MIN = 0.98
FAIL_A1 = 0.93
FAIL_A2 = 0.85
MAX_LEN = 128
EVAL_BATCH = 128
TRAIN_INPUTS = ("canon", "raw")

SET_SMISHX = "smishx"
SET_GL = "groundlm_legit"
SET_DONOR = "test_donor"
SET_HELD = "heldout_legit"
SET_CAMP = "campaign_test"
SETS = (SET_SMISHX, SET_GL, SET_DONOR, SET_HELD, SET_CAMP)
FILLED_SETS = (SET_DONOR, SET_HELD, SET_CAMP)
VERSIONS = ("scrubbed", "filled")
MODELS = ("teacher", "v0")
VIEWS = ("all", "clean")

DECISION_RULE = (
    "transfers if A1 >= 0.98 and A2 >= 0.98; does_not_transfer if A1 < 0.93 or A2 < 0.85; "
    "partial otherwise. A1 = SmishX label-1 vs GroundLM legit, A2 = SmishX label-1 vs filled "
    "test-donor legit, both on the rows not in the teacher's training partition."
)
NEXT_PLAN = {
    "transfers": "Distil a larger student from this teacher on filled data, with gates G1 to G5.",
    "does_not_transfer": (
        "Fine-tune the teacher on the S1 data (filled campaign train plus the inbox train donor) "
        "to test whether a pretrained encoder can use the inbox negatives."
    ),
    "partial": ("The same fine-tune. Distillation is not started from the current checkpoint."),
}
# Read by kaggle/build_kernel.py: errors are logged as type and frames only, never the message.
REDACT_ERRORS = True


# ---- pure helpers (unit-tested in tests/test_sms_teacher_xsource.py) ----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--teacher-dir", default=DEFAULT_ROOT)
    p.add_argument("--teacher-run", default=None)
    p.add_argument("--train-input", choices=TRAIN_INPUTS, default="canon")
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH)
    p.add_argument("--v0-onnx", default=DEFAULT_ROOT)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args(list(argv or []))
    if args.max_len < 1 or args.eval_batch_size < 1 or args.n_boot < 1:
        p.error("--max-len, --eval-batch-size and --n-boot must be positive")
    return args


def teacher_meta(metrics: dict) -> dict:
    """The provenance fields of the teacher job's metrics.json (``meta``). Raises if absent."""
    meta = metrics.get("meta") or {}
    best = meta.get("best_checkpoint") or {}
    if not best.get("run") or "train_input" not in meta:
        raise ValueError("teacher metrics.json lacks meta.best_checkpoint.run or meta.train_input")
    return {
        "run": str(best["run"]),
        "model": str(best.get("model")),
        "seed": int(best.get("seed", 0)),
        "cal_pr_auc": float(best.get("cal_pr_auc", float("nan"))),
        "train_input": str(meta["train_input"]),
    }


def resolve_teacher_run(from_metrics: str | None, from_args: str | None) -> str:
    """The teacher's split run. Both sources must agree when both are given."""
    if from_metrics and from_args and from_metrics != from_args:
        raise ValueError("--teacher-run differs from the run recorded in the teacher metrics.json")
    run = from_metrics or from_args
    if not run:
        raise ValueError(
            "no teacher run: metrics.json is not next to best_checkpoint, so pass --teacher-run"
        )
    if ":" not in run:
        raise ValueError("--teacher-run must be split:partition, e.g. source_holdout:holdout_x")
    return run


def check_train_input(recorded: str, requested: str) -> None:
    if recorded != requested:
        raise ValueError(
            f"--train-input {requested} differs from the teacher's recorded {recorded}"
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_teacher(root: Path) -> Path:
    """The best_checkpoint folder under ``root`` whose model.safetensors has the pinned sha256.
    Errors carry names and digest prefixes only, never file contents."""
    if not root.exists():
        raise FileNotFoundError(f"--teacher-dir {root} does not exist")
    folders = sorted(
        {
            Path(d)
            for d, _dirs, files in os.walk(root)
            if Path(d).name == TEACHER_DIR and TEACHER_WEIGHTS in files
        }
    )
    if not folders:
        raise FileNotFoundError(f"no {TEACHER_DIR}/{TEACHER_WEIGHTS} under {root}")
    found = {str(f): file_sha256(f / TEACHER_WEIGHTS) for f in folders}
    good = [f for f in folders if found[str(f)] == TEACHER_SHA]
    if not good:
        shas = sorted(v[:12] for v in found.values())
        raise ValueError(
            f"teacher checkpoint sha256 mismatch: no {TEACHER_DIR}/{TEACHER_WEIGHTS} "
            f"under {root} has the pinned digest {TEACHER_SHA[:12]} (found {shas})"
        )
    if len(good) > 1:
        raise FileNotFoundError(f"{len(good)} {TEACHER_DIR} folders match the pinned digest")
    return good[0]


def canon_keys(texts: Sequence[str], canon=D.canon_text) -> set[str]:
    """Canonical forms of ``texts`` (empty strings skipped)."""
    return {canon(t) for t in texts if t}


def contamination_mask(
    texts: Sequence[str], train_keys: set[str], canon=D.canon_text
) -> np.ndarray:
    """True for each text whose canonical form is in the training partition (exact match)."""
    return np.fromiter((canon(t) in train_keys for t in texts), dtype=bool, count=len(texts))


def auc_binary(y: np.ndarray, s: np.ndarray) -> float | None:
    """roc_auc_score on binary labels, or None when one class is absent."""
    y = np.asarray(y).astype(np.int64)
    if y.size == 0 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(s, dtype=np.float64)))


def auc_of(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """AUC of positive scores against negative scores (positives ranked above)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    y = np.concatenate([np.ones(pos.size, dtype=np.int64), np.zeros(neg.size, dtype=np.int64)])
    return float(roc_auc_score(y, np.concatenate([pos, neg])))


def bootstrap_auc(pos: np.ndarray, neg: np.ndarray, n_boot: int, seed: int) -> dict:
    """Percentile 95 % interval of auc_of, resampling positives and negatives within class."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return {"point": None, "ci95": None, "n_boot": 0, "seed": int(seed)}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        p = pos[rng.integers(0, pos.size, pos.size)]
        n = neg[rng.integers(0, neg.size, neg.size)]
        vals[b] = auc_of(p, n)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "point": auc_of(pos, neg),
        "ci95": [float(lo), float(hi)],
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def keep_threshold(pos: np.ndarray, keep_pct: int) -> float:
    """Threshold keeping at least ``keep_pct`` % of the positive scores. With n positives,
    k = ceil(n * keep_pct / 100) by integer arithmetic; the threshold is the k-th largest score.
    A row is flagged when score >= threshold."""
    s = np.sort(np.asarray(pos, dtype=np.float64))[::-1]
    n = int(s.size)
    if n == 0:
        raise ValueError("keep_threshold needs at least one positive score")
    if not 0 < keep_pct <= 100:
        raise ValueError("keep_pct must be in (0, 100]")
    k = -((-n * int(keep_pct)) // 100)
    return float(s[k - 1])


def rate_of(k: int, n: int) -> dict:
    """k of n with a Wilson 95 % interval; rate None when n is 0."""
    if n <= 0:
        return {"k": 0, "n": 0, "rate": None, "wilson95": None}
    return d0.rate(int(k), int(n))


def scam_recall_of(k: int, n: int) -> dict:
    """Recall with a Wilson and an exact Clopper-Pearson 95 % interval."""
    out = rate_of(k, n)
    cp = inbox.clopper_pearson(int(k), int(n))
    out["clopper_pearson95"] = None if cp is None else [round(x, 4) for x in cp]
    return out


def decide(a1: float | None, a2: float | None) -> dict:
    """The plan section 4 verdict. Boundaries: >= for transfer, strict < for the fail rule."""
    if a1 is None or a2 is None:
        raise ValueError("headline AUC is undefined: no rows on one side (A1 or A2)")
    if a1 >= TRANSFER_MIN and a2 >= TRANSFER_MIN:
        verdict = "transfers"
    elif a1 < FAIL_A1 or a2 < FAIL_A2:
        verdict = "does_not_transfer"
    else:
        verdict = "partial"
    return {
        "verdict": verdict,
        "a1": float(a1),
        "a2": float(a2),
        "rule": DECISION_RULE,
        "next_plan": NEXT_PLAN[verdict],
    }


def _rows(sets: dict, scores: dict, name: str, version: str, clean_only: bool):
    """(labels, scores) of one set and version, restricted to clean rows when asked."""
    entry = sets[name]
    keep = entry["clean"] if clean_only else np.ones(len(entry["labels"]), dtype=bool)
    return entry["labels"][keep], scores[name][version][keep]


def view_block(sets: dict, scores: dict, clean_only: bool, n_boot: int) -> dict:
    """Counts, AUCs, thresholds with FPR and recall, and (clean view) bootstrap intervals for one
    model. Only counts, rates, AUCs and thresholds leave this function."""
    y_sx, s_sx = _rows(sets, scores, SET_SMISHX, "scrubbed", clean_only)
    pos = s_sx[y_sx == 1]
    ham = s_sx[y_sx == 0]
    _, gl = _rows(sets, scores, SET_GL, "scrubbed", clean_only)
    leg = {}
    scam = {}
    held = {}
    for v in VERSIONS:
        y_d, s_d = _rows(sets, scores, SET_DONOR, v, clean_only)
        leg[v] = s_d[y_d == 0]
        scam[v] = s_d[y_d == 1]
    for v in VERSIONS:
        y_h, s_h = _rows(sets, scores, SET_HELD, v, clean_only)
        held[v] = s_h[y_h == 0]
    camp = {}
    for v in VERSIONS:
        camp[v] = _rows(sets, scores, SET_CAMP, v, clean_only)
    auc = {
        "A1_smishx_vs_groundlm": auc_of(pos, gl),
        "A2_smishx_vs_donor_filled": auc_of(pos, leg["filled"]),
        "smishx_vs_donor_scrubbed": auc_of(pos, leg["scrubbed"]),
        "smishx_vs_smishx_ham": auc_of(pos, ham),
        "campaign_test_scrubbed": auc_binary(*camp["scrubbed"]),
        "campaign_test_filled": auc_binary(*camp["filled"]),
    }
    keep: dict[str, Any] = {}
    for pct in KEEP_PCTS:
        if pos.size == 0:
            keep[str(pct)] = None
            continue
        thr = keep_threshold(pos, pct)
        keep[str(pct)] = {
            "threshold": thr,
            "smishx_positives_kept": rate_of(int((pos >= thr).sum()), int(pos.size)),
            "groundlm_legit_fpr": rate_of(int((gl >= thr).sum()), int(gl.size)),
            "donor_legit_fpr": {
                v: rate_of(int((leg[v] >= thr).sum()), int(leg[v].size)) for v in VERSIONS
            },
            "heldout_legit_fpr": {
                v: rate_of(int((held[v] >= thr).sum()), int(held[v].size)) for v in VERSIONS
            },
            "donor_scam_recall": {
                v: scam_recall_of(int((scam[v] >= thr).sum()), int(scam[v].size)) for v in VERSIONS
            },
        }
    boot = None
    if clean_only:
        boot = {
            "A1_smishx_vs_groundlm": bootstrap_auc(pos, gl, n_boot, BOOT_SEED),
            "A2_smishx_vs_donor_filled": bootstrap_auc(pos, leg["filled"], n_boot, BOOT_SEED),
        }
    return {
        "rows": {
            "smishx_positives": int(pos.size),
            "smishx_ham": int(ham.size),
            "groundlm_legit": int(gl.size),
            "donor_legit": int(leg["filled"].size),
            "donor_scam": int(scam["filled"].size),
            "heldout_legit": int(held["filled"].size),
            "campaign_test": int(camp["filled"][0].size),
        },
        "auc": auc,
        "thresholds": keep,
        "bootstrap": boot,
    }


def set_counts(sets: dict) -> dict:
    """Per set: rows, label counts, and the contamination count. Counts only."""
    out: dict[str, Any] = {}
    for name in SETS:
        entry = sets[name]
        y = np.asarray(entry["labels"]).astype(np.int64)
        n = int(y.size)
        contaminated = int((~entry["clean"]).sum())
        out[name] = {
            "n": n,
            "n_label1": int((y == 1).sum()),
            "n_label0": int((y == 0).sum()),
            "contaminated": {"k": contaminated, "n": n, "share": (contaminated / n) if n else None},
            "clean_n": int(entry["clean"].sum()),
        }
    return out


def summarise(sets: dict, scores: dict, n_boot: int) -> dict:
    """The per-model views and the teacher's decision. ``scores[model][set][version]`` holds the
    probability of class 1 per row (None for a version a set does not have)."""
    models: dict[str, Any] = {}
    for model in MODELS:
        models[model] = {
            "all": view_block(sets, scores[model], False, n_boot),
            "clean": view_block(sets, scores[model], True, n_boot),
        }
    t = models["teacher"]["clean"]["auc"]
    decision = decide(t["A1_smishx_vs_groundlm"], t["A2_smishx_vs_donor_filled"])
    return {"sets": set_counts(sets), "models": models, "decision": decision}


def render_report(payload: dict) -> str:
    """Markdown report (report.md). Counts, rates, AUCs and thresholds only."""
    lines = ["# sms_teacher_xsource (plan section 4)", ""]
    if payload.get("status") != "scored":
        lines += [f"Status: **{payload.get('status')}**", "", payload.get("reason", ""), ""]
        return "\n".join(lines)
    t = payload["teacher"]
    cal = t["cal_pr_auc"]
    cal_txt = "n/a" if cal is None or cal != cal else f"{cal:.6f}"
    lines += [
        f"Teacher checkpoint sha256 {TEACHER_SHA[:12]}, model {t['model']}, run {t['run']}, "
        f"seed {t['seed']}, input form {t['train_input']}, max_len {t['max_len']}, "
        f"eval batch {t['eval_batch_size']}. Teacher cal PR-AUC {cal_txt} "
        f"(from metrics.json). v0 sha256 {V0_SHA[:12]}.",
        "",
        "## Sets",
        "",
        "| Set | n | label 1 | label 0 | rows in training partition | clean rows |",
        "|---|---|---|---|---|---|",
    ]
    for name in SETS:
        s = payload["sets"][name]
        c = s["contaminated"]
        lines.append(
            f"| {name} | {s['n']} | {s['n_label1']} | {s['n_label0']} | "
            f"{c['k']} ({_pct(c['share'])}) | {s['clean_n']} |"
        )
    for model in MODELS:
        for view in VIEWS:
            blk = payload["models"][model][view]
            a = blk["auc"]
            lines += ["", f"## {model}, {view} rows", "", "| Measure | Value |", "|---|---|"]
            for key, val in a.items():
                lines.append(f"| AUC {key} | {_fmt(val)} |")
            if blk["bootstrap"]:
                for key, b in blk["bootstrap"].items():
                    ci = b["ci95"]
                    txt = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                    lines.append(f"| bootstrap 95 % {key} | {txt} |")
            for pct in KEEP_PCTS:
                th = blk["thresholds"][str(pct)]
                if th is None:
                    lines.append(f"| keep {pct} % | n/a (no positives) |")
                    continue
                lines.append(f"| keep {pct} % threshold | {th['threshold']:.6f} |")
                lines.append(
                    f"| keep {pct} % SmishX positives kept | "
                    f"{_fmt_rate(th['smishx_positives_kept'])} |"
                )
                lines.append(
                    f"| keep {pct} % GroundLM legit FPR | {_fmt_rate(th['groundlm_legit_fpr'])} |"
                )
                for v in VERSIONS:
                    lines.append(
                        f"| keep {pct} % donor legit FPR {v} | "
                        f"{_fmt_rate(th['donor_legit_fpr'][v])} |"
                    )
                    lines.append(
                        f"| keep {pct} % donor scam recall {v} | "
                        f"{_fmt_rate(th['donor_scam_recall'][v])} |"
                    )
    d = payload["decision"]
    lines += [
        "",
        "## Decision (teacher, clean rows)",
        "",
        d["rule"],
        "",
        f"A1 {d['a1']:.4f}, A2 {d['a2']:.4f}. **Verdict: {d['verdict']}.** {d['next_plan']}",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100 * value:.2f} %"


def _fmt_rate(r: dict) -> str:
    if r["rate"] is None:
        return "n/a"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.4f} [{lo:.4f}, {hi:.4f}]"


# ---- torch side (imported inside functions: the pure helpers stay importable without torch) ----


def load_teacher(folder: Path):
    """The fine-tuned classifier and its tokenizer, eval mode, on cuda when available."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(folder))
    model = AutoModelForSequenceClassification.from_pretrained(str(folder))
    model = model.to(device).eval()
    return model, tokenizer, device


def teacher_probs(
    model,
    tokenizer,
    texts: Sequence[str],
    *,
    device,
    canon: bool,
    max_len: int,
    eval_batch_size: int,
) -> np.ndarray:
    """Probability of class 1 per text, scored as sms_teacher.predict_logits does: canon(text)
    for the canon form, fp16 autocast, the same batching by length."""
    import sms_teacher as T

    cfg = T.Config(
        fp16=True,
        max_len=max_len,
        batch_size=eval_batch_size,
        eval_batch_size=eval_batch_size,
        patience=0,
        adv_aug=0.0,
        train_canon=canon,
        log=lambda _msg: None,
    )
    prepped = [D.canon_text(t) for t in texts] if canon else list(texts)
    return T.probs_from_logits(T.predict_logits(model, tokenizer, prepped, cfg, device))


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


def _run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seed_everything(inbox.DATA_SEED)
    env = env_report()
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})
    work = ctx.work

    # 1. teacher checkpoint and its provenance: a wrong file stops the job before any data is read
    teacher_dir = locate_teacher(Path(args.teacher_dir))
    metrics_path = teacher_dir.parent / "metrics.json"
    from_metrics = (
        teacher_meta(json.loads(metrics_path.read_text())) if metrics_path.is_file() else None
    )
    run_name = resolve_teacher_run(from_metrics["run"] if from_metrics else None, args.teacher_run)
    if from_metrics:
        check_train_input(from_metrics["train_input"], args.train_input)
    ctx.log(f"teacher checkpoint found with the pinned sha256 {TEACHER_SHA[:12]}; run {run_name}")

    # 2. v0 ONNX, the same pin and file search as sms_surrogate_d0
    v0_path = d0.locate_v0(Path(args.v0_onnx))
    ctx.log(f"v0 ONNX found with the pinned sha256 {V0_SHA[:12]}")

    # 3. campaign data, as sms_surrogate_d0 loads it
    root = D.resolve_data_root(None)
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    sms_texts = base.str_texts(sms["text"])
    sms_ids = sms["id"].astype(str).to_numpy()
    mask = D.eval_mask(sms)
    runs = D.load_runs(
        root, sms, patterns=["campaign", *base.SHIFT_PREFIXES], labels=labels, mask=mask
    )
    campaign_runs = [r for r in runs if r.name.split(":")[0] == "campaign"]
    if not campaign_runs:
        raise RuntimeError("no campaign split run found")
    run_ = campaign_runs[0]
    shift_runs = [r for r in runs if r.name.split(":")[0] in base.SHIFT_PREFIXES]
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
    gl_all_texts = base.str_texts(gl["text"].to_numpy()[gl_avail_idx])
    gl_g1_texts = [t for t, keep in zip(gl_all_texts, neg_eval[gl_avail_idx], strict=True) if keep]
    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    sx_avail_idx = np.flatnonzero(sx_avail)
    sx_texts = base.str_texts(sx["text"].to_numpy()[sx_avail_idx])
    sx_label = sx["label_binary"].to_numpy().astype(np.int64)[sx_avail_idx]

    # 4. inbox donors and pools (same calls and caps as sms_surrogate_d0)
    roots = [Path(DEFAULT_ROOT)]
    train_df, train_counts = inbox.label_inbox(
        pd.read_parquet(inbox.find_one(inbox.TRAIN_FILE, roots))
    )
    test_df, test_counts = inbox.label_inbox(
        pd.read_parquet(inbox.find_one(inbox.TEST_FILE, roots))
    )
    ctx.log(f"train donor counts: {train_counts}")
    ctx.log(f"test donor counts: {test_counts}")
    test_only = [sms_texts[i] for i in run_.test]
    for r in shift_runs:
        test_only += [sms_texts[i] for i in r.test]
    test_only += list(gl_all_texts) + list(sx_texts) + [t for t in test_df["text"] if t]
    test_only = [t for t in test_only if t]
    with timed("prep_s", timings):
        pools = inbox.build_inbox_pools(train_df, test_only, int((labels[run_.train] == 0).sum()))
    reason = inbox.stop_reason(pools["report"])
    if reason is not None:
        payload = {"job": JOB, "status": inbox.STOP_STATUS, "reason": reason}
        _write(work, payload)
        ctx.results["summary"] = f"{JOB}: {inbox.STOP_STATUS}: {reason}"
        ctx.log(inbox.STOP_STATUS + ": " + reason)
        return False

    # 5. the teacher's training partition, resolved the way sms_teacher resolves splits
    teacher_runs = D.load_runs(
        root, sms, patterns=[run_name.split(":")[0]], labels=labels, mask=mask
    )
    matched = [r for r in teacher_runs if r.name == run_name]
    if len(matched) != 1:
        raise ValueError(f"teacher run {run_name} is not in its split files")
    train_keys = canon_keys([sms_texts[i] for i in matched[0].train])
    ctx.log(f"teacher training partition: {len(matched[0].train)} rows")

    # 6. the sets, scrubbed texts and fills
    legit = pools["val_legit"]
    legit = legit.assign(id=d0.attach_row_ids(legit, train_df))
    test_ids = test_df["id"].astype(str).to_numpy()
    test_texts = [str(t) for t in test_df["text"]]
    test_y = test_df["label"].to_numpy().astype(np.int64)
    held_ids = legit["id"].astype(str).to_numpy()
    held_texts = [str(t) for t in legit["text"]]
    camp_idx = run_.test
    camp_ids = sms_ids[camp_idx]
    camp_texts = [sms_texts[i] for i in camp_idx]
    camp_y = labels[camp_idx]

    raw = {
        SET_SMISHX: {"labels": sx_label, "texts": sx_texts, "ids": None},
        SET_GL: {
            "labels": np.zeros(len(gl_g1_texts), dtype=np.int64),
            "texts": gl_g1_texts,
            "ids": None,
        },
        SET_DONOR: {"labels": test_y, "texts": test_texts, "ids": test_ids},
        SET_HELD: {
            "labels": np.zeros(len(held_texts), dtype=np.int64),
            "texts": held_texts,
            "ids": held_ids,
        },
        SET_CAMP: {"labels": camp_y, "texts": camp_texts, "ids": camp_ids},
    }
    sets: dict[str, dict] = {}
    for name in SETS:
        entry = raw[name]
        texts = list(entry["texts"])
        if not texts:
            raise ValueError(f"set {name} is empty")
        clean = ~contamination_mask(texts, train_keys)
        sets[name] = {
            "labels": np.asarray(entry["labels"]).astype(np.int64),
            "clean": clean,
            "texts": texts,
        }
        if name in FILLED_SETS:
            filled = S.fill_many(list(entry["ids"]), texts)
            S.assert_no_tags(filled)
            sets[name]["filled"] = filled
    ctx.log("sets built: " + ", ".join(f"{n} {len(sets[n]['texts'])}" for n in SETS))

    # 7. scoring: the teacher with its training form, the v0 student on the same rows
    model, tokenizer, device = load_teacher(teacher_dir)
    session = _onnx_session(v0_path)
    scores: dict[str, dict[str, dict[str, Any]]] = {"teacher": {}, "v0": {}}
    with timed("score_s", timings):
        for name in SETS:
            texts = sets[name]["texts"]
            filled = sets[name].get("filled")
            scores["teacher"][name] = {
                "scrubbed": teacher_probs(
                    model,
                    tokenizer,
                    texts,
                    device=device,
                    canon=args.train_input == "canon",
                    max_len=args.max_len,
                    eval_batch_size=args.eval_batch_size,
                ),
                "filled": None
                if filled is None
                else teacher_probs(
                    model,
                    tokenizer,
                    filled,
                    device=device,
                    canon=args.train_input == "canon",
                    max_len=args.max_len,
                    eval_batch_size=args.eval_batch_size,
                ),
            }
            scores["v0"][name] = {
                "scrubbed": score_sms_texts(session, texts),
                "filled": None if filled is None else score_sms_texts(session, filled),
            }
            ctx.log(f"{name}: scored")
    # The texts leave the sets here; only the labels, masks and scores are summarised.
    for name in SETS:
        sets[name].pop("texts", None)
        sets[name].pop("filled", None)

    result = summarise(sets, scores, args.n_boot)
    payload = {
        "job": JOB,
        "status": "scored",
        "plan": PLAN,
        "surrogate_seed": S.SURROGATE_SEED,
        "teacher": {
            "sha256": TEACHER_SHA,
            "model": from_metrics["model"] if from_metrics else None,
            "run": run_name,
            "seed": from_metrics["seed"] if from_metrics else None,
            "cal_pr_auc": from_metrics["cal_pr_auc"] if from_metrics else None,
            "train_input": args.train_input,
            "metrics_json_found": from_metrics is not None,
            "max_len": args.max_len,
            "eval_batch_size": args.eval_batch_size,
            "precision": "fp16 autocast over fp32 weights",
            "training_partition_rows": int(len(matched[0].train)),
        },
        "v0": {"sha256": V0_SHA},
        "args": {"n_boot": args.n_boot, "bootstrap_seed": BOOT_SEED, "keep_pcts": list(KEEP_PCTS)},
        "env": env,
        "prep": {
            "train_donor": train_counts,
            "test_donor": test_counts,
            "stages": pools["report"]["stages"],
        },
        "timings": timings,
        **result,
    }
    _write(work, payload)
    d = result["decision"]
    ctx.results["summary"] = (
        f"{JOB}: teacher verdict {d['verdict']} (A1 {d['a1']:.4f}, A2 {d['a2']:.4f}) "
        f"on clean rows of run {run_name}"
    )
    return True


def _write(work: Path, payload: dict) -> None:
    """metrics.json and report.md in ``work``. Nothing row-level is written."""
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(work / "metrics.json")
    (work / "report.md").write_text(render_report(payload))
