"""One fold of the encoder teacher, trained with the exact sms_teacher recipe (SOTA push W1b).

Trains ``jhu-clsp/mmBERT-small`` or ``jhu-clsp/mmBERT-base`` on ONE split partition (default
``source_holdout:holdout_scamshield``) with the training function of ``kaggle/jobs/sms_teacher.py``:
canonicalised input, AutoModelForSequenceClassification with 2 labels, fp16 autocast, class-balanced
cross-entropy, AdamW with linear warmup and decay, lr 3e-5, up to 4 epochs, weight decay 0.01,
warmup 0.1, max_len 128, batch 32, early stopping on the fold's cal PR-AUC with patience 2.

Scores ONLY the cal split. The GroundLM, SmishX and campaign test sets are not touched here; the
suite
job (kaggle/jobs/sms_teacher_suite.py) scores them from the checkpoint.

Output, in the kernel output folder:
- best_checkpoint/: the best-epoch model and tokenizer from save_pretrained (model.safetensors,
  config.json, tokenizer files). This is the same layout as the existing teacher checkpoint, so
  AutoModelForSequenceClassification.from_pretrained loads it unchanged.
- cal_scores.parquet: id, split, logit_0, logit_1, label_binary, p_scam for the cal rows.
- manifest.json: pinned model id and revision, seed, split, checkpoint sha256, parameter count,
  train
  and cal row counts, best epoch, cal PR-AUC, the thresholds (tau_max_f1 and tau_cal1pct, both from
  cal only), precision, versions. The export job reads it to check the checkpoint.

Thresholds (cal only): alarm rule is p_scam >= tau.
- tau_max_f1: the cal threshold with the highest F1.
- tau_cal1pct: the lowest threshold whose cal false-positive rate is at most 1 percent of cal
  negatives (alarm p > tau_cal1pct_prev, i.e. at most floor(0.01 * n_neg) cal negatives alarm).

The model revision is pinned to a full commit sha. The snapshot is downloaded at that sha and the
training function reads it from the local folder. Nothing is written to any other place.

Flags go through ``kaggle/launch.sh sms_encoder_fold --gpu t4 --internet --datasets
maadhavvh/rakshak-data
--job-args="..."``:
  --model {jhu-clsp/mmBERT-small,jhu-clsp/mmBERT-base}   required
  --seed INT               training seed (default 0)
  --split NAME             split partition (default source_holdout:holdout_scamshield)
  --epochs INT --lr FLOAT --weight-decay FLOAT --warmup-ratio FLOAT --patience INT
  --max-len INT --batch-size INT --eval-batch-size INT   (defaults match sms_teacher)
  --data-root PATH         data root (default: as in rakshak.eval.data.resolve_data_root)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _common import env_report, single_class_reason, timed

JOB = "sms_encoder_fold"
# Full commit shas of the Hugging Face repos (public API, read 2026-11-12). Any other revision is
# refused.
MODEL_REVISIONS: dict[str, str] = {
    "jhu-clsp/mmBERT-small": "abc32620dd4f6ab06f5fbe905dc25f310618e09f",
    "jhu-clsp/mmBERT-base": "c5955035435e2bf121cde7f3c8863ef52ff35d82",
}
# mmBERT repos ship pytorch_model.bin (no safetensors), so *.bin is required.
SNAPSHOT_PATTERNS = ["*.json", "*.safetensors", "*.bin", "*.txt"]
DEFAULT_SPLIT = "source_holdout:holdout_scamshield"
BEST_DIR = "best_checkpoint"
WEIGHTS = "model.safetensors"
CAL_SCORES = "cal_scores.parquet"
MANIFEST = "manifest.json"
CAL_FPR = 0.01
DEFAULTS = {
    "lr": 3e-5,
    "epochs": 4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "patience": 2,
    "max_len": 128,
    "batch_size": 32,
    "eval_batch_size": 128,
}


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--model", required=True, choices=sorted(MODEL_REVISIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--warmup-ratio", type=float, default=DEFAULTS["warmup_ratio"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--eval-batch-size", type=int, default=DEFAULTS["eval_batch_size"])
    p.add_argument("--data-root", default=None)
    args = p.parse_args(list(argv or []))
    if args.seed < 0:
        p.error("--seed must be 0 or more")
    if args.lr <= 0 or args.epochs < 1 or args.patience < 1:
        p.error("--lr must be positive; --epochs and --patience at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        p.error("--warmup-ratio must be in [0, 1)")
    return args


def probs_scam(logits: np.ndarray) -> np.ndarray:
    """Softmax probability of class 1 (scam) from 2-column logits, numerically stable."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def operating_point(y: Sequence[int], p: Sequence[float], tau: float) -> dict[str, float | int]:
    """Counts at the alarm rule p >= tau. Recall, FPR and precision are None when undefined."""
    y = np.asarray(y, dtype=np.int64)
    alarm = np.asarray(p, dtype=np.float64) >= tau
    tp = int(np.sum(alarm & (y == 1)))
    fp = int(np.sum(alarm & (y == 0)))
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    return {
        "tau": float(tau),
        "tp": tp,
        "fp": fp,
        "recall": tp / n_pos if n_pos else None,
        "fpr": fp / n_neg if n_neg else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def threshold_candidates(y: Sequence[int], p: Sequence[float]) -> dict[str, Any]:
    """tau_max_f1 and tau_cal1pct from calibration rows only (see the module docstring).

    tau_max_f1 is the candidate with the highest F1 (the first one on ties). tau_cal1pct is
    the nextafter of the (k+1)-th largest negative score, k = floor(CAL_FPR * n_neg), so that
    at most k cal negatives satisfy p >= tau.
    """
    from sklearn.metrics import precision_recall_curve

    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("cal needs both classes to pick thresholds")
    prec, rec, thr = precision_recall_curve(y, p)
    # precision/recall have one extra trailing point with no threshold
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    tau_f1 = float(thr[int(np.argmax(f1))])
    neg_desc = np.sort(p[y == 0])[::-1]
    k = int(math.floor(CAL_FPR * n_neg))
    tau_fpr = float(np.nextafter(neg_desc[k], np.inf))
    return {
        "rule": "alarm if p_scam >= tau",
        "cal_rows": int(len(y)),
        "cal_pos": n_pos,
        "cal_neg": n_neg,
        "max_f1": {
            "tau": tau_f1,
            "cal": operating_point(y, p, tau_f1),
            "cal_f1": float(np.max(f1)),
        },
        "cal1pct": {
            "tau": tau_fpr,
            "cal": operating_point(y, p, tau_fpr),
            "max_cal_fpr": CAL_FPR,
            "negatives_above_k": k,
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    model: str,
    revision: str,
    seed: int,
    split: str,
    checkpoint_sha256: str,
    param_count: int,
    train_rows: int,
    train_pos: int,
    cal_rows: int,
    cal_pos: int,
    best_epoch: int,
    cal_pr_auc: float,
    history: list[dict],
    hyperparameters: dict,
    thresholds: dict,
    precision: str,
    env: dict,
    seconds: float,
) -> dict[str, Any]:
    """The manifest the export job and the suite read. Pure: it only arranges its inputs."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a full 40-character commit sha")
    if len(checkpoint_sha256) != 64:
        raise ValueError("checkpoint_sha256 must be a sha256 hex digest")
    return {
        "job": JOB,
        "schema": 1,
        "model": model,
        "revision": revision,
        "seed": int(seed),
        "split": split,
        "train_input": "canon",
        "checkpoint": {
            "folder": BEST_DIR,
            "file": WEIGHTS,
            "sha256": checkpoint_sha256,
            "param_count": int(param_count),
        },
        "rows": {
            "train": int(train_rows),
            "train_pos": int(train_pos),
            "cal": int(cal_rows),
            "cal_pos": int(cal_pos),
        },
        "training": {
            "best_epoch": int(best_epoch),
            "cal_pr_auc": float(cal_pr_auc),
            "history": history,
            "hyperparameters": hyperparameters,
            "precision": precision,
            "seconds": round(float(seconds), 1),
        },
        "thresholds": thresholds,
        "env": env,
    }


def run(ctx) -> bool | str:
    args = parse_args(ctx.argv)
    import sms_teacher as teacher  # bundled sibling: the sms_teacher training recipe
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from rakshak.eval import data as D

    if not torch.cuda.is_available():
        raise RuntimeError(f"{JOB} needs a CUDA GPU; launch with --gpu t4")
    device = torch.device("cuda")
    revision = MODEL_REVISIONS[args.model]
    cfg = teacher.Config(
        fp16=True,
        max_len=args.max_len,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        patience=args.patience,
        adv_aug=0.0,
        train_canon=True,
        log=ctx.log,
    )
    env = env_report()
    ctx.log(f"env: {env}")
    ctx.log(f"model {args.model} @ {revision}; split {args.split}; seed {args.seed}")

    root = D.resolve_data_root(args.data_root)
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    ids = sms["id"].to_numpy()
    texts = sms["text"].to_numpy(dtype=object)
    stem = args.split.split(":", 1)[0]
    runs = {
        r.name: r
        for r in D.load_runs(root, sms, patterns=[stem], labels=labels, mask=D.eval_mask(sms))
    }
    if args.split not in runs:
        raise ValueError(f"split {args.split!r} not found; have {sorted(runs)}")
    run_ = runs[args.split]
    # Only the train and cal partitions are read here. The test partition is never scored.
    tr, ca = run_.train, run_.cal
    for part, rows in (("train", tr), ("cal", ca)):
        reason = single_class_reason(labels[rows], f"{args.split} {part}")
        if reason:
            raise ValueError(reason)

    with timed("download", ctx.results.setdefault("timings", {})):
        local = snapshot_download(
            repo_id=args.model, revision=revision, allow_patterns=SNAPSHOT_PATTERNS
        )
    tokenizer = AutoTokenizer.from_pretrained(local)
    data = {
        "train_raw": list(texts[tr]),
        "train_y": labels[tr],
        "cal_y": labels[ca],
        "cal_prepped": [D.canon_text(t) for t in texts[ca]],
    }
    hp = {
        "lr": args.lr,
        "epochs": args.epochs,
        "warmup": args.warmup_ratio,
        "weight_decay": args.weight_decay,
    }
    ctx.log(f"train {len(tr)} rows, cal {len(ca)} rows; hp {hp}")
    out = teacher.train_once(cfg, hp, local, args.seed, tokenizer, device, data, math.inf)
    model = out["model"]
    ctx.log(
        f"best epoch {out['best_epoch']} cal PR-AUC {out['cal_pr_auc']:.4f} ({out['seconds']} s)"
    )

    # Cal scores with the best-epoch weights, on the same canonicalised input the model trained on.
    model.to(device)
    cal_logits = teacher.predict_logits(model, tokenizer, data["cal_prepped"], cfg, device)
    model.to("cpu")
    p_cal = probs_scam(cal_logits)
    thresholds = threshold_candidates(labels[ca], p_cal)
    frame = pd.DataFrame(
        {
            "id": ids[ca],
            "split": "cal",
            "logit_0": cal_logits[:, 0].astype(np.float64),
            "logit_1": cal_logits[:, 1].astype(np.float64),
            "label_binary": labels[ca],
            "p_scam": p_cal,
        }
    )
    frame.to_parquet(Path(ctx.work) / CAL_SCORES, index=False)

    ckpt = Path(ctx.work) / BEST_DIR
    shutil.rmtree(ckpt, ignore_errors=True)
    with timed("save", ctx.results["timings"]):
        model.save_pretrained(ckpt, safe_serialization=True)
        tokenizer.save_pretrained(ckpt)
    checksum = file_sha256(ckpt / WEIGHTS)
    param_count = int(sum(p.numel() for p in model.parameters()))
    manifest = build_manifest(
        model=args.model,
        revision=revision,
        seed=args.seed,
        split=args.split,
        checkpoint_sha256=checksum,
        param_count=param_count,
        train_rows=len(tr),
        train_pos=int(np.sum(labels[tr] == 1)),
        cal_rows=len(ca),
        cal_pos=int(np.sum(labels[ca] == 1)),
        best_epoch=out["best_epoch"],
        cal_pr_auc=out["cal_pr_auc"],
        history=out["history"],
        hyperparameters={
            **hp,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "fp16": True,
        },
        thresholds=thresholds,
        precision=cfg.dtype_label,
        env=env,
        seconds=out["seconds"],
    )
    (Path(ctx.work) / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ctx.results["manifest"] = manifest
    ctx.results["summary"] = (
        f"{args.model} seed {args.seed} on {args.split}: best epoch "
        f"{out['best_epoch']}, cal PR-AUC {out['cal_pr_auc']:.4f}, "
        f"checkpoint sha256 {checksum[:12]}"
    )
    ctx.log(ctx.results["summary"])
    ctx.save()
    return True
