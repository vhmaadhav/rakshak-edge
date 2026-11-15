"""Cross-fitted real-notice negatives for the shipped dual-base student (docs/DETECTOR_FIX_PLAN.md).

Same recipe as the shipped ``dual-base`` seed 1 (campaign run only, rules on, same optimiser,
epochs, patience and cal-PR-AUC early stopping). The only change is the training pool. Four models:
- ctrl: the campaign train split as is (same code path, no GroundLM rows);
- m0 / m1: the train split plus the GroundLM scoreable policy negatives of fold 0 / fold 1, each
  repeated --copies times (label 0);
- full: the train split plus all scoreable GroundLM negatives (the deployment candidate).

Cross-fit: m0 scores the fold-1 negatives and m1 scores the fold-0 negatives, so every scoreable
negative is scored by a model that never trained on it. Folds are group-disjoint template clusters
(rakshak.eval.groupneg). Every decision uses one threshold: the campaign-cal max-F1 threshold of
``full``. Nothing is tuned on GroundLM, SmishX or the campaign test.

Outputs (in /kaggle/working): probs.parquet (per-row probabilities for every set), metrics.json, and
best_student.onnx (``full``) with its torch-vs-ONNX parity.

Flags (``kaggle/launch.sh sms_groupneg --job-args="..."``):
  --copies INT          repeats of each training negative (default 5)
  --epochs INT          max epochs (default 15)
  --patience INT        early-stop patience on cal PR-AUC (default 3)
  --batch-size INT      (default 128)
  --lr FLOAT            (default 2e-3)
  --weight-decay FLOAT  (default 1e-4)
  --n-boot INT          bootstrap resamples (default 1000)
  --no-export           skip ONNX export
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from _common import env_report, job_status, seed_everything, timed

from rakshak.eval import data as D
from rakshak.eval.groupneg import cross_fit_folds
from rakshak.eval.protocol import evaluate_scores, headline
from rakshak.models.features import Featurized, featurize
from rakshak.models.student import (
    ModelSpec,
    TrainConfig,
    export_onnx,
    onnx_parity,
    param_count,
    pick_device,
    predict_scores,
    train_student,
)

SEED = 1  # the shipped dual-base seed
BOOT_SEED = 0  # the shipped bootstrap seed
N_FOLDS = 2


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sms_groupneg")
    p.add_argument("--copies", type=int, default=5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args(argv or [])
    if args.copies < 0:
        p.error("--copies must be non-negative")
    return args


def concat(parts: list[Featurized]) -> Featurized:
    return Featurized(
        texts=[t for part in parts for t in part.texts],
        raw_bytes=np.concatenate([part.raw_bytes for part in parts]),
        canon_bytes=np.concatenate([part.canon_bytes for part in parts]),
        side_raw=np.concatenate([part.side_raw for part in parts]),
        side_canon=np.concatenate([part.side_canon for part in parts]),
        buckets=None,
    )


def run(ctx) -> str:
    args = parse_args(ctx.argv)
    seed_everything(SEED)
    device = pick_device("auto")
    env = env_report()
    env["device"] = str(device)
    ctx.log(f"env: {env}")

    root = D.resolve_data_root(None)
    ctx.log(f"data root: {root}")
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    sms_texts = [t if isinstance(t, str) else "" for t in sms["text"]]
    sms_ids = sms["id"].astype(str).to_numpy()
    sms_lang = sms["lang"].astype(str).to_numpy()
    mask = D.eval_mask(sms)
    runs = D.load_runs(root, sms, patterns=["campaign"], labels=labels, mask=mask)
    run_ = next(r for r in runs if r.name.split(":")[0] == "campaign")
    cluster_col = D.CLUSTER_COLUMNS["sms"]

    # GroundLM: scoreable policy negatives are the cross-fit population. Positives go to full.
    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    gl = gl[gl["text_available"].fillna(False).astype(bool)].reset_index(drop=True)
    gl_texts = [str(t) for t in gl["text"]]
    neg_flag = (
        gl["policy_negative"].fillna(False).astype(bool) & gl["eval_ok"].fillna(False).astype(bool)
    ).to_numpy()
    neg_idx = np.flatnonzero(neg_flag)
    keys = gl.loc[neg_idx, "template_key"].fillna("").astype(str).tolist()
    clusters, fold = cross_fit_folds(keys, n_folds=N_FOLDS)
    fold = np.asarray(fold)
    ctx.log(
        f"GroundLM scoreable negatives {len(neg_idx)}, clusters {len(set(clusters))}, "
        f"fold sizes {np.bincount(fold, minlength=N_FOLDS).tolist()}"
    )
    if len(neg_idx) != 1133:
        raise RuntimeError(f"expected 1133 scoreable negatives, got {len(neg_idx)}")

    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    sx = sx[sx["eval_ok"].fillna(False).astype(bool)].reset_index(drop=True)

    with timed("featurize_s", ctx.results.setdefault("timings", {})):
        fz_sms = featurize(sms_texts, buckets=None)
        fz_gl = featurize(gl_texts, buckets=None)
        fz_neg = fz_gl.take(neg_idx)
        fz_sx = featurize([str(t) for t in sx["text"]], buckets=None)

    spec = ModelSpec("dual", "base", "dual", True)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        adv_p=0.0,
        kd_alpha=0.7,
        kd_temp=2.0,
        seed=SEED,
    )
    base = fz_sms.take(run_.train)
    y_base = labels[run_.train]
    cal = fz_sms.take(run_.cal)
    y_cal = labels[run_.cal]
    test = fz_sms.take(run_.test)
    y_test = labels[run_.test]
    groups = D.group_values(sms, run_.test)
    cluster_ids = D.cluster_values(sms, run_.test, cluster_col)

    def fit(name: str, neg_positions: np.ndarray | None):
        parts, ys = [base], [y_base]
        if neg_positions is not None and len(neg_positions) and args.copies:
            rep = np.tile(neg_positions, args.copies)
            parts.append(fz_neg.take(rep))
            ys.append(np.zeros(len(rep), dtype=np.int64))
        train = concat(parts) if len(parts) > 1 else base
        y_train = np.concatenate(ys)
        ctx.log(
            f"{name}: train rows {len(y_train)} (added negatives "
            f"{len(y_train) - len(y_base)}), positives {int(y_train.sum())}"
        )
        model, info = train_student(
            spec, train, y_train, cal, y_cal, cfg, device=device, teacher=None, log=ctx.log
        )
        model = model.to(device)
        s_cal = predict_scores(model, spec, cal, device)
        s_test = predict_scores(model, spec, test, device)
        res = evaluate_scores(
            y_cal,
            s_cal,
            y_test,
            s_test,
            groups=groups,
            n_boot=args.n_boot,
            seed=BOOT_SEED,
            cluster_ids=cluster_ids,
        )
        thr = float(res["thresholds"]["max_f1"]["threshold"])
        out = {
            "cal": s_cal,
            "test": s_test,
            "smishx": predict_scores(model, spec, fz_sx, device),
            "groundlm": predict_scores(model, spec, fz_gl, device),
            "threshold": thr,
            "headline": headline(res),
            "result": res,
            "train": info,
            "params": param_count(model.cpu()),
            "model": model,
        }
        ctx.log(f"{name}: campaign test max-F1 threshold {thr:.6f}, headline {out['headline']}")
        return out

    ctrl = fit("ctrl", None)
    m0 = fit("m0", np.flatnonzero(fold == 0))
    m1 = fit("m1", np.flatnonzero(fold == 1))
    full = fit("full", np.arange(len(neg_idx)))

    # Cross-fit probabilities for the scoreable negatives (each by a model not trained on it).
    p_cross = np.full(len(gl), np.nan)
    p_cross[neg_idx[fold == 0]] = m1["groundlm"][neg_idx[fold == 0]]
    p_cross[neg_idx[fold == 1]] = m0["groundlm"][neg_idx[fold == 1]]
    thr = full["threshold"]

    gl_fold = np.full(len(gl), -1)
    gl_fold[neg_idx] = fold
    frames = []
    frames.append(
        pd.DataFrame(
            {
                "set": "campaign_cal",
                "id": sms_ids[run_.cal],
                "label": labels[run_.cal],
                "lang": sms_lang[run_.cal],
                "p_ctrl": ctrl["cal"],
                "p_full": full["cal"],
                "p_cross": np.nan,
            }
        )
    )
    frames.append(
        pd.DataFrame(
            {
                "set": "campaign_test",
                "id": sms_ids[run_.test],
                "label": labels[run_.test],
                "lang": sms_lang[run_.test],
                "p_ctrl": ctrl["test"],
                "p_full": full["test"],
                "p_cross": np.nan,
            }
        )
    )
    frames.append(
        pd.DataFrame(
            {
                "set": "smishx",
                "id": sx["id"].astype(str).to_numpy(),
                "label": pd.to_numeric(sx["label_binary"]).to_numpy(),
                "label_scam": pd.to_numeric(sx["label_scam"], errors="coerce").to_numpy(),
                "label_fine": sx["label_fine"].astype(str).to_numpy(),
                "lang": "",
                "p_ctrl": ctrl["smishx"],
                "p_full": full["smishx"],
                "p_cross": np.nan,
            }
        )
    )
    frames.append(
        pd.DataFrame(
            {
                "set": "groundlm",
                "id": gl["id"].astype(str).to_numpy(),
                "label": pd.to_numeric(gl["label_binary"]).to_numpy(),
                "label_fine": gl["label_fine"].astype(str).to_numpy(),
                "lang": gl["lang"].astype(str).to_numpy(),
                "category": gl["category"].astype(str).to_numpy(),
                "policy_negative": gl["policy_negative"].fillna(False).astype(bool).to_numpy(),
                "neg_eval": neg_flag,
                "fold": gl_fold,
                "p_ctrl": ctrl["groundlm"],
                "p_full": full["groundlm"],
                "p_cross": p_cross,
            }
        )
    )
    probs = pd.concat(frames, ignore_index=True)
    probs.to_parquet(ctx.work / "probs.parquet", index=False)
    ctx.log(f"wrote probs.parquet ({len(probs)} rows)")

    onnx_info = None
    if not args.no_export:
        cpu = full["model"].cpu()
        onnx_path = ctx.work / "best_student.onnx"
        exported = export_onnx(cpu, onnx_path)
        parity = onnx_parity(cpu, onnx_path, test)
        onnx_info = {"onnx": exported, "parity": parity, "threshold": thr}
        ctx.log(f"exported full: {exported} parity {parity}")

    def strip(d: dict) -> dict:
        return {
            k: v
            for k, v in d.items()
            if k not in ("cal", "test", "smishx", "groundlm", "model", "result")
        }

    payload = {
        "protocol": "docs/DETECTOR_FIX_PLAN.md",
        "copies": args.copies,
        "fold_sizes": np.bincount(fold, minlength=N_FOLDS).tolist(),
        "clusters": len(set(clusters)),
        "threshold_full": thr,
        "models": {"ctrl": strip(ctrl), "m0": strip(m0), "m1": strip(m1), "full": strip(full)},
        "campaign_test_full_result": full["result"],
        "campaign_test_ctrl_result": ctrl["result"],
        "onnx": onnx_info,
        "env": env,
    }
    (ctx.work / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    ctx.results["summary"] = (
        f"groupneg copies={args.copies}: full threshold {thr:.4f}, "
        f"4 models, probs rows {len(probs)}"
    )
    return job_status(ok=4, failed=0)
