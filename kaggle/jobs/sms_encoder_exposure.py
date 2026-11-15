"""Encoder exposure W1e (docs/ENCODER_EXPOSURE_PLAN.md, frozen 2026-11-12).

mmBERT-base trained on the teacher fold's training partition
(source_holdout:holdout_scamshield) plus
the arm C extra rows of rakshak.data.retrain_prep.build_pools (CloveAI with GroundLM-touching split
groups removed, capped FinEE), minus a 20 % calibration hold-out of CloveAI and FinEE split groups.
The hold-out is used only for the threshold (logit margin, FPR <= 1 %) and for the recall report.

Per seed: train (sms_encoder_fold recipe, early stopping on the fold's own cal split), score the
calibration sets, write thresholds.json, THEN score the test sets (sms_retrain.run_seed_pipeline
enforces the order). Scoring reuses sms_teacher_suite (build_sets, arm_block, fold_protocol).

Output, in the kernel output folder: prep_report.json, thresholds.json (written before any test
scoring), per-seed best_checkpoint/ (safetensors), metrics.json, report.md, probs.parquet (logit
margins and probabilities, no text).

Launch (T4, internet on; one kernel per seed or all seeds in one kernel):
  kaggle/launch.sh sms_encoder_exposure --gpu t4 --internet --datasets maadhavvh/rakshak-data \
      --job-args="--seeds 0 1 2"
  or one kernel per seed, with --run-name s0 / s1 / s2 and --job-args="--seeds 0".

Flags (--job-args="FLAGS"):
  --seeds INT ...           training seeds (default 0 1 2)
  --max-len INT             token limit (default 128)
  --eval-batch-size INT     inference batch (default 128)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import sms_encoder_fold as F
import sms_retrain as base
import sms_teacher_suite as S
import sms_teacher_xsource as X
from _common import env_report, seed_everything, timed
from baseline_v0 import rate_row
from scipy.special import expit

from rakshak.data import retrain_prep as RP
from rakshak.eval import data as D

JOB = "sms_encoder_exposure"
PLAN = "docs/ENCODER_EXPOSURE_PLAN.md"
MODEL = "jhu-clsp/mmBERT-base"
DATA_SEED = 0
CAL_SEED = 4
HOLDOUT_FRAC = 0.2
CH_MIN_LEGIT = 100
FPR_CAL = 0.01
DEFAULT_SEEDS = (0, 1, 2)
G3_MIN_RECALL = 0.90  # W1e gate (W1a used 0.95 on the same 91 rows)
SET_CH_LEGIT = base.SET_LEGIT_VAL
SET_FOLD_CAL = base.SET_CAL
SET_CH_SCAM = "calib_scam"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--max-len", type=int, default=F.DEFAULTS["max_len"])
    p.add_argument("--eval-batch-size", type=int, default=F.DEFAULTS["eval_batch_size"])
    args = p.parse_args(list(argv or []))
    if any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must be distinct non-negative integers")
    if args.max_len < 1 or args.eval_batch_size < 1:
        p.error("--max-len and --eval-batch-size must be positive")
    return args


# ---- pure helpers (unit-tested in tests/test_sms_encoder_exposure.py) ----


def margins(logits: np.ndarray) -> np.ndarray:
    """Logit margin logit_1 - logit_0 (float64); the logit of p_scam, monotone in p."""
    z = np.asarray(logits, dtype=np.float64)
    return z[:, 1] - z[:, 0]


def calibration_holdout(
    extra: pd.DataFrame, seed: int = CAL_SEED, frac: float = HOLDOUT_FRAC
) -> tuple[np.ndarray, dict]:
    """True for each extra row whose split group is held out for calibration. For each source
    present (sorted), round(frac x n_groups) of its split groups are drawn with default_rng(seed).
    Asserts that no split group is on both sides."""
    rng = np.random.default_rng(seed)
    source = extra["source"].astype(str).to_numpy()
    held: set = set()
    per_source: dict[str, dict] = {}
    for src in sorted(set(source.tolist())):
        groups = sorted(set(extra.loc[source == src, "cluster"].tolist()))
        n_hold = int(round(frac * len(groups)))
        picked = rng.permutation(len(groups))[:n_hold].tolist()
        chosen = {groups[i] for i in picked}
        held |= chosen
        per_source[src] = {"split_groups": len(groups), "held_groups": len(chosen)}
    mask = extra["cluster"].isin(held).to_numpy()
    inside = set(extra.loc[mask, "cluster"].tolist())
    outside = set(extra.loc[~mask, "cluster"].tolist())
    if not inside.isdisjoint(outside):
        raise ValueError("calibration hold-out shares a split group with training")
    return mask, {"by_source": per_source, "held_split_groups": len(held), "rows": int(mask.sum())}


def logit_threshold(neg_margins: Sequence[float], max_fpr: float = FPR_CAL) -> float:
    """Smallest margin tau with at most floor(max_fpr x n) negatives at margin >= tau (-inf when
    every negative may flag). Uses negatives only."""
    return S.fpr_threshold(neg_margins, max_fpr)


GATE_SPECS = (
    ("G1", "fp_neg", lambda v: v <= S.G1_MAX_FP),
    ("G2", "tp_smish", lambda v: v >= S.G2_MIN_TP),
    ("G3", "recall_pos", lambda v: v >= G3_MIN_RECALL),
    ("G4", "f1_clean", lambda v: v >= S.G4_MIN_F1),
)


def aggregate_gates(per_seed: Sequence[dict]) -> dict:
    """Each gate holds when the seed median passes and at least a majority of seeds pass
    (2 of 3). ``per_seed`` rows carry fp_neg, tp_smish, recall_pos and f1_clean."""
    need = len(per_seed) // 2 + 1
    out: dict[str, Any] = {}
    for gate, key, ok in GATE_SPECS:
        vals = [float(r[key]) for r in per_seed]
        med = float(statistics.median(vals))
        n_ok = int(sum(bool(ok(v)) for v in vals))
        out[gate] = {
            "key": key,
            "values": vals,
            "median": med,
            "seeds_pass": n_ok,
            "n_seeds": len(vals),
            "pass": bool(ok(med) and n_ok >= need),
        }
    out["arm_pass"] = bool(all(out[g]["pass"] for g, _k, _f in GATE_SPECS))
    return out


def verdict(per_seed: Sequence[dict], expected_seeds: int = 3) -> str:
    if len(per_seed) != expected_seeds:
        return f"incomplete ({len(per_seed)} of {expected_seeds} seeds)"
    return "pass" if aggregate_gates(per_seed)["arm_pass"] else "fail"


def seed_gate_values(flags: dict[str, np.ndarray], block: dict) -> dict:
    """The four gate inputs for one seed: GroundLM negatives flagged, SmishX smishing flagged,
    GroundLM positives recall, and campaign clean-row F1."""
    neg = flags[S.SET_GL_NEG]
    pos = flags[S.SET_GL_POS]
    smish = flags[S.SET_SX_SMISH]
    return {
        "fp_neg": int(neg.sum()),
        "tp_smish": int(smish.sum()),
        "tp_pos": int(pos.sum()),
        "n_pos": int(len(pos)),
        "recall_pos": float(pos.sum()) / len(pos),
        "f1_clean": float(block["campaign_clean"]["f1"]),
    }


# ---- the job ----


def run(ctx) -> bool:
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
    import sms_teacher as T  # the sms_teacher training recipe
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    seed_everything(DATA_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{JOB} needs a CUDA GPU; launch with --gpu t4")
    device = torch.device("cuda")
    revision = F.MODEL_REVISIONS[MODEL]
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

    # 1. data and the teacher fold partitions (sizes asserted as in the suite)
    root = D.resolve_data_root(None)
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    # object array: fancy indexing with fold index arrays (a list raised TypeError on Kaggle)
    texts = np.asarray(base.str_texts(sms["text"]), dtype=object)
    ids = sms["id"].astype(str).to_numpy()
    runs = D.load_runs(
        root,
        sms,
        patterns=["campaign", "lang_shift", "temporal", "source_holdout"],
        labels=labels,
        mask=D.eval_mask(sms),
    )

    def one(stem: str) -> Any:
        found = [r for r in runs if r.name.split(":")[0] == stem]
        if len(found) != 1:
            raise ValueError(f"expected one {stem} run, found {len(found)}")
        return found[0]

    fold = [r for r in runs if r.name == S.RUN_NAME]
    if len(fold) != 1:
        raise ValueError(f"fold run {S.RUN_NAME} is not in the split files")
    fold = fold[0]
    if len(fold.train) != S.TRAIN_ROWS or len(fold.cal) != S.CAL_ROWS:
        raise ValueError(f"fold partitions train {len(fold.train)} cal {len(fold.cal)}")
    camp, lang, temp = one("campaign"), one("lang_shift"), one("temporal")

    # 2. build_pools arm C once (data seed), then the calibration hold-out
    sources = RP.load_indian_sources(root)
    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    gl_avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    gl_texts = base.str_texts(gl["text"].to_numpy()[gl_avail])
    test_only = (
        [texts[i] for i in camp.test]
        + [texts[i] for i in lang.test]
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
    ch_mask, ch_info = calibration_holdout(extra)
    ch = extra[ch_mask]
    tr_extra = extra[~ch_mask]
    ch_legit = ch[ch["label"].astype(int) == 0]
    ch_scam = ch[ch["label"].astype(int) == 1]
    if len(ch_legit) < CH_MIN_LEGIT:
        raise ValueError(
            f"calibration hold-out has {len(ch_legit)} legit rows, need {CH_MIN_LEGIT}"
        )
    report = {
        "arm": "C",
        "data_seed": DATA_SEED,
        "cal_seed": CAL_SEED,
        "test_only_texts": len(test_only),
        "exposure_holdout": ch_info,
        "ch_legit": int(len(ch_legit)),
        "ch_scam": int(len(ch_scam)),
        "train_extra_used": int(len(tr_extra)),
        "build_pools_unused_pools": {
            "legit_val": int(len(prep.legit_val)),
            "val_scam": int(len(prep.val_scam)),
        },
        **prep.report,
    }
    (work / "prep_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    ctx.log(
        f"prep: train_extra {len(extra)}, hold-out legit {len(ch_legit)}, scam {len(ch_scam)}, "
        f"training extra {len(tr_extra)}"
    )

    # 3. training data: fold train + extra rows; early stopping on the fold's own cal split
    cal_idx = fold.cal
    tr_idx = fold.train
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

    # 4. the scored sets (same definitions as the suite), contamination against the NEW training set
    sets, join = S.build_sets(
        root, sms, labels, texts, ids, camp, lang, temp, train_ids, train_keys
    )
    folds, fold_info = S.negative_folds(sets[S.SET_GL_NEG]["texts"])
    report["amendment2_folds"] = fold_info
    ctx.log(f"XLM-R join: {join}; amendment 2 folds {fold_info}")

    with timed("download", timings):
        local = snapshot_download(
            repo_id=MODEL, revision=revision, allow_patterns=F.SNAPSHOT_PATTERNS
        )
    tokenizer = AutoTokenizer.from_pretrained(local)
    ch_legit_texts = base.str_texts(ch_legit["text"])
    ch_scam_texts = base.str_texts(ch_scam["text"])
    cal_texts = [texts[i] for i in cal_idx]
    text_of = {SET_CH_LEGIT: ch_legit_texts, SET_FOLD_CAL: cal_texts, SET_CH_SCAM: ch_scam_texts}
    text_of.update({n: sets[n]["texts"] for n in S.TEST_SETS})
    contamination = {
        n: {"n": len(s["texts"]), "contaminated": int((~s["clean"]).sum())} for n, s in sets.items()
    }

    per_seed: dict[int, dict] = {}
    thresholds_all: dict[str, Any] = {
        "plan": PLAN,
        "rule": "alarm if logit margin >= tau",
        "seeds": {},
    }
    frames: list[pd.DataFrame] = []
    ctx_pack = {
        "args": args,
        "cfg": cfg,
        "hp": hp,
        "data": data,
        "T": T,
        "device": device,
        "local": local,
        "tokenizer": tokenizer,
        "text_of": text_of,
        "sets": sets,
        "folds": folds,
        "labels": labels,
        "cal_idx": cal_idx,
        "ch_legit": ch_legit,
        "ch_scam": ch_scam,
        "work": work,
        "timings": timings,
        "thresholds_all": thresholds_all,
        "ch_scam_texts": ch_scam_texts,
        "ids_cal": ids[cal_idx],
        "torch": torch,
    }
    for seed in args.seeds:
        row, frame = _run_seed(ctx, seed, ctx_pack)
        per_seed[seed] = row
        frames.append(frame)

    rows = [per_seed[s]["gate_inputs"] for s in sorted(per_seed)]
    gates = aggregate_gates(rows) if len(rows) == 3 else None
    payload = {
        "job": JOB,
        "status": "scored",
        "plan": PLAN,
        "model": MODEL,
        "revision": revision,
        "seeds": sorted(per_seed),
        "verdict": verdict(rows),
        "gates": gates,
        "gate_note": "seed median and at least 2 of 3 seeds, per gate",
        "thresholds": thresholds_all,
        "prep": {
            k: report[k]
            for k in (
                "exposure_holdout",
                "ch_legit",
                "ch_scam",
                "train_extra_used",
                "amendment2_folds",
            )
        },
        "contamination": contamination,
        "xlmr_join": join,
        "label": "same source family (CloveAI), template-disjoint",
        "per_seed": {str(s): per_seed[s] for s in sorted(per_seed)},
        "env": env,
        "timings": timings,
    }
    (work / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    (work / "report.md").write_text(_render(payload))
    pd.concat(frames, ignore_index=True).to_parquet(work / "probs.parquet", index=False)
    ctx.results["summary"] = f"{JOB}: seeds {sorted(per_seed)}; verdict {payload['verdict']}"
    return True


def _run_seed(ctx, seed: int, pack: dict) -> tuple[dict, pd.DataFrame]:
    """Train one seed, write thresholds before scoring, score the test sets, run the gates inputs,
    Amendment 2 on logits, and save the checkpoint. Returns (per-seed row, probs frame)."""
    args, cfg, hp, data, T = pack["args"], pack["cfg"], pack["hp"], pack["data"], pack["T"]
    device, local, tokenizer = pack["device"], pack["local"], pack["tokenizer"]
    text_of, sets, folds, labels = pack["text_of"], pack["sets"], pack["folds"], pack["labels"]
    cal_idx, ch_legit, ch_scam = pack["cal_idx"], pack["ch_legit"], pack["ch_scam"]
    work, timings, thresholds_all = pack["work"], pack["timings"], pack["thresholds_all"]
    sdir = work / f"seed{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    with timed(f"train_s{seed}", timings):
        out = T.train_once(cfg, hp, local, seed, tokenizer, device, data, math.inf)
    model = out["model"].to(device).eval()
    ctx.log(f"seed {seed}: best epoch {out['best_epoch']} cal PR-AUC {out['cal_pr_auc']:.4f}")
    thr: dict[str, float] = {}

    def score(_model, name: str) -> np.ndarray:
        return S.teacher_margins(
            model,
            tokenizer,
            text_of[name],
            device=device,
            max_len=args.max_len,
            eval_batch_size=args.eval_batch_size,
        )

    def rule(s_legit: np.ndarray, s_cal: np.ndarray, y_cal: np.ndarray) -> tuple[float, bool]:
        thr["tau"] = logit_threshold(s_legit)
        thr["tau_cal"] = logit_threshold(s_cal[y_cal == 0])
        return thr["tau"], False

    def on_threshold(tau: float, fallback: bool) -> None:
        thresholds_all["seeds"][str(seed)] = {
            "tau": tau,
            "tau_cal_fold_1pct": thr["tau_cal"],
            "fallback": bool(fallback),
            "ch_legit_rows": int(len(ch_legit)),
            "fold_cal_negatives": int(np.sum(labels[cal_idx] == 0)),
        }
        (work / "thresholds.json").write_text(
            json.dumps(thresholds_all, indent=2, sort_keys=True) + "\n"
        )
        ctx.log(
            f"seed {seed}: thresholds.json written before any test set: tau {tau:.6f}, "
            f"tau_cal {thr['tau_cal']:.6f}"
        )

    with timed(f"score_s{seed}", timings):
        pipe = base.run_seed_pipeline(
            fit=lambda: model,
            score=score,
            rule=rule,
            y_cal=labels[cal_idx],
            post_sets=[SET_CH_SCAM, *S.TEST_SETS],
            on_threshold=on_threshold,
        )
    events = pipe["events"]
    first_test = min(
        i
        for i, e in enumerate(events)
        if e.startswith("score:") and e[len("score:") :] in S.TEST_SETS
    )
    if events.index("threshold") > first_test:
        raise RuntimeError("a test set was scored before the threshold")
    scores = pipe["scores"]
    tau, tau_cal = thr["tau"], thr["tau_cal"]
    flags_main = {n: scores[n] >= tau for n in S.TEST_SETS}
    flags_cal = {n: scores[n] >= tau_cal for n in S.TEST_SETS}
    block_main = S.arm_block(sets, flags_main, min_recall=G3_MIN_RECALL)
    block_cal = S.arm_block(sets, flags_cal, min_recall=G3_MIN_RECALL)
    ext = {
        "smishx_smishing": scores[S.SET_SX_SMISH],
        "smishx_ham": scores[S.SET_SX_HAM],
        "groundlm_pos": scores[S.SET_GL_POS],
    }
    amend2 = S.fold_protocol(scores[S.SET_GL_NEG], folds, ext)
    ckpt = sdir / "best_checkpoint"
    model.to("cpu")
    model.save_pretrained(ckpt, safe_serialization=True)
    tokenizer.save_pretrained(ckpt)
    sha = X.file_sha256(ckpt / "model.safetensors")
    gate_inputs = seed_gate_values(flags_main, block_main)
    row = {
        "seed": seed,
        "best_epoch": int(out["best_epoch"]),
        "cal_pr_auc": float(out["cal_pr_auc"]),
        "checkpoint_sha256": sha,
        "tau": tau,
        "tau_cal_fold_1pct": tau_cal,
        "events": events,
        "gate_inputs": gate_inputs,
        "calib_scam_recall_at_tau": rate_row(
            int((scores[SET_CH_SCAM] >= tau).sum()), len(pack["ch_scam_texts"])
        ),
        "arms": {"tau_ch": block_main, "tau_cal_fold": block_cal},
        "amendment2_logits": amend2,
    }
    frame = _probs_frame(
        seed,
        sets,
        scores,
        ch_legit,
        ch_scam,
        labels[cal_idx],
        pack["ids_cal"],
        margins_cal=scores[SET_FOLD_CAL],
    )
    ctx.log(
        f"seed {seed}: G1 {gate_inputs['fp_neg']}, G2 {gate_inputs['tp_smish']}, "
        f"G3 {gate_inputs['recall_pos']:.3f}, G4 {gate_inputs['f1_clean']:.4f}"
    )
    torch_empty_cache(pack["torch"])
    return row, frame


def torch_empty_cache(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _probs_frame(
    seed: int,
    sets: dict,
    scores: dict,
    ch_legit: pd.DataFrame,
    ch_scam: pd.DataFrame,
    cal_labels: np.ndarray,
    cal_ids: np.ndarray,
    margins_cal: np.ndarray,
) -> pd.DataFrame:
    """Margins and probabilities per row for every set of one seed. No text."""
    parts = [
        pd.DataFrame(
            {
                "seed": seed,
                "set": SET_FOLD_CAL,
                "id": cal_ids.astype(str),
                "label": cal_labels,
                "margin": margins_cal,
            }
        ),
        pd.DataFrame(
            {
                "seed": seed,
                "set": SET_CH_LEGIT,
                "id": ch_legit["id"].astype(str).to_numpy(),
                "label": 0,
                "margin": scores[SET_CH_LEGIT],
            }
        ),
        pd.DataFrame(
            {
                "seed": seed,
                "set": SET_CH_SCAM,
                "id": ch_scam["id"].astype(str).to_numpy(),
                "label": 1,
                "margin": scores[SET_CH_SCAM],
            }
        ),
    ]
    for name in S.TEST_SETS:
        s = sets[name]
        parts.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "set": name,
                    "id": s["ids"].astype(str),
                    "label": s["labels"],
                    "category": s["category"],
                    "clean": s["clean"],
                    "xlmr_pred": (
                        np.full(len(s["texts"]), np.nan)
                        if s["xlmr"] is None
                        else s["xlmr"].astype(float)
                    ),
                    "margin": scores[name],
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out["p_scam"] = expit(out["margin"].to_numpy(dtype=np.float64))
    return out


def _render(payload: dict) -> str:
    lines = [
        f"# {JOB} (plan {PLAN})",
        "",
        f"Label: {payload['label']}. Verdict: {payload['verdict']}.",
        "",
        "| seed | tau (margin) | G1 GL-neg FP | G2 SmishX TP | G3 GL-pos recall | G4 clean F1 |",
        "|---|---|---|---|---|---|",
    ]
    for s, row in payload["per_seed"].items():
        g = row["gate_inputs"]
        lines.append(
            f"| {s} | {row['tau']:.4f} | {g['fp_neg']}/1133 | {g['tp_smish']}/226 | "
            f"{g['tp_pos']}/{g['n_pos']} ({g['recall_pos']:.3f}) | {g['f1_clean']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
