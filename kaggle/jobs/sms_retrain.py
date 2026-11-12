"""Retrain the shipped dual-base student with Indian legit exposure (docs/SMS_RETRAIN_PLAN.md).

The plan (sections 2 to 6) is frozen. One arm per kernel, three seeds per arm:
- A: campaign train + licensed synthetic legit rows (FinEE, capped). Cross-source.
- B: A + minimal pairs (a legit template and its scam twin).
- C: A + CloveAI (GroundLM-stripped). In-distribution adjacent, not cross-source.

Pipeline:
1. Data. sms.parquet and the campaign split exactly as sms_groupneg loads them. lang_shift and
   temporal test rows and the external sets come from the same data root (rakshak-data). No
   internet is needed for the data.
2. Pools. rakshak.data.retrain_prep.build_pools runs once with DATA_SEED (0), so every seed trains
   on the same extra rows. It returns the extra training rows, the Indian legit validation pool
   (threshold only) and the scam validation pool (reported recall).
3. Per seed. Train on campaign train + extra rows, early-stop on campaign cal PR-AUC. Score the
   legit validation pool and campaign cal, then threshold_rule gives tau. tau is written to
   thresholds.json and logged BEFORE any test set is scored (run_seed_pipeline enforces the order).
4. Scoring after tau: campaign test, lang_shift, temporal, GroundLM (text available), SmishX,
   SMS-Shield eval, IEEE ham (if present), val_scam. Metrics reuse baseline_v0's sms_run_table,
   groundlm_table, smishx_table and rate_row.
5. GroundLM row by row against the released XLM-R decision (groundlm_xlmr_probs, joined on the
   GroundLM row key): 2x2 table, exact McNemar, per-category counts.
6. Gates (plan section 5): G1 GroundLM FP <= 11 of 1133, G2 SmishX smishing TP >= 157 of 226,
   G3 campaign F1 >= 0.9713, G4 ONNX <= 10 MB. A gate holds for an arm when the seed median passes
   and at least 2 of 3 seeds pass. Each seed's ONNX export is parity-checked against torch.
7. Outputs: prep_report.json, thresholds.json, probs.parquet (every set, every seed), metrics.json,
   report.md, best_student_seed<N>.onnx per seed. results.json comes from the runner.

Launch (one kernel per arm, T4; the data is the rakshak-data dataset):
  kaggle/launch.sh sms_retrain --run-name a --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data --job-args="--arm A"
  kaggle/launch.sh sms_retrain --run-name b --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data --job-args="--arm B"
  kaggle/launch.sh sms_retrain --run-name c --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data --job-args="--arm C"
--internet is not needed for the data. It enables the pinned pip set and the completion notice,
and wait_event.sh refuses a kernel built without it.

Flags (``--job-args="FLAGS"``):
  --arm {A,B,C}        required
  --seeds INT ...      default 1 2 3
  --epochs INT         max epochs (default 15)
  --patience INT       early-stop patience on cal PR-AUC (default 3)
  --batch-size INT     (default 128)
  --lr FLOAT           (default 2e-3)
  --weight-decay FLOAT (default 1e-4)
  --n-boot INT         bootstrap resamples (default 1000)
  --no-export          skip ONNX export (gate G4 then cannot pass)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from _common import Units, env_report, job_status, seed_everything, timed, write_metrics
from baseline_v0 import (
    SMS_CLUSTER_COLUMN,
    groundlm_table,
    rate_row,
    rules_any,
    smishx_table,
    sms_run_table,
)
from scipy.stats import binomtest

from rakshak.eval import data as D
from rakshak.models.features import Featurized, featurize

ARMS = ("A", "B", "C")
DEFAULT_SEEDS = (1, 2, 3)
DATA_SEED = 0  # build_pools seed: every training seed shares the same extra rows
BOOT_SEED = 0
TRAIN_ADV_P = 0.0
KD_ALPHA = 0.7
KD_TEMP = 2.0

GROUNDLM_SCOREABLE = 1133
GROUNDLM_MAX_FP = 11
SMISHX_SMISHING = 226
SMISHX_MIN_TP = 157
CAMPAIGN_MIN_F1 = 0.9713
ONNX_MAX_BYTES = 10 * 1024 * 1024

SET_LEGIT_VAL = "legit_val"
SET_CAL = "campaign_cal"
SET_CAMPAIGN_TEST = "campaign_test"
SET_GROUNDLM = "groundlm"
SET_SMISHX = "smishx"
SET_SMSSHIELD = "smsshield_eval"
SET_IEEE_HAM = "ieee_ham"
SET_VAL_SCAM = "val_scam"
PRE_THRESHOLD_SETS = (SET_LEGIT_VAL, SET_CAL)
SHIFT_PREFIXES = ("lang_shift", "temporal")
LANG_SHIFT_V0_F1 = 0.986  # plan section 5, reported not gated
LANG_SHIFT_DROP_CALLOUT = 0.03

# (gate id, key in the per-seed row, pass rule on a single value)
GATES: tuple[tuple[str, str, Callable[[float], bool]], ...] = (
    ("G1_groundlm_fp", "groundlm_fp", lambda v: v <= GROUNDLM_MAX_FP),
    ("G2_smishx_smishing_tp", "smishx_tp", lambda v: v >= SMISHX_MIN_TP),
    ("G3_campaign_f1", "campaign_f1", lambda v: v >= CAMPAIGN_MIN_F1),
    ("G4_onnx_bytes", "onnx_bytes", lambda v: v <= ONNX_MAX_BYTES),
)


# ---- pure helpers (unit-tested in tests/test_sms_retrain_job.py) ----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sms_retrain")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args(list(argv or []))
    if len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must not repeat a value")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        p.error("--epochs, --patience and --batch-size must be positive")
    return args


def _majority(count: int, n: int) -> bool:
    """At least 2 of 3 (and 1 of 1, 2 of 2): more than half the seeds."""
    return count >= n // 2 + 1


def _gate(values: list[float | None], ok: Callable[[float], bool]) -> dict:
    """One gate over the seeds. Passes when the seed median passes and most seeds pass. A missing
    value (failed seed, or ONNX not exported) fails the gate."""
    flags = [v is not None and bool(ok(v)) for v in values]
    if any(v is None for v in values):
        median, median_ok = None, False
    else:
        median = float(statistics.median(values))
        median_ok = bool(ok(median))
    n, count = len(values), sum(flags)
    return {
        "values": values,
        "median": median,
        "median_pass": median_ok,
        "seeds_pass": count,
        "seeds": n,
        "passes": bool(median_ok and _majority(count, n)),
    }


def gate_eval(rows: Sequence[dict]) -> dict:
    """Gates G1 to G4 over one row per requested seed. Each row holds groundlm_fp, smishx_tp,
    campaign_f1 and onnx_bytes (None when unknown). arm_pass is True only when every gate passes."""
    if not rows:
        raise ValueError("gate_eval needs at least one seed row")
    out: dict[str, Any] = {}
    for gate, key, ok in GATES:
        out[gate] = _gate([r.get(key) for r in rows], ok)
    out["arm_pass"] = all(out[gate]["passes"] for gate, _, _ in GATES)
    return out


def paired_table(ours: np.ndarray, ref: np.ndarray) -> dict:
    """2x2 table of two boolean decisions over the same rows, with the exact two-sided McNemar
    p-value on the discordant pairs (binomial, p = 0.5). p = 1.0 with no discordant pairs."""
    ours = np.asarray(ours, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if ours.shape != ref.shape:
        raise ValueError(f"paired decisions differ in shape: {ours.shape} vs {ref.shape}")
    ours_only = int((ours & ~ref).sum())
    ref_only = int((~ours & ref).sum())
    discordant = ours_only + ref_only
    p = 1.0 if discordant == 0 else float(binomtest(ours_only, discordant, 0.5).pvalue)
    return {
        "n": int(len(ours)),
        "both": int((ours & ref).sum()),
        "ours_only": ours_only,
        "ref_only": ref_only,
        "neither": int((~ours & ~ref).sum()),
        "ours_total": int(ours.sum()),
        "ref_total": int(ref.sum()),
        "mcnemar_exact_p": p,
    }


def bucket_counts(categories: Sequence[str], ours: np.ndarray, ref: np.ndarray) -> dict:
    """Per-category counts of flagged rows for ours and the reference (GroundLM negatives)."""
    cats = np.asarray([str(c) for c in categories])
    ours = np.asarray(ours, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    out: dict[str, dict] = {}
    for cat in sorted(set(cats.tolist())):
        m = cats == cat
        out[cat] = {
            "n": int(m.sum()),
            "ours_flagged": int(ours[m].sum()),
            "ref_flagged": int(ref[m].sum()),
            "ours_only": int((ours[m] & ~ref[m]).sum()),
            "ref_only": int((~ours[m] & ref[m]).sum()),
        }
    return out


def rate_on(scores: np.ndarray, mask: np.ndarray, tau: float) -> dict:
    """Share of the masked rows scored at or above tau, with a Wilson interval."""
    s = np.asarray(scores, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return rate_row(int((s >= tau).sum()), int(len(s)))


def float_labels(series: pd.Series) -> np.ndarray:
    """0.0 / 1.0 with NaN for missing labels (Int64 with NA, or object columns)."""
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Float64")
        .to_numpy(dtype="float64", na_value=np.nan)
    )


def expand(n: int, idx: np.ndarray, values: np.ndarray, fill: Any) -> np.ndarray:
    """A full-length array with ``values`` at ``idx`` and ``fill`` elsewhere. Boolean values keep a
    boolean array (the rule flags feed groundlm_table's ``&``)."""
    values = np.asarray(values)
    dtype = bool if values.dtype == bool else np.float64
    out = np.full(n, fill, dtype=dtype)
    out[np.asarray(idx, dtype=np.int64)] = values
    return out


def run_seed_pipeline(
    *,
    fit: Callable[[], Any],
    score: Callable[[Any, str], np.ndarray],
    rule: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[float, bool]],
    y_cal: np.ndarray,
    post_sets: Sequence[str],
    on_threshold: Callable[[float, bool], None],
) -> dict:
    """Fit one model, derive tau from the legit validation pool and campaign cal, persist it, and
    only then score ``post_sets``. ``events`` records the order, so the caller and the tests can
    check that no test set is scored before the threshold is written."""
    events: list[str] = []
    model = fit()
    events.append("fit")
    scores: dict[str, np.ndarray] = {}
    for name in PRE_THRESHOLD_SETS:
        scores[name] = np.asarray(score(model, name), dtype=np.float64)
        events.append(f"score:{name}")
    tau, fallback = rule(scores[SET_LEGIT_VAL], scores[SET_CAL], y_cal)
    tau, fallback = float(tau), bool(fallback)
    on_threshold(tau, fallback)
    events.append("threshold")
    for name in post_sets:
        if name in scores:
            raise ValueError(f"set {name!r} is scored twice")
        scores[name] = np.asarray(score(model, name), dtype=np.float64)
        events.append(f"score:{name}")
    return {"model": model, "tau": tau, "fallback": fallback, "scores": scores, "events": events}


def build_pools_once(
    arm: str,
    sources: dict,
    test_only_texts: list[str],
    n_neg: int,
    groundlm_texts: list[str],
    build: Callable[..., Any],
) -> Any:
    """Call build_pools exactly once with the data seed, so every training seed shares the rows.
    The clustering thresholds are left to the module defaults."""
    return build(
        arm=arm,
        sources=sources,
        test_only_texts=test_only_texts,
        n_campaign_train_neg=n_neg,
        seed=DATA_SEED,
        groundlm_texts=groundlm_texts,
    )


def jaccard_defaults(build: Callable[..., Any]) -> dict[str, float]:
    """The *_jaccard defaults of the prep function, read from its signature (for the record)."""
    import inspect

    out = {}
    for name, par in inspect.signature(build).parameters.items():
        if name.endswith("_jaccard") and par.default is not inspect.Parameter.empty:
            out[name] = par.default
    return out


def str_texts(values: Sequence[Any]) -> list[str]:
    return [v if isinstance(v, str) else "" for v in values]


def concat(parts: list[Featurized]) -> Featurized:
    """Row-wise concatenation of featurized parts (as sms_groupneg does)."""
    return Featurized(
        texts=[t for part in parts for t in part.texts],
        raw_bytes=np.concatenate([part.raw_bytes for part in parts]),
        canon_bytes=np.concatenate([part.canon_bytes for part in parts]),
        side_raw=np.concatenate([part.side_raw for part in parts]),
        side_canon=np.concatenate([part.side_canon for part in parts]),
        buckets=None,
    )


def xlmr_reference(
    gl: pd.DataFrame, xl: pd.DataFrame, needed: np.ndarray
) -> tuple[np.ndarray, dict]:
    """The released GroundLM XLM-R decision (``pred``, tau 0.5) for each GroundLM row, joined on
    ``idx`` = the GroundLM ``row_key``. Raises if a needed row has no match. Returns the full-length
    decision array (False where not needed) and join statistics."""
    xl_idx = pd.to_numeric(xl["idx"], errors="raise").astype(np.int64).to_numpy()
    if len(set(xl_idx.tolist())) != len(xl_idx):
        raise RuntimeError("groundlm_xlmr_probs has duplicate idx values")
    pos_of = {int(k): i for i, k in enumerate(xl_idx)}
    keys = pd.to_numeric(gl["row_key"], errors="coerce").to_numpy(dtype=np.float64)
    rows = np.full(len(gl), -1, dtype=np.int64)
    for i, k in enumerate(keys):
        if np.isfinite(k):
            rows[i] = pos_of.get(int(k), -1)
    missing = int(((rows < 0) & needed).sum())
    if missing:
        raise RuntimeError(f"{missing} needed GroundLM rows have no groundlm_xlmr_probs match")
    pred_all = xl["pred"].astype(bool).to_numpy()
    ref = np.zeros(len(gl), dtype=bool)
    ref[needed] = pred_all[rows[needed]]
    gold = pd.to_numeric(xl["gold"], errors="coerce").to_numpy()
    label = gl["label_binary"].to_numpy().astype(np.int64)
    return ref, {
        "needed": int(needed.sum()),
        "matched": int(needed.sum()) - missing,
        "gold_mismatch": int((gold[rows[needed]] != label[needed]).sum()),
    }


def render_report(payload: dict) -> str:
    """A short markdown summary of one arm (report.md)."""
    gates = payload["gates"]
    lines = [
        f"# sms_retrain arm {payload['arm']}",
        "",
        f"Seeds {payload['seeds']}. Arm pass: **{gates['arm_pass']}**. "
        f"Kill and reference rows are in metrics.json.",
        "",
        "| Gate | Per seed | Median | Seeds passing | Pass |",
        "|---|---|---|---|---|",
    ]
    for gate, _, _ in GATES:
        g = gates[gate]
        vals = ", ".join("n/a" if v is None else f"{v:g}" for v in g["values"])
        med = "n/a" if g["median"] is None else f"{g['median']:g}"
        pass_txt = f"{g['seeds_pass']}/{g['seeds']}"
        lines.append(f"| {gate} | {vals} | {med} | {pass_txt} | {g['passes']} |")
    lines += [
        "",
        "| Seed | tau | fallback | GroundLM FP / 1133 | SmishX TP / 226 | "
        "Campaign F1 | McNemar p (GroundLM neg vs XLM-R) |",
        "|---|---|---|---|---|---|---|",
    ]
    for seed, block in payload["per_seed"].items():
        if block is None:
            lines.append(f"| {seed} | failed | | | | | |")
            continue
        mc = block["groundlm_paired"]["negatives"].get("mcnemar_exact_p", float("nan"))
        lines.append(
            f"| {seed} | {block['tau']:.6f} | {block['fallback_flag']} | "
            f"{block['gate_row']['groundlm_fp']} | {block['gate_row']['smishx_tp']} | "
            f"{block['gate_row']['campaign_f1']:.4f} | {mc:.3g} |"
        )
    lines += [
        "",
        f"lang_shift F1 (v0 reference {LANG_SHIFT_V0_F1}; a drop above "
        f"{LANG_SHIFT_DROP_CALLOUT} is called out):",
    ]
    for seed, block in payload["per_seed"].items():
        if block is None:
            continue
        for name, shift in block["shift"].items():
            if not name.startswith("lang_shift"):
                continue
            f1 = shift["shipped_threshold"]["f1"]
            flag = " **CALL-OUT**" if LANG_SHIFT_V0_F1 - f1 > LANG_SHIFT_DROP_CALLOUT else ""
            lines.append(f"- seed {seed} {name}: F1 {f1:.4f}{flag}")
    lines.append("")
    return "\n".join(lines)


# ---- the job ----


def _set_meta_frame(name, ids, labels, extra=None):
    return {
        "ids": np.asarray(ids).astype(str),
        "labels": np.asarray(labels, dtype=np.float64),
        "extra": extra or {},
    }


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    # Imported here: torch and the prep module are needed only on the kernel, not for the pure
    # helpers the local tests import.
    from rakshak.data import retrain_prep as RP
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

    seed_everything(DATA_SEED)
    device = pick_device("auto")
    env = env_report()
    env["device"] = str(device)
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})

    root = D.resolve_data_root(None)
    ctx.log(f"data root: {root}")
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    sms_texts = str_texts(sms["text"])
    sms_ids = sms["id"].astype(str).to_numpy()
    mask = D.eval_mask(sms)
    runs = D.load_runs(root, sms, patterns=["campaign", *SHIFT_PREFIXES], labels=labels, mask=mask)
    campaign_runs = [r for r in runs if r.name.split(":")[0] == "campaign"]
    if not campaign_runs:
        raise RuntimeError("no campaign split run found")
    run_ = campaign_runs[0]
    shift_runs = [r for r in runs if r.name.split(":")[0] in SHIFT_PREFIXES]
    cluster_col = SMS_CLUSTER_COLUMN
    y_cal = labels[run_.cal]
    y_test = labels[run_.test]
    clusters_test = D.cluster_values(sms, run_.test, cluster_col)
    ctx.log(
        f"campaign train {len(run_.train)}, cal {len(run_.cal)}, test {len(run_.test)}; "
        f"shift runs {[r.name for r in shift_runs]}"
    )

    # GroundLM: 1 133 scoreable policy negatives; positives with text go to the paired recall table.
    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    gl_avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    gl_ok = gl["eval_ok"].fillna(False).astype(bool).to_numpy()
    gl_pol = gl["policy_negative"].fillna(False).astype(bool).to_numpy()
    gl_label = gl["label_binary"].to_numpy().astype(np.int64)
    gl_avail_idx = np.flatnonzero(gl_avail)
    neg_eval = gl_pol & gl_avail & gl_ok
    pos_eval = (gl_label == 1) & gl_avail & gl_ok
    neg_idx = np.flatnonzero(neg_eval)
    pos_idx = np.flatnonzero(pos_eval)
    if len(neg_idx) != GROUNDLM_SCOREABLE:
        raise RuntimeError(f"expected {GROUNDLM_SCOREABLE} scoreable negatives, got {len(neg_idx)}")
    gl_texts = str_texts(gl["text"].to_numpy()[gl_avail_idx])

    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    sx_avail_idx = np.flatnonzero(sx_avail)
    sx_texts = str_texts(sx["text"].to_numpy()[sx_avail_idx])

    # Sources and the test-only pool. Every text here is removed from every train and validation
    # pool by build_pools.
    sources = RP.load_indian_sources(root)
    # Optional sources: a missing one is recorded in metrics.json as skipped, not an error.
    xl = sources.get("groundlm_xlmr_probs")
    ss = sources.get(SET_SMSSHIELD)
    ieee = sources.get(SET_IEEE_HAM)
    skipped: dict[str, str] = {}
    for name, frame in (("groundlm_xlmr_probs", xl), (SET_SMSSHIELD, ss), (SET_IEEE_HAM, ieee)):
        if frame is None:
            skipped[name] = "skipped: not available"
            ctx.log(f"{name}: skipped: not available")

    test_only: list[str] = [sms_texts[i] for i in run_.test]
    for r in shift_runs:
        test_only += [sms_texts[i] for i in r.test]
    test_only += gl_texts
    test_only += sx_texts
    if ss is not None:
        test_only += [t for t in str_texts(ss["text"]) if t]
    if ieee is not None:
        test_only += [t for t in str_texts(ieee["text"]) if t]
    test_only = [t for t in test_only if t]

    with timed("prep_s", timings):
        n_campaign_neg = int((labels[run_.train] == 0).sum())
        prep = build_pools_once(
            args.arm, sources, test_only, n_campaign_neg, gl_texts, RP.build_pools
        )
    report = {
        "arm": args.arm,
        "data_seed": DATA_SEED,
        "test_only_texts": len(test_only),
        "jaccard_defaults": jaccard_defaults(RP.build_pools),
        **prep.report,
    }
    (ctx.work / "prep_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    ctx.log(
        f"prep: train_extra {len(prep.train_extra)}, legit_val {len(prep.legit_val)}, "
        f"val_scam {len(prep.val_scam)}"
    )

    # Featurize once: every set is scored with the same featurization as sms_groupneg.
    extra_texts = str_texts(prep.train_extra["text"])
    y_extra = pd.to_numeric(prep.train_extra["label"]).to_numpy().astype(np.int64)
    legit_texts = str_texts(prep.legit_val["text"])
    y_legit = (
        float_labels(prep.legit_val["label"])
        if "label" in prep.legit_val
        else np.zeros(len(legit_texts))
    )
    scam_texts = str_texts(prep.val_scam["text"])
    y_scam = (
        float_labels(prep.val_scam["label"])
        if "label" in prep.val_scam
        else np.ones(len(scam_texts))
    )

    with timed("featurize_s", timings):
        fz_sms = featurize(sms_texts, buckets=None)
        fz_extra = featurize(extra_texts, buckets=None)
        fz_base = fz_sms.take(run_.train)
        train_feats = concat([fz_base, fz_extra]) if len(extra_texts) else fz_base
        y_train = np.concatenate([labels[run_.train], y_extra])
        ctx.log(
            f"train rows {len(y_train)} (base {len(fz_base)} + extra {len(fz_extra)}), "
            f"positives {int(y_train.sum())}"
        )

        feats: dict[str, Featurized] = {
            SET_LEGIT_VAL: featurize(legit_texts, buckets=None),
            SET_CAL: fz_sms.take(run_.cal),
            SET_CAMPAIGN_TEST: fz_sms.take(run_.test),
            SET_GROUNDLM: featurize(gl_texts, buckets=None),
            SET_SMISHX: featurize(sx_texts, buckets=None),
            SET_VAL_SCAM: featurize(scam_texts, buckets=None),
        }
        if ss is not None:
            feats[SET_SMSSHIELD] = featurize(str_texts(ss["text"]), buckets=None)
        if ieee is not None:
            feats[SET_IEEE_HAM] = featurize(str_texts(ieee["text"]), buckets=None)
        for r in shift_runs:
            feats[r.name] = fz_sms.take(r.test)
        gl_rules_av = rules_any(gl_texts)
        sx_rules_av = rules_any(sx_texts)

    post_sets = [
        n
        for n in [
            SET_CAMPAIGN_TEST,
            *[r.name for r in shift_runs],
            SET_GROUNDLM,
            SET_SMISHX,
            SET_SMSSHIELD,
            SET_IEEE_HAM,
            SET_VAL_SCAM,
        ]
        if n in feats
    ]

    # Labels, ids and extra columns per set, for the probs table and the metrics.
    sx_label = sx["label_binary"].to_numpy().astype(np.int64)
    sx_fine = sx["label_fine"].astype(str).to_numpy()
    gl_id = gl["id"].astype(str).to_numpy()
    gl_cat = gl["category"].astype(str).to_numpy()
    ss_label = (
        (float_labels(ss["label"]) if "label" in ss else np.full(len(ss), np.nan))
        if ss is not None
        else np.zeros(0)
    )
    meta = {
        SET_LEGIT_VAL: _set_meta_frame(
            SET_LEGIT_VAL, [f"legit_val_{i}" for i in range(len(legit_texts))], y_legit
        ),
        SET_CAL: _set_meta_frame(SET_CAL, sms_ids[run_.cal], y_cal),
        SET_CAMPAIGN_TEST: _set_meta_frame(SET_CAMPAIGN_TEST, sms_ids[run_.test], y_test),
        SET_GROUNDLM: _set_meta_frame(
            SET_GROUNDLM,
            gl_id[gl_avail_idx],
            gl_label[gl_avail_idx],
            {"category": gl_cat[gl_avail_idx]},
        ),
        SET_SMISHX: _set_meta_frame(
            SET_SMISHX,
            sx["id"].astype(str).to_numpy()[sx_avail_idx],
            sx_label[sx_avail_idx],
            {"label_fine": sx_fine[sx_avail_idx]},
        ),
        SET_SMSSHIELD: _set_meta_frame(
            SET_SMSSHIELD,
            [f"smsshield_{i}" for i in range(len(ss_label))],
            ss_label,
        ),
        SET_VAL_SCAM: _set_meta_frame(
            SET_VAL_SCAM, [f"val_scam_{i}" for i in range(len(scam_texts))], y_scam
        ),
    }
    if ieee is not None:
        ieee_label = float_labels(ieee["label"]) if "label" in ieee else np.zeros(len(ieee))
        meta[SET_IEEE_HAM] = _set_meta_frame(
            SET_IEEE_HAM, [f"ieee_{i}" for i in range(len(ieee))], ieee_label
        )
    for r in shift_runs:
        meta[r.name] = _set_meta_frame(r.name, sms_ids[r.test], labels[r.test])

    # Reference decision and paired tables are fixed before any seed is trained.
    if xl is not None:
        ref_gl, join_stats = xlmr_reference(gl, xl, neg_eval | pos_eval)
    else:
        ref_gl, join_stats = None, {"skipped": "skipped: not available"}
    shift_clusters = {r.name: D.cluster_values(sms, r.test, cluster_col) for r in shift_runs}
    ctx.log(f"XLM-R join: {join_stats}")

    spec = ModelSpec("dual", "base", "dual", True)
    cfg_common = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        adv_p=TRAIN_ADV_P,
        kd_alpha=KD_ALPHA,
        kd_temp=KD_TEMP,
    )
    thresholds: dict[str, Any] = {
        "arm": args.arm,
        "rule": "rakshak.data.retrain_prep.threshold_rule",
        "seeds": {},
    }
    seed_out: dict[int, dict] = {}
    frames: list[pd.DataFrame] = []
    onnx_dir = ctx.work

    def run_seed(seed: int) -> dict:
        seed_everything(seed)
        state: dict[str, Any] = {}

        def fit():
            cfg = TrainConfig(seed=seed, **cfg_common)
            with timed(f"train_seed{seed}_s", timings):
                model, info = train_student(
                    spec,
                    train_feats,
                    y_train,
                    feats[SET_CAL],
                    y_cal,
                    cfg,
                    device=device,
                    teacher=None,
                    log=ctx.log,
                )
            state["info"] = info
            return model.to(device)

        def score(model, name: str) -> np.ndarray:
            return predict_scores(model, spec, feats[name], device)

        def on_threshold(tau: float, fallback: bool) -> None:
            thresholds["seeds"][str(seed)] = {
                "tau": tau,
                "fallback_flag": fallback,
                "n_legit_val": len(legit_texts),
                "n_campaign_cal": int(len(y_cal)),
            }
            (ctx.work / "thresholds.json").write_text(
                json.dumps(thresholds, indent=2, sort_keys=True) + "\n"
            )
            ctx.log(
                f"seed {seed}: tau {tau:.6f} fallback {fallback} written to thresholds.json "
                "before any test set is scored"
            )

        pipe = run_seed_pipeline(
            fit=fit,
            score=score,
            rule=RP.threshold_rule,
            y_cal=y_cal,
            post_sets=post_sets,
            on_threshold=on_threshold,
        )
        tau = pipe["tau"]
        model = pipe["model"]
        sc = pipe["scores"]
        ctx.log(f"seed {seed}: events {pipe['events']}")

        gl_full = expand(len(gl), gl_avail_idx, sc[SET_GROUNDLM], -1.0)
        gl_rules = expand(len(gl), gl_avail_idx, gl_rules_av, False)
        sx_full = expand(len(sx), sx_avail_idx, sc[SET_SMISHX], -1.0)
        sx_rules = expand(len(sx), sx_avail_idx, sx_rules_av, False)

        campaign = sms_run_table(
            y_cal,
            sc[SET_CAL],
            y_test,
            sc[SET_CAMPAIGN_TEST],
            clusters_test,
            threshold=tau,
            n_boot=args.n_boot,
        )
        shifts = {}
        for r in shift_runs:
            shifts[r.name] = sms_run_table(
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                labels[r.test],
                sc[r.name],
                shift_clusters[r.name],
                threshold=tau,
                n_boot=args.n_boot,
            )
        gl_block = groundlm_table(gl, gl_full, gl_rules, tau)
        sx_block = smishx_table(sx, sx_full, sx_rules, tau)

        if ref_gl is not None:
            ours_neg = gl_full[neg_idx] >= tau
            ref_neg = ref_gl[neg_idx]
            neg_paired = paired_table(ours_neg, ref_neg)
            neg_paired["buckets"] = bucket_counts(gl_cat[neg_idx], ours_neg, ref_neg)
            ours_pos = gl_full[pos_idx] >= tau
            pos_paired = paired_table(ours_pos, ref_gl[pos_idx])
        else:
            neg_paired = pos_paired = {"skipped": "skipped: not available"}

        smsshield: Any = {"skipped": "skipped: not available"}
        if ss is not None:
            ss_lab = meta[SET_SMSSHIELD]["labels"]
            ss_scores = sc[SET_SMSSHIELD]
            smsshield = {
                "n_safe": int((ss_lab == 0).sum()),
                "safe_fpr": rate_on(ss_scores, ss_lab == 0, tau),
                "n_scam": int((ss_lab == 1).sum()),
                "scam_recall": rate_on(ss_scores, ss_lab == 1, tau),
                "suspicious_reported": rate_on(ss_scores, np.isnan(ss_lab), tau),
            }
        block: dict[str, Any] = {
            "seed": seed,
            "tau": tau,
            "fallback_flag": pipe["fallback"],
            "train": state.get("info"),
            "campaign": campaign,
            "shift": shifts,
            "groundlm": gl_block,
            "groundlm_paired": {"negatives": neg_paired, "positives": pos_paired},
            "smishx": sx_block,
            "smsshield_eval": smsshield,
            "legit_val_fpr": rate_on(sc[SET_LEGIT_VAL], meta[SET_LEGIT_VAL]["labels"] == 0, tau),
            "val_scam_recall": rate_on(sc[SET_VAL_SCAM], y_scam == 1, tau),
            "join": join_stats,
        }
        if ieee is not None:
            block["ieee_ham_fpr"] = rate_on(
                sc[SET_IEEE_HAM], meta[SET_IEEE_HAM]["labels"] == 0, tau
            )
        block["params"] = param_count(model.cpu())
        block["sets_scored"] = sorted(sc)

        onnx_info = None
        if not args.no_export:
            with timed(f"export_seed{seed}_s", timings):
                path = onnx_dir / f"best_student_seed{seed}.onnx"
                exported = export_onnx(model.cpu(), path)
                parity = onnx_parity(model.cpu(), path, feats[SET_CAMPAIGN_TEST])
                exported["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                onnx_info = {**exported, "parity": parity}
            ctx.log(f"seed {seed}: exported {exported['bytes']} bytes, parity {parity}")
        block["onnx"] = onnx_info

        block["gate_row"] = {
            "seed": seed,
            "groundlm_fp": gl_block["student_at_threshold"]["k"],
            "smishx_tp": sx_block["smishing_recall"]["student_at_threshold"]["k"],
            "campaign_f1": campaign["shipped_threshold"]["f1"],
            "onnx_bytes": onnx_info["bytes"] if onnx_info else None,
        }

        # Per-row scores for every set at this seed (probs.parquet). Written after tau, so each
        # row is one set's score from the model that the thresholds.json entry describes.
        parts = []
        for name, values in sc.items():
            m = meta[name]
            df = pd.DataFrame(
                {
                    "seed": np.full(len(values), seed, dtype=np.int64),
                    "set": name,
                    "id": m["ids"],
                    "label": m["labels"],
                    "score": values,
                }
            )
            for key, col in m["extra"].items():
                df[key] = col
            parts.append(df)
        seed_frame = pd.concat(parts, ignore_index=True)
        frames.append(seed_frame)
        return block

    u = Units(ctx)
    for seed in args.seeds:
        seed_out[seed] = u.run({"seed": seed}, run_seed, seed)

    rows = []
    for seed in args.seeds:
        block = seed_out.get(seed)
        rows.append(block["gate_row"] if block else {"seed": seed})
    gates = gate_eval(rows)

    probs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    probs.to_parquet(ctx.work / "probs.parquet", index=False)
    ctx.log(f"wrote probs.parquet ({len(probs)} rows)")

    payload = {
        "job": "sms_retrain",
        "arm": args.arm,
        "seeds": list(args.seeds),
        "data_seed": DATA_SEED,
        "args": vars(args),
        "env": env,
        "prep": report,
        "thresholds": thresholds,
        "gates": gates,
        "per_seed": {str(s): seed_out.get(s) for s in args.seeds},
        "failures": ctx.results.get("failures", []),
        "skipped": skipped,
    }
    write_metrics("metrics.json", payload)
    (ctx.work / "report.md").write_text(render_report(payload))

    ctx.results["summary"] = (
        f"sms_retrain arm {args.arm}: seeds ok {u.ok}/{len(args.seeds)}, "
        f"arm_pass {gates['arm_pass']}, groundlm fp median {gates['G1_groundlm_fp']['median']}, "
        f"smishx tp median {gates['G2_smishx_smishing_tp']['median']}, "
        f"campaign f1 median {gates['G3_campaign_f1']['median']}"
    )
    status = job_status(ok=u.ok, failed=u.failed)
    return status if status != "fail" else False
