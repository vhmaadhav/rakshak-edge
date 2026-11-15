"""Inbox calibration of the W1e encoder (docs/INBOX_CALIBRATION_PLAN.md, pre-registered 2026-11-12).

No model is trained here. Three W1e checkpoints (seeds 0, 1, 2) and the base teacher (reference
only) are scored on the consented train donor's legit rows (phones 2 and 3). The threshold is the
smallest logit margin with at most floor(1 % x n) legit calibration rows flagged. The primary arm
uses the surrogate-filled text (the S-arm form); the secondary arm uses the raw scrubbed text. Then
the test sets are scored once: GroundLM, SmishX, campaign (clean and filled), lang_shift, temporal,
and phone 1 (the test donor, filled and raw).

Order (enforced): every model scores both calibration sets, both thresholds are computed, and
thresholds.json is written; only then is any test set scored (two_pass, unit-tested with fakes).

Gates G1 to G5 (primary arm, W1e seeds only; seed median and at least 2 of 3 seeds). The teacher
row is a reference and has no verdict.

PRIVACY (binding). No inbox text, sender string or inbox row id is written anywhere. Inbox rows are
indexed calib_legit_<i> and phone1_<i>. Logs, prep_report.json, thresholds.json, metrics.json,
report.md and probs.parquet carry counts, labels, margins, probabilities, thresholds and intervals
only. Errors are logged as exception type and frames (REDACT_ERRORS).

Launch (T4, internet on; the data, the W1e kernel output and the teacher kernel output):
  kaggle/launch.sh sms_inbox_calibration --gpu t4 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-encoder-exposure-w1e maadhavvh/rakshak-sms-teacher \
      --job-args="--seeds 0 1 2"

Flags (--job-args="FLAGS"):
  --seeds INT ...           W1e seeds to report (default 0 1 2; any other value is an error)
  --max-len INT             token limit (default 128)
  --eval-batch-size INT     inference batch (default 128)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sms_encoder_exposure as EX
import sms_inbox_retrain as IN
import sms_retrain as base
import sms_teacher_suite as S
import sms_teacher_xsource as X
from _common import env_report, seed_everything, timed

from rakshak.data import surrogates as SUR
from rakshak.eval import data as D
from rakshak.eval import metrics as M

JOB = "sms_inbox_calibration"
PLAN = "docs/INBOX_CALIBRATION_PLAN.md"
W1E_KERNEL = "sms-encoder-exposure-w1e"  # substring of the attached W1e kernel output path
# The W1e checkpoint digests (checkpoint_sha256 in the W1e kernel's metrics.json).
W1E_SHAS = {
    0: "355b6519680a79d3d05ee5917692b3cfa18b878c0610b494ca6f2b2231458189",
    1: "91cb21ee392bbab5ef08f63433d0215ceed3bb4ec4394ad37ba68caf6f029313",
    2: "2408940091dd9cc765c39804494f4f758119bacfeac2afef28df01a13128eed1",
}
TEACHER_SHA = X.TEACHER_SHA
DEFAULT_SEEDS = (0, 1, 2)
FPR_CAL = 0.01
G1_MAX_FP = S.G1_MAX_FP
G2_MIN_TP = S.G2_MIN_TP
G3_MIN_RECALL = EX.G3_MIN_RECALL
G4_MIN_F1 = S.G4_MIN_F1
G5_MAX_FPR = IN.G5_MAX_FPR
MIN_CAL_LEGIT = IN.MIN_DONOR_LEGIT
DEFAULT_ROOT = Path("/kaggle/input")
STOP_STATUS = "stopped: insufficient data"
# Read by kaggle/build_kernel.py: an error is logged and stored as its type and frames only.
REDACT_ERRORS = True

CAL_FILLED = "calib_legit_filled"
CAL_RAW = "calib_legit_raw"
P1_FILLED = "phone1_filled"
P1_RAW = "phone1_raw"
CAMP_FILLED = "campaign_filled"
CAL_NAMES = (CAL_FILLED, CAL_RAW)
TEST_NAMES = (*S.TEST_SETS, P1_FILLED, P1_RAW, CAMP_FILLED)
TEACHER = "teacher"

# (gate, key, rule) for G1 to G5. G1 to G4 are the W1e and S-arm limits; G5 is phone-1 legit FPR.
GATE_SPECS = (
    ("G1_groundlm_neg_fp", "fp_neg", lambda v: v <= G1_MAX_FP),
    ("G2_smishx_smishing_tp", "tp_smish", lambda v: v >= G2_MIN_TP),
    ("G3_groundlm_pos_recall", "recall_pos", lambda v: v >= G3_MIN_RECALL),
    ("G4_campaign_clean_f1", "f1_clean", lambda v: v >= G4_MIN_F1),
    ("G5_phone1_legit_fpr", "donor_fpr", lambda v: v <= G5_MAX_FPR),
)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--max-len", type=int, default=S.MAX_LEN)
    p.add_argument("--eval-batch-size", type=int, default=S.EVAL_BATCH)
    args = p.parse_args(list(argv or []))
    if any(s not in W1E_SHAS for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        p.error(f"--seeds must be distinct values from {sorted(W1E_SHAS)}")
    if args.max_len < 1 or args.eval_batch_size < 1:
        p.error("--max-len and --eval-batch-size must be positive")
    return args


# ---- pure helpers (unit-tested in tests/test_sms_inbox_calibration.py) ----


def model_names(seeds: Sequence[int]) -> dict[str, str]:
    """Model name to pinned sha256: one per W1e seed, then the teacher reference."""
    out = {f"w1e-s{s}": W1E_SHAS[s] for s in seeds}
    out[TEACHER] = TEACHER_SHA
    return out


def calibration_tau(margins: Sequence[float], max_fpr: float = FPR_CAL) -> dict:
    """Smallest margin with at most floor(max_fpr x n) legit rows flagged (W1e's rule, reused).
    Reports the count flagged at tau; -inf when every row may flag."""
    m = np.asarray(margins, dtype=np.float64)
    tau = EX.logit_threshold(m, max_fpr)
    return {
        "tau": float(tau),
        "n": int(m.size),
        "k_allowed": int(math.floor(max_fpr * m.size)),
        "flagged": int(np.sum(m >= tau)),
    }


def clean_from_probs(probs: pd.DataFrame, camp_ids: Sequence[str], seed: int = 0) -> np.ndarray:
    """The W1e campaign clean mask (its own training partition), read from its probs.parquet
    campaign rows. Asserts the ids match this job's campaign ids in order, and that every W1e seed
    records the same mask."""
    rows = probs[probs["set"] == S.SET_CAMP]
    base_rows = rows[rows["seed"] == seed]
    if base_rows["id"].astype(str).tolist() != [str(i) for i in camp_ids]:
        raise ValueError("W1e campaign ids differ from this job's campaign ids")
    for other in sorted(set(rows["seed"].tolist()) - {seed}):
        o = rows[rows["seed"] == other]["clean"].to_numpy(dtype=bool)
        if not np.array_equal(o, base_rows["clean"].to_numpy(dtype=bool)):
            raise ValueError("W1e seeds disagree on the campaign clean mask")
    return base_rows["clean"].to_numpy(dtype=bool)


def phone_block(labels: np.ndarray, scores: np.ndarray, tau: float, novel: np.ndarray) -> dict:
    """Phone 1 (test donor): legit FPR, template-novel legit FPR, and scam recall at tau."""
    y = np.asarray(labels).astype(np.int64)
    legit = y == 0
    scam = y == 1
    nov = np.asarray(novel, dtype=bool)
    return {
        "legit_fpr": base.rate_on(scores, legit, tau),
        "legit_fpr_template_novel": base.rate_on(scores, legit & nov, tau),
        "n_template_novel_legit": int((legit & nov).sum()),
        "scam_recall": base.rate_on(scores, scam, tau),
    }


def campaign_filled_block(
    labels: np.ndarray, scores: np.ndarray, tau: float, clean: np.ndarray, codes: np.ndarray
) -> dict:
    """Campaign F1 on the filled text (reported only, the S-arm G3 surface), all and clean rows."""
    y = np.asarray(labels).astype(np.int64)
    flags = np.asarray(scores, dtype=np.float64) >= tau
    cl = np.asarray(clean, dtype=bool)
    return {
        "all": {
            **M.metrics_from_counts(*M.confusion_counts(y, flags)),
            "f1_ci95": S.cluster_ci(y, flags, codes, "f1"),
        },
        "clean": {
            **M.metrics_from_counts(*M.confusion_counts(y[cl], flags[cl])),
            "n_clean": int(cl.sum()),
            "f1_ci95": S.cluster_ci(y[cl], flags[cl], np.asarray(codes)[cl], "f1"),
        },
    }


def gate_inputs(block: dict, phone: dict, groundlm_pos_n: int) -> dict:
    """The five gate inputs of one seed: the arm block (G1 to G4) and the phone-1 block (G5)."""
    neg = block["groundlm_neg_fp"]
    pos = block["groundlm_pos_recall"]
    smish = block["smishx_smishing_recall"]
    donor = phone["legit_fpr"]
    return {
        "fp_neg": int(neg["k"]),
        "tp_smish": int(smish["k"]),
        "tp_pos": int(pos["k"]),
        "n_pos": int(groundlm_pos_n),
        "recall_pos": float(pos["k"]) / float(groundlm_pos_n),
        "f1_clean": float(block["campaign_clean"]["f1"]),
        "donor_fpr": float(donor["rate"]),
        "donor_k": int(donor["k"]),
        "donor_n": int(donor["n"]),
    }


def aggregate_gates(per_seed: Sequence[dict]) -> dict:
    """Each gate holds when the seed median passes and at least a majority of seeds pass."""
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


def two_pass(
    scorers: dict[str, Callable[[str], np.ndarray]],
    write_thresholds: Callable[[dict], None],
) -> dict:
    """Calibration first for every model, then thresholds (written through ``write_thresholds``),
    then every test set. ``events`` records the order; the caller and the tests check it."""
    events: list[str] = []
    cal: dict[str, dict[str, np.ndarray]] = {}
    for model, score in scorers.items():
        cal[model] = {}
        for name in CAL_NAMES:
            cal[model][name] = np.asarray(score(name), dtype=np.float64)
            events.append(f"cal:{model}:{name}")
    taus = {
        model: {
            "filled": calibration_tau(cal[model][CAL_FILLED]),
            "raw": calibration_tau(cal[model][CAL_RAW]),
        }
        for model in scorers
    }
    write_thresholds(taus)
    events.append("threshold")
    test: dict[str, dict[str, np.ndarray]] = {}
    for model, score in scorers.items():
        test[model] = {}
        for name in TEST_NAMES:
            test[model][name] = np.asarray(score(name), dtype=np.float64)
            events.append(f"test:{model}:{name}")
    first_test = min(i for i, e in enumerate(events) if e.startswith("test:"))
    if events.index("threshold") > first_test:
        raise RuntimeError("a test set was scored before the threshold")
    return {"cal": cal, "taus": taus, "test": test, "events": events}


def probs_frame(
    model: str,
    sets: dict[str, dict],
    scores: dict[str, np.ndarray],
    phone_labels: dict[str, np.ndarray],
    clean: np.ndarray,
) -> pd.DataFrame:
    """Margins and probabilities per row for one model. Inbox rows carry index ids only."""
    parts = []

    def add(name: str, ids, labels, margins, category=None, clean_col=None, xlmr=None) -> None:
        n = len(margins)
        parts.append(
            pd.DataFrame(
                {
                    "model": model,
                    "set": name,
                    "id": np.asarray(ids).astype(str),
                    "label": np.asarray(labels, dtype=np.float64),
                    "margin": np.asarray(margins, dtype=np.float64),
                    "category": np.asarray(category if category is not None else [""] * n).astype(
                        str
                    ),
                    "clean": (
                        np.full(n, np.nan)
                        if clean_col is None
                        else np.asarray(clean_col, dtype=np.float64)
                    ),
                    "xlmr_pred": (
                        np.full(n, np.nan) if xlmr is None else np.asarray(xlmr, dtype=np.float64)
                    ),
                }
            )
        )

    for name in CAL_NAMES:
        n = len(scores[name])
        add(name, [f"calib_legit_{i}" for i in range(n)], np.zeros(n), scores[name])
    for name in (P1_FILLED, P1_RAW):
        n = len(scores[name])
        add(name, [f"phone1_{i}" for i in range(n)], phone_labels[name], scores[name])
    camp = sets[S.SET_CAMP]
    add(CAMP_FILLED, camp["ids"], camp["labels"], scores[CAMP_FILLED], clean_col=clean)
    for name in S.TEST_SETS:
        s = sets[name]
        add(
            name,
            s["ids"],
            s["labels"],
            scores[name],
            category=s["category"],
            clean_col=(clean if name == S.SET_CAMP else s["clean"]),
            xlmr=s["xlmr"],
        )
    out = pd.concat(parts, ignore_index=True)
    out["p_scam"] = 1.0 / (1.0 + np.exp(-out["margin"].to_numpy(dtype=np.float64)))
    return out


def arm_result(
    sets: dict[str, dict],
    scores: dict[str, np.ndarray],
    tau: float,
    phone_key: str,
    phone_labels: np.ndarray,
    novel: np.ndarray,
    clean: np.ndarray,
) -> dict:
    """All blocks of one model at one arm's threshold."""
    flags = {n: scores[n] >= tau for n in S.TEST_SETS}
    sets_m = dict(sets)
    sets_m[S.SET_CAMP] = {**sets[S.SET_CAMP], "clean": clean}
    block = S.arm_block(sets_m, flags, min_recall=G3_MIN_RECALL)
    phone = phone_block(phone_labels, scores[phone_key], tau, novel)
    camp_f = campaign_filled_block(
        sets[S.SET_CAMP]["labels"], scores[CAMP_FILLED], tau, clean, sets[S.SET_CAMP]["codes"]
    )
    gi = gate_inputs(block, phone, int(len(flags[S.SET_GL_POS])))
    return {
        "tau": float(tau),
        "block": block,
        "phone1": phone,
        "campaign_filled": camp_f,
        "gate_inputs": gi,
    }


def render_report(payload: dict) -> str:
    if "models" not in payload:  # a stopped run: the status and the counts only
        status, reason = payload.get("status"), payload.get("reason", "")
        return f"# {JOB} (plan {PLAN})\n\nStatus: {status}. {reason}\n"
    lines = [
        f"# {JOB} (plan {PLAN})",
        "",
        f"Verdict (W1e seeds, primary arm): {payload['verdict']}. Teacher row is a reference only.",
        "",
        "| model | arm | tau | G1 GL-neg FP | G2 SmishX TP | G3 GL-pos recall | "
        "G4 clean F1 | G5 phone-1 legit FPR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, res in payload["models"].items():
        for arm in ("primary", "secondary"):
            r = res[arm]
            g = r["gate_inputs"]
            lines.append(
                f"| {name} | {arm} | {r['tau']:.4f} | {g['fp_neg']}/1133 | {g['tp_smish']}/226 "
                f"| {g['tp_pos']}/{g['n_pos']} ({g['recall_pos']:.3f}) | {g['f1_clean']:.4f} "
                f"| {g['donor_k']}/{g['donor_n']} ({g['donor_fpr']:.4f}) |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(work: Path, payload: dict, probs: pd.DataFrame | None, prep: dict) -> None:
    """prep_report.json, probs.parquet (when scored), metrics.json and report.md in ``work``."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "prep_report.json").write_text(json.dumps(prep, indent=2, default=str) + "\n")
    if probs is not None:
        probs.to_parquet(work / "probs.parquet", index=False)
    tmp = work / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(work / "metrics.json")
    (work / "report.md").write_text(render_report(payload))


def index_checkpoints(root: Path, wanted: dict[str, str]) -> dict[str, Path]:
    """The best_checkpoint folder under ``root`` for each model name, by its pinned sha256. Each
    checkpoint is hashed once. Raises unless each pin matches exactly one folder."""
    if not root.exists():
        raise FileNotFoundError(f"input root {root} does not exist")
    folders = sorted(
        {
            Path(d)
            for d, _dirs, files in os.walk(root)
            if Path(d).name == X.TEACHER_DIR and X.TEACHER_WEIGHTS in files
        }
    )
    by_sha: dict[str, list[Path]] = {}
    for f in folders:
        by_sha.setdefault(X.file_sha256(f / X.TEACHER_WEIGHTS), []).append(f)
    out: dict[str, Path] = {}
    for name, sha in wanted.items():
        hits = by_sha.get(sha, [])
        if len(hits) != 1:
            raise FileNotFoundError(
                f"{name}: expected one checkpoint with the pinned digest, found {len(hits)}"
            )
        out[name] = hits[0]
    return out


def novel_full(legit_mask: np.ndarray, legit_novel: np.ndarray) -> np.ndarray:
    """Full-length mask: True where a legit row has no near-duplicate among calibration rows."""
    out = np.zeros(len(legit_mask), dtype=bool)
    out[np.asarray(legit_mask, dtype=bool)] = legit_novel
    return out


def _novel_full(
    texts: Sequence[str], legit_mask: np.ndarray, cal_texts: Sequence[str]
) -> np.ndarray:
    legit_texts = [t for t, k in zip(texts, legit_mask, strict=True) if k]
    return novel_full(legit_mask, IN.novel_mask(legit_texts, cal_texts))


def calib_frame_pair(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """(filled, raw) labelled frames of one donor file, with the same rows in the same order.
    Filled uses the S-arm fill (IN.fill_inbox_frame, keyed by row id). Counts only leave here."""
    filled_raw, tags = IN.fill_inbox_frame(raw)
    IN.assert_filled(list(filled_raw["text"]), "inbox donor")
    filled, counts_f = IN.label_inbox(filled_raw)
    raw_lab, counts_r = IN.label_inbox(raw)
    if filled["id"].astype(str).tolist() != raw_lab["id"].astype(str).tolist():
        raise ValueError("filled and raw donor rows differ in ids or order")
    return filled, raw_lab, {"filled": counts_f, "raw": counts_r}, tags


# ---- the job ----


class RedactedError(RuntimeError):
    """The job failed; the cause is in the log as type and frames, not as a message."""


def run(ctx) -> bool:
    """Runs _run; any exception leaves here as RedactedError (type and frames logged)."""
    failure = None
    try:
        return _run(ctx)
    except Exception as exc:
        failure = exc
        for line in IN.redacted_frames(exc):
            ctx.log(f"  at {line}")
        ctx.log(f"ERROR {IN.redacted_error(exc)}")
    raise RedactedError(IN.redacted_error(failure)) from None


def _run(ctx) -> bool:
    args = parse_args(ctx.argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{JOB} needs a CUDA GPU; launch with --gpu t4")
    seed_everything(0)
    env = env_report()
    ctx.log(f"env: {env}")
    timings = ctx.results.setdefault("timings", {})
    work = ctx.work
    names = model_names(args.seeds)

    # 1. data and the teacher fold partitions (as sms_teacher_suite)
    root = D.resolve_data_root(None)
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    texts = base.str_texts(sms["text"])
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

    # 2. the scored sets (no model yet). Teacher-clean masks come from the fold train partition.
    train_ids = set(ids[fold.train].tolist())
    train_keys = X.canon_keys([texts[i] for i in fold.train])
    with timed("sets_s", timings):
        sets, join = S.build_sets(
            root, sms, labels, texts, ids, camp, lang, temp, train_ids, train_keys
        )
    camp_ids = sets[S.SET_CAMP]["ids"]
    w1e_clean = clean_from_probs(
        pd.read_parquet(
            IN.find_probs(W1E_KERNEL, DEFAULT_ROOT), columns=["seed", "set", "id", "clean"]
        ),
        camp_ids,
    )
    camp_filled = SUR.fill_many(list(camp_ids), [texts[i] for i in camp.test])
    IN.assert_filled(camp_filled, "campaign test (filled)")
    ctx.log(
        "sets: "
        + ", ".join(f"{n} {len(sets[n]['texts'])}" for n in S.TEST_SETS)
        + f"; campaign filled {len(camp_filled)}; XLM-R join {join}"
    )

    # 3. inbox donors: calibration = train donor legit rows; test = phone 1
    train_raw = pd.read_parquet(IN.find_one(IN.TRAIN_FILE, [DEFAULT_ROOT]))
    test_raw = pd.read_parquet(IN.find_one(IN.TEST_FILE, [DEFAULT_ROOT]))
    cal_f_df, cal_r_df, cal_counts, cal_tags = calib_frame_pair(train_raw)
    p1_f_df, p1_r_df, p1_counts, p1_tags = calib_frame_pair(test_raw)
    del train_raw, test_raw
    cal_f = cal_f_df[cal_f_df["label"] == 0]
    cal_r = cal_r_df[cal_r_df["label"] == 0]
    prep: dict[str, Any] = {
        "job": JOB,
        "data_seed": 0,
        "plan": PLAN,
        "cal_counts": cal_counts,
        "phone1_counts": p1_counts,
        "cal_tags_before_fill": cal_tags,
        "phone1_tags_before_fill": p1_tags,
        "cal_legit": int(len(cal_f)),
        "cal_legit_raw": int(len(cal_r)),
        "phone1_legit": int((p1_f_df["label"] == 0).sum()),
        "phone1_scam": int((p1_f_df["label"] == 1).sum()),
        "test_only_contamination": {
            n: {"n": len(sets[n]["texts"]), "contaminated": int((~sets[n]["clean"]).sum())}
            for n in S.TEST_SETS
        },
        "campaign_clean_rows_w1e": int(w1e_clean.sum()),
        "campaign_clean_rows_teacher": int(sets[S.SET_CAMP]["clean"].sum()),
    }
    if len(cal_f) < MIN_CAL_LEGIT:
        payload = {
            "job": JOB,
            "status": STOP_STATUS,
            "plan": PLAN,
            "reason": f"calibration legit {len(cal_f)} (need {MIN_CAL_LEGIT})",
            "counts": prep,
        }
        write_outputs(work, payload, None, prep)
        ctx.results["summary"] = f"{JOB}: {STOP_STATUS}"
        ctx.log(STOP_STATUS)
        return False

    cal_f_texts = list(cal_f["text"])
    cal_r_texts = list(cal_r["text"])
    p1_f_texts = list(p1_f_df["text"])
    p1_r_texts = list(p1_r_df["text"])
    p1_labels = {
        P1_FILLED: p1_f_df["label"].to_numpy().astype(np.int64),
        P1_RAW: p1_r_df["label"].to_numpy().astype(np.int64),
    }
    p1_leg = p1_labels[P1_FILLED] == 0
    novel = {
        "filled": _novel_full(p1_f_texts, p1_leg, cal_f_texts),
        "raw": _novel_full(p1_r_texts, p1_leg, cal_r_texts),
    }
    prep["phone1_template_novel_legit"] = {
        "filled": int(novel["filled"].sum()),
        "raw": int(novel["raw"].sum()),
    }
    texts_of: dict[str, list[str]] = {
        CAL_FILLED: cal_f_texts,
        CAL_RAW: cal_r_texts,
        P1_FILLED: p1_f_texts,
        P1_RAW: p1_r_texts,
        CAMP_FILLED: camp_filled,
    }
    texts_of.update({n: sets[n]["texts"] for n in S.TEST_SETS})

    # 4. checkpoints by pinned sha256, then the two-pass scoring
    paths = index_checkpoints(DEFAULT_ROOT, names)
    loaded: dict[str, tuple] = {}
    for name in names:
        model, tokenizer, device = X.load_teacher(paths[name])
        loaded[name] = (model, tokenizer, device)
    prep["checkpoints"] = {n: {"sha256": names[n]} for n in names}

    def scorer(name: str) -> Callable[[str], np.ndarray]:
        model, tokenizer, device = loaded[name]

        def score(set_name: str) -> np.ndarray:
            return S.teacher_margins(
                model,
                tokenizer,
                texts_of[set_name],
                device=device,
                max_len=args.max_len,
                eval_batch_size=args.eval_batch_size,
            )

        return score

    def write_thresholds(taus: dict) -> None:
        payload = {
            "plan": PLAN,
            "rule": "alarm if logit margin >= tau; smallest tau with at most "
            "floor(0.01 x n) calibration legit rows flagged",
            "calibration": {"filled_legit": int(len(cal_f)), "raw_legit": int(len(cal_r))},
            "models": {n: {"sha256": names[n], **taus[n]} for n in names},
        }
        (work / "thresholds.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        ctx.log(
            "thresholds.json written before any test set is scored: "
            + ", ".join(f"{n} {taus[n]['filled']['tau']:.6f}" for n in names)
        )

    with timed("score_s", timings):
        passes = two_pass({n: scorer(n) for n in names}, write_thresholds)
    loaded.clear()
    torch_cache(torch)

    # 5. per model: both arms, probs, gate inputs
    models_out: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []
    for name in names:
        scores = passes["test"][name]
        cal_s = passes["cal"][name]
        taus = passes["taus"][name]
        clean = w1e_clean if name != TEACHER else sets[S.SET_CAMP]["clean"]
        primary = arm_result(
            sets,
            scores,
            taus["filled"]["tau"],
            P1_FILLED,
            p1_labels[P1_FILLED],
            novel["filled"],
            clean,
        )
        secondary = arm_result(
            sets, scores, taus["raw"]["tau"], P1_RAW, p1_labels[P1_RAW], novel["raw"], clean
        )
        models_out[name] = {
            "sha256": names[name],
            "tau": taus,
            "primary": primary,
            "secondary": secondary,
            "cal_scores_summary": {
                CAL_FILLED: len(cal_s[CAL_FILLED]),
                CAL_RAW: len(cal_s[CAL_RAW]),
            },
        }
        phone_labels_all = {P1_FILLED: p1_labels[P1_FILLED], P1_RAW: p1_labels[P1_RAW]}
        frame_scores = {**cal_s, **scores}
        frames.append(probs_frame(name, sets, frame_scores, phone_labels_all, clean))

    rows = [models_out[f"w1e-s{s}"]["primary"]["gate_inputs"] for s in sorted(args.seeds)]
    gates = aggregate_gates(rows) if rows else None
    gate_verdict = verdict(rows, expected_seeds=len(W1E_SHAS))
    payload = {
        "job": JOB,
        "status": "scored",
        "plan": PLAN,
        "label": "W1e same family; calibration "
        "on consented train donor (phones 2+3), test donor phone 1 (same person)",
        "verdict": gate_verdict,
        "gates": gates,
        "gate_note": "seed median and at least 2 of 3 seeds",
        "seeds": sorted(args.seeds),
        "models": models_out,
        "thresholds": passes["taus"],
        "xlmr_join": join,
        "prep": prep,
        "events": passes["events"],
        "env": env,
        "timings": timings,
    }
    prob_frame = pd.concat(frames, ignore_index=True)
    write_outputs(work, payload, prob_frame, prep)
    ctx.results["summary"] = f"{JOB}: seeds {sorted(args.seeds)}; verdict {gate_verdict}"
    return True


def torch_cache(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
