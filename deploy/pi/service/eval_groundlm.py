#!/usr/bin/env python3
"""False-positive rate of the full edge decision path on GroundLM policy negatives (laptop, no LLM).

The decision is ``edge_decision.decide_edge``, the same function ``edge_pipeline.EdgeAnalyzer``
calls (no copy): the SMS student ONNX score (``rbench.runners.onnx.SmsRunner``, the repo's
featurisation), the rule evidence (``rakshak.rules.run_rules``), the decision state,
``rakshak.evidence.build_package``, the detector-only mode (quiet or verify), and ``choose_action``.
No Llama, no explainer text. ``tests/test_eval_groundlm.py`` checks that ``edge_pipeline`` gives the
same decision on the same inputs.

    eval_groundlm.py run --out-dir DIR [--mode quiet|verify] [--threads 1] [--limit N] [--rescore]

The user-visible warning is ``action != IGNORE`` (alert f). In quiet mode a detector-only flag is
IGNORE, so it is not a warning; in verify mode it is an UNCERTAIN / VERIFY notice.

``run`` checks the GroundLM counts (1,294 policy negatives, 1,133 scoreable), scores GroundLM text
rows, SmishX eval rows and the campaign SMS test partition, caches the per-row decisions in
``DIR/scores.parquet``, and writes ``DIR/metrics.json`` plus a printed summary. The threshold
analysis is descriptive. It is not a shipped threshold, and GroundLM is in-sample (see
docs/GROUNDLM_FPR.md), so any threshold chosen on it is not an independent test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "deploy" / "pi" / "bench")]

from edge_decision import (  # noqa: E402
    DEFAULT_DETECTOR_ONLY_MODE,
    DETECTOR_ONLY_MODES,
    check_mode,
    decide_edge,
)
from rbench.datasets import _read_split_ids  # noqa: E402
from rbench.metrics import wilson_interval  # noqa: E402
from rbench.runners.onnx import SmsRunner  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.ela_eval import proxy_scam_state  # noqa: E402
from rakshak.eval.data import eval_mask  # noqa: E402
from rakshak.evidence import Action, build_package  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402

GROUNDLM_PATH = REPO / "data/processed/external/external_groundlm.parquet"
SMISHX_PATH = REPO / "data/processed/external/external_smishx.parquet"
SMS_PATH = REPO / "data/processed/sms.parquet"
CAMPAIGN_SPLIT = REPO / "data/processed/splits/campaign.json"
REGISTRY = REPO / "deploy/pi/bench/registry.toml"
MODEL_PATH = REPO / "kaggle/out/sms_student/output/best_student.onnx"

EXPECTED_COUNTS = {"rows": 1393, "policy_negatives": 1294, "scoreable": 1133, "hash_only": 169}
TARGET_FPR = 0.01
RISK_RANK = {"LOW": 0, "UNCERTAIN": 1, "MEDIUM": 2, "HIGH": 3}
SCORE_BANDS = [0.0, 0.05, 0.2, None, 0.5, 0.8, 1.0000001]  # None = the shipped threshold
ALERT_NAMES = [
    "a_student_score_ge_threshold",
    "b_any_rule_evidence",
    "c_risk_ge_MEDIUM",
    "d_risk_ge_UNCERTAIN",
    "e_risk_HIGH_only",
    "f_action_not_IGNORE",
]


# ----------------------------------------------------------------- decision (edge_decision)


def decide(text: str, prob: float, threshold: float, mode: str = DEFAULT_DETECTOR_ONLY_MODE):
    """Edge decision for one message given its student probability (no LLM)."""
    canon = canonicalize(text)
    evidence = run_rules(canon)
    dec = decide_edge(canon, evidence, prob, threshold, mode)
    # Rules-only cross-check: the label-free proxy state with no student score.
    rules_only_risk = build_package(canon, evidence, proxy_scam_state(evidence)).risk.value
    ranked = sorted(enumerate(evidence), key=lambda iv: (-iv[1].weight, iv[0]))
    primary = ranked[0][1].type if ranked else ""
    return {
        "state": dec.state,
        "risk": dec.package.risk.value,
        "action": dec.action.value,
        "detector_only": dec.detector_only,
        "n_evidence": len(evidence),
        "types": ",".join(sorted({e.type for e in evidence})),
        "primary_type": primary,
        "rules_only_risk": rules_only_risk,
    }


def apply_mode(scored: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Set risk, action and detector_only for a detector-only mode, from the cached scores.

    Only rows with scam state and no rule evidence depend on the mode: their package and action
    use no text or evidence (``build_package`` with no evidence and state "scam"). Every other
    row keeps its decision, so the cache (probabilities and evidence-derived fields) is reused for
    both modes. Checked against a fresh ``decide`` in ``tests/test_eval_groundlm.py``.
    """
    check_mode(mode)
    out = scored.copy()
    sel = (out["state"] == "scam") & (out["n_evidence"] == 0)
    if sel.any():
        d = decide_edge(canonicalize(""), [], 1.0, 0.5, mode)
        out.loc[sel, "risk"] = d.package.risk.value
        out.loc[sel, "action"] = d.action.value
        out.loc[sel, "detector_only"] = d.detector_only
    return out


# ----------------------------------------------------------------- counting (pure, unit-tested)


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    """Wilson 95% interval (rbench.metrics), with None for an empty denominator."""
    lo, hi = wilson_interval(int(k), int(n))
    return (None, None) if lo is None else (float(lo), float(hi))


def rate_row(name: str, flags: np.ndarray) -> dict[str, Any]:
    """Count, denominator, rate and Wilson 95% CI for a boolean alert vector."""
    flags = np.asarray(flags, dtype=bool)
    n = int(flags.size)
    k = int(flags.sum())
    lo, hi = wilson(k, n)
    return {
        "alert": name,
        "k": k,
        "n": n,
        "rate": (k / n) if n else None,
        "ci_lo": lo,
        "ci_hi": hi,
        "meets_1pct": bool(k / n <= TARGET_FPR) if n else None,
    }


def alert_flags(df: pd.DataFrame, threshold: float) -> dict[str, np.ndarray]:
    """The six alert definitions, evaluated per row of a scored frame."""
    rank = df["risk"].map(RISK_RANK).to_numpy()
    return {
        "a_student_score_ge_threshold": df["prob"].to_numpy(dtype=float) >= threshold,
        "b_any_rule_evidence": df["n_evidence"].to_numpy(dtype=int) > 0,
        "c_risk_ge_MEDIUM": rank >= RISK_RANK["MEDIUM"],
        "d_risk_ge_UNCERTAIN": rank >= RISK_RANK["UNCERTAIN"],
        "e_risk_HIGH_only": rank == RISK_RANK["HIGH"],
        "f_action_not_IGNORE": df["action"].to_numpy() != Action.IGNORE.value,
    }


def score_band_edges(threshold: float) -> list[float]:
    return [threshold if e is None else e for e in SCORE_BANDS]


def score_band_table(prob: np.ndarray, flags_a: np.ndarray, flags_d: np.ndarray, threshold: float):
    """Per score band over the negatives: n, flagged under (a) and under (d)."""
    edges = score_band_edges(threshold)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (prob >= lo) & (prob < hi)
        rows.append(
            {
                "band": f"[{lo:.4g}, {min(hi, 1.0):.4g})",
                "n": int(m.sum()),
                "fp_a": int((m & flags_a).sum()),
                "fp_d": int((m & flags_d).sum()),
            }
        )
    return rows


def type_breakdown(df: pd.DataFrame, flagged: np.ndarray) -> dict[str, Any]:
    """FP counts by rule type among flagged negatives: rows with the type, and primary type."""
    sub = df.loc[flagged]
    contains: Counter[str] = Counter()
    for types in sub["types"]:
        for t in filter(None, str(types).split(",")):
            contains[t] += 1
    primary = Counter(t for t in sub["primary_type"] if t)
    return {
        "rows_flagged": int(len(sub)),
        "rows_with_no_evidence": int(sub["n_evidence"].eq(0).sum()),
        "contains_type": dict(contains.most_common()),
        "primary_type": dict(primary.most_common()),
    }


def top_patterns(df: pd.DataFrame, flagged: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
    """Most common flagged-negative template groups (repo template_key), with their rule types."""
    sub = df.loc[flagged]
    out = []
    for key, grp in sub.groupby(sub["template_key"].fillna("(none)"), sort=False):
        out.append(
            {
                "template": key,
                "count": int(len(grp)),
                "types": sorted(
                    {t for ts in grp["types"] for t in filter(None, str(ts).split(","))}
                ),
                "prob_min": round(float(grp["prob"].min()), 4),
                "prob_max": round(float(grp["prob"].max()), 4),
                "category": str(grp["category"].iloc[0]),
            }
        )
    out.sort(key=lambda r: (-r["count"], r["template"]))
    return out[:k]


def min_threshold_for_fpr(neg_prob: np.ndarray, target: float = TARGET_FPR) -> dict[str, Any]:
    """Smallest threshold whose FP count over the negatives is <= floor(target * n).

    The threshold is the next float above the (k+1)-th largest negative score, so exactly the
    scores strictly above it (at most k of them) are flagged. Descriptive only.
    """
    n = int(neg_prob.size)
    k = math.floor(target * n + 1e-12)
    if n <= k:
        return {"threshold": 0.0, "k_allowed": k, "fp": 0, "n": n}
    s = np.sort(neg_prob)[::-1]
    t = float(np.nextafter(s[k], np.inf))
    fp = int((neg_prob >= t).sum())
    return {"threshold": t, "k_allowed": k, "fp": fp, "n": n}


def _ratio(k: int, n: int) -> float | None:
    return (k / n) if n else None


def threshold_sweep(
    neg_prob: np.ndarray,
    gl_fraud_prob: np.ndarray,
    sx_scam_prob: np.ndarray,
    sx_ham_prob: np.ndarray,
    camp_pos_prob: np.ndarray,
    camp_neg_prob: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """GroundLM FPR and the recall and FPR costs on the other sets, at each threshold."""
    rows = []
    for t in thresholds:
        fp_gl = int((neg_prob >= t).sum())
        rec_gl = int((gl_fraud_prob >= t).sum())
        rec_sx = int((sx_scam_prob >= t).sum())
        fp_sx = int((sx_ham_prob >= t).sum())
        rec_camp = int((camp_pos_prob >= t).sum())
        fp_camp = int((camp_neg_prob >= t).sum())
        rows.append(
            {
                "threshold": float(t),
                "groundlm_fp": fp_gl,
                "groundlm_fpr": _ratio(fp_gl, neg_prob.size),
                "groundlm_fraud_recall": _ratio(rec_gl, gl_fraud_prob.size),
                "smishx_scam_recall": _ratio(rec_sx, sx_scam_prob.size),
                "smishx_ham_fpr": _ratio(fp_sx, sx_ham_prob.size),
                "campaign_test_recall": _ratio(rec_camp, camp_pos_prob.size),
                "campaign_test_fpr": _ratio(fp_camp, camp_neg_prob.size),
            }
        )
    return rows


# ----------------------------------------------------------------- data loading


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def registry_sha(dataset_or_model_id: str, section: str) -> str | None:
    doc = tomllib.loads(REGISTRY.read_text(encoding="utf-8"))
    for entry in doc.get(section, []):
        if entry.get("id") == dataset_or_model_id:
            return entry.get("sha256")
    return None


def groundlm_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Policy negatives, scoreable (text and eval_ok) negatives, and the text-bearing fraud rows."""
    neg = df["policy_negative"].fillna(False).astype(bool).to_numpy()
    text = df["text_available"].fillna(False).astype(bool).to_numpy()
    ok = df["eval_ok"].fillna(False).astype(bool).to_numpy()
    return {
        "neg": neg,
        "scoreable": neg & text & ok,
        "fraud_text": (~neg) & text & ok,
    }


def count_check(df: pd.DataFrame, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    got = {
        "rows": int(len(df)),
        "policy_negatives": int(masks["neg"].sum()),
        "scoreable": int(masks["scoreable"].sum()),
        "hash_only": int((~df["text_available"].fillna(False).astype(bool)).sum()),
    }
    diffs = {
        k: {"expected": EXPECTED_COUNTS[k], "got": got[k]}
        for k in got
        if got[k] != EXPECTED_COUNTS[k]
    }
    return {"got": got, "expected": EXPECTED_COUNTS, "match": not diffs, "differences": diffs}


def load_campaign_test(sms_path: Path, split_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(sms_path, columns=["id", "text", "label_binary", "eval_ok", "synthetic"])
    ids = set(_read_split_ids(split_path, "test"))
    df = df[df["id"].astype(str).isin(ids)]
    return df[eval_mask(df)].reset_index(drop=True)


# ----------------------------------------------------------------- scoring


class Student:
    """The laptop copy of the SMS student, scored one message at a time (batch 1, as on the Pi)."""

    def __init__(self, path: Path, threads: int) -> None:
        self.runner = SmsRunner()
        self.runner.load(path, threads, {})

    def prob(self, text: str) -> float:
        feed = self.runner.make_feed({"text": text})
        p, _logit = self.runner.score(self.runner.run(feed))
        return float(p)


def score_frame(
    student: Student, texts: list[str], meta: pd.DataFrame, threshold: float, label: str
) -> pd.DataFrame:
    rows = []
    t0 = time.perf_counter()
    for i, (text, m) in enumerate(zip(texts, meta.to_dict("records"), strict=True)):
        p = student.prob(text)
        # Cached decisions use verify mode; apply_mode sets the requested mode afterwards.
        d = decide(text, p, threshold, mode="verify")
        rows.append({**m, "prob": p, **d})
        if (i + 1) % 2000 == 0:
            print(
                f"  {label}: {i + 1}/{len(texts)} ({time.perf_counter() - t0:.0f}s)",
                file=sys.stderr,
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.parquet"
    threshold = float(
        next(
            m["threshold"]
            for m in tomllib.loads(REGISTRY.read_text())["models"]
            if m["id"] == "sms_student"
        )
    )
    mode = check_mode(args.mode)
    model_sha = sha256_file(MODEL_PATH)
    if model_sha != registry_sha("sms_student", "models"):
        raise SystemExit(f"student ONNX sha256 {model_sha} does not match the registry")

    gl = pd.read_parquet(GROUNDLM_PATH)
    masks = groundlm_masks(gl)
    check = count_check(gl, masks)
    print(f"counts: {check['got']} match={check['match']}", file=sys.stderr)

    if scores_path.exists() and not args.rescore:
        scored = pd.read_parquet(scores_path)
        print(f"reusing {scores_path} ({len(scored)} rows)", file=sys.stderr)
    else:
        student = Student(MODEL_PATH, args.threads)
        frames = []

        gl_rows = gl[gl["text_available"].fillna(False).astype(bool)]
        if args.limit:
            gl_rows = gl_rows.head(args.limit)
        gl_cols = [
            "id",
            "label_binary",
            "label_fine",
            "policy_negative",
            "eval_ok",
            "template_key",
            "category",
        ]
        meta = gl_rows[gl_cols].assign(dataset="groundlm", label_scam=None, label_policy=None)
        frames.append(score_frame(student, gl_rows["text"].tolist(), meta, threshold, "groundlm"))

        sx = pd.read_parquet(SMISHX_PATH)
        sx = sx[sx["eval_ok"].fillna(False).astype(bool)]
        if args.limit:
            sx = sx.head(args.limit)
        sx_cols = [
            "id",
            "label_binary",
            "label_fine",
            "label_scam",
            "label_policy",
            "template_key",
            "eval_ok",
        ]
        meta = sx[sx_cols].assign(dataset="smishx", policy_negative=None, category=None)
        meta["label_scam"] = meta["label_scam"].astype("float")
        frames.append(score_frame(student, sx["text"].tolist(), meta, threshold, "smishx"))

        camp = load_campaign_test(SMS_PATH, CAMPAIGN_SPLIT)
        if args.limit:
            camp = camp.head(args.limit)
        meta = camp[["id", "label_binary", "eval_ok"]].assign(
            dataset="campaign_test",
            label_fine=None,
            label_scam=None,
            label_policy=None,
            policy_negative=None,
            template_key=None,
            category=None,
        )
        frames.append(score_frame(student, camp["text"].tolist(), meta, threshold, "campaign"))

        scored = pd.concat(frames, ignore_index=True)
        scored.to_parquet(scores_path, index=False)

    scored = apply_mode(scored, mode)
    metrics = analyse(scored, threshold)
    metrics["detector_only_mode"] = mode
    metrics["inputs"] = {
        "student_onnx_sha256": model_sha,
        "groundlm_sha256": sha256_file(GROUNDLM_PATH),
        "smishx_sha256": sha256_file(SMISHX_PATH),
        "campaign_split_sha256": sha256_file(CAMPAIGN_SPLIT),
        "limit": args.limit,
    }
    metrics["counts_check"] = check
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_summary(metrics)
    return 0


def analyse(scored: pd.DataFrame, threshold: float) -> dict[str, Any]:
    gl = scored[scored["dataset"] == "groundlm"].reset_index(drop=True)
    sx = scored[scored["dataset"] == "smishx"].reset_index(drop=True)
    cp = scored[scored["dataset"] == "campaign_test"].reset_index(drop=True)

    # Same denominator as groundlm_masks: policy negative, text available (scored), eval_ok.
    neg = (
        gl["policy_negative"].fillna(False).astype(bool).to_numpy()
        & gl["eval_ok"].fillna(False).astype(bool).to_numpy()
    )
    neg_df = gl[neg].reset_index(drop=True)
    flags = alert_flags(neg_df, threshold)
    fp_table = [rate_row(name, flags[name]) for name in ALERT_NAMES]

    sx_scam = sx[sx["label_fine"] == "smishing"].reset_index(drop=True)
    sx_ham = sx[sx["label_fine"] == "legitimate"].reset_index(drop=True)
    sx_promo = sx[sx["label_fine"] == "spam"].reset_index(drop=True)
    gl_fraud = gl[
        (~gl["policy_negative"].fillna(False).astype(bool).to_numpy())
        & gl["eval_ok"].fillna(False).astype(bool).to_numpy()
    ].reset_index(drop=True)
    cp_pos = cp[cp["label_binary"] == 1].reset_index(drop=True)
    cp_neg = cp[cp["label_binary"] == 0].reset_index(drop=True)

    sx_scam_flags = alert_flags(sx_scam, threshold)
    sx_ham_flags = alert_flags(sx_ham, threshold)
    sx_promo_flags = alert_flags(sx_promo, threshold)
    smishx = {
        "scam_recall": [rate_row(n, sx_scam_flags[n]) for n in ALERT_NAMES],
        "ham_fpr": [rate_row(n, sx_ham_flags[n]) for n in ALERT_NAMES],
        "promotion_flagged_info": [rate_row(n, sx_promo_flags[n]) for n in ALERT_NAMES],
    }

    d_flag = flags["d_risk_ge_UNCERTAIN"]
    camp_flags = alert_flags(cp.reset_index(drop=True), threshold)
    campaign = {
        "positives_recall": [
            rate_row(n, camp_flags[n][(cp["label_binary"] == 1).to_numpy()]) for n in ALERT_NAMES
        ],
        "negatives_fpr": [
            rate_row(n, camp_flags[n][(cp["label_binary"] == 0).to_numpy()]) for n in ALERT_NAMES
        ],
    }

    neg_prob = neg_df["prob"].to_numpy(dtype=float)
    detector_only_rows = {
        "groundlm_policy_negatives": int(neg_df["detector_only"].sum()),
        "smishx_smishing": int(sx_scam["detector_only"].sum()),
        "smishx_legitimate": int(sx_ham["detector_only"].sum()),
        "campaign_positives": int(cp_pos["detector_only"].sum()),
        "campaign_negatives": int(cp_neg["detector_only"].sum()),
    }
    thr_analysis = min_threshold_for_fpr(neg_prob)
    shipped = threshold
    grid = sorted({shipped, 0.5, 0.7, 0.8, 0.9, 0.95, thr_analysis["threshold"]})
    sweep = threshold_sweep(
        neg_prob,
        gl_fraud["prob"].to_numpy(dtype=float),
        sx_scam["prob"].to_numpy(dtype=float),
        sx_ham["prob"].to_numpy(dtype=float),
        cp_pos["prob"].to_numpy(dtype=float),
        cp_neg["prob"].to_numpy(dtype=float),
        grid,
    )

    rules_only_flag = neg_df["rules_only_risk"].ne("LOW").to_numpy()
    return {
        "threshold": threshold,
        "denominators": {
            "groundlm_policy_negatives_scored": int(len(neg_df)),
            "groundlm_fraud_text": int(len(gl_fraud)),
            "smishx_scam_eval": int(len(sx_scam)),
            "smishx_ham_eval": int(len(sx_ham)),
            "smishx_promotion_eval": int(len(sx_promo)),
            "campaign_positives": int(len(cp_pos)),
            "campaign_negatives": int(len(cp_neg)),
        },
        "detector_only_rows": detector_only_rows,
        "groundlm_fpr": fp_table,
        "groundlm_by_type_at_d": type_breakdown(neg_df, d_flag),
        "groundlm_by_type_at_b": type_breakdown(neg_df, flags["b_any_rule_evidence"]),
        "groundlm_by_score_band": score_band_table(
            neg_prob, flags["a_student_score_ge_threshold"], d_flag, threshold
        ),
        "groundlm_top_patterns_at_d": top_patterns(neg_df, d_flag),
        "rules_only_cross_check": {
            "any_evidence": int(flags["b_any_rule_evidence"].sum()),
            "risk_above_LOW_proxy_state": int(rules_only_flag.sum()),
            "note": "label-free proxy state, no student; docs/GROUNDLM_FPR.md reports 12 / 1",
        },
        "smishx": smishx,
        "campaign_test": campaign,
        "threshold_descriptive": {
            "target_fpr": TARGET_FPR,
            "min_threshold": thr_analysis,
            "sweep": sweep,
            "caveat": "GroundLM is in-sample (docs/GROUNDLM_FPR.md). A threshold chosen on it "
            "is not an independent test. Descriptive only, not shipped.",
        },
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2%}"


def print_summary(m: dict[str, Any]) -> None:
    print(f"threshold {m['threshold']:.6f}; denominators {m['denominators']}")
    print("GroundLM FPR on scoreable policy negatives (alert, k/n, rate, Wilson 95%):")
    for r in m["groundlm_fpr"]:
        print(
            f"  {r['alert']:32s} {r['k']:5d}/{r['n']}  {_pct(r['rate'])}  "
            f"[{_pct(r['ci_lo'])}, {_pct(r['ci_hi'])}]  <=1%: {r['meets_1pct']}"
        )
    print("SmishX recall (smishing, eval_ok) and ham FPR (legitimate, eval_ok):")
    for r, h in zip(m["smishx"]["scam_recall"], m["smishx"]["ham_fpr"], strict=True):
        print(
            f"  {r['alert']:32s} recall {r['k']}/{r['n']} {_pct(r['rate'])}   "
            f"ham FPR {h['k']}/{h['n']} {_pct(h['rate'])}"
        )
    print("Campaign test (eval_mask) recall / FPR:")
    for r, h in zip(
        m["campaign_test"]["positives_recall"], m["campaign_test"]["negatives_fpr"], strict=True
    ):
        print(f"  {r['alert']:32s} recall {_pct(r['rate'])}   FPR {_pct(h['rate'])}")
    print("Descriptive threshold sweep:")
    for row in m["threshold_descriptive"]["sweep"]:
        print(
            "  "
            + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        )
    print("Counts check:", m["counts_check"]["match"], m["counts_check"]["differences"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="score and write metrics")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--limit", type=int, default=0, help="first N rows per set (smoke only)")
    r.add_argument("--rescore", action="store_true", help="ignore cached scores.parquet")
    r.add_argument(
        "--mode",
        choices=DETECTOR_ONLY_MODES,
        default=DEFAULT_DETECTOR_ONLY_MODE,
        help="detector-only mode (default quiet)",
    )
    r.set_defaults(func=run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
