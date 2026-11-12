"""SMS surrogate diagnostic D0 (docs/SURROGATE_RETRAIN_PLAN.md, section 4). No training.

Scores the shipped v0 SMS student (ONNX sha 5341e14c) on three sets, each scrubbed as stored and
filled with rakshak.data.surrogates (keyed by the row id):
  (a) phone 1, the test donor: all labelled rows (inbox label_inbox rules)
  (b) the held-out legit set of phones 2+3 (sender-disjoint 20 %, built by sms_inbox_retrain)
  (c) the IMC25 rows (source imc25_fishing_for_smishing) of the campaign TEST split
Per set and version (scrubbed, filled): n, legit and scam counts, legit FPR and scam recall at v0's
shipped tau and at each R0 tau (Wilson 95 %), score quantiles, and the share of rows holding a tag
before filling.

H1 (plan section 2): the filled phone-1 legit FPR is at least 10 times lower than the scrubbed one.
Verdict rule, fixed here before any number is seen:
- At one tau, H1 holds iff the scrubbed FPR is above 0 and the filled FPR is at most one tenth of
  it (exact fractions). A filled FPR of 0 with a scrubbed FPR above 0 holds. A scrubbed FPR of 0
  does not hold (nothing to reduce).
- The verdict uses the median of the three R0 taus (the middle value by size). H1 holds iff it
  holds at that tau. Every tau is still reported.
- Stop rule 1 (plan section 4): H1 fails, so S0 to S2 do not run.

PRIVACY (binding). No inbox text, sender string, or inbox row id leaves memory. metrics.json and
report.md carry only counts, rates, Wilson intervals, score quantiles, tau values, the v0 sha256 and
the surrogate seed. Row ids are used in memory only, to seed the surrogate realisation.

Pipeline:
1. Campaign data, lang_shift and temporal splits, GroundLM and SmishX text, as sms_inbox_retrain
   loads them (only used to build the same test-only text list, so the hold-out matches).
2. Inbox files train_donor.parquet and test_donor.parquet, found under /kaggle/input.
3. The train donor's pools by sms_inbox_retrain.build_inbox_pools (same seed, same caps). The
   held-out legit rows are recovered with their row id by (text, sender) matching in memory.
4. The v0 ONNX is located under --v0-onnx by sha256 (default /kaggle/input, the attached
   maadhavvh/rakshak-sms-student output). A file with another sha stops the job.
5. Scoring with onnxruntime on CPU, via baseline_v0.score_sms_texts. Filled texts are checked
   with surrogates.assert_no_tags before scoring.
6. Outputs: metrics.json, report.md, ctx.results["summary"].

Launch (CPU, internet on, the two datasets, the sms-student kernel; the R0 taus are the three
seed thresholds of kaggle/out/sms_inbox_retrain-r0/output/thresholds.json, seeds 1, 2, 3):
  kaggle/launch.sh sms_surrogate_d0 --internet \
      --datasets maadhavvh/rakshak-data maadhavvh/rakshak-inbox \
      --kernels maadhavvh/rakshak-sms-student \
      --job-args="--v0-onnx /kaggle/input --r0-taus 0.5127761960029602 \
      0.9999979734420776 0.7533388137817383"

Flags (``--job-args="FLAGS"``):
  --v0-onnx PATH        folder searched for best_student.onnx with the pinned sha
                        (default /kaggle/input)
  --r0-taus T1 T2 T3    required: the three R0 seed thresholds, each in (0, 1]
"""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sms_inbox_retrain as inbox
import sms_retrain as base
from _common import env_report, seed_everything, timed
from baseline_v0 import ARTIFACT_PINS, SMS_THRESHOLD, _onnx_session, score_sms_texts

from rakshak.data import retrain_prep as RP
from rakshak.data import surrogates as S
from rakshak.ela_eval import wilson_interval
from rakshak.eval import data as D

JOB = "sms_surrogate_d0"
V0_NAME = "best_student.onnx"
V0_SHA = ARTIFACT_PINS[V0_NAME][0]
SHIPPED_TAU = SMS_THRESHOLD
IMC25_SOURCE = "imc25_fishing_for_smishing"
SET_PHONE1 = "phone1_test_donor"
SET_LEGIT = "phones23_heldout_legit"
SET_IMC25 = "imc25_campaign_test"
SET_NAMES = (SET_PHONE1, SET_LEGIT, SET_IMC25)
VERSIONS = ("scrubbed", "filled")
R0_COUNT = 3
H1_FACTOR = 10
DEFAULT_ROOT = "/kaggle/input"
QUANTILES = (
    ("min", 0),
    ("p5", 5),
    ("p25", 25),
    ("p50", 50),
    ("p75", 75),
    ("p95", 95),
    ("max", 100),
)
H1_RULE = (
    "At one tau, H1 holds iff the scrubbed phone-1 legit FPR is above 0 and the filled FPR is at "
    "most one tenth of it (exact fractions). A filled FPR of 0 with a scrubbed FPR above 0 holds. "
    "A scrubbed FPR of 0 does not hold. The verdict is the median of the three R0 taus (the middle "
    "value by size); H1 holds iff it holds at that tau. Every tau is reported."
)
# Read by kaggle/build_kernel.py: errors are logged as type and frames only, never the message.
REDACT_ERRORS = True


# ---- pure helpers (unit-tested in tests/test_sms_surrogate_d0.py) ----


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=JOB)
    p.add_argument("--v0-onnx", default=DEFAULT_ROOT)
    p.add_argument("--r0-taus", type=float, nargs=R0_COUNT, required=True)
    args = p.parse_args(list(argv or []))
    if not all(0.0 < t <= 1.0 for t in args.r0_taus):
        p.error("--r0-taus values must each be in (0, 1]")
    return args


def tau_map(r0_taus: Sequence[float]) -> dict[str, float]:
    """Named taus in report order: shipped, then r0_1 to r0_3 in the order given."""
    if len(r0_taus) != R0_COUNT:
        raise ValueError(f"expected {R0_COUNT} R0 taus, got {len(r0_taus)}")
    out = {"shipped": float(SHIPPED_TAU)}
    for i, t in enumerate(r0_taus, start=1):
        out[f"r0_{i}"] = float(t)
    return out


def tag_share(texts: Sequence[str]) -> dict[str, Any]:
    """Rows containing a <TAG> placeholder before filling. Counts only."""
    n = len(texts)
    k = sum(1 for t in texts if S.TAG_RE.search(t))
    return {"rows": n, "with_tag": k, "share": (k / n) if n else None}


def rate(k: int, n: int) -> dict[str, Any]:
    """k of n with a Wilson 95 % interval (None when n is 0)."""
    lo, hi = wilson_interval(int(k), int(n))
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "wilson95": [lo, hi]}


def quantiles(values: np.ndarray) -> dict[str, Any]:
    """min, 5, 25, 50, 75, 95 and max of a score array (None when empty)."""
    v = np.asarray(values, dtype=np.float64)
    out: dict[str, Any] = {"n": int(v.size)}
    for name, q in QUANTILES:
        out[name] = float(np.percentile(v, q)) if v.size else None
    return out


def rates_at(labels: np.ndarray, scores: np.ndarray, tau: float) -> dict[str, Any]:
    """Legit FPR (score >= tau is a flag) and scam recall at tau."""
    y = np.asarray(labels).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    legit, scam = y == 0, y == 1
    flagged = s >= tau
    return {
        "legit_fpr": rate(int((flagged & legit).sum()), int(legit.sum())),
        "scam_recall": rate(int((flagged & scam).sum()), int(scam.sum())),
    }


def version_table(labels: np.ndarray, scores: np.ndarray, taus: dict[str, float]) -> dict:
    y = np.asarray(labels).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    return {
        "legit_scores": quantiles(s[y == 0]),
        "scam_scores": quantiles(s[y == 1]),
        "at": {name: {"tau": t, **rates_at(y, s, t)} for name, t in taus.items()},
    }


def holds_at(scrubbed: dict, filled: dict) -> tuple[bool, str]:
    """H1 at one tau from two legit_fpr rate dicts (k, n). Returns (holds, reason)."""
    if not scrubbed["n"] or not filled["n"]:
        return False, "no legit rows"
    s = Fraction(scrubbed["k"], scrubbed["n"])
    f = Fraction(filled["k"], filled["n"])
    if s == 0:
        return False, "scrubbed FPR is 0: nothing to reduce"
    if f == 0:
        return True, "filled FPR is 0 (ratio unbounded)"
    return f * H1_FACTOR <= s, "ratio test"


def h1_rule(scrubbed_at: dict[str, dict], filled_at: dict[str, dict]) -> dict[str, Any]:
    """H1 per R0 tau and its median verdict. ``*_at`` are version_table()["at"] of phone 1."""
    names = [n for n in scrubbed_at if n.startswith("r0_")]
    rows = []
    for name in sorted(names):
        s_fpr = scrubbed_at[name]["legit_fpr"]
        f_fpr = filled_at[name]["legit_fpr"]
        ok, reason = holds_at(s_fpr, f_fpr)
        rows.append(
            {
                "tau_name": name,
                "tau": scrubbed_at[name]["tau"],
                "scrubbed_fpr": s_fpr["rate"],
                "filled_fpr": f_fpr["rate"],
                "scrubbed_k_n": [s_fpr["k"], s_fpr["n"]],
                "filled_k_n": [f_fpr["k"], f_fpr["n"]],
                "holds": ok,
                "reason": reason,
            }
        )
    if len(rows) != R0_COUNT:
        raise ValueError(f"H1 needs {R0_COUNT} R0 taus, found {len(rows)}")
    by_tau = sorted(rows, key=lambda r: (r["tau"], r["tau_name"]))
    median_row = by_tau[len(by_tau) // 2]
    holds = bool(median_row["holds"])
    stop = (
        "H1 holds: stop rule 1 not triggered"
        if holds
        else "stop rule 1 triggered: H1 fails, S0 to S2 do not run"
    )
    return {
        "rule": H1_RULE,
        "per_threshold": rows,
        "verdict_tau_name": median_row["tau_name"],
        "verdict_tau": median_row["tau"],
        "holds": holds,
        "stop_rule_1": stop,
    }


def build_sets(phone1: pd.DataFrame, legit: pd.DataFrame, imc25: pd.DataFrame) -> dict:
    """The three sets, each with id, text and int label columns. Legit must be all label 0."""
    if (legit["label"].to_numpy() != 0).any():
        raise ValueError("the held-out legit set holds a non-legit label")
    keep = ["id", "text", "label"]
    out = {
        SET_PHONE1: phone1[keep].reset_index(drop=True),
        SET_LEGIT: legit[keep].reset_index(drop=True),
        SET_IMC25: imc25[keep].reset_index(drop=True),
    }
    for name, frame in out.items():
        frame["label"] = frame["label"].astype(np.int64)
        frame["id"] = frame["id"].astype(str)
        frame["text"] = frame["text"].astype(str)
        if frame.empty:
            raise ValueError(f"set {name} is empty")
    return out


def attach_row_ids(sub: pd.DataFrame, source: pd.DataFrame) -> list[str]:
    """Row ids for the rows of ``sub`` (build_inbox_pools drops ids). Matched on (text, sender) in
    source order, so identical rows take their ids in turn. Raises if a row cannot be matched."""
    pool: dict[tuple[str, str], collections.deque] = collections.defaultdict(collections.deque)
    for rid, text, sender in zip(source["id"], source["text"], source["sender"], strict=True):
        pool[(str(text), str(sender))].append(str(rid))
    ids = []
    for text, sender in zip(sub["text"], sub["sender"], strict=True):
        queue = pool.get((str(text), str(sender)))
        if not queue:
            raise ValueError("a held-out legit row could not be matched back to its row id")
        ids.append(queue.popleft())
    return ids


def summarise(scored: dict[str, dict], r0_taus: Sequence[float]) -> dict[str, Any]:
    """The per-set, per-version tables and the H1 verdict. ``scored[name]`` holds ``labels``,
    ``texts`` (scrubbed, used only for the tag share) and ``scores`` = {"scrubbed", "filled"}.
    Output carries counts and statistics only."""
    taus = tau_map(r0_taus)
    sets_out: dict[str, Any] = {}
    for name in SET_NAMES:
        entry = scored[name]
        y = np.asarray(entry["labels"]).astype(np.int64)
        sets_out[name] = {
            "n": int(len(y)),
            "n_legit": int((y == 0).sum()),
            "n_scam": int((y == 1).sum()),
            "tag_share_before_fill": tag_share(entry["texts"]),
        }
        for version in VERSIONS:
            sets_out[name][version] = version_table(y, entry["scores"][version], taus)
    h1 = h1_rule(sets_out[SET_PHONE1]["scrubbed"]["at"], sets_out[SET_PHONE1]["filled"]["at"])
    return {"sets": sets_out, "taus": taus, "h1": h1}


def render_report(payload: dict) -> str:
    """Markdown report (report.md). Counts, rates and scores only."""
    lines = ["# sms_surrogate_d0 (plan section 4)", ""]
    if payload.get("status") != "scored":
        lines += [f"Status: **{payload.get('status')}**", "", payload.get("reason", ""), ""]
        return "\n".join(lines)
    taus = payload["taus"]
    names = list(taus)
    lines += [
        f"v0 sha256 pinned and checked: {payload['v0']['sha256']}. Surrogate seed "
        f"{payload['surrogate_seed']}. Shipped tau {taus['shipped']:.6f}; R0 taus "
        + ", ".join(f"{taus[n]:.6f}" for n in names if n.startswith("r0_"))
        + ".",
        "",
        "## Sets",
        "",
        "| Set | n | legit | scam | rows with a tag before fill |",
        "|---|---|---|---|---|",
    ]
    for name in SET_NAMES:
        s = payload["sets"][name]
        share = s["tag_share_before_fill"]
        lines.append(
            f"| {name} | {s['n']} | {s['n_legit']} | {s['n_scam']} | "
            f"{share['with_tag']} / {share['rows']} |"
        )
    header = (
        "| Set | Version | "
        + " | ".join(f"legit FPR {n}" for n in names)
        + " | "
        + " | ".join(f"scam recall {n}" for n in names)
        + " |"
    )
    lines += ["", "## Rates (Wilson 95 %)", "", header, "|" + "---|" * (2 + 2 * len(names))]
    for name in SET_NAMES:
        for version in VERSIONS:
            at = payload["sets"][name][version]["at"]
            fpr = [_fmt_rate(at[n]["legit_fpr"]) for n in names]
            rec = [_fmt_rate(at[n]["scam_recall"]) for n in names]
            lines.append(f"| {name} | {version} | " + " | ".join(fpr + rec) + " |")
    lines += [
        "",
        "## Score quantiles",
        "",
        "| Set | Version | Class | n | min | p5 | p25 | p50 | p75 | p95 | max |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in SET_NAMES:
        for version in VERSIONS:
            v = payload["sets"][name][version]
            for cls, key in (("legit", "legit_scores"), ("scam", "scam_scores")):
                q = v[key]
                if not q["n"]:
                    continue
                vals = [_fmt(q[k]) for k in ("min", "p5", "p25", "p50", "p75", "p95", "max")]
                lines.append(
                    f"| {name} | {version} | {cls} | {q['n']} | " + " | ".join(vals) + " |"
                )
    h1 = payload["h1"]
    lines += [
        "",
        "## H1 (filled phone-1 legit FPR at least 10x lower than scrubbed)",
        "",
        h1["rule"],
        "",
        "| Tau | Value | Scrubbed FPR | Filled FPR | Holds | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for r in h1["per_threshold"]:
        lines.append(
            f"| {r['tau_name']} | {r['tau']:.6f} | {_fmt(r['scrubbed_fpr'])} | "
            f"{_fmt(r['filled_fpr'])} | {r['holds']} | {r['reason']} |"
        )
    lines += [
        "",
        f"Verdict tau: {h1['verdict_tau_name']} ({h1['verdict_tau']:.6f}). "
        f"**H1 holds: {h1['holds']}.** {h1['stop_rule_1']}.",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_rate(r: dict) -> str:
    if r["rate"] is None:
        return "n/a"
    lo, hi = r["wilson95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.4f} [{lo:.4f}, {hi:.4f}]"


def locate_v0(root: Path) -> Path:
    """The best_student.onnx under ``root`` whose sha256 is the pin. Raises with the digests found
    (never a message from the data) when none matches."""
    if not root.exists():
        raise FileNotFoundError(f"--v0-onnx folder {root} does not exist")
    # The search lives in rakshak.data.retrain_prep so that sms_inbox_retrain can use it too
    # without importing this job (that import would be a cycle).
    return RP.locate_pinned(root, V0_NAME, V0_SHA)


# ---- the job ----


def redacted_error(exc: BaseException) -> str:
    return inbox.redacted_error(exc)


def redacted_frames(exc: BaseException) -> list[str]:
    return inbox.redacted_frames(exc)


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

    # 1. v0 ONNX first: a wrong file stops the job before any data is read.
    try:
        v0_path = locate_v0(Path(args.v0_onnx))
    except (FileNotFoundError, ValueError) as exc:
        ctx.log(f"v0 ONNX check failed: {exc}")
        raise
    ctx.log(f"v0 ONNX found with the pinned sha256 {V0_SHA[:12]}")

    # 2. campaign data, as sms_inbox_retrain loads it
    root = D.resolve_data_root(None)
    sms = D.load_sms(root)
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
    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    gl_avail = gl["text_available"].fillna(False).astype(bool).to_numpy()
    gl_texts = base.str_texts(gl["text"].to_numpy()[np.flatnonzero(gl_avail)])
    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    sx_avail = sx["text_available"].fillna(False).astype(bool).to_numpy()
    sx_texts = base.str_texts(sx["text"].to_numpy()[np.flatnonzero(sx_avail)])

    # 3. inbox donors and the train-donor pools (same call as sms_inbox_retrain)
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
    test_only += gl_texts + sx_texts + [t for t in test_df["text"] if t]
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

    # 4. the three sets, with row ids for seeding
    legit = pools["val_legit"]
    legit_ids = attach_row_ids(legit, train_df)
    legit = legit.assign(id=legit_ids)
    imc = sms.iloc[run_.test]
    imc = imc[imc["source"].astype(str) == IMC25_SOURCE]
    imc_frame = pd.DataFrame(
        {
            "id": imc["id"].astype(str).to_numpy(),
            "text": base.str_texts(imc["text"]),
            "label": imc["label_binary"].to_numpy().astype(np.int64),
        }
    )
    phone1 = pd.DataFrame(
        {
            "id": test_df["id"].astype(str).to_numpy(),
            "text": test_df["text"].to_numpy(),
            "label": test_df["label"].to_numpy().astype(np.int64),
        }
    )
    sets = build_sets(phone1, legit, imc_frame)

    # 5. score each set scrubbed and filled, with onnxruntime on CPU
    session = _onnx_session(v0_path)
    scored: dict[str, dict] = {}
    for name in SET_NAMES:
        frame = sets[name]
        texts = list(frame["text"])
        filled = S.fill_many(list(frame["id"]), texts)
        S.assert_no_tags(filled)
        with timed(f"score_{name}_s", timings):
            s_scrub = score_sms_texts(session, texts)
            s_fill = score_sms_texts(session, filled)
        scored[name] = {
            "labels": frame["label"].to_numpy(),
            "texts": texts,
            "scores": {"scrubbed": s_scrub, "filled": s_fill},
        }
        ctx.log(f"{name}: scored {len(texts)} rows, scrubbed and filled")

    result = summarise(scored, args.r0_taus)
    payload = {
        "job": JOB,
        "status": "scored",
        "plan": "docs/SURROGATE_RETRAIN_PLAN.md section 4",
        "surrogate_seed": S.SURROGATE_SEED,
        "v0": {"sha256": V0_SHA, "shipped_tau": SHIPPED_TAU},
        "args": {"r0_taus": list(args.r0_taus)},
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
    h1 = result["h1"]
    ctx.results["summary"] = (
        f"{JOB}: H1 holds {h1['holds']} at {h1['verdict_tau_name']} "
        f"({h1['verdict_tau']:.6f}); {h1['stop_rule_1']}"
    )
    return True


def _write(work: Path, payload: dict) -> None:
    """metrics.json and report.md in ``work``. Nothing row-level is written."""
    work.mkdir(parents=True, exist_ok=True)
    tmp = work / "metrics.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(work / "metrics.json")
    (work / "report.md").write_text(render_report(payload))
