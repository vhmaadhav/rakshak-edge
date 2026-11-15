"""URL threshold tau_url for the frozen URL student, and its offline numbers at that threshold.

CPU job (no GPU). Spec: docs/URL_SERVICE_PLAN.md (frozen).
Launch (build, then push):
  kaggle/launch.sh url_threshold --internet --datasets maadhavvh/rakshak-data \
      --kernels maadhavvh/rakshak-url-student

Order, enforced in code by ThresholdGate:
1. Pins. best_url_student.onnx and url_side_scaler.json from maadhavvh/rakshak-url-student, and
   the data files, by sha256. Any mismatch raises before a score is computed.
2. Cal carve, rebuilt as kaggle/jobs/url_student.py builds it: the 5 % grouped carve of PhreshPhish
   official.train (rakshak.eval.phreshphish.grouped_cal, seed 0, groups = registered domain). The
   sizes must be 18087 cal and 480168 fit, and the fit rows must reproduce the shipped scaler mean
   and std.
3. Cal rows scored. tau_url = the smallest observed cal score whose benign FPR is at most 1 %.
   thresholds.json is written and logged here. Nothing else is scored before this point.
4. Everything else at tau_url: PhreshPhish official.test and the benchmark protocol, url_domain
   test, GroundLM and SmishX (message score = max over canonicalize(text).urls, absent when no
   URL), a 2,000-URL feature skew sample, and the 300-row parity sample.

Every URL is scored on the runtime feature path (rakshak.url_features.url_features, then the
shipped scaler), including the PhreshPhish rows, so the numbers are the service path.

Outputs (under /kaggle/working): thresholds.json, metrics.json, url_scores.parquet,
parity_300.parquet, report.md, and results.json from the runner.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import baseline_v0 as base
import numpy as np
import pandas as pd
from _common import Units, env_report, out_dir, seed_everything, timed, write_metrics

import rakshak
from rakshak.canonicalize import canonicalize
from rakshak.eval import data as D
from rakshak.eval import phreshphish as P
from rakshak.models.encoding import fit_standardizer
from rakshak.url_features import url_features

# ---- pins and fixed protocol values ----

KERNEL = "maadhavvh/rakshak-url-student"
ONNX_NAME = "best_url_student.onnx"
ONNX_SHA = "1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7"
SCALER_NAME = "url_side_scaler.json"
SCALER_SHA = "c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e"
DATA_PINS = {  # relative to the data root (rakshak-data == data/processed)
    "urls_phreshphish.parquet": "af9441871e9b542aad03e0c94dd702485a0ab41a786d5f0b0547836c5e322844",
    "splits/url_phreshphish.json": "58d8c8c81866491db4f54b1db9a46c70b059b5f261e78be4518d3c5e05640bc5",  # noqa: E501
    "splits/url_domain.json": "170a810a0071d65edf2718607b0e88e94ef81ab9aa06a541cf1feecd164f5a47",
    "urls.parquet": "ff389178dd752a0240268e550e7c892164392d6c180372d537f6fc299e16efa2",
    "external/external_groundlm.parquet": "979aaca4c6a922912d223515ec6a3373276a7f5308aaa8d72e17b9f092850f73",  # noqa: E501
    "external/external_smishx.parquet": "872d08e8885e101b33ac83732ec776e36d8cf183a669733baaf15165d3ec7780",  # noqa: E501
}
CAL_FRACTION = 0.05
SPLIT_SEED = 0
CAL_N = 18087
FIT_N = 480168
TARGET_FPR = 0.01
U1_FPR_MAX = 0.015
SKEW_N = 2000
PARITY_SEED = 1
PARITY_N = 300
TEMPLATE = "Your parcel is held, visit {url} to update"
GROUNDLM_U2_MAX = 4


# ---- pure helpers (unit-tested in tests/test_url_threshold.py) ----


def check_data_pins(root: Path) -> dict[str, str]:
    """sha256 of every pinned data file under root. Raises on a missing or changed file."""
    actual = {name: base.sha256_file(root / name) for name in DATA_PINS if (root / name).exists()}
    problems = base.check_pins(actual, DATA_PINS)
    if problems:
        raise RuntimeError("data pin mismatch: " + "; ".join(problems))
    return actual


def check_carve_sizes(n_cal: int, n_fit: int) -> None:
    """The carve must match the trained sizes, else fail with the numbers."""
    if n_cal != CAL_N or n_fit != FIT_N:
        raise RuntimeError(
            f"cal carve mismatch: cal {n_cal} (expected {CAL_N}), fit {n_fit} (expected {FIT_N})"
        )


def threshold_rule(y, s, target: float = TARGET_FPR) -> dict:
    """tau = the smallest observed score in ``s`` whose benign FPR is at most ``target``.

    The candidates are the observed cal scores, as sklearn's ROC thresholds are. A flag is
    score >= tau. Raises when a class is missing or no candidate meets the target (for example a
    tie block larger than the allowed false positives).
    """
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("labels and scores differ in shape")
    neg = np.sort(s[y == 0])
    pos = np.sort(s[y == 1])
    n_neg, n_pos = int(neg.size), int(pos.size)
    if n_neg == 0 or n_pos == 0:
        raise ValueError(f"threshold rule needs both classes (benign {n_neg}, phish {n_pos})")
    max_fp = math.floor(target * n_neg + 1e-9)
    cand = np.unique(s)
    # benign rows at or above each candidate
    cnt_neg = n_neg - np.searchsorted(neg, cand, side="left")
    ok = np.flatnonzero(cnt_neg <= max_fp)
    if ok.size == 0:
        raise ValueError(f"no cal score meets FPR <= {target} (max false positives {max_fp})")
    tau = float(cand[ok[0]])
    fp = int(n_neg - np.searchsorted(neg, tau, side="left"))
    tp = int(n_pos - np.searchsorted(pos, tau, side="left"))
    return {
        "tau": tau,
        "rule": "smallest observed cal score with benign FPR <= target (flag = score >= tau)",
        "target_fpr": float(target),
        "cal_n_benign": n_neg,
        "cal_n_phish": n_pos,
        "cal_max_false_positives": int(max_fp),
        "cal_fp": fp,
        "cal_tp": tp,
        "cal_fpr": fp / n_neg,
        "cal_recall": tp / n_pos,
    }


class ThresholdGate:
    """Enforces the order. Pre sets (the cal rows) are scored only before the threshold is
    written; post sets only after. Each set name is scored once."""

    def __init__(self, log: Callable[[str], None], write: Callable[[dict], object]):
        self._log = log
        self._write = write
        self.tau: float | None = None
        self.events: list[str] = []
        self._seen: set[str] = set()

    def _score(self, name: str, fn: Callable[[], object]):
        if name in self._seen:
            raise ValueError(f"set {name!r} is scored twice")
        self._seen.add(name)
        self.events.append(f"score:{name}")
        return fn()

    def score_pre(self, name: str, fn: Callable[[], object]):
        if self.tau is not None:
            raise RuntimeError(
                f"set {name!r} is a calibration set; the threshold is already written"
            )
        return self._score(name, fn)

    def write(self, record: dict) -> None:
        if self.tau is not None:
            raise RuntimeError("the threshold is already written")
        tau = float(record["tau"])
        if not math.isfinite(tau):
            raise ValueError(f"tau is not finite: {tau}")
        self._write(record)
        self.tau = tau
        self.events.append("threshold")
        self._log(f"threshold written before any other set: tau_url={tau!r} ({json.dumps(record)})")

    def adopt(self, record: dict, source: str) -> None:
        """Take a threshold an earlier run already wrote (score-only). Nothing is written and no
        calibration set is scored again; the post sets then run at this tau."""
        if self.tau is not None:
            raise RuntimeError("the threshold is already set")
        tau = float(record["tau"])
        if not math.isfinite(tau):
            raise ValueError(f"tau is not finite: {tau}")
        self.tau = tau
        self.events.append("threshold:read")
        self._log(f"threshold read from {source}, not recomputed: tau_url={tau!r}")

    def score_post(self, name: str, fn: Callable[[], object]):
        if self.tau is None:
            raise RuntimeError(f"set {name!r} scored before the threshold was written")
        return self._score(name, fn)


def aggregate_message_scores(
    url_lists: Sequence[Sequence[str]], url_scores: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Per message: the max URL score over its URLs. A message with no URL is absent (NaN,
    has_url False), never a zero score."""
    scores = np.full(len(url_lists), np.nan, dtype=np.float64)
    has = np.zeros(len(url_lists), dtype=bool)
    for i, urls in enumerate(url_lists):
        if not urls:
            continue
        has[i] = True
        scores[i] = max(float(url_scores[u]) for u in urls)
    return scores, has


def message_url_scores(
    texts: Sequence[str], score_urls: Callable[[list[str]], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """FrozenURLExpert semantics: URLs from canonicalize(text).urls, each scored once, max per
    message."""
    url_lists = [list(canonicalize(str(t)).urls) for t in texts]
    uniq = sorted({u for urls in url_lists for u in urls})
    vals = score_urls(uniq) if uniq else np.empty(0)
    return aggregate_message_scores(
        url_lists, dict(zip(uniq, [float(v) for v in vals], strict=True))
    )


def flags_at(scores: np.ndarray, has: np.ndarray, tau: float) -> np.ndarray:
    """Flag = has a URL and score >= tau. No-URL rows are never flagged."""
    return has & (np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf) >= tau)


def score_url_strings(session, urls: Sequence[str], cols: list[str], mean, std) -> np.ndarray:
    """Probability for each URL string on the runtime path: url_features, then the shipped
    scaler (via baseline_v0.url_side_matrix), then the ONNX (baseline_v0.score_url_rows)."""
    urls = [str(u) for u in urls]
    out = np.empty(len(urls), dtype=np.float64)
    chunk = 4096
    for start in range(0, len(urls), chunk):
        part = urls[start : start + chunk]
        feats = [url_features(u) for u in part]
        absent = [c for c in cols if any(c not in f for f in feats)]
        if absent:
            raise ValueError(f"url_features lacks scaler columns: {absent}")
        side = base.url_side_matrix(pd.DataFrame(feats), cols, mean, std)
        out[start : start + len(part)] = base.score_url_rows(session, part, side)
    if not np.isfinite(out).all():
        raise ValueError("non-finite URL score from the ONNX")
    return out


def at_threshold_counts(y, s, tau: float) -> dict:
    """Confusion counts, FPR and recall with Wilson 95 % intervals, precision, at score >= tau."""
    y = np.asarray(y).astype(np.int64)
    pred = np.asarray(s, dtype=np.float64) >= tau
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    fn = n_pos - tp
    tn = n_neg - fp
    fpr_lo, fpr_hi = base.wilson_interval(fp, n_neg)
    rec_lo, rec_hi = base.wilson_interval(tp, n_pos)
    return {
        "tau": float(tau),
        "n": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "fpr": fp / n_neg if n_neg else math.nan,
        "fpr_wilson95": [fpr_lo, fpr_hi],
        "recall": tp / n_pos if n_pos else math.nan,
        "recall_wilson95": [rec_lo, rec_hi],
        "precision": tp / (tp + fp) if (tp + fp) else math.nan,
    }


def skew_summary(urls: Sequence[str], ds_scores, score_urls, tau: float) -> dict:
    """Dataset URL string against what canonicalize extracts from TEMPLATE around it."""
    urls = [str(u) for u in urls]
    messages = [TEMPLATE.format(url=u) for u in urls]
    url_lists = [list(canonicalize(m).urls) for m in messages]
    uniq = sorted({u for us in url_lists for u in us})
    vals = score_urls(uniq) if uniq else np.empty(0)
    ext, has = aggregate_message_scores(
        url_lists, dict(zip(uniq, [float(v) for v in vals], strict=True))
    )
    ds = np.asarray(ds_scores, dtype=np.float64)
    equal = np.asarray([us == [u] for us, u in zip(url_lists, urls, strict=True)], dtype=bool)
    diff = np.abs(ds[has] - ext[has])
    flips = flags_at(ds, np.ones(len(ds), dtype=bool), tau) != flags_at(ext, has, tau)
    n = len(urls)
    return {
        "n": n,
        "n_extracted_equals_dataset_string": int(equal.sum()),
        "frac_extracted_equals_dataset_string": float(equal.mean()) if n else math.nan,
        "n_no_url_extracted": int((~has).sum()),
        "n_multiple_urls_extracted": int(sum(len(us) > 1 for us in url_lists)),
        "n_compared": int(has.sum()),
        "max_abs_score_diff": float(diff.max()) if diff.size else math.nan,
        "n_flips_at_tau": int(flips.sum()),
    }


def parity_frame(ids, urls, labels, scores) -> pd.DataFrame:
    """The Pi parity rows: score as float32, and its little-endian bytes as hex (exact compare)."""
    s32 = np.asarray(scores, dtype=np.float32)
    return pd.DataFrame(
        {
            "id": [str(i) for i in ids],
            "url": [str(u) for u in urls],
            "label": np.asarray(labels, dtype=np.int64),
            "score": s32,
            "score_f32_hex": [struct.pack("<f", float(v)).hex() for v in s32],
        }
    )


def grouped_carve(data: P.PhreshData) -> tuple[np.ndarray, np.ndarray]:
    """(keep, cal) as kaggle/jobs/url_student.py builds them: registered domain, else the URL."""
    df = data.df
    url = df["url"].astype(str).to_numpy(dtype=object)
    reg = (
        df["reg_domain"].fillna("").astype(str).to_numpy(dtype=object)
        if "reg_domain" in df.columns
        else np.full(len(df), "", dtype=object)
    )
    norm = df["url_norm"].astype(str).to_numpy(dtype=object) if "url_norm" in df.columns else url
    group_key = np.where(reg != "", reg, norm)
    return P.grouped_cal(data.train, data.labels, group_key[data.train], CAL_FRACTION, SPLIT_SEED)


def _external_block(u: Units, gate: ThresholdGate, name: str, frame: pd.DataFrame, score_urls):
    texts = [t if isinstance(t, str) else "" for t in frame["text"]]
    avail = frame["text_available"].fillna(False).astype(bool).to_numpy()
    idx = np.flatnonzero(avail)

    def _score():
        return message_url_scores([texts[i] for i in idx], score_urls)

    pair = gate.score_post(f"external_{name}", lambda: u.run({"split": name}, _score))
    if pair is None:
        return None
    ms, mh = pair
    scores = np.full(len(frame), np.nan)
    has = np.zeros(len(frame), dtype=bool)
    scores[idx] = ms
    has[idx] = mh
    return scores, has


def _groundlm_table(frame: pd.DataFrame, scores, has, tau: float) -> dict:
    pol = frame["policy_negative"].fillna(False).astype(bool).to_numpy()
    avail = frame["text_available"].fillna(False).astype(bool).to_numpy()
    ok = frame["eval_ok"].fillna(False).astype(bool).to_numpy()
    scoreable = pol & avail & ok
    cats = frame["category"].fillna("").astype(str).to_numpy()
    flag = flags_at(scores, has, tau)
    per = {}
    for c in sorted(set(cats[scoreable])):
        m = scoreable & (cats == c)
        per[c] = {
            "n": int(m.sum()),
            "with_url": int((m & has).sum()),
            "flagged": int((m & flag).sum()),
        }
    total = {
        "n": int(scoreable.sum()),
        "with_url": int((scoreable & has).sum()),
        "flagged": int((scoreable & flag).sum()),
    }
    total["u2_flagged_le_4"] = bool(total["flagged"] <= GROUNDLM_U2_MAX)
    return {"scoreable": total, "per_category": per}


def _smishx_table(frame: pd.DataFrame, scores, has, tau: float) -> dict:
    ok = frame["eval_ok"].fillna(False).astype(bool).to_numpy()
    fine = frame["label_fine"].astype(str).to_numpy()
    flag = flags_at(scores, has, tau)
    per = {}
    for lab in sorted(set(fine[ok])):
        m = ok & (fine == lab)
        per[lab] = {
            "n": int(m.sum()),
            "with_url": int((m & has).sum()),
            "flagged": int((m & flag).sum()),
        }
    return {
        "eval_ok": {
            "n": int(ok.sum()),
            "with_url": int((ok & has).sum()),
            "flagged": int((ok & flag).sum()),
        },
        "per_label_fine": per,
    }


def _report(metrics: dict, tau: float) -> str:
    t = metrics["phreshphish"]["test"]
    d = metrics["url_domain"].get("test", {})
    g = metrics["external"].get("groundlm", {})
    x = metrics["external"].get("smishx", {})
    cal = metrics["calibration"]
    lines = [
        "# URL threshold (tau_url) report",
        "",
        f"tau_url = {tau!r} (smallest cal score with benign FPR <= 1 %).",
        f"Cal: {cal['cal_n_benign']} benign, {cal['cal_n_phish']} phish; "
        f"cal FPR {cal['cal_fpr']:.4f}, cal recall {cal['cal_recall']:.4f}.",
        "",
        "## PhreshPhish official.test",
        f"- n {t['n']} (pos {t['n_pos']}, neg {t['n_neg']}); FPR {t['fpr']:.4f} "
        f"(Wilson {t['fpr_wilson95'][0]:.4f} to {t['fpr_wilson95'][1]:.4f}); "
        f"recall {t['recall']:.4f}; precision {t['precision']:.4f}.",
        f"- U1 (FPR <= 1.5 %): {'PASS' if t['fpr'] <= U1_FPR_MAX else 'FAIL'}.",
    ]
    if d:
        lines += [
            "",
            "## url_domain test (out of domain, reported)",
            f"- n {d['n']} (pos {d['n_pos']}, neg {d['n_neg']}); FPR {d['fpr']:.4f}; "
            f"recall {d['recall']:.4f}.",
        ]
    if g:
        s = g["scoreable"]
        lines += [
            "",
            "## GroundLM scoreable negatives",
            f"- scoreable {s['n']}, with URL {s['with_url']}, flagged {s['flagged']} "
            f"(U2 <= {GROUNDLM_U2_MAX}: {'yes' if s['u2_flagged_le_4'] else 'no'}).",
        ]
        for c, v in g["per_category"].items():
            lines.append(f"- {c}: n {v['n']}, with URL {v['with_url']}, flagged {v['flagged']}")
    if x:
        lines += ["", "## SmishX (eval_ok)"]
        for lab, v in x["per_label_fine"].items():
            lines.append(f"- {lab}: n {v['n']}, with URL {v['with_url']}, flagged {v['flagged']}")
    sk = metrics["skew"]
    if sk:
        lines += [
            "",
            "## Feature skew (template SMS around dataset URL)",
            f"- n {sk['n']}; extracted equals dataset string "
            f"{sk['n_extracted_equals_dataset_string']} "
            f"({sk['frac_extracted_equals_dataset_string']:.4f}); "
            f"no URL extracted {sk['n_no_url_extracted']}; "
            f"multiple URLs {sk['n_multiple_urls_extracted']}.",
            f"- max abs score diff {sk['max_abs_score_diff']:.6g}; "
            f"flips at tau {sk['n_flips_at_tau']}.",
        ]
    lines += ["", "## PhreshPhish base-rate protocol (AP mean over subsets)"]
    for rate, v in metrics["phreshphish"]["protocol"]["per_rate"].items():
        lines.append(
            f"- {rate}: AP {v['ap_mean']:.4f} (sd {v['ap_std']:.4f}), "
            f"P@R=0.90 {v['p_at_r90_mean']:.4f} over {v['n_subsets']} subsets"
        )
    return "\n".join(lines) + "\n"


# ---- the job ----


def run(ctx) -> bool:
    seed_everything(SPLIT_SEED)
    env = env_report()
    ctx.log(f"env: {env}")
    timings: dict = {}

    # 1. pins
    with timed("pins_s", timings):
        search = [Path("/kaggle/input")]
        onnx_path = base.locate_artifact(search, ONNX_NAME, ONNX_SHA)
        scaler_path = base.locate_artifact(search, SCALER_NAME, SCALER_SHA)
        root = D.resolve_data_root(None, marker=P.PHRESH_FILE)
        data_actual = check_data_pins(root)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    cols = list(scaler["columns"])
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    ctx.log(f"onnx: {onnx_path}  scaler: {scaler_path}  data root: {root}")

    data = P.load_phreshphish(root)
    if list(data.feature_cols) != cols:
        raise RuntimeError(
            "PhreshPhish feature columns differ from the scaler: "
            f"{[c for c in cols if c not in data.feature_cols]}"
        )
    df = data.df
    urls_all = df["url"].astype(str).to_numpy()

    # 2. cal carve, sizes, and the shipped scaler reproduced from the carve's fit rows
    with timed("carve_s", timings):
        keep, cal = grouped_carve(data)
        fit = P.subsample_rows(keep, None, SPLIT_SEED)
        check_carve_sizes(len(cal), len(fit))
        m_fit, s_fit = fit_standardizer(df[data.feature_cols].fillna(0.0).to_numpy(np.float32), fit)
        dm = float(np.max(np.abs(m_fit - mean)))
        ds = float(np.max(np.abs(s_fit - std)))
        if not (
            np.allclose(m_fit, mean, rtol=1e-5, atol=1e-6)
            and np.allclose(s_fit, std, rtol=1e-5, atol=1e-6)
        ):
            raise RuntimeError(
                f"shipped scaler does not match the carve's fit rows: max |mean diff| {dm:.3g}, "
                f"max |std diff| {ds:.3g}"
            )
    ctx.log(
        f"carve: cal {len(cal)} fit {len(fit)}; scaler reproduced (max diffs {dm:.2g}, {ds:.2g})"
    )

    session = base._onnx_session(onnx_path)

    def score_urls(urls):
        return score_url_strings(session, urls, cols, mean, std)

    # 3. cal rows, then tau_url, written before any other set is scored
    gate = ThresholdGate(ctx.log, lambda rec: write_metrics("thresholds.json", rec))
    with timed("cal_score_s", timings):
        s_cal = gate.score_pre("phreshphish_cal", lambda: score_urls(urls_all[cal]))
    rec = threshold_rule(data.labels[cal], s_cal)
    rec.update(
        {
            "seed": SPLIT_SEED,
            "cal_fraction": CAL_FRACTION,
            "cal_n": int(len(cal)),
            "fit_n": int(len(fit)),
            "kernel": KERNEL,
            "onnx_sha256": ONNX_SHA,
            "scaler_sha256": SCALER_SHA,
            "data_pins": data_actual,
            "scorer": "rakshak.url_features.url_features + shipped scaler + ONNX (runtime path)",
        }
    )
    gate.write(rec)
    tau = gate.tau

    # 4. everything else, at tau
    u = Units(ctx)
    metrics: dict = {
        "job": "url_threshold",
        "rakshak_version": getattr(rakshak, "__version__", None),
        "env": env,
        "tau": tau,
        "calibration": rec,
        "phreshphish": {},
        "url_domain": {},
        "external": {},
        "skew": {},
        "parity": {},
    }

    test = data.test
    bench = data.bench_full
    union = np.unique(np.concatenate([test, bench] + [pos for _rate, pos in data.subsets]))

    def _union():
        full = np.full(len(df), np.nan)
        full[union] = score_urls(urls_all[union])
        return full

    full = gate.score_post(
        "phreshphish_test_and_benchmark", lambda: u.run({"split": "phreshphish"}, _union)
    )
    if full is None:
        raise RuntimeError("PhreshPhish scoring failed; see failures")
    t = at_threshold_counts(data.labels[test], full[test], tau)
    t["gate_U1"] = {"fpr_max": U1_FPR_MAX, "pass": bool(t["fpr"] <= U1_FPR_MAX)}
    metrics["phreshphish"]["test"] = t
    metrics["phreshphish"]["protocol"] = P.protocol_section(
        data.labels, full, test, bench, data.subsets
    )

    ud = D.load_urls(root)
    ud_labels = ud["label"].to_numpy().astype(np.int64)
    ud_runs = D.load_runs(root, ud, patterns=["url_domain"], labels=ud_labels, mask=None)
    run_ud = next((r for r in ud_runs if r.name == "url_domain"), None)
    if run_ud is None:
        raise RuntimeError("url_domain split has no 'url_domain' partition")
    ud_test = np.asarray(run_ud.test, dtype=np.int64)
    ud_urls = ud["url"].to_numpy()

    def _ud():
        return score_urls(ud_urls[ud_test])

    s_ud = gate.score_post("url_domain_test", lambda: u.run({"split": "url_domain"}, _ud))
    if s_ud is not None:
        metrics["url_domain"]["test"] = at_threshold_counts(ud_labels[ud_test], s_ud, tau)

    gl = pd.read_parquet(root / "external" / "external_groundlm.parquet")
    sx = pd.read_parquet(root / "external" / "external_smishx.parquet")
    gl_block = _external_block(u, gate, "groundlm", gl, score_urls)
    sx_block = _external_block(u, gate, "smishx", sx, score_urls)
    if gl_block is not None:
        metrics["external"]["groundlm"] = _groundlm_table(gl, *gl_block, tau)
    if sx_block is not None:
        metrics["external"]["smishx"] = _smishx_table(sx, *sx_block, tau)
    if gl_block is not None and sx_block is not None:
        frames = [
            pd.DataFrame(
                {
                    "id": gl["id"].astype(str).to_numpy(),
                    "set": "groundlm",
                    "score": gl_block[0],
                    "has_url": gl_block[1],
                }
            ),
            pd.DataFrame(
                {
                    "id": sx["id"].astype(str).to_numpy(),
                    "set": "smishx",
                    "score": sx_block[0],
                    "has_url": sx_block[1],
                }
            ),
        ]
        scores_path = out_dir() / "url_scores.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(scores_path, index=False)
        ctx.log(f"wrote {scores_path}")

    rng = np.random.default_rng(SPLIT_SEED)
    skew_rows = rng.choice(test, size=min(SKEW_N, len(test)), replace=False)

    def _skew():
        return skew_summary(urls_all[skew_rows], full[skew_rows], score_urls, tau)

    sk = u.run({"split": "feature_skew"}, _skew)
    if sk is not None:
        metrics["skew"] = sk

    rng_p = np.random.default_rng(PARITY_SEED)
    parity_rows = rng_p.choice(test, size=min(PARITY_N, len(test)), replace=False)
    parity = parity_frame(
        df["id"].to_numpy()[parity_rows],
        urls_all[parity_rows],
        data.labels[parity_rows],
        full[parity_rows],
    )
    parity_path = out_dir() / "parity_300.parquet"
    parity.to_parquet(parity_path, index=False)
    metrics["parity"] = {
        "n": int(len(parity)),
        "n_flagged_at_tau": int((parity["score"] >= tau).sum()),
        "path": parity_path.name,
        "seed": PARITY_SEED,
    }

    out_path = write_metrics("metrics.json", metrics)
    (out_dir() / "report.md").write_text(_report(metrics, tau), encoding="utf-8")
    ctx.log(f"wrote {out_path}")
    ctx.results["summary"] = (
        f"tau_url {tau:.6g} phresh test fpr {t['fpr']:.4f} (U1 "
        f"{'pass' if t['gate_U1']['pass'] else 'fail'}) recall {t['recall']:.4f} "
        f"units ok {u.ok} failed {u.failed}"
    )
    ctx.results["url_threshold_timings_s"] = timings
    ctx.results["threshold_events"] = gate.events
    status = u.status()
    return status if status != "fail" else False
