"""Frozen baseline v0: score the shipped ONNX artifacts on the protocol splits, write the table.

Launch: kaggle/launch.sh baseline_v0 --internet --datasets maadhavvh/rakshak-data \
            --kernels maadhavvh/rakshak-sms-student maadhavvh/rakshak-url-student
(CPU; no GPU needed.)

What it does, in order:
1. Pins. Verifies the source files it runs (rules, canonicalize, evidence, verifier at ac45aea), the
   data tables it reads, and the three shipped artifacts (SMS student ONNX, URL student ONNX, URL
   side scaler). Any sha256 mismatch raises before a single score is computed.
2. SMS v0 (dual-view char-CNN ONNX, shipped threshold 0.35127517580986023): scores every row any
   split run uses (campaign, lang_shift, temporal, source holdouts, legacy folds), reports the
   free metrics, the shipped-threshold operating point with campaign-clustered bootstrap CIs, and
   recall at a calibration FPR of 1 percent.
3. SMS external sets at the shipped threshold: GroundLM policy negatives (1,133 scoreable of 1,294)
   and SmishX (smishing recall, legitimate FPR). Wilson intervals. Rule-evidence rate from
   rakshak.rules over the canonical text. The edge decision path (visible warning, quiet mode) is
   NOT reproduced here; it lives in deploy/pi/service and is not in this bundle.
4. SMS robustness at the shipped threshold, rakshak.eval.robustness over the campaign test split,
   through the ONNX model with canonicalisation on (the shipped pipeline).
5. URL v0 (char-CNN ONNX + 29 side features): PhreshPhish benchmark protocol (AP and P@R=0.90 per
   base rate, mean and SD over subsets; full 24% benchmark; full test split) and the url_domain test
   split (AP, ROC-AUC, recall at a 1 percent calibration FPR, clustered by registered domain).

Writes metrics.json (every table, with denominators) and results.json (via the runner).
Units: a failing split is recorded in results.json failures and the job goes on (status partial).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from _common import Units, env_report, record_skip, seed_everything, timed, write_metrics

import rakshak
from rakshak.canonicalize import canonicalize
from rakshak.eval import data as D
from rakshak.eval import metrics as M
from rakshak.eval import phreshphish as P
from rakshak.eval import robustness as R
from rakshak.models.encoding import apply_standardizer, encode_batch
from rakshak.models.features import featurize
from rakshak.rules import run_rules
from rakshak.url_features import url_features

# ---- pins: source from `git show ac45aea:<path> | sha256sum`; artifacts and data by sha256sum ----

SMS_THRESHOLD = 0.35127517580986023
SMS_CLUSTER_COLUMN = "campaign_id"
# SMS split files: the set sms_student uses. Loading every split file would also pick up URL splits.
SMS_SPLITS = ("campaign", "temporal", "source_holdout", "lang_shift", "legacy_*")
URL_CLUSTER_COLUMN = "reg_domain"
URL_MAX_LEN = 200
SMS_MAX_LEN = 256
FEATURE_BATCH = 4096
SCORE_BATCH = 512

SOURCE_PINS = {  # relative to the rakshak package, at git commit ac45aea
    "rules.py": "9bde4ae25a44e5721daa4a1bd1c97304376d75765e3bed5d952dc5e845078e5a",
    "canonicalize.py": "892a89ad7619181e9d41f609bf46df52dc76622441b3c789ffb2c95228e783cb",
    "evidence.py": "df8300c6f9ee1d0392640df3cfcc674d60fdf2e4bca95bf1334f9b90d85302f2",
    "verifier.py": "f10890b50d981a5b660d48620400bb6b8075d0b224a1559d1a3f2843e1c4558a",
}
ARTIFACT_PINS = {  # filename -> (sha256, bytes, producing kernel)
    "best_student.onnx": (
        "5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466",
        388127,
        "maadhavvh/rakshak-sms-student",
    ),
    "best_url_student.onnx": (
        "1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7",
        2741232,
        "maadhavvh/rakshak-url-student",
    ),
    "url_side_scaler.json": (
        "c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e",
        2083,
        "maadhavvh/rakshak-url-student",
    ),
}
DATA_PINS = {  # relative to the data root (kaggle/out/rakshak-data == data/processed)
    "sms.parquet": "430ded382b22ea1c3c14469c80ff9b34a3214ab82257e89be20d67439053e135",
    "urls.parquet": "ff389178dd752a0240268e550e7c892164392d6c180372d537f6fc299e16efa2",
    "urls_phreshphish.parquet": "af9441871e9b542aad03e0c94dd702485a0ab41a786d5f0b0547836c5e322844",
    "external/external_groundlm.parquet": "979aaca4c6a922912d223515ec6a3373276a7f5308aaa8d72e17b9f092850f73",  # noqa: E501
    "external/external_smishx.parquet": "872d08e8885e101b33ac83732ec776e36d8cf183a669733baaf15165d3ec7780",  # noqa: E501
}


# ---- pure helpers (unit-tested in tests/test_baseline_v0.py) ----


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_pins(actual: dict[str, str], expected: dict[str, str]) -> list[str]:
    """One message per name whose digest differs from the pin or is missing. [] means all pass."""
    problems = []
    for name, want in expected.items():
        got = actual.get(name)
        if got is None:
            problems.append(f"{name}: not found")
        elif got != want:
            problems.append(f"{name}: sha256 {got} != pinned {want}")
    return problems


def locate_artifact(search_roots: list[Path], filename: str, expected_sha: str) -> Path:
    """The copy of ``filename`` under the roots whose sha256 is the pin. Raises if none matches."""
    seen = []
    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, _dirs, files in os.walk(root):
            if filename in files:
                path = Path(dirpath) / filename
                seen.append(path)
                if sha256_file(path) == expected_sha:
                    return path
    detail = ", ".join(str(p) for p in seen) or "none found"
    raise FileNotFoundError(f"{filename}: no copy with sha256 {expected_sha} (seen: {detail})")


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for k successes out of n. (nan, nan) when n is 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -60.0, 60.0)))


def url_side_matrix(df: pd.DataFrame, columns: list[str], mean, std) -> np.ndarray:
    """The 29 URL side features in scaler order, standardised as at training (NaN as 0, clip 10)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"URL feature columns missing from the table: {missing}")
    raw = df[columns].fillna(0.0).to_numpy(np.float32)
    return apply_standardizer(
        raw, np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    )


def rate_row(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else float("nan"), "wilson95": [lo, hi]}


def sms_run_table(
    y_cal, s_cal, y_test, s_test, clusters, threshold: float = SMS_THRESHOLD, n_boot: int = 1000
) -> dict:
    """Threshold-free metrics, the shipped-threshold operating point with a campaign-clustered
    bootstrap, and recall at a 1 percent FPR threshold chosen on the calibration split."""
    y_test = np.asarray(y_test).astype(np.int64)
    s_test = np.asarray(s_test, dtype=np.float64)
    out = {
        "n_test": int(len(y_test)),
        "n_test_pos": int((y_test == 1).sum()),
        "n_test_neg": int((y_test == 0).sum()),
        "threshold_free": {
            "pr_auc": M.pr_auc(y_test, s_test),
            "roc_auc": M.roc_auc(y_test, s_test),
        },
        "shipped_threshold": {
            "threshold": float(threshold),
            **M.binary_metrics(y_test, s_test, threshold),
        },
        "bootstrap_shipped": M.bootstrap_intervals(
            y_test, s_test, threshold, n_boot=n_boot, cluster_ids=list(clusters)
        ),
    }
    if (np.asarray(y_cal) == 1).any() and (np.asarray(y_cal) == 0).any():
        fpr1 = M.threshold_for_fpr(np.asarray(y_cal), np.asarray(s_cal, dtype=np.float64), 0.01)
        if fpr1["threshold"] == fpr1["threshold"]:
            out["cal_fpr_1pct"] = {
                "threshold": fpr1["threshold"],
                "cal_fpr": fpr1["fpr"],
                "cal_tpr": fpr1["tpr"],
                **{
                    f"test_{k}": v
                    for k, v in M.binary_metrics(y_test, s_test, fpr1["threshold"]).items()
                    if k in ("recall", "fpr", "precision", "f1")
                },
            }
    return out


def groundlm_table(
    df: pd.DataFrame, scores: np.ndarray, rules_fired: np.ndarray, threshold: float = SMS_THRESHOLD
) -> dict:
    """GroundLM policy negatives. Denominator: policy_negative and text available and eval_ok."""
    pol = df["policy_negative"].fillna(False).astype(bool).to_numpy()
    avail = df["text_available"].fillna(False).astype(bool).to_numpy()
    ok = df["eval_ok"].fillna(False).astype(bool).to_numpy()
    scoreable = pol & avail & ok
    s = scores[scoreable]
    r = rules_fired[scoreable]
    n = int(scoreable.sum())
    return {
        "policy_negatives_total": int(pol.sum()),
        "policy_negatives_hash_only": int((pol & ~avail).sum()),
        "scoreable": n,
        "student_at_threshold": rate_row(int((s >= threshold).sum()), n),
        "rule_evidence_any": rate_row(int(r.sum()), n),
        "rule_evidence_and_student": rate_row(int(((s >= threshold) & r).sum()), n),
    }


def smishx_table(
    df: pd.DataFrame, scores: np.ndarray, rules_fired: np.ndarray, threshold: float = SMS_THRESHOLD
) -> dict:
    """SmishX: smishing recall (eval_ok, label_fine smishing) and legitimate FPR (eval_ok)."""
    ok = df["eval_ok"].fillna(False).astype(bool).to_numpy()
    fine = df["label_fine"].astype(str).to_numpy()
    scam = ok & (fine == "smishing")
    ham = ok & (fine == "legitimate")
    out = {}
    for name, mask in (("smishing_recall", scam), ("legitimate_fpr", ham)):
        n = int(mask.sum())
        s, r = scores[mask], rules_fired[mask]
        out[name] = {
            "student_at_threshold": rate_row(int((s >= threshold).sum()), n),
            "rule_evidence_any": rate_row(int(r.sum()), n),
        }
    return out


def sms_feeds(texts: list[str]) -> dict[str, np.ndarray]:
    """ONNX inputs raw, canon and side for a batch of messages, as the student was trained."""
    feats = featurize(texts, max_len=SMS_MAX_LEN)
    return {
        "raw": feats.raw_bytes.astype(np.int64),
        "canon": feats.canon_bytes.astype(np.int64),
        "side": feats.side_canon.astype(np.float32),
    }


def run_onnx(session, feeds: dict[str, np.ndarray], n: int, batch: int = SCORE_BATCH) -> np.ndarray:
    """Probability (sigmoid of the logit output) for n rows, batched."""
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch):
        stop = min(n, start + batch)
        chunk = {k: v[start:stop] for k, v in feeds.items()}
        logit = session.run(["logit"], chunk)[0].reshape(-1)
        out[start:stop] = sigmoid(logit)
    return out


def score_sms_texts(session, texts: list[str]) -> np.ndarray:
    scores = np.empty(len(texts), dtype=np.float64)
    for start in range(0, len(texts), FEATURE_BATCH):
        chunk = texts[start : start + FEATURE_BATCH]
        feeds = sms_feeds(chunk)
        scores[start : start + len(chunk)] = run_onnx(session, feeds, len(chunk))
    return scores


def rules_any(texts: list[str]) -> np.ndarray:
    """True where rakshak.rules fires on the canonical text (any evidence type)."""
    return np.asarray([len(run_rules(canonicalize(t))) > 0 for t in texts], dtype=bool)


def score_url_rows(session, urls: list[str], side: np.ndarray) -> np.ndarray:
    raw = encode_batch(urls, URL_MAX_LEN).astype(np.int64)
    feeds = {"raw": raw, "side": side.astype(np.float32)}
    return run_onnx(session, feeds, len(urls))


def protocol_scores(labels: np.ndarray, scores_full: np.ndarray, data) -> dict:
    """The PhreshPhish protocol section for one score vector (NaN outside the scored rows)."""
    return P.protocol_section(labels, scores_full, data.test, data.bench_full, data.subsets)


def url_domain_table(y_cal, s_cal, y_test, s_test, clusters, n_boot: int = 1000) -> dict:
    """url_domain test: AP, ROC-AUC, P@R=0.90, and recall at a 1 percent FPR threshold chosen on the
    calibration split. No threshold is registered for the PhreshPhish-trained student, so the
    operating point is this calibration threshold, and the bootstrap is campaign-clustered by
    registered domain."""
    y_cal = np.asarray(y_cal).astype(np.int64)
    y_test = np.asarray(y_test).astype(np.int64)
    s_test = np.asarray(s_test, dtype=np.float64)
    fpr1 = M.threshold_for_fpr(y_cal, np.asarray(s_cal, dtype=np.float64), 0.01)
    thr = fpr1["threshold"]
    out = {
        "n_test": int(len(y_test)),
        "n_test_pos": int((y_test == 1).sum()),
        "n_test_neg": int((y_test == 0).sum()),
        "pr_auc": M.pr_auc(y_test, s_test),
        "roc_auc": M.roc_auc(y_test, s_test),
        "p_at_r90": P.precision_at_recall(y_test, s_test),
        "cal_fpr_1pct": {"threshold": thr, "cal_fpr": fpr1["fpr"], "cal_tpr": fpr1["tpr"]},
    }
    if thr == thr:
        out["at_cal_fpr_1pct"] = M.binary_metrics(y_test, s_test, thr)
        out["bootstrap_at_cal_fpr_1pct"] = M.bootstrap_intervals(
            y_test, s_test, thr, n_boot=n_boot, cluster_ids=list(clusters)
        )
    return out


# ---- the job ----


def _data_root_pins(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in DATA_PINS if (root / name).exists()}


def _onnx_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run(ctx) -> bool:
    seed_everything(0)
    env = env_report()
    ctx.log(f"env: {env}")
    timings: dict = {}
    pkg = Path(rakshak.__file__).resolve().parent

    # 1. pins: source, artifacts, data. A mismatch stops the job before any score.
    with timed("pins_s", timings):
        src_actual = {name: sha256_file(pkg / name) for name in SOURCE_PINS}
        problems = check_pins(src_actual, SOURCE_PINS)
        if problems:
            raise RuntimeError("source pin mismatch: " + "; ".join(problems))
        search = [Path("/kaggle/input")]
        artifact_paths = {}
        for name, (sha, _size, _kernel) in ARTIFACT_PINS.items():
            artifact_paths[name] = locate_artifact(search, name, sha)
        data_root = D.resolve_data_root(None)
        data_actual = _data_root_pins(data_root)
        problems = check_pins(data_actual, DATA_PINS)
        if problems:
            raise RuntimeError("data pin mismatch: " + "; ".join(problems))
    ctx.log(f"artifacts: { {k: str(v) for k, v in artifact_paths.items()} }")
    ctx.log(f"data root: {data_root}")

    metrics: dict = {
        "job": "baseline_v0",
        "rakshak_version": getattr(rakshak, "__version__", None),
        "env": env,
        "pins": {
            "source": src_actual,
            "data": data_actual,
            "artifacts": {
                k: {
                    "path": str(v),
                    "sha256": ARTIFACT_PINS[k][0],
                    "bytes": v.stat().st_size,
                    "kernel": ARTIFACT_PINS[k][2],
                }
                for k, v in artifact_paths.items()
            },
            "sms_threshold": SMS_THRESHOLD,
        },
        "sms": {},
        "groundlm_smishx": {},
        "robustness": {},
        "url": {},
        "url_domain": {},
    }
    u = Units(ctx)
    ctx.results["baseline_pins_ok"] = True

    # 2-4. SMS v0.
    sms = D.load_sms(data_root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    sms_texts = [t if isinstance(t, str) else "" for t in sms["text"]]
    mask = D.eval_mask(sms)
    runs = D.load_runs(data_root, sms, patterns=list(SMS_SPLITS), labels=labels, mask=mask)
    sms_session = _onnx_session(artifact_paths["best_student.onnx"])
    needed = np.unique(
        np.concatenate([np.concatenate([r.cal, r.test]) for r in runs]).astype(np.int64)
    )
    with timed("sms_score_s", timings):
        s_needed = u.run(
            {"stage": "sms_score"}, score_sms_texts, sms_session, [sms_texts[i] for i in needed]
        )
    if s_needed is None:
        raise RuntimeError("SMS scoring failed; see failures")
    sms_scores = np.full(len(sms), np.nan)
    sms_scores[needed] = s_needed
    metrics["sms"]["runs_scored_rows"] = int(len(needed))
    for run_ in runs:
        cluster = D.cluster_values(sms, run_.test, SMS_CLUSTER_COLUMN)

        def _table(run_=run_, cluster=cluster):
            return sms_run_table(
                labels[run_.cal],
                sms_scores[run_.cal],
                labels[run_.test],
                sms_scores[run_.test],
                cluster,
            )

        table = u.run({"split": run_.name}, _table)
        if table is not None:
            metrics["sms"][run_.name] = table

    # consistency: the ONNX campaign operating point against the stored .pt-based campaign numbers.
    campaign = next((r for r in runs if r.name == "campaign"), None)
    if campaign is not None:
        t = metrics["sms"].get("campaign", {}).get("shipped_threshold", {})
        metrics["sms"]["onnx_vs_stored_campaign_f1"] = {
            "onnx_f1_at_shipped": t.get("f1"),
            "stored_pt_f1_at_shipped": 0.9748635835047857,
            "note": "stored value: kaggle/out/sms_student metrics.json, operating_points.max_f1",
        }

    # 3. GroundLM and SmishX at the shipped threshold.
    gl = pd.read_parquet(data_root / "external" / "external_groundlm.parquet")
    sx = pd.read_parquet(data_root / "external" / "external_smishx.parquet")
    for name, frame in (("groundlm", gl), ("smishx", sx)):
        texts = [t if isinstance(t, str) else "" for t in frame["text"]]
        avail = frame["text_available"].fillna(False).astype(bool).to_numpy()
        idx = np.flatnonzero(avail)

        n_rows = len(frame)

        def _ext(texts=texts, idx=idx, n=n_rows):
            sub = [texts[i] for i in idx]
            s = score_sms_texts(sms_session, sub)
            r = rules_any(sub)
            full_s = np.full(n, np.nan)
            full_r = np.zeros(n, dtype=bool)
            full_s[idx] = s
            full_r[idx] = r
            return full_s, full_r

        pair = u.run({"split": name}, _ext)
        if pair is None:
            continue
        ext_scores, ext_rules = pair
        ext_scores = np.where(np.isnan(ext_scores), -1.0, ext_scores)  # unscored rows never count
        if name == "groundlm":
            metrics["groundlm_smishx"]["groundlm"] = groundlm_table(frame, ext_scores, ext_rules)
        else:
            metrics["groundlm_smishx"]["smishx"] = smishx_table(frame, ext_scores, ext_rules)

    # 4. Robustness of the shipped ONNX student, shipped threshold, campaign test split.
    if campaign is not None:
        te = campaign.test
        pos = [sms_texts[i] for i in te if labels[i] == 1]
        neg = [sms_texts[i] for i in te if labels[i] == 0]

        def _predict(texts):
            return score_sms_texts(sms_session, list(texts))

        report = u.run(
            {"split": "robustness_campaign"}, R.robustness_report, _predict, pos, neg, SMS_THRESHOLD
        )
        if report is not None:
            metrics["robustness"]["sms_campaign_canon_on"] = report
            metrics["robustness"]["sms_campaign_canon_off"] = {
                "status": "MISSING",
                "reason": "the shipped ONNX reads canonical bytes; no raw-only ONNX is pinned",
            }

    # 5. URL v0 on PhreshPhish (protocol) and url_domain (out-of-training check).
    url_session = _onnx_session(artifact_paths["best_url_student.onnx"])
    scaler = json.loads(artifact_paths["url_side_scaler.json"].read_text(encoding="utf-8"))
    cols = list(scaler["columns"])
    data = P.load_phreshphish(data_root)
    if list(data.feature_cols) != cols:
        raise RuntimeError(
            f"PhreshPhish feature columns differ from the scaler: "
            f"{[c for c in cols if c not in data.feature_cols]}"
        )
    df = data.df
    needed_url = np.zeros(len(df), dtype=bool)
    needed_url[data.test] = True
    needed_url[data.bench_full] = True
    for _rate, pos_rows in data.subsets:
        needed_url[pos_rows] = True
    rows = np.flatnonzero(needed_url)
    side_all = url_side_matrix(df.iloc[rows], cols, scaler["mean"], scaler["std"])
    urls_ph = [str(x) for x in df["url"].to_numpy()[rows]]

    def _url_ph():
        return score_url_rows(url_session, urls_ph, side_all)

    s_rows = u.run({"split": "phreshphish"}, _url_ph)
    if s_rows is not None:
        full = np.full(len(df), np.nan)
        full[rows] = s_rows
        # consistency: parquet side features against the runtime url_features() on a sample
        probe = data.test[:200]
        diffs = []
        for i in probe:
            feats = url_features(str(df["url"].iloc[i]))
            want = np.asarray([float(feats.get(c, 0.0)) for c in cols])
            have = df[cols].iloc[i].fillna(0.0).to_numpy(np.float64)
            diffs.append(float(np.max(np.abs(want - have))))
        metrics["url"]["side_feature_parity_max_abs_diff_200"] = float(np.max(diffs))
        metrics["url"]["phreshphish_protocol"] = protocol_scores(data.labels, full, data)
        metrics["url"]["phreshphish_protocol"]["n_test"] = int(len(data.test))
        metrics["url"]["phreshphish_protocol"]["n_scored"] = int(len(rows))

    ud = D.load_urls(data_root)
    ud_labels = ud["label"].to_numpy().astype(np.int64)
    try:
        ud_runs = D.load_runs(data_root, ud, patterns=["url_domain"], labels=ud_labels, mask=None)
    except FileNotFoundError as exc:
        record_skip(ctx, "url_domain", str(exc))
        ud_runs = []
    if ud_runs:
        if not set(cols).issubset(ud.columns):
            raise RuntimeError("url_domain table lacks scaler feature columns")
        need = np.unique(np.concatenate([np.concatenate([r.cal, r.test]) for r in ud_runs]))
        side_ud = url_side_matrix(ud.iloc[need], cols, scaler["mean"], scaler["std"])
        urls_ud = [str(x) for x in ud["url"].to_numpy()[need]]
        s_ud = u.run({"split": "url_domain"}, score_url_rows, url_session, urls_ud, side_ud)
        if s_ud is not None:
            full_ud = np.full(len(ud), np.nan)
            full_ud[need] = s_ud
            for r in ud_runs:
                if r.name != "url_domain":
                    continue
                reg = ud[URL_CLUSTER_COLUMN].fillna("").astype(str).to_numpy()[r.test]
                metrics["url_domain"]["test"] = url_domain_table(
                    ud_labels[r.cal],
                    full_ud[r.cal],
                    ud_labels[r.test],
                    full_ud[r.test],
                    list(reg),
                    n_boot=1000,
                )

    out_path = write_metrics("metrics.json", metrics)
    ctx.log(f"wrote {out_path}")
    ctx.results["summary"] = _summary(metrics, u)
    ctx.results["baseline_timings_s"] = timings
    status = u.status()
    return status if status != "fail" else False


def _summary(metrics: dict, u: Units) -> str:
    camp = metrics["sms"].get("campaign", {}).get("shipped_threshold", {})
    ph = metrics["url"].get("phreshphish_protocol", {}).get("per_rate", {})
    ap05 = ph.get("0.05%", {}).get("ap_mean")
    return (
        f"v0 sms campaign f1 {camp.get('f1', float('nan')):.4f} "
        f"url phresh ap0.05 {ap05 if ap05 is not None else float('nan'):.4f} "
        f"units ok {u.ok} failed {u.failed}"
    )
