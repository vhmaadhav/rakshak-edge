"""URL general student: PhreshPhish plus url_domain training.

Pre-registered in docs/URL_GENERAL_PLAN.md.

One arm, three seeds, on a Kaggle T4. Training rows are the PhreshPhish carve fit rows plus the
url_domain train rows, minus every row whose registered domain is in either test set.
Calibration is the PhreshPhish carve plus the url_domain cal. Test is PhreshPhish official.test
and url_domain test, scored only after the seed's threshold is written (ThresholdGate enforces it).

Model, training and export reuse kaggle/jobs/url_student.py. Scoring, the threshold rule and the
GroundLM/SmishX tables reuse kaggle/jobs/url_threshold.py, so every number is on the
service path (url_features, then the shipped-style scaler, then the ONNX).

Launch (build, then push):
  kaggle/launch.sh url_general --internet --datasets maadhavvh/rakshak-data \
      --kernels maadhavvh/rakshak-url-student

Score-only (no training; the three seeds' files come from the finished url_general run):
  kaggle/launch.sh url_general --run-name score --internet --datasets maadhavvh/rakshak-data \
      --kernels maadhavvh/rakshak-url-general maadhavvh/rakshak-url-student \
      --job-args="--score-only"
  It locates best_url_general_s<seed>.onnx, url_general_scaler_s<seed>.json and
  thresholds_s<seed>.json under /kaggle/input, records their sha256, reads each threshold (never
  recomputes it) and runs the same scoring, gate and report code as the full run. Scoring uses the
  CPU execution provider, so no GPU is needed.

Outputs (under /kaggle/working): thresholds_s<seed>.json, metrics.json, report.md,
url_scores.parquet, best_url_general_s<seed>.onnx and url_general_scaler_s<seed>.json per seed,
results.json from the runner.

torch is imported only inside run(), so the pure helpers below are testable without torch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import baseline_v0 as base
import numpy as np
import pandas as pd
import url_threshold as ut
from _common import Units, env_report, out_dir, seed_everything, timed, write_metrics

from rakshak.eval import data as D
from rakshak.eval import metrics as M
from rakshak.eval import phreshphish as P
from rakshak.models.encoding import apply_standardizer, encode_batch, fit_standardizer

# ---- pre-registered values (docs/URL_GENERAL_PLAN.md; frozen at launch) ----

SEEDS = (0, 1, 2)
SEED_PASS_MIN = 2
UD_SIZES = {"train": 106512, "cal": 15215, "test": 30432}
N_SIDE = 29
URL_MAX_LEN = 200
CNN_SIZE = "wide"
TRAIN = {
    "epochs": 8,
    "batch_size": 512,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "patience": 3,
    "kd_alpha": 0.5,
    "kd_temp": 2.0,
}
# Set sizes logged by the full run (both modes assert them).
EXPECTED_SIZES = {
    "train": 568924,
    "cal": 33302,
    "test_pp": 168060,
    "test_ud": 30432,
    "domain_novel": 76644,
}
# tau per seed as logged by the full run; score-only checks the saved threshold against these.
LOGGED_TAU = {0: 0.91737, 1: 0.64630, 2: 0.74102}
TAU_LOG_TOL = 5e-6
INPUT_ROOT = Path("/kaggle/input")
PARITY_ROWS = 2000
PARITY_SEED = 0
PARITY_MAX = 1e-3
G1_FPR_MAX = 0.015
G1_RECALL_MIN = 0.85
G2_FPR_MAX = 0.05
CURRENT_AP_1PCT = 0.8647
G3_MAX_DROP = 0.02
G3_AP_MIN = CURRENT_AP_1PCT - G3_MAX_DROP
CURRENT_TAU = 0.6308487409530461
RATE = "1%"
PP = "PP"
UD = "UD"


# ---- pure helpers (unit-tested in tests/test_url_general.py) ----


def registered_domains(df: pd.DataFrame) -> np.ndarray:
    """Registered domain per row, "" when missing or when the table has no reg_domain column."""
    if "reg_domain" not in df.columns:
        return np.full(len(df), "", dtype=object)
    return df["reg_domain"].fillna("").astype(str).to_numpy(dtype=object)


def _in_domains(values, domains) -> np.ndarray:
    """True where a non-empty registered domain is one of ``domains``."""
    arr = np.asarray(values, dtype=object)
    wanted = [d for d in set(domains) if d]
    if not wanted:
        return np.zeros(len(arr), dtype=bool)
    return pd.Series(arr, dtype=object).isin(wanted).to_numpy(dtype=bool) & (arr != "")


def train_keep_mask(train_domains, test_domains) -> np.ndarray:
    """Train rows kept: a row is dropped when its non-empty registered domain is in the test set."""
    return ~_in_domains(train_domains, test_domains)


def domain_overlap(cells: dict[str, np.ndarray]) -> list[dict]:
    """For every unordered pair of cells: shared distinct registered domains, and the rows on each
    side whose domain is shared. Empty domains are never shared."""
    rows: list[dict] = []
    names = list(cells)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            da = np.asarray(cells[a], dtype=object)
            db = np.asarray(cells[b], dtype=object)
            shared = {d for d in da if d} & {d for d in db if d}
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "shared_domains": len(shared),
                    "rows_a": int(_in_domains(da, shared).sum()),
                    "rows_b": int(_in_domains(db, shared).sum()),
                }
            )
    return rows


@dataclass(frozen=True)
class Splits:
    """Combined row positions. PhreshPhish rows keep their table positions; url_domain row p is
    stored at n_pp + p."""

    train: np.ndarray
    cal: np.ndarray
    test_pp: np.ndarray
    test_ud: np.ndarray
    dropped_pp: int
    dropped_ud: int
    overlap: list[dict]


def combined_splits(
    pp_reg, pp_train, pp_cal, pp_test, ud_reg, ud_train, ud_cal, ud_test, n_pp: int
) -> Splits:
    """The training, calibration and test sets of the arm, and the overlap report.

    Amended 2026-11-12 (docs/URL_GENERAL_PLAN.md): train drops a row only when its non-empty
    registered domain is in the url_domain test set. PhreshPhish official train and test are kept as
    published. Cal and test are never dropped. Raises when two sets share a row.
    """
    pp_reg = np.asarray(pp_reg, dtype=object)
    ud_reg = np.asarray(ud_reg, dtype=object)
    pp_train, pp_cal, pp_test = (np.asarray(x, dtype=np.int64) for x in (pp_train, pp_cal, pp_test))
    ud_train, ud_cal, ud_test = (np.asarray(x, dtype=np.int64) for x in (ud_train, ud_cal, ud_test))
    overlap = domain_overlap(
        {
            "PP train": pp_reg[pp_train],
            "PP cal": pp_reg[pp_cal],
            "PP test": pp_reg[pp_test],
            "UD train": ud_reg[ud_train],
            "UD cal": ud_reg[ud_cal],
            "UD test": ud_reg[ud_test],
        }
    )
    test_domains = ud_reg[ud_test]
    keep_pp = train_keep_mask(pp_reg[pp_train], test_domains)
    keep_ud = train_keep_mask(ud_reg[ud_train], test_domains)
    train = np.sort(np.concatenate([pp_train[keep_pp], n_pp + ud_train[keep_ud]]))
    cal = np.sort(np.concatenate([pp_cal, n_pp + ud_cal]))
    test_pp = np.sort(pp_test)
    test_ud = np.sort(n_pp + ud_test)
    for name, a, b in (
        ("train/cal", train, cal),
        ("train/test", train, np.concatenate([test_pp, test_ud])),
        ("cal/test", cal, np.concatenate([test_pp, test_ud])),
    ):
        if np.intersect1d(a, b).size:
            raise ValueError(f"combined sets share rows: {name}")
    return Splits(
        train=train,
        cal=cal,
        test_pp=test_pp,
        test_ud=test_ud,
        dropped_pp=int((~keep_pp).sum()),
        dropped_ud=int((~keep_ud).sum()),
        overlap=overlap,
    )


def training_domains(splits: Splits, pp_reg, ud_reg, n_pp: int) -> set:
    """Non-empty registered domains of the final training rows, both sources."""
    pp_reg = np.asarray(pp_reg, dtype=object)
    ud_reg = np.asarray(ud_reg, dtype=object)
    tr = np.asarray(splits.train, dtype=np.int64)
    doms = np.concatenate([pp_reg[tr[tr < n_pp]], ud_reg[tr[tr >= n_pp] - n_pp]])
    return {d for d in doms if d}


def domain_novel_flag(pp_reg, pp_test, train_domains) -> np.ndarray:
    """Per PhreshPhish test row (in pp_test order): non-empty registered domain not in training."""
    dom = np.asarray(pp_reg, dtype=object)[np.asarray(pp_test, dtype=np.int64)]
    return (dom != "") & ~_in_domains(dom, train_domains)


def novel_subset(y, s_new, s_old, tau_new: float, tau_old: float) -> dict:
    """AP and paired flag counts on the domain-novel rows (reported, not gated)."""
    y = np.asarray(y, dtype=np.int64)
    s_new = np.asarray(s_new, dtype=np.float64)
    s_old = np.asarray(s_old, dtype=np.float64)
    out = paired_counts(s_new >= tau_new, s_old >= tau_old, y)
    out.update(
        {
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "ap_new": M.pr_auc(y, s_new),
            "ap_old": M.pr_auc(y, s_old),
            "tau_new": float(tau_new),
            "tau_old": float(tau_old),
        }
    )
    return out


def calibrate(y, s, source) -> dict:
    """tau on the combined cal rows (url_threshold.threshold_rule), plus the cal benign FPR at tau
    for each source. A flag is score >= tau."""
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    src = np.asarray(source, dtype=object)
    rec = ut.threshold_rule(y, s)
    tau = rec["tau"]
    for tag in (PP, UD):
        benign = (y == 0) & (src == tag)
        n = int(benign.sum())
        fp = int((benign & (s >= tau)).sum())
        rec[f"cal_n_benign_{tag}"] = n
        rec[f"cal_fp_{tag}"] = fp
        rec[f"cal_fpr_{tag}"] = fp / n if n else math.nan
    return rec


def seed_gates(pp_fpr: float, pp_recall: float, ud_fpr: float, pp_ap1: float) -> dict:
    """G1 (PhreshPhish test FPR and recall), G2 (url_domain test FPR), G3 (PP AP at 1 %)."""
    g1 = bool(pp_fpr <= G1_FPR_MAX and pp_recall >= G1_RECALL_MIN)
    g2 = bool(ud_fpr <= G2_FPR_MAX)
    g3 = bool(pp_ap1 >= G3_AP_MIN)
    return {"G1": g1, "G2": g2, "G3": g3, "pass": bool(g1 and g2 and g3)}


def arm_decision(per_seed: dict[int, dict]) -> dict:
    """Arm passes when every seed ran, the median over seeds meets each gate's metric, and at least
    SEED_PASS_MIN seeds meet G1, G2 and G3 together."""
    keys = ("pp_fpr", "pp_recall", "ud_fpr", "pp_ap1")
    if len(per_seed) != len(SEEDS):
        return {"pass": False, "reason": f"{len(per_seed)} of {len(SEEDS)} seeds produced metrics"}
    med = {k: float(np.median([per_seed[s][k] for s in per_seed])) for k in keys}
    median_gates = {
        "G1_fpr": bool(med["pp_fpr"] <= G1_FPR_MAX),
        "G1_recall": bool(med["pp_recall"] >= G1_RECALL_MIN),
        "G2": bool(med["ud_fpr"] <= G2_FPR_MAX),
        "G3": bool(med["pp_ap1"] >= G3_AP_MIN),
    }
    seed_pass = {s: seed_gates(**per_seed[s])["pass"] for s in per_seed}
    n_pass = sum(seed_pass.values())
    return {
        "median": med,
        "median_gates": median_gates,
        "seed_pass": seed_pass,
        "seeds_passing": n_pass,
        "seeds_required": SEED_PASS_MIN,
        "pass": bool(all(median_gates.values()) and n_pass >= SEED_PASS_MIN),
    }


def mcnemar_exact(new_only: int, old_only: int) -> float:
    """Two-sided exact McNemar p on the discordant pairs (binomial, p = 0.5)."""
    n = int(new_only) + int(old_only)
    if n == 0:
        return 1.0
    k = min(int(new_only), int(old_only))
    # Log space: 2.0**n overflows a float for n > 1023 (it crashed on 168k paired test rows).
    from scipy.stats import binom

    return float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))


def paired_counts(new_flag, old_flag, y) -> dict:
    """Flags of the new arm and of the current model on the same rows, with discordant counts."""
    new = np.asarray(new_flag, dtype=bool)
    old = np.asarray(old_flag, dtype=bool)
    y = np.asarray(y, dtype=np.int64)
    neg, pos = y == 0, y == 1
    new_only = int((new & ~old).sum())
    old_only = int((old & ~new).sum())
    return {
        "n": int(len(y)),
        "new_only": new_only,
        "old_only": old_only,
        "both": int((new & old).sum()),
        "mcnemar_exact_p": mcnemar_exact(new_only, old_only),
        "new_fpr": float((new & neg).sum() / max(neg.sum(), 1)),
        "old_fpr": float((old & neg).sum() / max(neg.sum(), 1)),
        "new_recall": float((new & pos).sum() / max(pos.sum(), 1)),
        "old_recall": float((old & pos).sum() / max(pos.sum(), 1)),
    }


def scaler_payload(cols, mean, std) -> dict:
    """The side scaler in the layout url_student.py and url_threshold.py read."""
    return {
        "columns": list(cols),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "clip": 10.0,
        "missing_as": 0.0,
    }


def scores_frame(blocks: dict[str, dict]) -> pd.DataFrame:
    """GroundLM and SmishX message rows: id, set, has_url and one score column per seed.

    ``blocks`` maps a set name to {"frame": the external table, "has": URL presence per row,
    "scores": {seed: score per row}}. Rows with no URL carry NaN.
    """
    frames = []
    for name, block in blocks.items():
        d = pd.DataFrame(
            {
                "id": block["frame"]["id"].astype(str).to_numpy(),
                "set": name,
                "has_url": np.asarray(block["has"], dtype=bool),
            }
        )
        for seed, sc in sorted(block["scores"].items()):
            d[f"score_s{seed}"] = np.asarray(sc, dtype=np.float64)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def onnx_parity(US, model, path: Path, ds, rows) -> float:
    """Max |logit| difference between torch (model on CPU, eval) and onnxruntime on ``rows``."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {
        "raw": ds.ids[rows].astype(np.int64),
        "side": np.ascontiguousarray(ds.side[rows], dtype=np.float32),
    }
    got = session.run(None, {i.name: feed[i.name] for i in session.get_inputs()})[0]
    ref = US.predict_logits(model, "charcnn", ds.ids, ds.side, rows, "cpu", True)
    return float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))


# ---- score-only mode: the trained arm's files, from the attached kernel outputs ----


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="url_general")
    p.add_argument(
        "--score-only",
        action="store_true",
        help="skip training and export; score the trained seeds' ONNX files at the thresholds they "
        "already wrote (read, never recomputed)",
    )
    return p.parse_args(argv or [])


def check_arm_sizes(sizes: dict) -> None:
    """The set sizes against the logged run. Raises with every mismatch."""
    bad = [
        f"{k} {sizes.get(k)} (expected {v})" for k, v in EXPECTED_SIZES.items() if sizes.get(k) != v
    ]
    if bad:
        raise RuntimeError("arm sizes differ from the logged run: " + "; ".join(bad))


def find_named(root: Path, name: str) -> list[Path]:
    """Every file called ``name`` under ``root``, sorted. Empty when root does not exist."""
    if not root.exists():
        return []
    return sorted(Path(d) / name for d, _dirs, files in os.walk(root) if name in files)


def locate_seed_files(seed: int, root: Path = INPUT_ROOT) -> dict[str, Path]:
    """The seed's ONNX, scaler and thresholds under the kernel inputs. A name with two different
    contents is ambiguous and raises. Identical copies are fine."""
    names = {
        "onnx": f"best_url_general_s{seed}.onnx",
        "scaler": f"url_general_scaler_s{seed}.json",
        "thresholds": f"thresholds_s{seed}.json",
    }
    found: dict[str, Path] = {}
    for kind, name in names.items():
        copies = find_named(root, name)
        if not copies:
            raise FileNotFoundError(f"{name}: not found under {root}")
        digests = {base.sha256_file(p) for p in copies}
        if len(digests) > 1:
            where = ", ".join(str(p) for p in copies)
            raise RuntimeError(f"{name}: {len(copies)} copies with different sha256 ({where})")
        found[kind] = copies[0]
    return found


def read_thresholds(path: Path, seed: int) -> dict:
    """The record the training run wrote to thresholds_s<seed>.json. Read, never recomputed:
    the seed, the set sizes and the logged tau are checked against the pre-registered run."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("seed") != seed:
        raise RuntimeError(f"{path.name}: seed {rec.get('seed')!r}, expected {seed}")
    if rec.get("fit_n") != EXPECTED_SIZES["train"] or rec.get("cal_n") != EXPECTED_SIZES["cal"]:
        raise RuntimeError(
            f"{path.name}: fit_n {rec.get('fit_n')} and cal_n {rec.get('cal_n')} differ from the "
            "logged run"
        )
    tau = float(rec["tau"])
    if not math.isfinite(tau):
        raise ValueError(f"{path.name}: tau is not finite: {tau}")
    if abs(tau - LOGGED_TAU[seed]) > TAU_LOG_TOL:
        raise RuntimeError(f"{path.name}: tau {tau!r} differs from the logged {LOGGED_TAU[seed]}")
    return rec


def read_scaler(path: Path, cols, expected_mean, expected_std) -> tuple[np.ndarray, np.ndarray]:
    """The seed's shipped scaler. It must have the arm's columns, and the stats the training rows
    give when refit here."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if list(payload["columns"]) != list(cols):
        raise RuntimeError(f"{path.name}: columns differ from the PhreshPhish feature columns")
    mean = np.asarray(payload["mean"], dtype=np.float64)
    std = np.asarray(payload["std"], dtype=np.float64)
    close = np.allclose(mean, expected_mean, rtol=1e-6, atol=1e-7) and np.allclose(
        std, expected_std, rtol=1e-6, atol=1e-7
    )
    if not close:
        raise RuntimeError(
            f"{path.name}: mean and std differ from the scaler refit on the train rows"
        )
    return mean, std


def _url_scorer(session, cols, mean, std):
    """Probability per URL string on the runtime path (url_threshold.score_url_strings)."""

    def score(texts):
        return ut.score_url_strings(session, texts, cols, mean, std)

    return score


def _refuse_write(record: dict) -> None:
    raise RuntimeError("score-only never writes a threshold")


@dataclass
class Arm:
    """What every seed unit shares. Built once in run()."""

    ctx: object
    u: Units
    root: Path
    data: P.PhreshData
    splits: Splits
    labels: np.ndarray
    source: np.ndarray
    urls: np.ndarray
    cols: list
    n_pp: int
    pp_union: np.ndarray
    side_stats: tuple


def _post_sets(arm: Arm, seed: int, gate, score, rec: dict, files: dict[str, Path]) -> dict:
    """The post sets at the seed's tau, which the gate already holds. Shared by the training unit
    and the score-only unit, so both report through the same code."""
    tau = gate.tau
    data, splits, labels, urls = arm.data, arm.splits, arm.labels, arm.urls

    def _pp_union():
        full = np.full(arm.n_pp, np.nan)
        full[arm.pp_union] = score(urls[arm.pp_union])
        return full

    full_pp = gate.score_post("pp_test_and_benchmark", _pp_union)
    t_pp = ut.at_threshold_counts(labels[data.test], full_pp[data.test], tau)
    proto = P.protocol_section(data.labels, full_pp, data.test, data.bench_full, data.subsets)

    s_ud = gate.score_post("ud_test", lambda: score(urls[splits.test_ud]))
    y_ud = labels[splits.test_ud]
    t_ud = ut.at_threshold_counts(y_ud, s_ud, tau)
    t_ud["pr_auc"] = M.pr_auc(y_ud, s_ud)
    t_ud["roc_auc"] = M.roc_auc(y_ud, s_ud)

    gl = pd.read_parquet(arm.root / "external" / "external_groundlm.parquet")
    sx = pd.read_parquet(arm.root / "external" / "external_smishx.parquet")
    gl_block = ut._external_block(arm.u, gate, "groundlm", gl, score)
    sx_block = ut._external_block(arm.u, gate, "smishx", sx, score)
    ext = {}
    if gl_block is not None:
        ext["groundlm"] = ut._groundlm_table(gl, *gl_block, tau)
    if sx_block is not None:
        ext["smishx"] = ut._smishx_table(sx, *sx_block, tau)

    return {
        "seed": seed,
        "tau": tau,
        "cal": rec,
        "pp_test": t_pp,
        "pp_protocol": proto,
        "ud_test": t_ud,
        "external": ext,
        "gate_score_blocks": {"groundlm": gl_block, "smishx": sx_block, "gl": gl, "sx": sx},
        "arrays": {"pp_test_scores": full_pp[data.test], "ud_test_scores": s_ud},
        "onnx": files["onnx"].name,
        "scaler": files["scaler"].name,
        "files": files,
    }


def _seed_unit_train(arm: Arm, seed: int, US, device, ds) -> dict:
    """Train one seed, export it, check parity, write its threshold, then score the post sets."""
    ctx = arm.ctx
    cfg = US.TrainCfg(
        epochs=TRAIN["epochs"],
        batch_size=TRAIN["batch_size"],
        lr=TRAIN["lr"],
        weight_decay=TRAIN["weight_decay"],
        patience=TRAIN["patience"],
        seed=seed,
        kd_alpha=TRAIN["kd_alpha"],
        kd_temp=TRAIN["kd_temp"],
        amp=device.type == "cuda",
    )
    model = US.CharCNN(US.CONFIGS[CNN_SIZE], views=("raw",), n_side=len(arm.cols))
    info = US.train_model(model, "charcnn", ds, cfg, device, True, None, None, ctx.log)
    model = model.cpu().eval()

    onnx_path = out_dir() / f"best_url_general_s{seed}.onnx"
    exported = US.export_onnx(model, onnx_path, len(arm.cols), URL_MAX_LEN)
    rng = np.random.default_rng(PARITY_SEED)
    rows = np.sort(
        rng.choice(arm.splits.cal, size=min(PARITY_ROWS, len(arm.splits.cal)), replace=False)
    )
    parity = onnx_parity(US, model, onnx_path, ds, rows)
    ctx.log(f"seed {seed}: ONNX parity max |dlogit| {parity:.3g} on {len(rows)} cal rows")
    if not parity <= PARITY_MAX:
        raise RuntimeError(f"seed {seed}: ONNX parity {parity} exceeds {PARITY_MAX}")
    mean, std = arm.side_stats
    scaler_path = out_dir() / f"url_general_scaler_s{seed}.json"
    scaler_path.write_text(json.dumps(scaler_payload(arm.cols, mean, std), indent=2) + "\n")

    session = base._onnx_session(onnx_path)
    score = _url_scorer(session, arm.cols, mean, std)

    # cal first, tau written, then everything else
    splits = arm.splits
    gate = ut.ThresholdGate(ctx.log, lambda rec: write_metrics(f"thresholds_s{seed}.json", rec))
    s_cal = gate.score_pre("combined_cal", lambda: score(arm.urls[splits.cal]))
    rec = calibrate(arm.labels[splits.cal], s_cal, arm.source[splits.cal])
    rec.update(
        {
            "seed": seed,
            "fit_n": int(len(splits.train)),
            "cal_n": int(len(splits.cal)),
            "onnx": exported,
            "parity_max_abs_logit": parity,
            "parity_rows": int(len(rows)),
            "train": {
                k: info[k] for k in ("best_epoch", "best_cal_pr_auc", "epochs_run", "stopped")
            },
            "scorer": "rakshak.url_features + scaler (fit on train rows) + ONNX (runtime path)",
        }
    )
    gate.write(rec)
    files = {
        "onnx": onnx_path,
        "scaler": scaler_path,
        "thresholds": out_dir() / f"thresholds_s{seed}.json",
    }
    return _post_sets(arm, seed, gate, score, rec, files)


def _seed_unit_score_only(arm: Arm, seed: int, files: dict[str, Path]) -> dict:
    """Score one trained seed from its saved files. No training, no parity, no calibration
    scoring: the threshold is read from its file and the post sets are scored at it."""
    rec = read_thresholds(files["thresholds"], seed)
    mean, std = read_scaler(files["scaler"], arm.cols, *arm.side_stats)
    session = base._onnx_session(files["onnx"])
    score = _url_scorer(session, arm.cols, mean, std)
    gate = ut.ThresholdGate(arm.ctx.log, _refuse_write)
    gate.adopt(rec, files["thresholds"].name)
    return _post_sets(arm, seed, gate, score, rec, files)


def _report(metrics: dict) -> str:
    arm = metrics["arm"]
    lines = [
        "# URL general student (pre-registered arm)",
        "",
        "Plan: docs/URL_GENERAL_PLAN.md.",
        f"Mode: {metrics.get('mode', 'train')} (score-only reads each seed's saved threshold; "
        "it does not recompute it).",
        f"Arm: {'PASS' if arm.get('pass') else 'FAIL'}; seeds passing G1-G3 together "
        f"{arm.get('seeds_passing', 0)} of {len(SEEDS)} (need {SEED_PASS_MIN}).",
        "",
        "## Sets",
        f"- train {metrics['sizes']['train']}, cal {metrics['sizes']['cal']}, "
        f"PP test {metrics['sizes']['test_pp']}, UD test {metrics['sizes']['test_ud']}, "
        f"domain-novel {metrics['sizes']['domain_novel']}.",
        f"- dropped from train (domain in a test set): PP {metrics['dropped']['pp']}, "
        f"UD {metrics['dropped']['ud']}.",
        "",
        "## Overlap of registered domains (before the drop)",
    ]
    for o in metrics["overlap"]:
        lines.append(
            f"- {o['a']} vs {o['b']}: shared domains {o['shared_domains']}, rows "
            f"{o['rows_a']} / {o['rows_b']}"
        )
    lines += ["", "## Per seed (at its own tau)"]
    for s, r in metrics["seeds"].items():
        t, u_, c = r["pp_test"], r["ud_test"], r["cal"]
        lines.append(
            f"- seed {s}: tau {r['tau']!r}; cal FPR combined {c['cal_fpr']:.4f} "
            f"(PP {c['cal_fpr_PP']:.4f}, UD {c['cal_fpr_UD']:.4f}); PP test FPR {t['fpr']:.4f} "
            f"recall {t['recall']:.4f}; UD test FPR {u_['fpr']:.4f} recall {u_['recall']:.4f} "
            f"AP {u_['pr_auc']:.4f}; PP AP 1% {r['pp_protocol']['per_rate'][RATE]['ap_mean']:.4f}."
        )
    nov = metrics.get("domain_novel", {})
    if nov:
        lines += ["", "## Domain-novel PhreshPhish test rows (reported)", f"- n {nov['n']}"]
        for s, v in nov.get("paired", {}).items():
            lines.append(
                f"- seed {s}: AP new {v['ap_new']:.4f} vs current {v['ap_old']:.4f}; "
                f"FPR new {v['new_fpr']:.4f} vs current {v['old_fpr']:.4f}; recall new "
                f"{v['new_recall']:.4f} vs current {v['old_recall']:.4f}; McNemar p "
                f"{v['mcnemar_exact_p']:.3g}"
            )
    lines += ["", "## Arm decision"]
    lines.append(f"- median: {arm.get('median')}")
    lines.append(f"- median gates: {arm.get('median_gates')}")
    lines.append(f"- seed pass: {arm.get('seed_pass')}")
    return "\n".join(lines) + "\n"


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    mode = "score-only" if args.score_only else "train"
    seed_everything(0)
    env = env_report()
    ctx.log(f"env: {env}")
    ctx.log(f"mode: {mode}")
    timings: dict = {}

    # 1. pins: the comparator ONNX and scaler, and the data files
    with timed("pins_s", timings):
        search = [Path("/kaggle/input")]
        old_onnx = base.locate_artifact(search, ut.ONNX_NAME, ut.ONNX_SHA)
        old_scaler_path = base.locate_artifact(search, ut.SCALER_NAME, ut.SCALER_SHA)
        root = D.resolve_data_root(None, marker=P.PHRESH_FILE)
        data_actual = ut.check_data_pins(root)
    old_scaler = json.loads(old_scaler_path.read_text(encoding="utf-8"))
    cols = list(old_scaler["columns"])
    if len(cols) != N_SIDE:
        raise RuntimeError(f"scaler has {len(cols)} columns, expected {N_SIDE}")
    old_mean = np.asarray(old_scaler["mean"], dtype=np.float64)
    old_std = np.asarray(old_scaler["std"], dtype=np.float64)

    # 2. the two tables, the splits and the sizes
    data = P.load_phreshphish(root)
    if list(data.feature_cols) != cols:
        raise RuntimeError("PhreshPhish feature columns differ from the scaler")
    df = data.df
    n_pp = len(df)
    keep, cal_pp = ut.grouped_carve(data)
    ut.check_carve_sizes(len(cal_pp), len(keep))

    ud = D.load_urls(root)
    missing = [c for c in cols if c not in ud.columns]
    if missing:
        raise RuntimeError(f"url_domain table lacks side columns {missing}")
    ud_labels = ud["label"].to_numpy().astype(np.int64)
    ud_runs = D.load_runs(root, ud, patterns=["url_domain"], labels=ud_labels, mask=None)
    run_ud = next((r for r in ud_runs if r.name == "url_domain"), None)
    if run_ud is None:
        raise RuntimeError("url_domain split has no 'url_domain' partition")
    ud_sizes = {"train": len(run_ud.train), "cal": len(run_ud.cal), "test": len(run_ud.test)}
    if ud_sizes != UD_SIZES:
        raise RuntimeError(f"url_domain sizes {ud_sizes} differ from the pre-registered {UD_SIZES}")

    pp_reg_all = registered_domains(df)
    ud_reg_all = registered_domains(ud)
    splits = combined_splits(
        pp_reg_all,
        keep,
        cal_pp,
        data.test,
        ud_reg_all,
        run_ud.train,
        run_ud.cal,
        run_ud.test,
        n_pp,
    )
    train_doms = training_domains(splits, pp_reg_all, ud_reg_all, n_pp)
    novel_flag = domain_novel_flag(pp_reg_all, data.test, train_doms)
    sizes = {
        "train": int(len(splits.train)),
        "cal": int(len(splits.cal)),
        "test_pp": int(len(splits.test_pp)),
        "test_ud": int(len(splits.test_ud)),
        "domain_novel": int(novel_flag.sum()),
    }
    check_arm_sizes(sizes)
    ctx.log(f"domain-novel PhreshPhish test rows: {sizes['domain_novel']} of {len(data.test)}")
    ctx.log(
        f"combined: train {len(splits.train)} (dropped PP {splits.dropped_pp}, UD "
        f"{splits.dropped_ud}), cal {len(splits.cal)}, PP test {len(splits.test_pp)}, "
        f"UD test {len(splits.test_ud)}"
    )

    side_all = np.concatenate(
        [df[cols].fillna(0.0).to_numpy(np.float32), ud[cols].fillna(0.0).to_numpy(np.float32)]
    )
    labels = np.concatenate([data.labels, ud_labels])
    urls = np.concatenate(
        [df["url"].astype(str).to_numpy(dtype=object), ud["url"].astype(str).to_numpy(dtype=object)]
    )
    source = np.concatenate([np.full(n_pp, PP, dtype=object), np.full(len(ud), UD, dtype=object)])
    # report only: exact URL strings shared by the train and test sets after the domain drop
    train_urls = set(urls[splits.train].tolist())
    exact_after = {
        "PP_test": int(sum(u_ in train_urls for u_ in set(urls[splits.test_pp].tolist()))),
        "UD_test": int(sum(u_ in train_urls for u_ in set(urls[splits.test_ud].tolist()))),
    }
    ctx.log(f"exact URL strings shared with train after the drop: {exact_after}")
    mean, std = fit_standardizer(side_all, splits.train)
    pp_union = np.unique(
        np.concatenate([data.test, data.bench_full] + [pos for _r, pos in data.subsets])
    )
    arm = Arm(
        ctx=ctx,
        u=Units(ctx),
        root=root,
        data=data,
        splits=splits,
        labels=labels,
        source=source,
        urls=urls,
        cols=cols,
        n_pp=n_pp,
        pp_union=pp_union,
        side_stats=(mean, std),
    )

    # 3. the seeds: train each one (train mode), or locate its saved files (score-only)
    results: dict[int, dict] = {}
    if args.score_only:
        with timed("locate_s", timings):
            located = {seed: locate_seed_files(seed) for seed in SEEDS}
        for seed, files in located.items():
            ctx.log(f"seed {seed} files: " + ", ".join(f"{k} {p}" for k, p in files.items()))
        for seed in SEEDS:
            out = arm.u.run(
                {"split": "url_general", "seed": seed, "mode": mode},
                _seed_unit_score_only,
                arm,
                seed,
                located[seed],
            )
            if out is not None:
                results[seed] = out
            ctx.log(f"seed {seed} done: {'ok' if out is not None else 'FAILED'}")
    else:
        import torch as torch_mod
        import url_student as US

        side = apply_standardizer(side_all, mean, std)
        with timed("encode_s", timings):
            ids = encode_batch(list(urls), URL_MAX_LEN)
        ds = US.Dataset(
            name="url_general",
            ids=ids,
            labels=labels,
            row_ids=np.concatenate(
                [
                    df["id"].astype(str).to_numpy(dtype=object),
                    ud["id"].astype(str).to_numpy(dtype=object),
                ]
            ),
            fit=splits.train,
            cal=splits.cal,
            test=np.concatenate([splits.test_pp, splits.test_ud]),
            side=side,
            side_cols=cols,
            side_stats=(mean, std),
            urls=urls,
        )
        device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        ctx.log(f"device {device}; seeds {list(SEEDS)}")
        for seed in SEEDS:
            out = arm.u.run(
                {"split": "url_general", "seed": seed, "mode": mode},
                _seed_unit_train,
                arm,
                seed,
                US,
                device,
                ds,
            )
            if out is not None:
                results[seed] = out
            ctx.log(f"seed {seed} done: {'ok' if out is not None else 'FAILED'}")
    u = arm.u

    # 4. decision and the comparisons (all thresholds are written or read before this point)
    per_seed = {
        s: {
            "pp_fpr": r["pp_test"]["fpr"],
            "pp_recall": r["pp_test"]["recall"],
            "ud_fpr": r["ud_test"]["fpr"],
            "pp_ap1": r["pp_protocol"]["per_rate"][RATE]["ap_mean"],
        }
        for s, r in results.items()
    }
    per_seed_gates = {s: seed_gates(**m) for s, m in per_seed.items()}
    arm_dec = arm_decision(per_seed)

    paired: dict = {}
    novel: dict = {}
    with timed("comparator_s", timings):
        old_session = base._onnx_session(old_onnx)

        def old_score(texts):
            return ut.score_url_strings(old_session, texts, cols, old_mean, old_std)

        old_full = np.full(n_pp, np.nan)
        old_full[pp_union] = old_score(urls[pp_union])
        old_ud = old_score(urls[splits.test_ud])
    old_proto = P.protocol_section(data.labels, old_full, data.test, data.bench_full, data.subsets)
    comparator = {
        "tau": CURRENT_TAU,
        "pp_test": ut.at_threshold_counts(labels[data.test], old_full[data.test], CURRENT_TAU),
        "ud_test": ut.at_threshold_counts(labels[splits.test_ud], old_ud, CURRENT_TAU),
        "pp_protocol": old_proto,
        "ap_1pct_recorded": CURRENT_AP_1PCT,
    }
    for s, r in results.items():
        paired[s] = {
            "PP_test": paired_counts(
                r["arrays"]["pp_test_scores"] >= r["tau"],
                old_full[data.test] >= CURRENT_TAU,
                labels[data.test],
            ),
            "UD_test": paired_counts(
                r["arrays"]["ud_test_scores"] >= r["tau"],
                old_ud >= CURRENT_TAU,
                labels[splits.test_ud],
            ),
            "ap_new_minus_current": {
                rate: float(
                    r["pp_protocol"]["per_rate"][rate]["ap_mean"]
                    - old_proto["per_rate"][rate]["ap_mean"]
                )
                for rate in r["pp_protocol"]["per_rate"]
            },
        }

    novel_pos = data.test[novel_flag]
    novel_block = {
        "n": int(len(novel_pos)),
        "excluded_empty_domain": int((pp_reg_all[data.test] == "").sum()),
    }
    for s, r in results.items():
        novel[s] = novel_subset(
            labels[novel_pos],
            r["arrays"]["pp_test_scores"][novel_flag],
            old_full[novel_pos],
            r["tau"],
            CURRENT_TAU,
        )
    novel_block["paired"] = novel

    # 5. outputs
    blocks: dict[str, dict] = {}
    for name, frame_key in (("groundlm", "gl"), ("smishx", "sx")):
        ok = {
            s: r["gate_score_blocks"][name]
            for s, r in results.items()
            if r["gate_score_blocks"][name] is not None
        }
        if not ok:
            continue
        first = min(ok)
        blocks[name] = {
            "frame": results[first]["gate_score_blocks"][frame_key],
            "has": ok[first][1],
            "scores": {s: b[0] for s, b in ok.items()},
        }
    if blocks:
        scores_path = out_dir() / "url_scores.parquet"
        scores_frame(blocks).to_parquet(scores_path, index=False)
        ctx.log(f"wrote {scores_path}")

    artifacts = {
        str(s): {
            kind: {"path": str(p), "sha256": base.sha256_file(p)} for kind, p in r["files"].items()
        }
        for s, r in results.items()
    }
    metrics = {
        "job": "url_general",
        "mode": mode,
        "env": env,
        "timings": timings,
        "data_pins": data_actual,
        "artifacts": artifacts,
        "sizes": {
            **sizes,
            "fit_pp_before_drop": int(len(keep)),
            "ud_train_before_drop": UD_SIZES["train"],
        },
        "dropped": {"pp": splits.dropped_pp, "ud": splits.dropped_ud},
        "overlap": splits.overlap,
        "exact_url_shared_with_train_after_drop": exact_after,
        "seeds": {
            s: {k: v for k, v in r.items() if k not in ("gate_score_blocks", "arrays", "files")}
            for s, r in results.items()
        },
        "per_seed_gates": per_seed_gates,
        "arm": arm_dec,
        "comparator": comparator,
        "paired": paired,
        "domain_novel": novel_block,
    }
    out_path = write_metrics("metrics.json", metrics)
    (out_dir() / "report.md").write_text(_report(metrics), encoding="utf-8")
    ctx.log(f"wrote {out_path}")
    ctx.results["summary"] = (
        f"{mode}: seeds ok {len(results)}/{len(SEEDS)}; arm "
        f"{'pass' if arm_dec.get('pass') else 'fail'}; median PP FPR "
        f"{arm_dec.get('median', {}).get('pp_fpr')}, UD FPR "
        f"{arm_dec.get('median', {}).get('ud_fpr')}"
    )
    ctx.results["timings_s"] = timings
    status = u.status()
    return status if status != "fail" else False
