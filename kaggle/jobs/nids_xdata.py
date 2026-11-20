"""Cross-dataset generalisation job for the NetFlow-v2 network module (plan: docs/NIDS_XDATA_PLAN.md).

Modes, all run through ``runjob.py`` from the VM copy of the repo (PYTHONPATH=src:kaggle/jobs):

  stage      one raw NF-*-v2 parquet -> label-stratified sample (numeric columns, Label, Attack),
             with a JSON record (row counts, sha256). One dataset at a time.
               python ../runjob.py nids_xdata <workdir> --mode stage --raw <parquet> --key <key> \
                   --out-dir ../nids_data
  fold       one leave-one-dataset-out fold: --target <key>. Every arm (a, b, c, d; LightGBM and
             MLP), the inner leave-one-source-out choice of hyperparameter and threshold, seeds,
             and bootstrap draws. Writes <out-dir>/fold_<target>.json.
               python ../runjob.py nids_xdata <workdir> --mode fold --target <key> \
                   --data-dir ../nids_data --out-dir ../out/xdata --seeds 0 1 2
  aggregate  reads the four fold files, applies the pre-registered gate, writes metrics.json and
             summary.md under <out-dir>.
               python ../runjob.py nids_xdata <workdir> --mode aggregate --out-dir ../out/xdata
"""

from __future__ import annotations

import os

# Thread caps before numpy or any BLAS loads: the VM is shared, and unbounded threads
# oversubscribe the cores (LightGBM and BLAS both default to every core).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rakshak.eval import metrics as M
from rakshak.nids import baseline as B
from rakshak.nids import schema as S
from rakshak.nids import splits as SP
from rakshak.nids import xdata as X

DATASETS: dict[str, str] = {
    "unsw": "NF-UNSW-NB15-V2",
    "cse_cic_ids2018": "NF-CSE-CIC-IDS2018-V2",
    "bot_iot": "NF-BoT-IoT-V2",
    "ton_iot": "NF-ToN-IoT-V2",
}
ARMS: tuple[str, ...] = ("a", "b", "c", "d")
FAMILIES: tuple[str, ...] = ("lgbm", "mlp")
REFERENCE_CSE = "cse_cic_ids2018"
HEAD_K = 5000  # unlabelled site-calibration flows of a target (never tested on)
TRAIN_PER_CLASS = 30_000  # per source, balanced
EVAL_PER_CLASS = 100_000  # amendment (compute budget): validation and test rows per class
FPR_CHECK = 0.01
GATE_MEAN = 0.10
GATE_WORST = 0.05
GRID: dict[str, dict[str, list[dict[str, Any]]]] = {
    "lgbm": {
        "a": [{"trees": 300, "leaves": 63}],  # baseline: nids_baselines defaults
        "bcd": [{"trees": 200, "leaves": 15}, {"trees": 200, "leaves": 31}],
    },
    "mlp": {
        "a": [{"hidden": [64, 32], "alpha": 1e-4}],  # baseline: shipped student width
        "bcd": [{"hidden": [32, 32], "alpha": 1e-4}, {"hidden": [32, 32], "alpha": 1e-2}],
    },
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="nids_xdata")
    p.add_argument("--mode", choices=["stage", "fold", "aggregate"], required=True)
    p.add_argument("--raw", type=Path, default=None)
    p.add_argument("--key", choices=list(DATASETS), default=None)
    p.add_argument("--target", choices=list(DATASETS), default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--max-benign", type=int, default=300_000)
    p.add_argument("--max-attack", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--n-boot", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv or [])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


def stage(ctx, args: argparse.Namespace) -> bool:
    if args.raw is None or args.key is None or args.out_dir is None:
        raise SystemExit("stage mode needs --raw, --key and --out-dir")
    out = args.out_dir / f"{args.key}.parquet"
    t0 = time.perf_counter()
    info = X.stage_sample(args.raw, out, max_benign=args.max_benign, max_attack=args.max_attack,
                          seed=args.seed)
    record: dict[str, Any] = {
        "key": args.key, "source_file": DATASETS[args.key] + ".parquet",
        "sample_file": out.name, "sample_sha256": sha256(out),
        "sample_bytes": out.stat().st_size, "caps": {"benign": args.max_benign,
                                                     "attack": args.max_attack},
        "seed": args.seed, "seconds": round(time.perf_counter() - t0, 1), **info.as_dict(),
    }
    (args.out_dir / f"{args.key}.json").write_text(json.dumps(record, indent=2) + "\n")
    ctx.results["stage"] = record
    ctx.log(f"staged {args.key}: benign {info.benign_kept}/{info.benign_total}, attack "
            f"{info.attack_kept}/{info.attack_total}, columns {len(info.columns)}")
    return True


# ---------------------------------------------------------------------------
# fold: data, arms, models
# ---------------------------------------------------------------------------


@dataclass
class DS:
    key: str
    M: np.ndarray  # (rows, len(X.ALL_NAMES)) float32, after the shared-feature dedupe
    y: np.ndarray  # 0 benign, 1 attack
    n_raw: int
    n_dup: int


def load_dataset(root: Path, key: str) -> DS:
    import pandas as pd

    path = root / f"{key}.parquet"
    df = pd.read_parquet(path)
    y = df["Label"].to_numpy().astype(np.int64)
    raw = {c: df[c].to_numpy(dtype=np.float64) for c in df.columns if c not in ("Label", "Attack")}
    missing = [f for f in S.SHARED_FEATURES if f not in raw]
    if missing:
        raise KeyError(f"{key}: staged sample lacks shared features {missing}")
    mat = X.build_feature_matrix(raw)
    base = X.column_index(S.SHARED_FEATURES)
    _, first = np.unique(mat[:, base], axis=0, return_index=True)
    first = np.sort(first)
    if set(np.unique(y[first]).tolist()) != {0, 1}:
        raise ValueError(f"{key}: a class is missing after the dedupe")
    return DS(key, mat[first], y[first], int(len(y)), int(len(y) - len(first)))


def arm_names(arm: str, selected: list[str]) -> list[str]:
    if arm == "a":
        return list(S.SHARED_FEATURES)
    if arm == "b":
        return list(X.HYGIENE_FEATURES) + list(X.ENGINEERED)
    return list(selected)  # c and d


def grid_for(family: str, arm: str) -> list[dict[str, Any]]:
    return GRID[family]["a" if arm == "a" else "bcd"]


def fit_model(family: str, cfg: dict[str, Any], X_list: list[np.ndarray],
              y_list: list[np.ndarray], seed: int):
    Xtr = np.vstack(X_list)
    ytr = np.concatenate(y_list)
    if family == "lgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=cfg["trees"], num_leaves=cfg["leaves"], learning_rate=0.1,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
            n_jobs=2, force_col_wise=True)
        model.fit(Xtr, ytr)
        return model
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=tuple(cfg["hidden"]), alpha=cfg["alpha"], max_iter=40,
        early_stopping=True, n_iter_no_change=5, batch_size=512, random_state=seed))
    pipe.fit(Xtr, ytr)
    return pipe


class Scorer:
    """P(attack) for a fitted model. MLPs are scored through an INT8 ONNX graph, so every MLP
    number in the results comes from the exported model, not from sklearn."""

    def __init__(self, family: str, model, n_in: int, tmp: Path | None, tag: str,
                 threads: int = 4) -> None:
        self.family = family
        self.model = model
        self.session = None
        self.int8_path: Path | None = None
        self.fp32_path: Path | None = None
        if family == "mlp":
            self._export(n_in, tmp, tag, threads)

    def _export(self, n_in: int, tmp: Path | None, tag: str, threads: int) -> None:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        import nids_baselines as NB  # kaggle/jobs helpers: ONNX writer, folding, sessions

        scaler, net = self.model[0], self.model[1]
        layers = [NB.fold_first_layer(scaler.mean_, scaler.scale_, net.coefs_[0],
                                      net.intercepts_[0])]
        for i in range(1, len(net.coefs_) - 1):
            layers.append((net.coefs_[i], net.intercepts_[i]))
        W, b = net.coefs_[-1], net.intercepts_[-1]
        layers.append((np.hstack([np.zeros_like(W), W]), np.concatenate([[0.0], b])))
        assert tmp is not None
        tmp.mkdir(parents=True, exist_ok=True)
        meta = {"rakshak.schema": S.SCHEMA_VERSION, "rakshak.kind": "xdata MLP 2x32 (binary)",
                "rakshak.tag": tag}
        self.fp32_path = tmp / f"{tag}_fp32.onnx"
        self.int8_path = tmp / f"{tag}_int8.onnx"
        NB.build_student_onnx(layers, n_in, self.fp32_path, meta)
        quantize_dynamic(str(self.fp32_path), str(self.int8_path), weight_type=QuantType.QInt8)
        self.session = NB.onnx_session(self.int8_path, threads)

    def score(self, Xm: np.ndarray) -> np.ndarray:
        if self.family == "lgbm":
            return self.model.predict_proba(Xm)[:, 1]
        import nids_baselines as NB

        return NB.onnx_probs(self.session, Xm)[:, 1].astype(np.float64)


def size_bytes(scorer: Scorer) -> int:
    if scorer.family == "mlp":
        return int(scorer.int8_path.stat().st_size)
    return len(scorer.model.booster_.model_to_string().encode("utf-8"))


def single_flow_latency(scorer: Scorer, Xm: np.ndarray, n: int = 1000) -> dict[str, float]:
    rows = Xm[np.arange(min(n, len(Xm))) % len(Xm)]
    if scorer.family == "mlp":
        import nids_baselines as NB

        return NB.latency_ms(scorer.session, rows)
    import time as _t

    for r in rows[:50]:
        scorer.model.predict_proba(r[None, :])
    times = []
    for r in rows:
        t0 = _t.perf_counter()
        scorer.model.predict_proba(r[None, :])
        times.append((_t.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"n": int(len(arr)), "threads": "default", "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "mean_ms": float(arr.mean())}


def fp32_int8_parity(scorer: Scorer, Xm: np.ndarray) -> dict[str, float]:
    import nids_baselines as NB

    sess32 = NB.onnx_session(scorer.fp32_path, 1)
    sub = Xm[:4096].astype(np.float32)
    p32 = NB.onnx_probs(sess32, sub)[:, 1]
    p8 = NB.onnx_probs(scorer.session, sub)[:, 1]
    return {"n": int(len(sub)), "max_abs": float(np.max(np.abs(p32 - p8))),
            "decision_agreement_at_0.5": float(np.mean((p32 >= 0.5) == (p8 >= 0.5)))}


def normalise(Xs: np.ndarray, names: list[str], fit_rows: np.ndarray) -> np.ndarray:
    """Benign-quantile (or head) map fitted on ``fit_rows`` only, applied to all of ``Xs``."""
    smap = X.fit_benign_quantiles(fit_rows, names)
    return smap.transform(Xs)


def choose_on_sources(family: str, arm: str, seed: int, srcs: list[str],
                      Xtr: dict[str, np.ndarray], ytr: dict[str, np.ndarray],
                      Xval: dict[str, np.ndarray], yval: dict[str, np.ndarray],
                      tmp: Path, tag: str) -> tuple[dict[str, Any], int]:
    """Inner leave-one-source-out: per config, threshold from the three inner folds; the config
    with the best mean inner MCC is kept. Returns the per-config info and the chosen index."""
    grid = grid_for(family, arm)
    info: dict[str, dict[str, Any]] = {}
    for ci, cfg in enumerate(grid):
        folds = []
        for v in srcs:
            trn = [k for k in srcs if k != v]
            model = fit_model(family, cfg, [Xtr[k] for k in trn], [ytr[k] for k in trn], seed)
            scorer = Scorer(family, model, Xval[v].shape[1], tmp, f"{tag}_c{ci}_in{v}")
            s = scorer.score(Xval[v])
            folds.append((yval[v], s))
            _cleanup(scorer)
        entry = X.choose_threshold(folds)
        entry["config"] = cfg
        info[f"cfg{ci}"] = entry
    pick = X.choose_config(info)
    return info, int(pick.removeprefix("cfg"))


def _cleanup(scorer: Scorer) -> None:
    for p in (scorer.int8_path, scorer.fp32_path):
        if p is not None and p.exists():
            p.unlink()


def run_fold(ctx, args: argparse.Namespace, root: Path, out_dir: Path, target: str) -> dict:
    t_start = time.perf_counter()
    keys = list(DATASETS)
    ds = {k: load_dataset(root, k) for k in keys}
    srcs = [k for k in keys if k != target]
    T = ds[target]
    head, rest = B.calibration_slice(len(T.y), HEAD_K, seed=0)
    result: dict[str, Any] = {
        "target": target, "sources": srcs, "seeds": list(args.seeds), "n_boot": args.n_boot,
        "head_k": HEAD_K, "train_per_class": TRAIN_PER_CLASS,
        "datasets": {k: {"rows_raw": d.n_raw, "dups_removed": d.n_dup, "rows": int(len(d.y)),
                         "attack_rows": int(d.y.sum()), "prevalence": float(d.y.mean())}
                     for k, d in ds.items()},
        "target_test_rows": int(len(rest)), "target_head_rows": int(len(head)),
        "runs": [], "a1": [], "selection": {},
    }
    tmp = out_dir / "tmp_onnx"
    # amendment (compute budget): evaluation rows are capped per class with a fixed seed
    vrows = {k: X.training_subsample(ds[k].y, EVAL_PER_CLASS, 0) for k in srcs}
    te = rest[X.training_subsample(T.y[rest], EVAL_PER_CLASS, 0)]
    result["eval_rows"] = {"per_class_cap": EVAL_PER_CLASS,
                           "validation": {k: int(len(vrows[k])) for k in srcs},
                           "target_test": int(len(te))}
    for seed in args.seeds:
        tr = {k: X.training_subsample(ds[k].y, TRAIN_PER_CLASS, seed) for k in srcs}
        b_names, b_cols = arm_names("b", []), X.column_index(arm_names("b", []))
        sel = X.stable_features(
            {k: (ds[k].M[tr[k]][:, b_cols], ds[k].y[tr[k]]) for k in srcs}, b_names,
            k=20, min_keep=8, seed=seed)
        result["selection"][str(seed)] = sel
        if target != REFERENCE_CSE:
            result["a1"].append(_a1_reference(ds, target, seed, te))
        ref_scores: dict[str, np.ndarray] = {}
        for family in FAMILIES:
            for arm in ARMS:
                names = arm_names(arm, sel)
                cols = X.column_index(names)
                full = {k: ds[k].M[:, cols] for k in keys}
                if arm == "d":
                    for k in srcs:
                        fit_rows = full[k][ds[k].y == 0]
                        full[k] = normalise(full[k], names, fit_rows)
                    full[target] = normalise(full[target], names, full[target][head])
                Xtr = {k: full[k][tr[k]] for k in srcs}
                ytr = {k: ds[k].y[tr[k]] for k in srcs}
                Xval = {k: full[k][vrows[k]] for k in srcs}
                yval = {k: ds[k].y[vrows[k]] for k in srcs}
                Xt, yt = full[target][te], ds[target].y[te]
                tag = f"{target}_{family}_{arm}_s{seed}"
                t0 = time.perf_counter()
                info, pick = choose_on_sources(family, arm, seed, srcs, Xtr, ytr, Xval, yval,
                                               tmp, tag)
                cfg = grid_for(family, arm)[pick]
                thr = info[f"cfg{pick}"]["threshold"]
                model = fit_model(family, cfg, [Xtr[k] for k in srcs],
                                  [ytr[k] for k in srcs], seed)
                scorer = Scorer(family, model, Xt.shape[1], tmp, tag + "_outer")
                s_t = scorer.score(Xt)
                pred = (s_t >= thr).astype(np.int64)
                m = X.target_metrics(yt, pred, s_t)
                m["threshold"] = float(thr)
                run: dict[str, Any] = {
                    "family": family, "arm": arm, "seed": seed, "config": cfg,
                    "config_index": pick, "inner": info, "metrics": m,
                    "fpr_check_ok": bool(m["fpr"] <= FPR_CHECK),
                    "features": names, "seconds": round(time.perf_counter() - t0, 1),
                }
                if arm == "a":
                    ref_scores[family] = pred
                if seed == args.seeds[0]:
                    run["size_bytes"] = size_bytes(scorer)
                    run["latency"] = single_flow_latency(scorer, Xt, 1000)
                    if family == "mlp":
                        run["parity"] = fp32_int8_parity(scorer, Xt)
                    ref_pred = ref_scores[family]
                    draws = X.bootstrap_metric_draws(yt, [ref_pred, pred], args.n_boot, seed=0)
                    run["bootstrap_ref_arm"] = np.round(draws, 6).tolist()
                _cleanup(scorer)
                result["runs"].append(run)
                ctx.log(f"{tag}: mcc={m['mcc']:.4f} f1={m['f1']:.4f} fpr={m['fpr']:.4f} "
                        f"thr={thr:.5f} cfg={pick} ({run['seconds']}s)")
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    result["seconds"] = round(time.perf_counter() - t_start, 1)
    return result


def _a1_reference(ds: dict[str, DS], target: str, seed: int, rest: np.ndarray) -> dict[str, Any]:
    """Reference only: LightGBM 300/63 on the baseline features, trained on CSE-CIC-2018 alone
    (70% of its sample), threshold = max-F1 on CSE's own 15% held-out split, tested on the target."""
    cse = ds[REFERENCE_CSE]
    a_cols = X.column_index(S.SHARED_FEATURES)
    tr_rows, cal_rows, _ = SP.random_split(len(cse.y), (0.70, 0.15, 0.15), seed, strata=cse.y)
    sub = tr_rows[X.training_subsample(cse.y[tr_rows], TRAIN_PER_CLASS, seed)]
    model = fit_model("lgbm", {"trees": 300, "leaves": 63}, [cse.M[sub][:, a_cols]],
                      [cse.y[sub]], seed)
    s_cal = model.predict_proba(cse.M[cal_rows][:, a_cols])[:, 1]
    thr, f1_cal = M.threshold_max_f1(cse.y[cal_rows], s_cal)
    T = ds[target]
    s_t = model.predict_proba(T.M[rest][:, a_cols])[:, 1]
    m = X.target_metrics(T.y[rest], (s_t >= thr).astype(np.int64), s_t)
    m["threshold"] = float(thr)
    return {"seed": seed, "cal_f1": float(f1_cal), "metrics": m}


# ---------------------------------------------------------------------------
# aggregate: gate and tables
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}


def aggregate(ctx, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    folds: dict[str, dict] = {}
    for p in sorted(out_dir.glob("fold_*.json")):
        f = json.loads(p.read_text())
        folds[f["target"]] = f
    missing = [k for k in DATASETS if k not in folds]
    if missing:
        raise SystemExit(f"aggregate needs all four folds; missing {missing}")
    targets = list(DATASETS)
    table: dict[str, Any] = {}
    draws_by: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        for arm in ARMS:
            per_t: dict[str, Any] = {}
            for t in targets:
                runs = [r for r in folds[t]["runs"] if r["family"] == family and r["arm"] == arm]
                ms = [r["metrics"] for r in runs]
                per_t[t] = {
                    "mcc": _stats([m["mcc"] for m in ms]), "f1": _stats([m["f1"] for m in ms]),
                    "pr_auc": _stats([m["pr_auc"] for m in ms]),
                    "fpr": _stats([m["fpr"] for m in ms]), "tpr": _stats([m["recall"] for m in ms]),
                    "prevalence": ms[0]["prevalence"], "n": ms[0]["n"],
                    "thresholds": [m["threshold"] for m in ms],
                    "configs": [r["config_index"] for r in runs],
                    "fpr_ok_all_seeds": all(r["fpr_check_ok"] for r in runs),
                }
                seed0 = [r for r in runs if r["seed"] == folds[t]["seeds"][0]][0]
                draws_by.setdefault((family, arm), {})[t] = np.asarray(seed0["bootstrap_ref_arm"])
                if "latency" in seed0:
                    per_t[t]["latency_ms_p50"] = seed0["latency"]["p50_ms"]
                    per_t[t]["size_bytes"] = seed0["size_bytes"]
                    if "parity" in seed0:
                        per_t[t]["parity"] = seed0["parity"]
            mean_mcc = float(np.mean([per_t[t]["mcc"]["mean"] for t in targets]))
            worst_t = min(targets, key=lambda t: per_t[t]["mcc"]["mean"])
            table[f"{family}/{arm}"] = {
                "targets": per_t, "mean_mcc": mean_mcc,
                "worst_target": worst_t, "worst_mcc": per_t[worst_t]["mcc"]["mean"],
                "mean_f1": float(np.mean([per_t[t]["f1"]["mean"] for t in targets])),
                "mean_pr_auc": float(np.mean([per_t[t]["pr_auc"]["mean"] for t in targets])),
                "mean_fpr": float(np.mean([per_t[t]["fpr"]["mean"] for t in targets])),
            }
    gates: dict[str, Any] = {}
    for family in FAMILIES:
        ref_key = f"{family}/a"
        ref = table[ref_key]
        for arm in ARMS[1:]:
            key = f"{family}/{arm}"
            arm_row = table[key]
            diff_mean = arm_row["mean_mcc"] - ref["mean_mcc"]
            diff_worst = arm_row["worst_mcc"] - ref["worst_mcc"]
            # paired bootstrap of the mean-over-targets MCC difference (draw b of each target)
            boots = np.mean([draws_by[(family, arm)][t][:, 1] - draws_by[(family, arm)][t][:, 0]
                             for t in targets], axis=0)
            lo, hi = X.percentile_interval(boots)
            pass_mean = diff_mean >= GATE_MEAN
            pass_worst = diff_worst >= GATE_WORST
            pass_ci = lo > 0
            gates[key] = {
                "reference": ref_key, "diff_mean_mcc": diff_mean, "diff_worst_mcc": diff_worst,
                "ci95_diff_mean_mcc": [lo, hi], "pass_mean": bool(pass_mean),
                "pass_worst": bool(pass_worst), "pass_ci": bool(pass_ci),
                "pass": bool(pass_mean and pass_worst and pass_ci),
                "fpr_over_1pct_targets": [t for t in targets
                                          if not arm_row["targets"][t]["fpr_ok_all_seeds"]],
            }
    a1 = {t: _stats([r["metrics"]["mcc"] for r in folds[t]["a1"]])
          for t in targets if t != REFERENCE_CSE}
    summary = {
        "schema": S.SCHEMA_VERSION, "targets": targets, "table": table, "gates": gates,
        "gate_rule": {"mean_mcc_diff_min": GATE_MEAN, "worst_mcc_diff_min": GATE_WORST,
                      "ci": "paired bootstrap 95% of mean-over-targets MCC difference > 0"},
        "a1_reference": a1,
        "seeds": folds[targets[0]]["seeds"], "n_boot": folds[targets[0]]["n_boot"],
    }
    any_pass = [k for k, g in gates.items() if g["pass"]]
    summary["verdict"] = ("PASS: " + ", ".join(any_pass)) if any_pass else "FAIL: no arm passes the gate"
    (out_dir / "metrics.json").write_text(json.dumps(
        {**summary, "folds": folds}, indent=1, default=str) + "\n")
    (out_dir / "summary.md").write_text(_markdown(summary) + "\n")
    ctx.results["summary"] = summary["verdict"]
    return summary


def _markdown(s: dict[str, Any]) -> str:
    lines = ["| model/arm | " + " | ".join(f"{t} MCC" for t in s["targets"])
             + " | mean MCC | worst MCC | gate |", "|---" * (len(s["targets"]) + 4) + "|"]
    for key, row in s["table"].items():
        gate = s["gates"].get(key, {}).get("pass", "ref") if key.split("/")[1] != "a" else "ref"
        cells = [f"{row['targets'][t]['mcc']['mean']:.3f} +/- {row['targets'][t]['mcc']['sd']:.3f}"
                 for t in s["targets"]]
        lines.append(f"| {key} | " + " | ".join(cells)
                     + f" | {row['mean_mcc']:.3f} | {row['worst_mcc']:.3f} ({row['worst_target']})"
                     + f" | {gate} |")
    lines.append("")
    lines.append(f"Verdict: {s['verdict']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    if args.mode == "stage":
        return stage(ctx, args)
    if args.out_dir is None:
        raise SystemExit("--out-dir is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "aggregate":
        aggregate(ctx, args, args.out_dir)
        return True
    if args.data_dir is None or args.target is None:
        raise SystemExit("fold mode needs --data-dir and --target")
    result = run_fold(ctx, args, args.data_dir, args.out_dir, args.target)
    path = args.out_dir / f"fold_{args.target}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1, default=str) + "\n")
    tmp.replace(path)
    ctx.results["fold"] = {"target": args.target, "runs": len(result["runs"]),
                           "seconds": result["seconds"], "file": str(path)}
    return True
