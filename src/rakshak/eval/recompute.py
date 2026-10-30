"""Recompute every metric and interval from saved predictions, without re-running any model.

Input: parquet files with one row per prediction, keyed by ``id`` and ``split``. Rows with split
``cal`` pick thresholds and rows with split ``test`` give the reported numbers. The score is a
``score`` or ``prob`` column, or ``logit_1 - logit_0`` through a sigmoid. Labels, cluster ids and
groups come from the data table (joined by id), and the evaluation mask is applied. A
``label_binary`` (or ``label``) column in the file is checked against the table.

Each file is one model. ``--compare A:B`` runs the paired test (delta, CI, p-value) for PR-AUC,
ROC-AUC, F1 at each model's calibration threshold, and recall at 1% FPR, on the same test rows.

    python -m rakshak.eval.recompute --scores 'logits_*.parquet' --out DIR \
        [--table sms|urls] [--bootstrap cluster|iid] [--n-boot 1000] [--seed 0] \
        [--compare A:B ...] [--data-root PATH]

Writes ``DIR/metrics.json`` and ``DIR/report.md``. Robustness is not recomputed here: it needs
the models.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.eval import data as D
from rakshak.eval.metrics import PAIRED_METRICS, paired_bootstrap_test
from rakshak.eval.protocol import evaluate_scores
from rakshak.eval.report import write_report

SPLIT_NAME = "recomputed"
TABLES = {
    # table name: (loader, label column, default cluster column)
    "sms": (D.load_sms, "label_binary", D.CLUSTER_COLUMNS["sms"]),
    "urls": (D.load_urls, "label", D.CLUSTER_COLUMNS["urls"]),
}


def _scores(df: pd.DataFrame) -> np.ndarray:
    if "score" in df.columns:
        return df["score"].to_numpy(dtype=np.float64)
    if "prob" in df.columns:
        return df["prob"].to_numpy(dtype=np.float64)
    if {"logit_0", "logit_1"} <= set(df.columns):
        z = df["logit_1"].to_numpy(dtype=np.float64) - df["logit_0"].to_numpy(dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("predictions need a score, prob, or logit_0/logit_1 column")


def load_predictions(path: Path, table: pd.DataFrame, label_col: str,
                     mask: np.ndarray) -> pd.DataFrame:
    """One row per prediction in the eval mask: id, split, score, table position, label."""
    df = pd.read_parquet(path)
    missing = {"id", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    ids = df["id"].astype(str).to_numpy()
    if len(set(zip(ids, df["split"].astype(str), strict=True))) != len(ids):
        raise ValueError(f"{path.name} has duplicate (id, split) rows")
    pos = pd.Index(table["id"]).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f"{path.name}: {int((pos < 0).sum())} ids are not in the data table")
    labels = table[label_col].to_numpy().astype(np.int64)[pos]
    for col in ("label_binary", "label"):
        if col in df.columns and not np.array_equal(df[col].to_numpy().astype(np.int64), labels):
            raise ValueError(f"{path.name}: the {col} column disagrees with the data table")
    frame = pd.DataFrame({
        "id": ids,
        "split": df["split"].astype(str).to_numpy(),
        "score": _scores(df),
        "pos": pos,
        "y": labels,
    })
    keep = mask[pos]
    return frame.loc[keep].reset_index(drop=True)


def _evaluate(frame: pd.DataFrame, table: pd.DataFrame, cluster_col: str | None,
              method: str, n_boot: int, seed: int) -> tuple[dict, dict]:
    cal = frame[frame["split"] == "cal"]
    test = frame[frame["split"] == "test"]
    if cal.empty or test.empty:
        raise ValueError("predictions need both cal and test rows")
    if not test["id"].is_unique:
        raise ValueError("test rows repeat an id")
    test_pos = test["pos"].to_numpy()
    clusters = None
    if method != "iid":
        if cluster_col is None:
            raise ValueError("the cluster bootstrap needs a cluster column")
        clusters = D.cluster_values(table, test_pos, cluster_col)
    result = evaluate_scores(
        cal["y"].to_numpy(), cal["score"].to_numpy(),
        test["y"].to_numpy(), test["score"].to_numpy(),
        groups=D.group_values(table, test_pos),
        n_boot=n_boot, seed=seed,
        cluster_ids=clusters,
        bootstrap_method=method,
    )
    return result, {"test": test, "clusters": clusters}


def _compare(name_a: str, name_b: str, runs: dict, n_boot: int, seed: int,
             method: str) -> dict:
    ra, rb = runs[name_a], runs[name_b]
    test_a = ra["test"].set_index("id")
    test_b = rb["test"].set_index("id").reindex(test_a.index)
    if test_b["score"].isna().any():
        raise ValueError(f"{name_a} and {name_b} do not score the same test ids")
    y = test_a["y"].to_numpy()
    if not np.array_equal(test_b["y"].to_numpy(), y):
        raise ValueError(f"{name_a} and {name_b} disagree on test labels")
    thr_a = ra["result"]["thresholds"]["max_f1"]["threshold"]
    thr_b = rb["result"]["thresholds"]["max_f1"]["threshold"]
    clusters = ra["clusters"] if method != "iid" else None
    out: dict = {}
    for metric in PAIRED_METRICS:
        out[metric] = paired_bootstrap_test(
            y, test_a["score"].to_numpy(), test_b["score"].to_numpy(),
            metric=metric, threshold_a=thr_a, threshold_b=thr_b,
            cluster_ids=clusters, method=method, n_boot=n_boot, seed=seed,
        )
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m rakshak.eval.recompute",
                                description=__doc__.split("\n")[0])
    p.add_argument("--scores", nargs="+", required=True,
                   help="prediction parquet files or glob patterns; each file is one model")
    p.add_argument("--out", required=True, help="output directory for metrics.json and report.md")
    p.add_argument("--table", choices=sorted(TABLES), default="sms")
    p.add_argument("--cluster-col", default=None,
                   help="cluster column in the table (default: campaign_id for sms, reg_domain "
                        "for urls)")
    p.add_argument("--bootstrap", choices=("cluster", "iid"), default="cluster")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", nargs="*", default=[], metavar="A:B",
                   help="paired tests between two model files, named by file stem")
    p.add_argument("--title", default="Recomputed evaluation")
    p.add_argument("--data-root", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files: list[Path] = []
    for pattern in args.scores:
        hits = sorted(glob.glob(pattern)) or [pattern]
        files += [Path(h) for h in hits]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"no such prediction files: {missing}")
    loader, label_col, default_cluster = TABLES[args.table]
    marker = D.SMS_FILE if args.table == "sms" else D.URL_FILE
    root = D.resolve_data_root(args.data_root, marker=marker)
    table = loader(root)
    mask = D.eval_mask(table)
    cluster_col = args.cluster_col or default_cluster

    runs: dict[str, dict] = {}
    for path in files:
        frame = load_predictions(path, table, label_col, mask)
        result, extra = _evaluate(frame, table, cluster_col, args.bootstrap, args.n_boot, args.seed)
        runs[path.stem] = {"result": result, **extra}
        op = result["operating_points"]["max_f1"]
        ci = result["bootstrap"].get("pr_auc")
        ci_txt = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "[n/a]"
        print(f"{path.stem}: F1 {op['f1']:.4f}  PR-AUC {result['threshold_free']['pr_auc']:.4f} "
              f"{ci_txt} ({result['bootstrap']['method']})")

    comparisons: dict[str, dict] = {}
    for spec in args.compare:
        if ":" not in spec:
            raise ValueError(f"--compare expects A:B, got {spec!r}")
        a, b = spec.split(":", 1)
        for name in (a, b):
            if name not in runs:
                raise KeyError(f"--compare names {name!r}, which is not one of {sorted(runs)}")
        comparisons[f"{a} vs {b}"] = _compare(a, b, runs, args.n_boot, args.seed, args.bootstrap)

    payload = {
        "title": args.title,
        "meta": {"data_root": str(root), "table": args.table, "cluster_column": cluster_col,
                 "bootstrap": args.bootstrap, "n_boot": args.n_boot, "seed": args.seed,
                 "prediction_files": [f.name for f in files]},
        "splits": {SPLIT_NAME: {name: run["result"] for name, run in runs.items()}},
        "comparisons": comparisons,
    }
    paths = write_report(payload, Path(args.out))
    print(f"wrote {paths['metrics']} and {paths['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
