"""CPU URL baselines on the url_domain and url_temporal splits.

Models (trained on the run's train rows, thresholds chosen on its cal rows):
- lgbm_features: LightGBM on the numeric URL feature columns of urls.parquet.
- char_lr_url: char (3-6) TF-IDF on the raw URL + logistic regression.

Metrics come from the shared protocol with URL-style operating points: TPR at FPR 0.01%, 0.1% and
1% (URLTran-style). Each figure carries a ``resolvable`` flag. A 0.01% FPR needs at least
100k benign URLs to resolve, so a smaller test set reports the step size, not a measurement.
Test negatives are grouped by ``source`` for the breakdown.

Flags go through ``kaggle/launch.sh url_baselines --job-args="..."``:
  --data-root PATH          data root (default: $RAKSHAK_DATA, /kaggle/input/rakshak-data,
                            then data/processed, then a search of /kaggle/input)
  --splits GLOB...          split-file stems (default: url_domain url_temporal)
  --models NAME...          subset of lgbm_features char_lr_url
  --seed INT                (default 0)
  --n-boot INT              bootstrap resamples (default 1000)
  --fpr-targets FLOAT...    (default 0.0001 0.001 0.01)
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
from _common import env_report, log_input_tree, seed_everything, timed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rakshak.eval import data as D
from rakshak.eval import report
from rakshak.eval.protocol import evaluate_scores

MODELS = ("lgbm_features", "char_lr_url")
DEFAULT_SPLITS = ("url_domain", "url_temporal")
DEFAULT_FPR_TARGETS = (0.0001, 0.001, 0.01)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="url_baselines")
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--fpr-targets", nargs="*", type=float, default=list(DEFAULT_FPR_TARGETS))
    return p.parse_args(argv or [])


def fit_and_score(name: str, seed: int, feats: np.ndarray, url_text: np.ndarray,
                  labels: np.ndarray, tr: np.ndarray, ca: np.ndarray, te: np.ndarray):
    """Fit on the train rows. Return the scam probabilities for the cal and test rows."""
    if name == "lgbm_features":
        model = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.05, num_leaves=127, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
            random_state=seed, verbose=-1,
        ).fit(feats[tr], labels[tr])
        return model.predict_proba(feats[ca])[:, 1], model.predict_proba(feats[te])[:, 1]
    if name == "char_lr_url":
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=3, sublinear_tf=True,
                              max_features=1_000_000, dtype=np.float32)
        x_train = vec.fit_transform(url_text[tr])
        model = LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced",
                                   max_iter=1000).fit(x_train, labels[tr])
        return (model.predict_proba(vec.transform(url_text[ca]))[:, 1],
                model.predict_proba(vec.transform(url_text[te]))[:, 1])
    raise KeyError(f"unknown model {name!r}")


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seed_everything(args.seed)
    env = env_report()
    ctx.log(f"env: {env}")
    try:
        root = D.resolve_data_root(args.data_root, marker=D.URL_FILE)
    except FileNotFoundError:
        log_input_tree(ctx.log)
        raise
    ctx.log(f"data root: {root}")
    urls = D.load_urls(root)
    labels = urls["label"].to_numpy().astype(np.int64)
    url_text = urls["url"].to_numpy(dtype=object)
    feature_cols = D.url_feature_columns(urls)
    feats = urls[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    ctx.log(f"{len(urls)} urls, {len(feature_cols)} numeric features: {feature_cols}")
    runs = D.load_runs(root, urls, patterns=args.splits, labels=labels, mask=None)

    timings: dict[str, float] = {}
    splits: dict[str, dict] = {}
    for run_ in runs:
        tr, ca, te = run_.train, run_.cal, run_.test
        ctx.log(f"run {run_.name}: {run_.notes['sizes']}")
        groups = D.group_values(urls, te, columns=("source",))
        block: dict[str, dict] = {}
        for name in args.models:
            with timed(f"{run_.name}|{name}|fit_s", timings):
                s_cal, s_test = fit_and_score(name, args.seed, feats, url_text, labels, tr, ca, te)
            result = evaluate_scores(
                labels[ca], s_cal, labels[te], s_test,
                groups=groups, fpr_targets=tuple(args.fpr_targets),
                n_boot=args.n_boot, seed=args.seed,
            )
            block[name] = result
            op = result["operating_points"]["max_f1"]
            ctx.log(f"  {name}: F1 {op['f1']:.4f} PR-AUC {result['threshold_free']['pr_auc']:.4f}")
        splits[run_.name] = block
        report.write_report({"title": "URL baselines", "meta": {"data_root": str(root)},
                             "splits": splits}, ctx.work)

    meta = {"data_root": str(root), "seed": args.seed, "features": feature_cols, "env": env}
    report.write_report({"title": "URL baselines", "meta": meta, "splits": splits}, ctx.work)
    ctx.results["timings"] = timings
    ctx.results["summary"] = f"{len(runs)} URL runs, {len(args.models)} models"
    return True
