"""CPU URL baselines on the url_domain and url_temporal splits.

Models (trained on the run's train rows, thresholds chosen on its cal rows):
- lgbm_features: LightGBM on the numeric URL feature columns of urls.parquet.
- char_lr_url: char (3-6) TF-IDF on the raw URL + logistic regression.

Metrics come from the shared protocol with URL-style operating points: TPR at FPR 0.01%, 0.1% and
1% (URLTran-style). Each figure carries a ``resolvable`` flag. A 0.01% FPR needs at least
100k benign URLs to resolve, so a smaller test set reports the step size, not a measurement.
Test negatives are grouped by ``source`` for the breakdown. Confidence intervals are campaign
(registered-domain) clustered by default: the cluster ids are ``reg_domain``.

PhreshPhish protocol (arXiv 2507.10854 v2; docs/SOTA_TARGETS.md section 5), run after the splits
unless ``--no-phreshphish``. The models are trained on the official PhreshPhish train split (no
cal split, no thresholds). The protocol code lives in rakshak.eval.phreshphish and is shared with
url_student. Every benchmark subset (benign list plus one phish list, 975 in total) is scored, and
average precision (AP) and precision at 90% recall (P@R=0.90) are averaged per base rate, reported
as mean and sample std (ddof=1) over the subsets. Full-benchmark (24%) and full-test-split AP and
P@R are reported as well. The report.md tables carry the paper's GTE row (URL and HTML) for
reference only: GTE reads HTML, so it is not a like-for-like URL-only comparison.

Flags go through ``kaggle/launch.sh url_baselines --job-args="..."``:
  --data-root PATH          data root (default: $RAKSHAK_DATA, /kaggle/input/rakshak-data,
                            then data/processed, then a search of /kaggle/input)
  --splits GLOB...          split-file stems (default: url_domain url_temporal)
  --models NAME...          subset of lgbm_features char_lr_url
  --seed INT                (default 0)
  --n-boot INT              bootstrap resamples (default 1000)
  --fpr-targets FLOAT...    (default 0.0001 0.001 0.01)
  --no-phreshphish          skip the PhreshPhish protocol section
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
from _common import (
    Units,
    env_report,
    log_input_tree,
    record_skip,
    seed_everything,
    single_class_reason,
    timed,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rakshak.eval import data as D
from rakshak.eval import report
from rakshak.eval.phreshphish import (  # phreshphish_subsets re-exported for the tests
    PHRESH_FAMILY,
    PHRESH_FILE,
    load_phreshphish,
    paper_reference,
    phreshphish_subsets,  # noqa: F401
    protocol_section,
    protocol_tables,
)
from rakshak.eval.protocol import evaluate_scores

MODELS = ("lgbm_features", "char_lr_url")
DEFAULT_SPLITS = ("url_domain", "url_temporal")
DEFAULT_FPR_TARGETS = (0.0001, 0.001, 0.01)
CLUSTER_COLUMN = "reg_domain"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="url_baselines")
    p.add_argument("--data-root", default=None)
    p.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--fpr-targets", nargs="*", type=float, default=list(DEFAULT_FPR_TARGETS))
    p.add_argument("--no-phreshphish", action="store_true")
    return p.parse_args(argv or [])


def fit_models(
    name: str,
    seed: int,
    feats: np.ndarray,
    url_text: np.ndarray,
    labels: np.ndarray,
    tr: np.ndarray,
):
    """Fit on the train rows. Returns a scorer: ``score(rows) -> scam probabilities``."""
    if name == "lgbm_features":
        model = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=127,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        ).fit(feats[tr], labels[tr])
        return lambda rows: model.predict_proba(feats[rows])[:, 1]
    if name == "char_lr_url":
        vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            min_df=3,
            sublinear_tf=True,
            max_features=1_000_000,
            dtype=np.float32,
        )
        x_train = vec.fit_transform(url_text[tr])
        model = LogisticRegression(
            C=4.0, solver="liblinear", class_weight="balanced", max_iter=1000
        ).fit(x_train, labels[tr])
        return lambda rows: model.predict_proba(vec.transform(url_text[rows]))[:, 1]
    raise KeyError(f"unknown model {name!r}")


def fit_and_score(
    name: str,
    seed: int,
    feats: np.ndarray,
    url_text: np.ndarray,
    labels: np.ndarray,
    tr: np.ndarray,
    ca: np.ndarray,
    te: np.ndarray,
):
    """Fit on the train rows. Return the scam probabilities for the cal and test rows."""
    score = fit_models(name, seed, feats, url_text, labels, tr)
    return score(ca), score(te)


def evaluate_unit(
    split: str,
    name: str,
    args,
    feats,
    url_text,
    labels,
    tr,
    ca,
    te,
    groups,
    clusters,
    timings: dict,
) -> dict:
    """One (split, model) unit: fit, pick thresholds on cal, report protocol metrics on test."""
    with timed(f"{split}|{name}|fit_s", timings):
        s_cal, s_test = fit_and_score(name, args.seed, feats, url_text, labels, tr, ca, te)
    return evaluate_scores(
        labels[ca],
        s_cal,
        labels[te],
        s_test,
        groups=groups,
        fpr_targets=tuple(args.fpr_targets),
        n_boot=args.n_boot,
        seed=args.seed,
        cluster_ids=clusters,
    )


def phreshphish_unit(
    name: str, args, feats, url_text, labels, tr, te, subsets, bench_full, timings: dict
) -> dict:
    """One model on the PhreshPhish protocol: fit on the official train split, score the test
    split once, then average the subset metrics per base rate."""
    with timed(f"url_phreshphish|{name}|fit_s", timings):
        score = fit_models(name, args.seed, feats, url_text, labels, tr)
        scores = np.full(len(labels), np.nan)
        scores[te] = score(te)
    return protocol_section(labels, scores, te, bench_full, subsets)


def run_phreshphish(ctx, units, root: Path, args, timings: dict) -> dict | None:
    """The PhreshPhish protocol section. Skips with a recorded reason when the file is absent."""
    path = root / PHRESH_FILE
    if not path.is_file():
        record_skip(ctx, PHRESH_FAMILY, f"{PHRESH_FILE} not found under {root}")
        return None
    data = load_phreshphish(root)
    df = data.df
    feature_cols = data.feature_cols
    feats = df[feature_cols].fillna(0.0).to_numpy(np.float32)
    url_text = df["url"].to_numpy(dtype=object)
    ctx.log(
        f"PhreshPhish: {len(data.train)} train, {len(data.test)} test, "
        f"{len(data.bench_full)} full-benchmark rows, {len(data.subsets)} subsets, "
        f"{len(feature_cols)} features"
    )
    section: dict = {
        "rows": {
            "train": int(len(data.train)),
            "test": int(len(data.test)),
            "benchmark_full": int(len(data.bench_full)),
            "subsets": len(data.subsets),
        },
        "features": feature_cols,
        "paper_v2_gte_reference": paper_reference(),
        "models": {},
    }
    for name in args.models:
        result = units.run(
            {"split": PHRESH_FAMILY, "model": name},
            phreshphish_unit,
            name,
            args,
            feats,
            url_text,
            data.labels,
            data.train,
            data.test,
            data.subsets,
            data.bench_full,
            timings,
        )
        if result is not None:
            section["models"][name] = result
            rates = result["per_rate"]
            ctx.log(
                f"  PhreshPhish {name}: "
                + ", ".join(f"{r} AP {v['ap_mean']:.4f}" for r, v in rates.items())
            )
    return section


def _checkpoint(ctx, meta: dict, splits: dict, phresh: dict | None = None) -> None:
    payload = {"title": "URL baselines", "meta": meta, "splits": splits}
    if phresh is not None:
        payload["phreshphish"] = phresh
        payload["tables"] = protocol_tables(phresh)
    report.write_report(payload, ctx.work)


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

    units = Units(ctx)
    timings: dict[str, float] = {}
    splits: dict[str, dict] = {}
    meta = {"data_root": str(root), "seed": args.seed, "features": feature_cols, "env": env}
    for run_ in runs:
        tr, ca, te = run_.train, run_.cal, run_.test
        ctx.log(f"run {run_.name}: {run_.notes['sizes']}")
        reason = single_class_reason(labels[te], run_.name)
        if reason:
            record_skip(ctx, run_.name, reason)
            continue
        groups = D.group_values(urls, te, columns=("source",))
        clusters = D.cluster_values(urls, te, CLUSTER_COLUMN)
        block: dict[str, dict] = splits.setdefault(run_.name, {})
        for name in args.models:
            result = units.run(
                {"split": run_.name, "model": name},
                evaluate_unit,
                run_.name,
                name,
                args,
                feats,
                url_text,
                labels,
                tr,
                ca,
                te,
                groups,
                clusters,
                timings,
            )
            if result is not None:
                block[name] = result
                op = result["operating_points"]["max_f1"]
                pr_auc = result["threshold_free"]["pr_auc"]
                ctx.log(f"  {name}: F1 {op['f1']:.4f} PR-AUC {pr_auc:.4f}")
            _checkpoint(ctx, meta, splits)

    _checkpoint(ctx, meta, splits)
    phresh = None
    if not args.no_phreshphish:
        phresh = run_phreshphish(ctx, units, root, args, timings)
        if phresh is not None:
            ctx.results["phreshphish_protocol"] = phresh
    _checkpoint(ctx, meta, splits, phresh)
    ctx.results["timings"] = timings
    ctx.results["summary"] = (
        f"{len(runs)} URL runs, {units.ok} units ok, {units.failed} failed "
        f"({len(args.models)} models)"
    )
    return units.status()
