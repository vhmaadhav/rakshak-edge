"""CPU baselines for the SMS detector, run on every split file in data/splits.

Models (each trained on the run's train rows, calibrated or thresholded on its cal rows):
- rules_lr: logistic regression on the summed rule weight. Fit on cal, since it has one feature.
- char_lr: char_wb TF-IDF (1-5) + logistic regression.
- svc_cal: word (1-2) + char_wb (1-5) TF-IDF + LinearSVC, sigmoid-calibrated on cal only.
- cnb_cal: ComplementNB on char_wb TF-IDF, sigmoid-calibrated on cal only.
- lgbm_svd_rules: LightGBM on 256 SVD components of char TF-IDF plus rule features.

Each model runs twice: "raw" (the message as received) and "canon" (rakshak.canonicalize first).
Rule features come from the canonical form in "canon" and from a plain Canon in "raw".

Output: metrics.json and report.md in the work directory (per split and per model/variant, with
bootstrap CIs and robustness). The best model is saved as a joblib bundle. Selection uses the
mean cal PR-AUC across runs, so test is never used for selection. The bundle is the fit on the
campaign split (first run if there is none). It stores sklearn objects and plain values. Its
scorer is ``Detector.score`` in this file.

Flags go through ``kaggle/launch.sh sms_baselines --job-args="..."``:
  --data-root PATH          data root (default: $RAKSHAK_DATA, /kaggle/input/rakshak-data,
                            then data/processed, then a search of /kaggle/input)
  --splits GLOB...          split-file stems to run (default: campaign temporal
                            source_holdout lang_shift legacy_*)
  --models NAME...          subset of rules_lr char_lr svc_cal cnb_cal lgbm_svd_rules
  --variants raw canon     which input variants
  --seed INT                bootstrap and split seed (default 0)
  --n-boot INT             bootstrap resamples (default 1000; campaign-clustered bootstrap)
  --no-robustness           skip the attack harness
  --robust-seeds INT...     attack seeds (default 0 1 2)
  --max-robust-pos INT      cap the scam texts per run in the attack harness (default all)
  --latency-n INT           messages timed for CPU latency (default 1000)

The bundle, metrics.json and report.md are written to the work directory (/kaggle/working).
"""

from __future__ import annotations

import argparse
import os
import time

import joblib
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from rakshak.canonicalize import Canon, canonicalize, extract_urls
from rakshak.eval import data as D
from rakshak.eval import report
from rakshak.eval.protocol import evaluate_scores
from rakshak.eval.robustness import ATTACK_SEEDS, robustness_report
from rakshak.rules import EVIDENCE_TYPES, run_rules

MODELS = ("rules_lr", "char_lr", "svc_cal", "cnb_cal", "lgbm_svd_rules")
VARIANTS = ("raw", "canon")
DEFAULT_SPLITS = ("campaign", "temporal", "source_holdout", "lang_shift", "legacy_*")
PRIMARY_SPLIT = "campaign"
N_TYPES = len(EVIDENCE_TYPES)
TOTAL_COL = 2 * N_TYPES  # rule matrix layout: max weight per type, count per type, total weight
SVD_COMPONENTS = 256


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sms_baselines")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=MODELS)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--no-robustness", action="store_true")
    parser.add_argument("--robust-seeds", nargs="*", type=int, default=list(ATTACK_SEEDS))
    parser.add_argument("--max-robust-pos", type=int, default=None)
    parser.add_argument("--latency-n", type=int, default=1000)
    return parser.parse_args(argv or [])


def prep(text: str, variant: str, cached: bool = True) -> str:
    """The text a TF-IDF model sees: the message itself, or its canonical form."""
    if variant == "raw":
        return text
    return D.canon_of(text).normalized if cached else canonicalize(text).normalized


def canon_for_rules(text: str, variant: str, cached: bool = True) -> Canon:
    if variant == "canon":
        return D.canon_of(text) if cached else canonicalize(text)
    return Canon(original=text, normalized=text, urls=extract_urls(text))


def rule_matrix(texts, variant: str, cached: bool = True) -> np.ndarray:
    """Per message: max weight per evidence type, count per type, total weight, evidence count."""
    index = {t: i for i, t in enumerate(EVIDENCE_TYPES)}
    out = np.zeros((len(texts), 2 * N_TYPES + 2), dtype=np.float32)
    for row, text in enumerate(texts):
        evidence = run_rules(canon_for_rules(text, variant, cached))
        for ev in evidence:
            i = index[ev.type]
            out[row, i] = max(out[row, i], ev.weight)
            out[row, N_TYPES + i] += 1
            out[row, TOTAL_COL] += ev.weight
        out[row, TOTAL_COL + 1] = len(evidence)
    return out


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), min_df=2,
                           sublinear_tf=True, max_features=500_000, dtype=np.float32)


class Detector:
    """One baseline on one input variant. ``fit`` takes raw texts. ``score`` takes raw texts and
    returns scam probabilities."""

    def __init__(self, name: str, variant: str):
        self.name = name
        self.variant = variant
        self.vec = None
        self.model = None
        self.svd = None

    def _rules(self, texts, cached: bool) -> np.ndarray:
        return rule_matrix(texts, self.variant, cached)

    def fit(self, texts, y, cal_texts, cal_y) -> Detector:
        train = [prep(t, self.variant) for t in texts]
        if self.name == "rules_lr":
            x_cal = self._rules(cal_texts, True)[:, [TOTAL_COL]]
            self.model = LogisticRegression(C=1.0, class_weight="balanced").fit(x_cal, cal_y)
            return self
        if self.name == "char_lr":
            self.vec = _char_vectorizer().fit(train)
            self.model = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced")
            self.model.fit(self.vec.transform(train), y)
            return self
        if self.name == "svc_cal":
            self.vec = FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                         sublinear_tf=True, max_features=300_000,
                                         token_pattern=r"(?u)\b\w+\b", dtype=np.float32)),
                ("char", _char_vectorizer()),
            ]).fit(train)
            svc = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000)
            svc.fit(self.vec.transform(train), y)
            cal = [prep(t, self.variant) for t in cal_texts]
            self.model = CalibratedClassifierCV(FrozenEstimator(svc), method="sigmoid")
            self.model.fit(self.vec.transform(cal), cal_y)
            return self
        if self.name == "cnb_cal":
            self.vec = _char_vectorizer().fit(train)
            nb = ComplementNB(alpha=0.3).fit(self.vec.transform(train), y)
            cal = [prep(t, self.variant) for t in cal_texts]
            self.model = CalibratedClassifierCV(FrozenEstimator(nb), method="sigmoid")
            self.model.fit(self.vec.transform(cal), cal_y)
            return self
        if self.name == "lgbm_svd_rules":
            self.vec = _char_vectorizer().fit(train)
            x_char = self.vec.transform(train)
            self.svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=0).fit(x_char)
            x = np.hstack([self.svd.transform(x_char), self._rules(texts, True)])
            self.model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                random_state=0, verbose=-1,
            ).fit(x, y)
            return self
        raise KeyError(f"unknown model {self.name!r}")

    def score(self, texts, cached: bool = True) -> np.ndarray:
        texts = list(texts)
        if self.name == "rules_lr":
            return self.model.predict_proba(self._rules(texts, cached)[:, [TOTAL_COL]])[:, 1]
        prepped = [prep(t, self.variant, cached) for t in texts]
        if self.name in ("char_lr", "svc_cal", "cnb_cal"):
            return self.model.predict_proba(self.vec.transform(prepped))[:, 1]
        x_char = self.vec.transform(prepped)
        x = np.hstack([self.svd.transform(x_char), self._rules(texts, cached)])
        return self.model.predict_proba(x)[:, 1]

    def state(self) -> dict:
        """Picklable components. Score them with ``Detector.score`` in this file."""
        return {"name": self.name, "variant": self.variant, "vec": self.vec,
                "svd": self.svd, "model": self.model,
                "scorer": "kaggle/jobs/sms_baselines.py:Detector.score"}


def measure_latency(det: Detector, texts, n: int) -> dict:
    """Per-message CPU latency on one thread, with no caches (canonicalise, rules, featurise,
    predict)."""
    if len(texts) == 0:
        return {}
    sample = [texts[i % len(texts)] for i in range(n)]
    times = []
    with threadpool_limits(limits=1):
        for text in sample:
            start = time.perf_counter()
            det.score([text], cached=False)
            times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times)
    return {"n": n, "threads": 1, "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)), "mean_ms": float(arr.mean())}


def fit_and_evaluate(run_, name: str, variant: str, texts, labels, args, groups, clusters,
                     pos_test, neg_test, timings: dict) -> dict:
    """One (split, model, variant) unit: fit on train, threshold on cal, score test, then the
    attack harness. Returns the protocol result with the fitted detector and max-F1 threshold."""
    tr, ca, te = run_.train, run_.cal, run_.test
    key = f"{name}/{variant}"
    with timed(f"{run_.name}|{key}|fit_s", timings):
        det = Detector(name, variant).fit(texts[tr], labels[tr], texts[ca], labels[ca])
    s_cal = det.score(texts[ca])
    s_test = det.score(texts[te])
    result = evaluate_scores(
        labels[ca], s_cal, labels[te], s_test,
        groups=groups, n_boot=args.n_boot, seed=args.seed, cluster_ids=clusters,
    )
    thr = result["thresholds"]["max_f1"]["threshold"]
    if not args.no_robustness:
        with timed(f"{run_.name}|{key}|robust_s", timings):
            result["robustness"] = robustness_report(
                lambda ts, d=det: d.score(ts, cached=False),
                pos_test, neg_test, thr,
                seeds=args.robust_seeds, max_positives=args.max_robust_pos,
                subsample_seed=args.seed,
            )
    return {"result": result, "det": det, "thr": thr}


def _checkpoint(ctx, root, splits: dict) -> None:
    report.write_report({"title": "SMS baselines", "meta": {"data_root": str(root)},
                         "splits": splits}, ctx.work)


def run(ctx) -> bool:
    args = parse_args(ctx.argv)
    seed_everything(args.seed)
    env = env_report()
    ctx.log(f"env: {env}")
    try:
        root = D.resolve_data_root(args.data_root)
    except FileNotFoundError:
        log_input_tree(ctx.log)
        raise
    ctx.log(f"data root: {root}")
    sms = D.load_sms(root)
    labels = sms["label_binary"].to_numpy().astype(np.int64)
    texts = sms["text"].to_numpy(dtype=object)
    mask = D.eval_mask(sms)
    runs = D.load_runs(root, sms, patterns=args.splits, labels=labels, mask=mask)
    ctx.log(f"{len(runs)} runs: {[r.name for r in runs]}")

    timings: dict[str, float] = {}
    canon_agreement = None
    if "text_canon" in sms.columns:
        sample = np.arange(min(2000, len(sms)))
        same = [D.canon_text(texts[i]) == sms["text_canon"].iat[i] for i in sample]
        canon_agreement = float(np.mean(same))
        ctx.log(f"runtime canonicalize agrees with text_canon on {canon_agreement:.4f} of rows")

    units = Units(ctx)
    splits: dict[str, dict] = {}
    primary_name = next((r.name for r in runs if r.name.split(":")[0] == PRIMARY_SPLIT),
                        runs[0].name)
    primary_run = next(r for r in runs if r.name == primary_name)
    fitted: dict[tuple[str, str], Detector] = {}
    primary_thresholds: dict[tuple[str, str], float] = {}
    for run_ in runs:
        ctx.log(f"run {run_.name}: sizes {run_.notes['sizes']}")
        reason = single_class_reason(labels[run_.test], run_.name)
        if reason:
            record_skip(ctx, run_.name, reason)
            continue
        groups = D.group_values(sms, run_.test)
        clusters = D.cluster_values(sms, run_.test, "campaign_id")
        block: dict[str, dict] = splits.setdefault(run_.name, {})
        pos_test = texts[run_.test][labels[run_.test] == 1].tolist()
        neg_test = texts[run_.test][labels[run_.test] == 0].tolist()
        for variant in args.variants:
            for name in args.models:
                key = f"{name}/{variant}"
                out = units.run({"split": run_.name, "model": name, "variant": variant},
                                fit_and_evaluate, run_, name, variant, texts, labels, args,
                                groups, clusters, pos_test, neg_test, timings)
                if out is not None:
                    block[key] = out["result"]
                    op = out["result"]["operating_points"]["max_f1"]
                    ctx.log(f"  {key}: F1 {op['f1']:.4f} "
                            f"PR-AUC {out['result']['threshold_free']['pr_auc']:.4f}")
                    if run_.name == primary_name:
                        fitted[(name, variant)] = out["det"]
                        primary_thresholds[(name, variant)] = out["thr"]
                _checkpoint(ctx, root, splits)

    # Selection on cal only: mean cal PR-AUC across the runs whose unit finished.
    cal_scores: dict[str, list[float]] = {}
    for block in splits.values():
        for key, result in block.items():
            cal_scores.setdefault(key, []).append(result["cal_threshold_free"]["pr_auc"])
    mean_cal = {key: float(np.mean(values)) for key, values in cal_scores.items()}
    ranking = sorted(((mean_cal[k], k) for k in mean_cal), reverse=True)
    ctx.log(f"selection (mean cal PR-AUC): {ranking[:5]}")
    # The bundle needs the fit from the primary run, so only those models are candidates.
    best_key = next((k for _, k in ranking if tuple(k.split("/")) in fitted), None)

    latency: dict = {}
    size = None
    if best_key is None:
        ctx.log("no selected model has a primary-run fit; the model bundle is not written")
    else:
        best_name, best_variant = best_key.split("/")
        bundle_path = ctx.work / "best_model.joblib"
        det = fitted[(best_name, best_variant)]
        latency = measure_latency(det, texts[primary_run.test], args.latency_n)
        bundle = {"model": best_key, "run": primary_name,
                  "threshold": primary_thresholds[(best_name, best_variant)],
                  "train_rows": int(len(primary_run.train)), "state": det.state()}
        joblib.dump(bundle, bundle_path, compress=3)
        size = os.path.getsize(bundle_path)
        ctx.log(f"best model {best_key}: {size} bytes, latency {latency}")

    payload = {
        "title": "SMS baselines",
        "meta": {
            "data_root": str(root),
            "runs": len(runs),
            "seed": args.seed,
            "best_model": best_key,
            "best_mean_cal_pr_auc": mean_cal.get(best_key) if best_key else None,
            "best_bundle_bytes": size,
            "best_cpu_latency": latency,
            "canon_agreement_with_text_canon": canon_agreement,
            "env": env,
        },
        "splits": splits,
    }
    report.write_report(payload, ctx.work)
    ctx.results["timings"] = timings
    best_text = f"{mean_cal[best_key]:.3f}" if best_key else "n/a"
    ctx.results["summary"] = (f"{len(runs)} runs, {units.ok} units ok, {units.failed} failed; "
                              f"best {best_key or 'none'} cal-PR-AUC {best_text}")
    return units.status()
