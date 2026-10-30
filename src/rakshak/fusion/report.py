"""Held-out fusion comparisons and campaign-clustered uncertainty intervals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rakshak.eval import metrics as M
from rakshak.eval.report import sanitize
from rakshak.fusion.calibrate import calibration_metrics, labelled
from rakshak.fusion.selective import (
    SelectiveThresholds,
    campaign_size_buckets,
    conformal_fpr,
    group_coverage,
    risk_coverage,
    search_thresholds,
    selective_metrics,
)

ALPHAS = (0.005, 0.01, 0.02)


def clustered_intervals(y, prob, *, cluster_ids, thresholds: dict,
                        n_boot: int = 1000, seed: int = 0) -> dict:
    """Reuse eval's stratified campaign draws; hold all fitted objects fixed.

    Intervals describe test-population sampling, not fitting/selection uncertainty.
    Undefined class risks are omitted per metric, with valid draw counts reported.
    """
    p, y = labelled(prob, y, probability=True)
    if cluster_ids is None or n_boot < 0:
        raise ValueError("campaign IDs and nonnegative n_boot required")
    index = M._cluster_index(y, cluster_ids)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = M._cluster_draw(index, rng)
        yy, pp = y[idx], p[idx]
        cal = calibration_metrics(yy, pp)
        stats = {k: cal[k] for k in ("brier", "ece_width", "ece_mass")}
        stats["aurc"] = risk_coverage(yy, pp)["aurc"]
        stats["pr_auc"] = M.pr_auc(yy, pp)
        for name, threshold in thresholds.items():
            met = selective_metrics(yy, pp, threshold)
            stats.update({f"{name}/{k}": met[k]
                          for k in ("coverage", "risk", "fpr", "miss_rate")})
        for key, value in stats.items():
            if np.isfinite(value):
                values.setdefault(key, []).append(float(value))
    return {"method": M.BOOTSTRAP_CLUSTER, "n_clusters": index["n_clusters"],
            "n_boot": n_boot, "seed": seed, "confidence": 0.95,
            "intervals": {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in values.items()},
            "n_valid": {k: len(v) for k, v in values.items()}}


def evaluate_fusion(y_cal, p_cal, y_test, p_before, p_after, *, cluster_ids,
                    groups: dict | None = None, crc_y=None, crc_prob=None,
                    beta: float = 0.05, n_boot: int = 1000, seed: int = 0) -> dict:
    """Cal here means threshold-only holdout; CRC uses another independent holdout."""
    p, y = labelled(p_after, y_test, probability=True)
    before, _ = labelled(p_before, y, probability=True)
    groups = dict(groups or {})
    groups["campaign_size_bucket"] = campaign_size_buckets(cluster_ids)
    selected, fixed = {}, {}
    for alpha in ALPHAS:
        key = f"{alpha:g}"
        entry = search_thresholds(y_cal, p_cal, alpha=alpha, beta=beta)
        if entry["status"] == "ok":
            threshold = SelectiveThresholds.from_dict(entry["thresholds"])
            fixed[key] = threshold
            entry["test"] = selective_metrics(y, p, threshold)
            entry["groups"] = group_coverage(p, threshold, groups)
            if crc_y is not None and crc_prob is not None:
                crc = conformal_fpr(crc_y, crc_prob, t_safe=threshold.t_safe, alpha=alpha)
                if crc["status"] == "ok":
                    ct = SelectiveThresholds.from_dict(crc["thresholds"])
                    fixed[f"crc_{key}"] = ct
                    crc["test"] = selective_metrics(y, p, ct)
                    crc["groups"] = group_coverage(p, ct, groups)
                entry["crc"] = crc
        selected[key] = entry
    return {"n_test": len(y), "n_threshold_cal": len(y_cal),
            "calibration_before": calibration_metrics(y, before),
            "calibration_after": calibration_metrics(y, p),
            "pr_auc": M.pr_auc(y, p), "roc_auc": M.roc_auc(y, p),
            "selective": selected, "risk_coverage": risk_coverage(y, p),
            "bootstrap_before": clustered_intervals(
                y, before, cluster_ids=cluster_ids, thresholds={}, n_boot=n_boot, seed=seed),
            "bootstrap": clustered_intervals(y, p, cluster_ids=cluster_ids, thresholds=fixed,
                                             n_boot=n_boot, seed=seed)}


def render_markdown(payload: dict) -> str:
    lines = ["# Fusion evaluation", "",
             "Fusion, probability calibration, empirical thresholds and CRC use four disjoint "
             "campaign subsets of cal. Test is used only for evaluation. Scores must originate "
             "from experts frozen independently of cal and test.", "",
             "FPR = FP / auto-decided benign; miss rate = FN / auto-decided scams. "
             "Undefined risks are null, not zero. Empirical constraints are not guarantees.", "",
             "CRC uses (sum loss + 1)/(n_benign + 1), with loss FP + alpha*abstention. "
             "Its expected marginal selective FPR bound requires exchangeable benign "
             "observations and an independently fixed scorer/safe threshold. No guarantee "
             "under campaign shift, within-campaign dependence, or conditioning on successful "
             "selection. CRC does not guarantee the scam miss constraint.", "",
             "95% percentile CIs resample campaigns using existing stratified evaluation "
             "helpers, holding fitted objects fixed. Valid resample counts are in JSON. "
             "Risk can decrease as coverage increases; it is not artificially monotonised.", ""]
    for name, block in payload.get("runs", {}).items():
        lines += [f"## {name}", "", "| Model | PR-AUC | Brier before → after | "
                  "ECE width before → after | ECE mass before → after | AURC |",
                  "|---|---|---|---|---|---|"]
        for model, result in block.items():
            before, after = result["calibration_before"], result["calibration_after"]
            cells = [f"{before[k]:.5f} → {after[k]:.5f}"
                     for k in ("brier", "ece_width", "ece_mass")]
            lines.append(f"| {model} | {result['pr_auc']:.5f} | " + " | ".join(cells)
                         + f" | {result['risk_coverage']['aurc']:.5f} |")
        for model, result in block.items():
            lines += ["", f"### {model}", "",
                      "| alpha | Variant | Status | Thresholds | Test coverage | Test FPR | "
                      "Test miss |", "|---|---|---|---|---|---|---|"]
            for alpha, entry in result["selective"].items():
                for variant, item in [("empirical", entry), ("CRC", entry.get("crc"))]:
                    if item is None:
                        continue
                    test = item.get("test", {})
                    cells = [str(test.get(k, "n/a")) for k in ("coverage", "fpr", "miss_rate")]
                    lines.append(f"| {alpha} | {variant} | {item['status']} | "
                                 f"{item['thresholds']} | " + " | ".join(cells) + " |")
            lines += ["", "| Metric | Before 95% CI | After 95% CI |", "|---|---|---|"]
            for key, ci in result["bootstrap"]["intervals"].items():
                prior = result["bootstrap_before"]["intervals"].get(key, "n/a")
                lines.append(f"| {key} | {prior} | {ci} |")
            lines += ["", "| Coverage | Risk |", "|---|---|"]
            curve = result["risk_coverage"]["curve"]
            lines += [f"| {c:.5f} | {r:.5f} |"
                      for c, r in zip(curve["coverage"], curve["risk"], strict=True)]
            lines += ["", "Group coverage, reliability bins and fitted parameters are in JSON.", ""]
    return "\n".join(lines)


def write_report(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fusion_metrics.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
    (out_dir / "fusion_report.md").write_text(render_markdown(payload) + "\n")
