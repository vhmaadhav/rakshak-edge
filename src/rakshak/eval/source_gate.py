"""Apply the predeclared source-control acceptance rule to completed paired statistics."""

import math


def source_gate(metrics):
    runs, comparisons = metrics["runs"], metrics["comparisons"]
    folds = sorted(name for name in runs if name.startswith("source_holdout:"))
    target = "source_holdout:holdout_scamshield"
    required = {f"source_holdout:holdout_{source}" for source in
                ("scamshield", "uci_sms_spam", "mishra_soni_smsphishing")}
    if set(folds) != required:
        raise ValueError("all three frozen source folds are required")
    for name, run in runs.items():
        for arm in ("erm", "balanced", "dro"):
            if set(run["arms"][arm]) != {"1", "2", "3"}:
                raise ValueError(f"{name}/{arm}: all three frozen seeds are required")
        if set(comparisons.get(name, {})) != {"balanced", "dro"}:
            raise ValueError(f"{name}: both paired intervention comparisons are required")
    decisions = {}
    for arm in ("balanced", "dro"):
        checks = {}
        for name in folds:
            row = comparisons[name][arm]
            values = [row["delta"], *row["delta_ci95"], *row["b_ci95"]]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                raise ValueError(f"{name}/{arm}: finite paired intervals are required")
            half_width = (row["b_ci95"][1] - row["b_ci95"][0]) / 2
            if half_width < 0 or row["delta_ci95"][0] > row["delta_ci95"][1]:
                raise ValueError("interval bounds are reversed")
            passed = (row["delta_ci95"][0] > 0 if name == target
                      else row["delta"] >= -half_width)
            checks[name] = {"passes": passed, "baseline_ci_half_width": half_width,
                            "comparison": row}
        decisions[arm] = {"decision": "go" if all(c["passes"] for c in checks.values())
                          else "kill", "source_checks": checks}
    return {
        "arms": decisions, "all_split_comparisons": comparisons,
        "skipped": metrics.get("skipped", {}),
        "criterion": "ScamShield delta lower CI >0; other source delta >= -ERM CI half-width",
        "scope": "fixed three seeds; unadjusted paired campaign comparisons; "
                 "in-sample for rule-feature design; no independent held-out claim",
    }
