"""A2/A3 frozen rules audit. Kaggle only; never changes the released split files.

GroundLM/ScamShield: in-sample, these rows informed rule design. SmishX: independent,
non-Indian-specific validation of these changes. No fitted thresholds and no training.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
from _common import env_report, write_metrics

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import cached_registered_domain
from rakshak.datasets.dedup import assign_campaigns, mark_exact_duplicates
from rakshak.eval import _rules_pre_widening, data
from rakshak.eval.rule_audit import THRESHOLDS, VARIANTS, audit_fold, cohort_report, score_text

IN_SAMPLE = "in-sample: these rows informed rule design"
INDEPENDENT = "independent, non-Indian-specific"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def external_campaigns(frame):
    """Same frozen MinHash/domain method as SMS; external-only IDs are namespaced."""
    frame = frame[frame["text_available"].fillna(False)].copy().reset_index(drop=True)
    frame["reg_domains"] = [
        sorted({cached_registered_domain(u) for u in canonicalize(t).urls})
        for t in frame["text"]
    ]
    frame = mark_exact_duplicates(frame)
    frame, diagnostics = assign_campaigns(frame)
    frame["campaign_id"] = "external:" + frame["campaign_id"].astype(str)
    return frame, diagnostics


def run(ctx):
    root = data.resolve_data_root()
    sms = data.load_sms(root)
    runs = data.load_runs(
        root, sms, patterns=["campaign", "source_holdout*", "lang_shift", "temporal"],
        labels=sms["label_binary"].to_numpy(dtype=int), mask=data.eval_mask(sms),
    )
    ext_paths = [root / "external" / f"external_{s}.parquet" for s in ("groundlm", "smishx")]
    raw_external = {p.stem: pd.read_parquet(p) for p in ext_paths}
    external, clustering = external_campaigns(pd.concat(raw_external.values(), ignore_index=True))
    frame = pd.concat([sms, external], ignore_index=True)
    needed = set(i for r in runs for i in r.test)
    needed.update(sms.index[(sms["source"] == "scamshield") & data.eval_mask(sms)])
    needed.update(range(len(sms), len(frame)))
    predictions = {}
    for j, i in enumerate(sorted(needed)):
        predictions[i] = score_text(frame.at[i, "text"])
        if j % 10000 == 0:
            ctx.log(f"scored {j}/{len(needed)} rows (three frozen variants)")

    reports, manifest = {}, []

    def add(name, positions, scope):
        subset = frame.loc[list(positions)]
        reports[name] = {"scope": scope, **cohort_report(subset, predictions)}
        ctx.log(f"reported {name}: {len(subset)} rows; {scope}")

    for r in runs:
        # These splits are preserved exactly. They were inspected in the reviewed work.
        add(f"split/{r.name}/test", r.test, IN_SAMPLE)
    scam = sms[(sms["source"] == "scamshield") & data.eval_mask(sms)]
    ext_start = len(sms)
    for name, subset in (
        ("scamshield", scam),
        ("groundlm", frame.loc[range(ext_start, len(frame))].query("policy_negative == True")),
    ):
        add(f"{name}/all_scoreable", subset.index, IN_SAMPLE)
        folds = subset["campaign_id"].map(audit_fold)
        for fold in ("audit", "remainder"):
            part = subset[folds == fold]
            add(f"{name}/{fold}", part.index, IN_SAMPLE)
        manifest.extend({"set": name, "id": str(r.id), "campaign_id": r.campaign_id,
                         "fold": audit_fold(r.campaign_id), "scope": IN_SAMPLE}
                        for r in subset.itertuples())
    for source, raw in raw_external.items():
        ids = set(raw["id"].astype(str))
        subset = frame.loc[range(ext_start, len(frame))]
        subset = subset[subset["id"].isin(ids) & subset["eval_ok"].fillna(False)]
        scope = INDEPENDENT if source == "external_smishx" else IN_SAMPLE
        add(f"{source}/eval_eligible", subset.index, scope)
        if source == "external_smishx":
            manifest.extend({"set": source, "id": str(r.id), "campaign_id": r.campaign_id,
                             "fold": "independent_external", "scope": scope}
                            for r in subset.itertuples())

    ground = raw_external["external_groundlm"]
    neg = ground["policy_negative"].fillna(False).astype(bool)
    count_audit = {
        "rows": len(ground), "policy_negatives": int(neg.sum()),
        "missing_text": int((~ground["text_available"].fillna(False)).sum()),
        "policy_negative_missing_text": int((neg & ~ground["text_available"].fillna(False)).sum()),
        "policy_negative_scoreable": int((neg & ground["text_available"].fillna(False)).sum()),
        "policy_negative_eval_eligible": int((neg & ground["eval_ok"].fillna(False)).sum()),
    }
    # The groundlm all-scoreable report includes all policy negatives, even if dup/ineligible;
    # the eval_eligible report above is distinct and never silently changes the denominator.
    artifacts = [root / "sms.parquet", *ext_paths, *sorted((root / "splits").glob("*.json"))]
    metrics = {
        "env": env_report(), "variants": VARIANTS, "thresholds": THRESHOLDS,
        "baseline_rules_sha256": sha256(_rules_pre_widening.__file__),
        "baseline_rules_commit": "e2cfa625e3ae48ada2c2358dfee3c3c41ba6dbfe",
        "groundlm_count_audit": count_audit, "external_clustering": clustering,
        "input_sha256": {str(p.relative_to(root)): sha256(p) for p in artifacts},
        "scope": "rules only; no model-side or URL/PhreshPhish claim",
        "independent_indian_validation": "OPEN: consented privacy-scrubbed post-freeze real SMS",
        "reports": reports,
    }
    write_metrics("metrics.json", metrics)
    write_metrics("audit_manifest.json", {"seed": 13, "rows": manifest})
    records = []
    for i, pred in predictions.items():
        records.append({"id": frame.at[i, "id"], "campaign_id": frame.at[i, "campaign_id"],
                        **{f"{v}/{k}": x for v, scores in pred.items() for k, x in scores.items()}})
    pd.DataFrame(records).to_parquet(ctx.work / "rule_predictions.parquet", index=False)
    (ctx.work / "protocol.json").write_text(json.dumps({
        "thresholds": THRESHOLDS, "scopes": [IN_SAMPLE, INDEPENDENT],
        "claim": "No held-out improvement claim; immutable frozen-rule audit only",
    }, indent=2))
    ctx.results["summary"] = f"frozen rules audit: {len(reports)} cohorts; {count_audit}"
    return True
