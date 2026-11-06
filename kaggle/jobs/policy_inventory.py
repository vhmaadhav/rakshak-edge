"""Inventory only: freeze train-only annotation scopes and a blinded human sample.

No model loading, inference, label changes, evaluation or paid API. Outputs are
private; review texts must never be committed or shown in public logs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from _common import write_metrics

from rakshak.eval import data as D
from rakshak.eval.clean_kd import FOLDS
from rakshak.llm.policy_training import inventory


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(ctx):
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    runs = D.load_runs(root, sms, patterns=["campaign", "source_holdout", "lang_shift", "temporal"],
                       labels=sms.label_binary.to_numpy(), mask=None)
    if len(runs) != 6 or {r.name for r in runs} != FOLDS:
        raise ValueError("exactly six original frozen SMS folds required")
    roles = {}
    for run in runs:
        if any(run.notes["missing_ids"].values()):
            raise ValueError("missing original split IDs; stop annotation inventory")
        roles[run.name] = {"train": sms.id.iloc[run.train].tolist(),
                           "cal": sms.id.iloc[run.cal].tolist(),
                           "test": sms.id.iloc[run.test_all].tolist()}
        if any(len(v) != len(set(v)) for v in roles[run.name].values()):
            raise ValueError("duplicate frozen partition IDs")
    plan = inventory(sms, roles, audit_n=200)
    plan["sms_sha256"] = sha(root / "sms.parquet")
    plan["split_files"] = {p.name: sha(p) for p in D.list_split_files(
        root, ["campaign", "source_holdout", "lang_shift", "temporal"])}
    plan["frozen_fold_roles"] = roles
    if len(plan["audit_ids"]) != 200:
        raise ValueError("fewer than 200 eligible training-only human audit rows")
    # The reviewer sees neither a source/legacy label nor an LLM proposal.
    audit = sms.set_index("id").loc[plan["audit_ids"], ["text"]].reset_index()
    audit["human_label"] = pd.Series([None] * len(audit), dtype="string")
    audit["human_reviewed"] = False
    audit_path = ctx.work / "policy_human_audit_200.parquet"
    audit.to_parquet(audit_path, index=False)
    plan["audit_file_sha256"] = sha(audit_path)
    (ctx.work / "policy_annotation_scope.json").write_text(json.dumps(plan, indent=2) + "\n")
    summary = {"folds": plan["folds"], "eligible_union": len(plan["eligible_union_ids"]),
               "audit_rows": len(audit), "model_inferences": 0,
               "approved_for_training": False, "eval_labels_modified": 0,
               "sms_sha256": plan["sms_sha256"], "split_files": plan["split_files"],
               "scope_manifest_sha256": sha(ctx.work / "policy_annotation_scope.json"),
               "audit_ids_sha256": plan["audit_ids_sha256"],
               "scope": "train-only per-fold proposals; no global policy rewrite",
               "independent_indian_validation": "OPEN; no new real data here"}
    write_metrics("policy_inventory.json", summary)
    ctx.results["summary"] = (
        f"annotation scope union={summary['eligible_union']}, audit200; "
        "no inference/approved labels or eval changes")
    return True
