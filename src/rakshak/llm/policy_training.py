"""Null-policy proposals bound to one frozen fold's training IDs only.

No function overwrites sms.parquet, an existing policy label or an eval label.
The same ID may have a different role in another fold; never apply a proposal
there without that other fold's explicit training-membership check.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import pandas as pd

from rakshak.datasets.loaders import SOURCE_GROUNDLM, SOURCE_SMISHX
from rakshak.llm.jobs import JobSpec, Partitions, stratified_sample

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LICENCE = "Apache-2.0"


def ids_hash(ids: Sequence[str]) -> str:
    raw = json.dumps(sorted(map(str, ids)), separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def train_null_ids(frame: pd.DataFrame, parts: Partitions) -> list[str]:
    """Select real kept null-policy training rows; raw cal/test are never eligible."""
    required = {"id", "text", "source", "dup_of", "label_policy"}
    if not required <= set(frame):
        raise ValueError("policy annotation requires explicit original metadata")
    if not frame.id.is_unique:
        raise ValueError("duplicate corpus IDs are forbidden")
    roles = {name: set(map(str, parts[name])) for name in ("train", "cal", "test")}
    if (roles["train"] & roles["cal"] or roles["train"] & roles["test"]
            or roles["cal"] & roles["test"]):
        raise ValueError("frozen fold roles overlap")
    if set.union(*roles.values()) - set(frame.id.astype(str)):
        raise ValueError("frozen fold holds missing corpus IDs")
    mask = frame.id.astype(str).isin(roles["train"]) & frame.dup_of.isna()
    mask &= frame.label_policy.isna()
    mask &= ~frame.source.isin([SOURCE_GROUNDLM, SOURCE_SMISHX])
    if "synthetic" in frame:
        mask &= ~frame.synthetic.fillna(False).astype(bool)
    return sorted(frame.loc[mask, "id"].astype(str))


def assert_proposal_scope(frame: pd.DataFrame, parts: Partitions, proposed_ids) -> None:
    """Apply-time guard: check the named fold again, not a cached global union."""
    proposed = list(map(str, proposed_ids))
    if len(set(proposed)) != len(proposed) or set(proposed) - set(train_null_ids(frame, parts)):
        raise ValueError("proposal outside this fold's eligible null-policy training IDs")


def annotation_spec(instructions: str, *, allowed_ids: Sequence[str] | None = None) -> JobSpec:
    """Keep the existing chunk schema/ledger but name the actual pinned open model."""
    allowed = None if allowed_ids is None else set(map(str, allowed_ids))

    def select(frame: pd.DataFrame, parts: Partitions) -> list[str]:
        eligible = train_null_ids(frame, parts)
        return eligible if allowed is None else sorted(set(eligible) & allowed)

    return JobSpec(name="POLICY_OPEN", kind="relabel",
                   description="Unapproved null-policy training proposals; human audit pending",
                   labeller=f"{MODEL}@{REVISION}", schema_name="annotate",
                   instructions=instructions, select=select)


def inventory(frame: pd.DataFrame, folds: Mapping[str, Partitions], *, audit_n: int = 200) -> dict:
    """Counts/hashes and a deterministic campaign-spread human-audit ID sample.

    audit_ids are training in at least one named fold, never globally relabelled.
    Review exports must hide the model proposal and source labels from the human.
    """
    eligible = {name: train_null_ids(frame, parts) for name, parts in folds.items()}
    union = sorted({rid for ids in eligible.values() for rid in ids})
    sub = frame[frame.id.astype(str).isin(union)]
    audit = stratified_sample(sub, audit_n, seed=13)
    return {"eligible_ids": eligible, "eligible_union_ids": union, "audit_ids": audit,
            "folds": {name: {"eligible": len(ids), "ids_sha256": ids_hash(ids),
                              "original_train_ids_sha256": ids_hash(folds[name]["train"])}
                      for name, ids in eligible.items()},
            "model": MODEL, "revision": REVISION, "licence": LICENCE,
            "audit_ids_sha256": ids_hash(audit), "approved_for_training": False,
            "eval_labels_modified": 0,
            "scope": "per-fold training sidecars only; no global label rewrite"}
