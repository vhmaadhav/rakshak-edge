"""Frozen rule comparisons for Kaggle audits; no training or threshold selection."""

from __future__ import annotations

import hashlib

import numpy as np

from rakshak.canonicalize import canonicalize
from rakshak.ela_eval import proxy_scam_state, wilson_interval
from rakshak.eval import _rules_pre_widening
from rakshak.eval.audit import clustered_rate, paired_rate_test
from rakshak.evidence import Risk, _risk
from rakshak.rules import run_rules

VARIANTS = ("pre_urgency", "pre_widening", "frozen_current")
THRESHOLDS = {"warn": "risk != LOW (includes abstention)", "high": "risk == HIGH (SCAM)"}


def audit_fold(campaign_id, seed=13):
    """Deterministic 20% campaign subset; a post-hoc audit, never an untouched holdout."""
    digest = hashlib.sha256(f"rule-audit:{seed}:{campaign_id}".encode()).digest()
    return "audit" if int.from_bytes(digest[:8], "big") % 5 == 0 else "remainder"


def legacy_risk(evidence, state):
    """Exact pre-f29521a _risk logic (source commit b2a5ab5)."""
    top = max((e.weight for e in evidence), default=0.0)
    if top >= 0.85:
        return Risk.HIGH
    if top >= 0.6:
        return Risk.MEDIUM
    if state == "scam" and evidence:
        return Risk.MEDIUM
    if state == "uncertain":
        return Risk.UNCERTAIN
    return Risk.LOW


def score_text(text):
    canon = canonicalize(text)
    old, new = _rules_pre_widening.run_rules(canon), run_rules(canon)
    result = {}
    for variant, evidence, risk_fn in (
        ("pre_urgency", old, legacy_risk),
        ("pre_widening", old, _risk),
        ("frozen_current", new, _risk),
    ):
        risk = risk_fn(evidence, proxy_scam_state(evidence))
        result[variant] = {
            "any_evidence": bool(evidence),
            "urgency_only": bool(evidence) and all(e.type == "urgency" for e in evidence),
            "warn": risk != Risk.LOW,
            "high": risk == Risk.HIGH,
        }
    return result


def describe_rate(values, campaigns, *, n_boot=1000):
    values = np.asarray(values, dtype=int)
    return {
        **clustered_rate(values, campaigns, n_boot=n_boot),
        "events": int(values.sum()),
        "wilson95": list(wilson_interval(int(values.sum()), len(values))),
    }


def cohort_report(frame, predictions, *, n_boot=1000):
    """All count denominators are explicit; FPR includes every labelled negative."""
    campaigns = frame["campaign_id"].astype(str).to_numpy()
    indices = frame.index.to_numpy()
    out = {"n": len(frame), "n_campaigns": len(set(campaigns)), "labels": {}}
    for label in ("label_binary", "label_scam"):
        if label not in frame:
            continue
        valid = frame[label].notna().to_numpy()
        labels = frame[label].fillna(-1).astype(int).to_numpy()
        slices = {"all_known": valid, "negative": valid & (labels == 0),
                  "positive": valid & (labels == 1)}
        report = {"unknown": int((~valid).sum()), "slices": {}}
        for name, mask in slices.items():
            positions, groups = indices[mask], campaigns[mask]
            variants = {}
            for variant in VARIANTS:
                variants[variant] = {
                    field: describe_rate(
                        [predictions[i][variant][field] for i in positions], groups, n_boot=n_boot
                    )
                    for field in ("any_evidence", "urgency_only", "warn", "high")
                }
            paired = {}
            for before, after in (("pre_urgency", "pre_widening"),
                                  ("pre_widening", "frozen_current")):
                paired[f"{after}_minus_{before}"] = {
                    field: paired_rate_test(
                        [predictions[i][after][field] for i in positions],
                        [predictions[i][before][field] for i in positions],
                        groups, n_boot=n_boot,
                    ) for field in ("any_evidence", "warn", "high")
                }
            report["slices"][name] = {"n": len(positions), "variants": variants, "paired": paired}
        out["labels"][label] = report
    return out
