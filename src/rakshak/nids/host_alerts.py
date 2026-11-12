"""Per-source-host alerts from flow decisions and rule hits: one advisory per host.

Two modes (``--net-model-mode``, docs/NIDS_EDGE.md "Rules-first policy"):

- ``quiet`` (default, rules first). A host with a rule hit (``host_rules`` or the per-flow
  port-scan window) gets the rule evidence only. Its family, action and allowed actions come from
  the rule. The model score is reported next to it and cannot change them. A host with model
  evidence and no rule hit is a model-only advisory: state UNCERTAIN, risk capped at UNCERTAIN,
  action VERIFY_INDEPENDENTLY, path ``model_only_quiet``. It does not count as an alert.
- ``alert`` (the 2026-11-06 behaviour, kept for comparison). A host is alerted when one of its
  flows is in the ALERT state (score >= the calibrated threshold) or a window or rule flags it.
  Evidence is one item per type (largest weight kept). A host whose flows are only UNCERTAIN from
  the model goes to ``watchlist``.

The package and template come from ``build_network_package`` / ``net_template`` (rule or alert)
or ``build_model_only_package`` / ``model_only_template`` (model-only). The advisory is verified
with ``rakshak.verifier.verify``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rakshak.evidence import EvidencePackage
from rakshak.nids.evidence import (
    MODEL_ONLY_PATH,
    PORT_SCAN_WEIGHT,
    STATE_LABEL,
    NetEvidence,
    build_model_only_package,
    build_network_package,
    model_only_template,
    net_template,
)
from rakshak.nids.host_rules import RuleHit
from rakshak.nids.pi_monitor import FlowDecision
from rakshak.verifier import verify

MAX_EXAMPLES = 5
NET_MODEL_MODES = ("quiet", "alert")


@dataclass
class HostAlert:
    src: str
    state: str  # "scam", "uncertain" (rule, window or model-only host)
    family: str  # rule family in quiet mode; "model_only" for a model-only advisory
    p_attack: float
    flows_from_host: int
    flagged_flows: int
    distinct_dst_ports: int
    evidence: list[NetEvidence]
    package: EvidencePackage
    template: dict[str, Any]
    verified: bool
    verify_reasons: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    path: str = "model"  # "rule", "model" (alert mode) or MODEL_ONLY_PATH
    model_family: str = "benign"  # most likely model family among the host's flagged flows
    model_flagged_flows: int = 0

    @property
    def is_model_only(self) -> bool:
        return self.path == MODEL_ONLY_PATH

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "decision": {
                "risk": self.package.risk.value,
                "state": STATE_LABEL[self.state],
                "family": self.family,
                "p_attack": round(self.p_attack, 4),
                "flows_from_host": self.flows_from_host,
                "flagged_flows": self.flagged_flows,
                "distinct_dst_ports": self.distinct_dst_ports,
            },
            "model": {
                "p_attack": round(self.p_attack, 4),
                "flagged_flows": self.model_flagged_flows,
                "top_family": self.model_family,
            },
            "evidence": [
                {
                    "id": e.id,
                    "type": e.type,
                    "weight": round(e.weight, 3),
                    "source": e.source,
                    "family": e.family,
                }
                for e in self.evidence
            ],
            "advisory": {
                "domain": "network",
                "mode": "template",
                "path": self.path,
                "summary": self.template["summary"],
                "action": self.template["action"],
                "allowed_actions": [a.value for a in self.package.allowed_actions],
            },
            "verifier": {"ok": self.verified, "reasons": self.verify_reasons},
            "examples": self.examples,
        }


def _renumber(items: Iterable[NetEvidence]) -> list[NetEvidence]:
    ordered = sorted(items, key=lambda e: (-e.weight, e.type))
    return [
        NetEvidence(id=f"N{i}", type=e.type, weight=e.weight, family=e.family, source=e.source)
        for i, e in enumerate(ordered, start=1)
    ]


def _best(items: Iterable[NetEvidence]) -> dict[str, NetEvidence]:
    best: dict[str, NetEvidence] = {}
    for e in items:
        prev = best.get(e.type)
        if prev is None or e.weight > prev.weight:
            best[e.type] = e
    return best


def _alert_flow(d: FlowDecision) -> bool:
    return d.state == "scam" or any(e.source == "window" for e in d.evidence)


def _model_flagged(d: FlowDecision) -> bool:
    """The model put the flow in ALERT or UNCERTAIN (a non-benign family is only set then)."""
    return d.family != "benign"


def _by_source(decisions: Iterable[FlowDecision]) -> dict[str, list[FlowDecision]]:
    by_src: dict[str, list[FlowDecision]] = defaultdict(list)
    for d in decisions:
        by_src[d.src].append(d)
    return by_src


def watchlist(decisions: Iterable[FlowDecision], limit: int = MAX_EXAMPLES) -> list[dict[str, Any]]:
    """Alert-mode only: hosts with UNCERTAIN model flows and no alert flow (not advisories)."""
    out = []
    for src, flows in _by_source(decisions).items():
        if any(_alert_flow(d) for d in flows):
            continue
        unc = [d for d in flows if d.state == "uncertain"]
        if not unc:
            continue
        top = max(unc, key=lambda d: d.p_attack)
        out.append(
            {
                "src": src,
                "flows_from_host": len(flows),
                "uncertain_flows": len(unc),
                "max_p_attack": round(top.p_attack, 4),
                "top_family": top.family,
            }
        )
    out.sort(key=lambda x: -x["max_p_attack"])
    return out[:limit]


def _window_items(flows: Sequence[FlowDecision]) -> list[NetEvidence]:
    """Host-window port-scan evidence as a rule: the rule's type, family and fixed weight."""
    if not any(e.source == "window" for d in flows for e in d.evidence):
        return []
    return [
        NetEvidence(
            id="", type="port_scan", weight=PORT_SCAN_WEIGHT, family="port_scan", source="window"
        )
    ]


def host_alerts(
    decisions: Iterable[FlowDecision],
    rule_hits: Iterable[RuleHit] = (),
    mode: str = "quiet",
) -> list[HostAlert]:
    """Advisories for the source hosts, rule and model-only (quiet) or alert (alert) mode.

    ``decisions`` and the ``flow_index`` of each rule hit use the same request order.
    Results are sorted with rule and alert hosts first, then by attack probability.
    """
    if mode not in NET_MODEL_MODES:
        raise ValueError(f"mode must be one of {NET_MODEL_MODES}, got {mode!r}")
    indexed: dict[str, list[tuple[int, FlowDecision]]] = defaultdict(list)
    for idx, d in enumerate(decisions):
        indexed[d.src].append((idx, d))
    hits_by_src: dict[str, list[RuleHit]] = defaultdict(list)
    for h in rule_hits:
        hits_by_src[h.src].append(h)

    alerts: list[HostAlert] = []
    for src, pairs in indexed.items():
        flows = [d for _, d in pairs]
        hits = hits_by_src.get(src, [])
        rule_flows: set[int] = set()
        for h in hits:
            rule_flows |= set(h.flow_index)
        window_flows = {i for i, d in pairs if any(e.source == "window" for e in d.evidence)}
        model_idx = {i for i, d in pairs if _model_flagged(d)}
        model_flagged = [d for d in flows if _model_flagged(d)]
        top_model = (
            max(model_flagged, key=lambda d: d.p_attack).family if model_flagged else "benign"
        )
        ports = {d.dst_port for d in flows if d.dst_port is not None}
        rule_items = [
            NetEvidence(id="", type=h.type, weight=h.weight, family=h.type, source=h.source)
            for h in hits
        ] + _window_items(flows)
        rule_best = _best(rule_items)
        p_all = max(d.p_attack for d in flows)

        if mode == "quiet":
            if rule_best:
                evidence = _renumber(rule_best.values())
                state = "uncertain"
                package = build_network_package(evidence, state)
                template = net_template(package)
                path = "rule"
                family = evidence[0].family or "benign"
                flagged = len(rule_flows | window_flows | model_idx)
                p_attack = p_all
            elif model_flagged:
                model_items = [e for d in flows for e in d.evidence if e.source == "model"]
                evidence = _renumber(_best(model_items).values())
                state = "uncertain"
                package = build_model_only_package(evidence)
                template = model_only_template(package)
                path = MODEL_ONLY_PATH
                family = "model_only"
                flagged = len(model_idx)
                p_attack = p_all
            else:
                continue
        else:
            flagged_flows = [d for d in flows if _alert_flow(d)]
            if not flagged_flows and not rule_best:
                continue
            items = [e for d in flagged_flows for e in d.evidence]
            items += list(rule_items)
            evidence = _renumber(_best(items).values())
            state = "scam" if any(d.state == "scam" for d in flagged_flows) else "uncertain"
            package = build_network_package(evidence, state)
            template = net_template(package)
            path = "rule" if rule_best else "model"
            family = evidence[0].family or "benign"
            flagged = len({i for i, d in pairs if _alert_flow(d)} | rule_flows | window_flows)
            p_attack = max(d.p_attack for d in flagged_flows) if flagged_flows else p_all

        verdict = verify(json.dumps(template), package)
        shown = _model_flagged if mode == "quiet" else _alert_flow
        examples_src = [(i, d) for i, d in pairs if shown(d)]
        examples = [
            {
                "dst": d.dst,
                "dst_port": d.dst_port,
                "family": d.family,
                "p_attack": round(d.p_attack, 4),
                "state": STATE_LABEL[d.state],
            }
            for _, d in sorted(examples_src, key=lambda x: -x[1].p_attack)[:MAX_EXAMPLES]
        ]
        alerts.append(
            HostAlert(
                src=src,
                state=state,
                family=family,
                p_attack=p_attack,
                flows_from_host=len(flows),
                flagged_flows=flagged,
                distinct_dst_ports=len(ports),
                evidence=evidence,
                package=package,
                template=template,
                verified=verdict.ok,
                verify_reasons=list(verdict.reasons),
                examples=examples,
                path=path,
                model_family=top_model,
                model_flagged_flows=len(model_idx),
            )
        )
    alerts.sort(key=lambda a: (a.is_model_only, -a.p_attack, a.src))
    return alerts
