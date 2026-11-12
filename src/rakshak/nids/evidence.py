"""Network evidence and advisory actions (Evidence-Locked Advisory for flows).

The detector emits a family per flow (rakshak.nids.schema.FAMILIES) and a probability. This
module turns that, plus optional host-window statistics, into ``NetEvidence`` items. The
``EvidencePackage`` built here is the same type as the SMS one, so ``rakshak.verifier.verify``
and the explainer prompt work on it unchanged. Nothing here blocks traffic. Every
``NetAction`` is advice the user applies.

Two evidence types are additions to docs/NETWORK_IDS_PLAN.md section 5.2, because the plan has
no evidence for two families the taxonomy needs: ``exploit_suspect`` (injection and exploit)
and ``anomalous_flow`` (other attacks).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rakshak.evidence import EvidencePackage, Risk
from rakshak.evidence import _risk as _sms_risk  # same thresholds as the SMS package
from rakshak.rules import Evidence

NETWORK_DOMAIN = "network"
PORT_SCAN_MIN_PORTS = 20  # distinct destination ports from one source inside the window
PORT_SCAN_WEIGHT = 0.9  # host-window port-scan evidence weight (rule, not model)
SSH_PORT = 22
MAX_SUMMARY = 280


class NetAction(StrEnum):
    BLOCK_SOURCE_IP = "BLOCK_SOURCE_IP"
    CHECK_ROUTER_FIREWALL = "CHECK_ROUTER_FIREWALL"
    CHANGE_PASSWORD = "CHANGE_PASSWORD"
    DISABLE_PASSWORD_LOGIN = "DISABLE_PASSWORD_LOGIN"
    RATE_LIMIT = "RATE_LIMIT"
    CONTACT_ISP = "CONTACT_ISP"
    ISOLATE_DEVICE = "ISOLATE_DEVICE"
    FACTORY_RESET_IOT = "FACTORY_RESET_IOT"
    USE_TRUSTED_DNS = "USE_TRUSTED_DNS"
    VERIFY_INDEPENDENTLY = "VERIFY_INDEPENDENTLY"  # shared with the SMS package
    IGNORE = "IGNORE"  # shared with the SMS package


NET_EVIDENCE_TYPES: tuple[str, ...] = (
    "port_scan",
    "brute_force_ssh",
    "brute_force_other",
    "dos_flood",
    "botnet_beaconing",
    "dns_tunnel",
    "data_exfil",
    "new_device_rare_port",
    "exploit_suspect",
    "anomalous_flow",
)

NET_ACTION_MAP: dict[str, tuple[NetAction, ...]] = {
    "port_scan": (
        NetAction.BLOCK_SOURCE_IP,
        NetAction.CHECK_ROUTER_FIREWALL,
        NetAction.VERIFY_INDEPENDENTLY,
    ),
    "brute_force_ssh": (
        NetAction.CHANGE_PASSWORD,
        NetAction.DISABLE_PASSWORD_LOGIN,
        NetAction.BLOCK_SOURCE_IP,
    ),
    "brute_force_other": (NetAction.CHANGE_PASSWORD, NetAction.BLOCK_SOURCE_IP),
    "dos_flood": (NetAction.RATE_LIMIT, NetAction.CONTACT_ISP, NetAction.CHECK_ROUTER_FIREWALL),
    "botnet_beaconing": (
        NetAction.ISOLATE_DEVICE,
        NetAction.FACTORY_RESET_IOT,
        NetAction.CHANGE_PASSWORD,
    ),
    "dns_tunnel": (
        NetAction.ISOLATE_DEVICE,
        NetAction.VERIFY_INDEPENDENTLY,
        NetAction.USE_TRUSTED_DNS,
    ),
    "data_exfil": (
        NetAction.ISOLATE_DEVICE,
        NetAction.CONTACT_ISP,
        NetAction.VERIFY_INDEPENDENTLY,
    ),
    "new_device_rare_port": (NetAction.VERIFY_INDEPENDENTLY, NetAction.ISOLATE_DEVICE),
    "exploit_suspect": (
        NetAction.BLOCK_SOURCE_IP,
        NetAction.CHECK_ROUTER_FIREWALL,
        NetAction.VERIFY_INDEPENDENTLY,
    ),
    "anomalous_flow": (NetAction.VERIFY_INDEPENDENTLY, NetAction.BLOCK_SOURCE_IP),
}

# Family -> evidence type. ``brute_force`` is resolved by destination port (see below).
FAMILY_EVIDENCE: dict[str, str] = {
    "port_scan": "port_scan",
    "dos_ddos": "dos_flood",
    "botnet_c2": "botnet_beaconing",
    "exfil_theft": "data_exfil",
    "injection_exploit": "exploit_suspect",
    "other": "anomalous_flow",
}

_PHRASE = {
    "port_scan": "probes many ports on your devices from one source",
    "brute_force_ssh": "makes repeated SSH login attempts to one host",
    "brute_force_other": "makes repeated login attempts to a service",
    "dos_flood": "sends a very high volume of traffic at one target",
    "botnet_beaconing": "calls an outside address on a regular schedule",
    "dns_tunnel": "sends DNS queries that look like tunnelled data",
    "data_exfil": "sends much more data out than it receives",
    "new_device_rare_port": "is a new device using an unusual port",
    "exploit_suspect": "matches an exploit or injection pattern",
    "anomalous_flow": "looks anomalous but matches no known pattern",
}
_BASE = {
    "scam": "Check the device and your router firewall. Do not ignore it.",
    "uncertain": "Not sure about this traffic. Check the device list.",
    "benign": "No network attack pattern was found.",
}
_ACTION_PRIORITY = [
    NetAction.ISOLATE_DEVICE,
    NetAction.BLOCK_SOURCE_IP,
    NetAction.CHANGE_PASSWORD,
    NetAction.DISABLE_PASSWORD_LOGIN,
    NetAction.RATE_LIMIT,
    NetAction.CHECK_ROUTER_FIREWALL,
    NetAction.CONTACT_ISP,
    NetAction.FACTORY_RESET_IOT,
    NetAction.USE_TRUSTED_DNS,
    NetAction.VERIFY_INDEPENDENTLY,
    NetAction.IGNORE,
]

STATE_LABEL = {"scam": "ALERT", "uncertain": "UNCERTAIN", "benign": "SAFE"}

# Rules-first policy (docs/NIDS_EDGE.md, 2026-11-07). A rule's type decides the advice; the model
# only adds a model_only advisory when no rule fired (quiet mode).
RULE_TYPES: frozenset[str] = frozenset(
    {"port_scan", "brute_force_ssh", "brute_force_other", "dos_flood", "dns_tunnel"}
)
MODEL_ONLY_PATH = "model_only_quiet"
MODEL_ONLY_SUMMARY = (
    "The traffic model marked some flows as unusual, but no known attack pattern was found. "
    "Check the device if you don't recognise it."
)


@dataclass(frozen=True)
class NetEvidence:
    """One evidence item. ``id`` is N1, N2, ... in the order of ``weight`` (largest first)."""

    id: str
    type: str
    weight: float
    family: str | None = None
    source: str = "model"  # "model" for flow-level evidence, "window" for host-window evidence

    def as_rule_evidence(self) -> Evidence:
        """The SMS ``Evidence`` shape, so the shared package code can carry it."""
        return Evidence(id=self.id, type=self.type, span=(0, 0), text="", weight=self.weight)


@dataclass(frozen=True)
class WindowStats:
    """Host-window statistics for one source address over a time window."""

    src_ip: str
    distinct_dst_ports: int
    flows: int


def _clamp(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def evidence_for_flow(
    family: str,
    p_attack: float,
    *,
    dst_port: int | None = None,
    window: WindowStats | None = None,
) -> list[NetEvidence]:
    """Evidence for one flow. Empty for benign flows that the window does not flag.

    The family evidence weight is the attack probability. Window evidence (``port_scan`` from
    many destination ports) gets weight 0.9 when the window passes ``PORT_SCAN_MIN_PORTS``.
    Items are de-duplicated by type (largest weight kept) and numbered by weight.
    """
    found: dict[str, tuple[float, str, str]] = {}  # type -> (weight, family, source)
    if family != "benign":
        if family == "brute_force":
            etype = "brute_force_ssh" if dst_port == SSH_PORT else "brute_force_other"
        elif family in FAMILY_EVIDENCE:
            etype = FAMILY_EVIDENCE[family]
        else:
            raise ValueError(f"unknown family {family!r}")
        found[etype] = (_clamp(p_attack), family, "model")
    if window is not None and window.distinct_dst_ports >= PORT_SCAN_MIN_PORTS:
        prev = found.get("port_scan")
        weight = max(prev[0] if prev else 0.0, PORT_SCAN_WEIGHT)
        found["port_scan"] = (weight, prev[1] if prev else "port_scan", "window")
    ordered = sorted(found.items(), key=lambda kv: (-kv[1][0], kv[0]))
    return [
        NetEvidence(id=f"N{i}", type=etype, weight=weight, family=fam, source=src)
        for i, (etype, (weight, fam, src)) in enumerate(ordered, start=1)
    ]


def _allowed_actions(evidence: list[NetEvidence], risk: Risk) -> list[NetAction]:
    acts: set[NetAction] = set()
    for e in evidence:
        acts.update(NET_ACTION_MAP.get(e.type, ()))
    if not acts:
        acts.add(NetAction.IGNORE if risk is Risk.LOW else NetAction.VERIFY_INDEPENDENTLY)
    if risk is Risk.LOW:
        acts.add(NetAction.IGNORE)
    if risk in (Risk.UNCERTAIN, Risk.MEDIUM):
        acts.add(NetAction.VERIFY_INDEPENDENTLY)
    return sorted(acts, key=lambda a: a.value)


def build_network_package(evidence: list[NetEvidence], state: str) -> EvidencePackage:
    """Evidence package for one flow. ``state`` is ``scam``, ``uncertain`` or ``benign``.

    Risk uses the SMS thresholds (top weight 0.85 gives HIGH, 0.6 MEDIUM), and the detector
    state can raise it as in the SMS builder.
    """
    if state not in STATE_LABEL:
        raise ValueError(f"state must be one of {sorted(STATE_LABEL)}, got {state!r}")
    rule_ev = [e.as_rule_evidence() for e in evidence]
    risk = _sms_risk(rule_ev, state)
    return EvidencePackage(
        risk=risk,
        confidence_state=state,
        evidence=rule_ev,
        allowed_actions=list(_allowed_actions(evidence, risk)),
        domain=NETWORK_DOMAIN,
    )


def build_model_only_package(evidence: list[NetEvidence]) -> EvidencePackage:
    """Quiet-mode package for a host with model evidence and no rule hit.

    Risk is capped at UNCERTAIN and the only allowed action is VERIFY_INDEPENDENTLY. The model
    evidence is kept for the record (its ids and weights), but it grants no other action.
    """
    return EvidencePackage(
        risk=Risk.UNCERTAIN,
        confidence_state="uncertain",
        evidence=[e.as_rule_evidence() for e in evidence],
        allowed_actions=[NetAction.VERIFY_INDEPENDENTLY],
        domain=NETWORK_DOMAIN,
    )


def model_only_template(package: EvidencePackage) -> dict[str, Any]:
    """Fixed neutral advisory for a model-only host (quiet mode). Names no evidence type."""
    return {
        "summary": MODEL_ONLY_SUMMARY,
        "reasons": [e.id for e in package.evidence],
        "action": NetAction.VERIFY_INDEPENDENTLY.value,
    }


def _pick_action(package: EvidencePackage) -> NetAction:
    """The advised action. A rule's type takes its first allowed action in its map entry;
    other types use the priority list."""
    allowed = set(package.allowed_actions)
    if package.evidence:
        primary = max(package.evidence, key=lambda e: e.weight).type
        if primary in RULE_TYPES:
            for act in NET_ACTION_MAP.get(primary, ()):
                if act in allowed:
                    return act
    return next((a for a in _ACTION_PRIORITY if a in allowed), package.allowed_actions[0])


def net_template(package: EvidencePackage) -> dict[str, Any]:
    """Deterministic explanation for a network package. Always passes ``verify``.

    Used when the explainer output is rejected. It names only evidence types and advice codes
    from the package, so the verifier accepts it for any package this module builds.
    """
    base = _BASE[package.confidence_state]
    seen: list[str] = []
    for e in package.evidence:
        if e.type not in seen:
            seen.append(e.type)
    if seen:
        described = "; ".join(_PHRASE[t] for t in seen[:3] if t in _PHRASE)
        summary = f"This flow {described}. {base}" if described else base
    else:
        summary = base
    summary = summary[:MAX_SUMMARY]
    action = _pick_action(package)
    return {
        "summary": summary,
        "reasons": [e.id for e in package.evidence],
        "action": action.value,
    }
