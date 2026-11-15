"""Evidence package: the only facts the explanation layer is allowed to talk about.

``build_package`` turns a canonicalised message plus its rule evidence into an
``EvidencePackage`` with a risk level, a confidence state, and an allowed-action set. The
Llama explainer may only cite evidence ids from this package and only pick actions from
``allowed_actions``; ``verifier.verify`` enforces both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum

from rakshak.canonicalize import Canon
from rakshak.rules import Evidence
from rakshak.url_features import mentioned_brand_domains, registered_domain
from rakshak.url_model import URL_MODEL_TYPE, url_model_fields


class Risk(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNCERTAIN = "UNCERTAIN"


class Action(StrEnum):
    DO_NOT_CLICK = "DO_NOT_CLICK"
    DO_NOT_OPEN_LINK = "DO_NOT_OPEN_LINK"
    DO_NOT_SHARE_OTP = "DO_NOT_SHARE_OTP"
    OPEN_OFFICIAL_APP = "OPEN_OFFICIAL_APP"
    CONTACT_BANK_OFFICIAL = "CONTACT_BANK_OFFICIAL"
    REPORT_1930_OR_CHAKSHU = "REPORT_1930_OR_CHAKSHU"
    BLOCK_SENDER = "BLOCK_SENDER"
    IGNORE = "IGNORE"
    VERIFY_INDEPENDENTLY = "VERIFY_INDEPENDENTLY"


CONFIDENCE_STATES = ("scam", "uncertain", "benign")

# Evidence type -> actions that are safe to recommend for that evidence.
ACTION_MAP: dict[str, tuple[Action, ...]] = {
    "credential_request": (Action.DO_NOT_SHARE_OTP, Action.CONTACT_BANK_OFFICIAL),
    "upi_collect_or_pay": (Action.DO_NOT_SHARE_OTP, Action.REPORT_1930_OR_CHAKSHU),
    "kyc_block_urgency": (
        Action.OPEN_OFFICIAL_APP,
        Action.CONTACT_BANK_OFFICIAL,
        Action.DO_NOT_CLICK,
    ),
    "account_suspension": (
        Action.CONTACT_BANK_OFFICIAL,
        Action.OPEN_OFFICIAL_APP,
        Action.DO_NOT_CLICK,
    ),
    "prize_lottery": (Action.DO_NOT_CLICK, Action.BLOCK_SENDER),
    "apk_install": (Action.DO_NOT_CLICK, Action.BLOCK_SENDER, Action.REPORT_1930_OR_CHAKSHU),
    "remote_access_app": (Action.DO_NOT_CLICK, Action.REPORT_1930_OR_CHAKSHU, Action.BLOCK_SENDER),
    "electricity_disconnect": (
        Action.VERIFY_INDEPENDENTLY,
        Action.OPEN_OFFICIAL_APP,
        Action.DO_NOT_CLICK,
    ),
    "parcel_customs": (Action.DO_NOT_CLICK, Action.VERIFY_INDEPENDENTLY),
    "job_task_scam": (Action.BLOCK_SENDER, Action.REPORT_1930_OR_CHAKSHU),
    "investment_returns": (
        Action.BLOCK_SENDER,
        Action.REPORT_1930_OR_CHAKSHU,
        Action.VERIFY_INDEPENDENTLY,
    ),
    "digital_arrest": (
        Action.REPORT_1930_OR_CHAKSHU,
        Action.VERIFY_INDEPENDENTLY,
        Action.BLOCK_SENDER,
        Action.DO_NOT_SHARE_OTP,
    ),
    "brand_domain_mismatch": (Action.DO_NOT_CLICK, Action.OPEN_OFFICIAL_APP),
    "shortened_url": (Action.DO_NOT_CLICK,),
    "suspicious_url": (Action.DO_NOT_CLICK,),
    URL_MODEL_TYPE: (Action.DO_NOT_OPEN_LINK,),
}

_PHONE_RE = re.compile(r"\+?\d[\d \-]{8,}\d")


@dataclass
class EvidencePackage:
    risk: Risk
    confidence_state: str
    evidence: list[Evidence]
    # Action codes from the package's domain: ``Action`` for SMS, ``NetAction`` for network.
    allowed_actions: list[Action | StrEnum]
    urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    phones: list[str] = field(default_factory=list)
    brand_domains: dict[str, list[str]] = field(default_factory=dict)
    # "sms" (default, the original behaviour) or "network" (rakshak.nids).
    domain: str = "sms"

    @property
    def evidence_ids(self) -> set[str]:
        return {e.id for e in self.evidence}

    def allowed_domain_names(self) -> set[str]:
        """Every domain the summary may mention: message domains plus official brand domains."""
        names = set(self.domains)
        for doms in self.brand_domains.values():
            names.update(doms)
        return {d.lower() for d in names}

    def to_prompt_json(self, include_raw: bool = False) -> str:
        """JSON view of the package for the explainer prompt.

        Raw message text is never included. Evidence spans and matched text are dropped;
        URL evidence is reduced to its registered domain unless ``include_raw`` is set.
        """
        ev_out = []
        for e in self.evidence:
            item: dict[str, object] = {"id": e.id, "type": e.type, "weight": round(e.weight, 3)}
            if e.type in ("brand_domain_mismatch", "shortened_url", "suspicious_url"):
                item["domain"] = registered_domain(e.text) or None
                if include_raw:
                    item["url"] = e.text
            elif e.type == URL_MODEL_TYPE:
                domain, score = url_model_fields(e.text)
                item["domain"] = domain or None
                item["url_score"] = score
            elif include_raw:
                item["text"] = e.text
            ev_out.append(item)
        payload: dict[str, object] = {
            "risk": self.risk.value,
            "confidence_state": self.confidence_state,
            "evidence": ev_out,
            "allowed_actions": [a.value for a in self.allowed_actions],
            "domains": sorted(set(self.domains)),
            "brand_domains": self.brand_domains,
        }
        if include_raw:
            payload["urls"] = self.urls
            payload["phones"] = self.phones
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _allowed_actions(evidence: list[Evidence], risk: Risk) -> list[Action]:
    acts: set[Action] = set()
    for e in evidence:
        acts.update(ACTION_MAP.get(e.type, ()))
    if not acts:
        acts.add(Action.IGNORE if risk is Risk.LOW else Action.VERIFY_INDEPENDENTLY)
    if risk is Risk.LOW:
        acts.add(Action.IGNORE)
    if risk in (Risk.UNCERTAIN, Risk.MEDIUM):
        acts.add(Action.VERIFY_INDEPENDENTLY)
    return sorted(acts, key=lambda a: a.value)


def _risk(evidence: list[Evidence], scam_state: str) -> Risk:
    top = max((e.weight for e in evidence), default=0.0)
    if top >= 0.85:
        return Risk.HIGH
    # The URL model item (phishing_url_model) is only present at or above its registered
    # threshold. With the SMS student also over its threshold it is HIGH; otherwise it only
    # makes the state at least UNCERTAIN (the rules below). Without the item nothing changes.
    if scam_state == "scam" and any(e.type == URL_MODEL_TYPE for e in evidence):
        return Risk.HIGH
    if top >= 0.6:
        return Risk.MEDIUM
    # ``urgency`` is the only generic, context-free type: it fires on "I am leaving early", "बहुत
    # जल्दी", "pay your bill now". It is deliberately weak (weight 0.4) and must not on its own
    # escalate a message the fusion layer has not called suspicious. This rule was designed
    # using GroundLM outcomes; its historical FPR is in-sample. Denominator-specific counts
    # and the frozen-rule audit are in docs/GROUNDLM_FPR.md, not duplicated in production code.
    if evidence and all(e.type == "urgency" for e in evidence):
        return Risk.UNCERTAIN if scam_state == "scam" else Risk.LOW
    if scam_state == "scam" and evidence:
        return Risk.MEDIUM
    if scam_state == "uncertain":
        return Risk.UNCERTAIN
    return Risk.LOW


def build_package(
    canon: Canon, evidence: list[Evidence], scam_state: str = "uncertain"
) -> EvidencePackage:
    """Build the evidence package.

    ``scam_state`` is the fusion output: ``"scam"``, ``"uncertain"`` or ``"benign"``. Rule
    evidence can raise the risk above what fusion alone would give, never lower it.
    """
    if scam_state not in CONFIDENCE_STATES:
        raise ValueError(f"scam_state must be one of {CONFIDENCE_STATES}, got {scam_state!r}")
    risk = _risk(evidence, scam_state)
    domains = sorted({registered_domain(u) for u in canon.urls if registered_domain(u)})
    phones = sorted({re.sub(r"\D", "", p) for p in _PHONE_RE.findall(canon.normalized)})
    return EvidencePackage(
        risk=risk,
        confidence_state=scam_state,
        evidence=list(evidence),
        allowed_actions=_allowed_actions(evidence, risk),
        urls=list(canon.urls),
        domains=domains,
        phones=phones,
        brand_domains=mentioned_brand_domains(canon.normalized),
    )


def choose_action(package: EvidencePackage) -> Action:
    """Advisory action for an SMS package, picked by the system and never by the explainer.

    Rule (documented, deterministic): walk the evidence from highest weight to lowest, keeping the
    list order for ties. The first item whose type has an ``ACTION_MAP`` entry gives the first
    action of that entry (the entry's order is its priority). ``urgency`` is the only evidence
    type with no entry, so it is skipped. With no such item, LOW risk gives ``IGNORE`` and any
    other risk gives ``VERIFY_INDEPENDENTLY``. The result is always one of ``allowed_actions``.
    """
    if package.domain != "sms":
        raise ValueError("choose_action covers SMS packages; network packages use NET_ACTION_MAP")
    ranked = sorted(enumerate(package.evidence), key=lambda iv: (-iv[1].weight, iv[0]))
    chosen: Action | None = None
    for _, e in ranked:
        entry = ACTION_MAP.get(e.type)
        if entry:
            chosen = entry[0]
            break
    if chosen is None:
        chosen = Action.IGNORE if package.risk is Risk.LOW else Action.VERIFY_INDEPENDENTLY
    if chosen not in package.allowed_actions:
        raise ValueError(f"chosen action {chosen} is not in the package's allowed actions")
    return chosen
