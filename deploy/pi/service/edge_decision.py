"""The edge decision, without the explainer: scam state, detector-only mode, package and action.

Pure functions with no model and no LLM import, so ``eval_groundlm.py`` and ``edge_pipeline.py``
call the same code (no copy that can drift). ``EdgeAnalyzer.analyse`` adds the explainer on top.

Detector-only mode (Fix B, see docs/EDGE_PIPELINE.md): a message whose student score is at or above
the threshold with no rule evidence has scam state "scam". ``evidence._risk`` gives it LOW, which is
not changed (Kaggle evals depend on it). The mode decides what the user sees:

- ``quiet`` (default, for the demo): the package stays LOW, the action is IGNORE, and the message
  gets the same no-warning path as a benign one. The detector's flag is kept in the response
  (``decision.detector_only``, ``decision.score``) so it can be counted.
- ``verify``: the package becomes UNCERTAIN with VERIFY_INDEPENDENTLY (the Fix B behaviour before
  the demo decision). The user sees a verify notice.
"""

from __future__ import annotations

import dataclasses

from rakshak.canonicalize import Canon
from rakshak.evidence import Action, EvidencePackage, Risk, build_package, choose_action
from rakshak.rules import Evidence
from rakshak.url_model import add_url_model_evidence
from rakshak.url_reputation import default_guard

DETECTOR_ONLY_MODES = ("quiet", "verify")
DEFAULT_DETECTOR_ONLY_MODE = "quiet"
# URL model evidence on SMS decisions is shown in verify mode only. The quiet-mode URL warning
# failed gate U2 (docs/URL_SERVICE_RESULTS.md), so quiet mode ignores it. /analyze_url is not
# affected: it has no mode.
URL_EVIDENCE_MODE = "verify"


@dataclasses.dataclass(frozen=True)
class EdgeDecision:
    state: str  # "scam", "uncertain" or "benign"
    package: EvidencePackage
    action: Action
    detector_only: bool  # state "scam" and no rule evidence (the Fix B case, either mode)


def check_mode(mode: str) -> str:
    if mode not in DETECTOR_ONLY_MODES:
        raise ValueError(f"detector_only_mode must be one of {DETECTOR_ONLY_MODES}, got {mode!r}")
    return mode


def url_model_active(mode: str, threshold: float | None) -> bool:
    """True when the URL model may add evidence to an SMS decision in this mode."""
    return check_mode(mode) == URL_EVIDENCE_MODE and threshold is not None


# Quiet-mode URL policy. "off": quiet mode ignores URL evidence (shipped default, U2 failed).
# "corroborated": quiet mode adds it only when the SMS student is over its threshold or a rule
# fired (docs/URL_SERVICE_PLAN.md, amendment 2026-11-12 night; ships only if U2 and U3 pass).
# "guarded": quiet mode adds it only when the URL is not *established* (popular registrable domain,
# not a shortener / user-content host / IP); docs/RECALL_V2_PLAN.md, pre-registered 2026-11-20.
URL_QUIET_POLICIES = ("off", "corroborated", "guarded")
URL_QUIET_POLICY = "off"


def add_url_evidence(
    evidence: list[Evidence],
    canon: Canon,
    hit,
    threshold: float | None,
    mode: str,
    *,
    p_scam: float | None = None,
    p_threshold: float | None = None,
    quiet_policy: str | None = None,
    guard=None,
) -> list[Evidence]:
    """``evidence`` plus the phishing_url_model item. Verify mode: always (when the URL is over
    threshold). Quiet mode: never under policy "off"; under "corroborated", only when rule evidence
    already fired or the student score is over its threshold.
    Shared by the service and eval_groundlm."""
    policy = URL_QUIET_POLICY if quiet_policy is None else quiet_policy
    if policy not in URL_QUIET_POLICIES:
        raise ValueError(f"quiet_policy must be one of {URL_QUIET_POLICIES}, got {policy!r}")
    if check_mode(mode) == URL_EVIDENCE_MODE:
        return add_url_model_evidence(evidence, canon, hit, threshold)
    if policy == "corroborated":
        student_over = p_scam is not None and p_threshold is not None and p_scam >= p_threshold
        if evidence or student_over:
            return add_url_model_evidence(evidence, canon, hit, threshold)
    if policy == "guarded" and hit is not None:
        g = guard if guard is not None else default_guard()
        if g is None:
            raise RuntimeError("url quiet policy 'guarded' needs the reputation list file")
        if not g.established(hit.url):
            return add_url_model_evidence(evidence, canon, hit, threshold)
    return list(evidence)


def scam_state_for(p_scam: float, threshold: float, evidence: list) -> str:
    """Decision state from the student score and the rule evidence (as NIDS assess_flow)."""
    if p_scam >= threshold:
        return "scam"
    return "uncertain" if evidence else "benign"


def detector_only_package(package: EvidencePackage) -> EvidencePackage:
    """Verify mode: the package for scam_state "scam" with no rule evidence. The risk becomes
    UNCERTAIN, which needs no reasons, and the only allowed action is VERIFY_INDEPENDENTLY."""
    return dataclasses.replace(
        package, risk=Risk.UNCERTAIN, allowed_actions=[Action.VERIFY_INDEPENDENTLY]
    )


def decide_edge(
    canon: Canon,
    evidence: list[Evidence],
    p_scam: float,
    threshold: float,
    mode: str = DEFAULT_DETECTOR_ONLY_MODE,
) -> EdgeDecision:
    """The edge decision for one message: state, package (after the mode), action."""
    check_mode(mode)
    state = scam_state_for(p_scam, threshold, evidence)
    package = build_package(canon, evidence, state)
    detector_only = state == "scam" and not evidence
    if detector_only and mode == "verify":
        package = detector_only_package(package)
    return EdgeDecision(
        state=state,
        package=package,
        action=choose_action(package),
        detector_only=detector_only,
    )
