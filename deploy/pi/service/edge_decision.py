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

DETECTOR_ONLY_MODES = ("quiet", "verify")
DEFAULT_DETECTOR_ONLY_MODE = "quiet"


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
