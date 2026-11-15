"""The edge SMS advisory pipeline: canonicalise, student score and rules, decision, advisory.

Detection is the shipped SMS path: the SMS student ONNX (``sms_student``, decision threshold
from the registry) and the rule evidence (``rakshak.rules``). The decision is
``edge_decision.decide_edge``, which ``eval_groundlm.py`` also calls. It mirrors the NIDS state
logic in ``rakshak.nids.pi_monitor.assess_flow``: score >= threshold gives ``scam``; otherwise
rule evidence raises ``benign`` to ``uncertain``; otherwise ``benign``. ``build_package`` then
gives the risk (LOW, UNCERTAIN, MEDIUM, HIGH) and the allowed actions, and ``choose_action`` the
advisory action. The explainer runs through ``rbench.slot`` with the edge guards on (G1 and G2,
see src/rakshak/verifier.py).

Detector-only flags (score >= threshold, no rule evidence) follow ``detector_only_mode``: "quiet"
(default, the demo decision) gives the benign-style no-warning path, "verify" the UNCERTAIN notice.

The detector and the LLM runner are injected, so tests run without a model.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable
from typing import Any, Protocol

from edge_decision import (  # noqa: F401 - scam_state_for and detector_only_package re-exported
    DEFAULT_DETECTOR_ONLY_MODE,
    add_url_evidence,
    check_mode,
    decide_edge,
    detector_only_package,
    scam_state_for,
    url_model_active,
)
from rbench.slot import HINGLISH_LANG, SLOT_MAX_NEW_TOKENS, generate_slot_advisory

from rakshak.canonicalize import canonicalize
from rakshak.datasets.lang import detect_lang
from rakshak.evidence import EvidencePackage, build_package, choose_action
from rakshak.remediation import remediation_for, remediation_problems
from rakshak.rules import run_rules
from rakshak.url_features import registered_domain
from rakshak.url_model import (
    URL_MODEL_TYPE,
    UrlScorer,
    add_url_model_evidence,
    best_url_hit,
)
from rakshak.verifier import check_points, template_only_advisory

LANG = "en"
# Per-request language (docs/EDGE_PIPELINE.md, "Hinglish"). The request field "lang" is "auto"
# (default: route on detect_lang), "en" or "hinglish". A text that detect_lang puts in hinglish
# routes to hinglish under "auto"; hi_deva and other route to en for now.
LANG_REQUESTS = ("auto", "en", "hinglish")
HINGLISH_MODES = ("template", "llm")
DEFAULT_HINGLISH_MODE = "template"
PATH_TEMPLATE_HINGLISH = "template_hinglish"
# Fix A: the explainer prompt variant (a PROMPT_VARIANTS name in rbench.slot), or None for the
# default slot prompt. "describe_kw" won the prompt trial (docs/EDGE_PIPELINE.md, Fix A).
EDGE_PROMPT_VARIANT: str | None = "describe_kw"
# Fix B, quiet mode (the default): the whole user-visible text for a detector-only flag. It is
# neutral: it does not say the message is safe and does not say "no known scam pattern".
DETECTOR_ONLY_QUIET_SUMMARY = "No specific scam pattern was found in this message."
# Fix B, verify mode: the summary for a scam-flagged message with no rule evidence. Deterministic.
# It never says "no known scam pattern", because the detector fired.
DETECTOR_ONLY_SUMMARY = (
    "Our on-device detector flagged this message, but no specific scam pattern was matched. "
    "Do not click links or give out codes until you have checked with the sender through an "
    "official channel."
)
# The same two summaries in Hinglish, for a detector-only flag that routes to hinglish.
DETECTOR_ONLY_QUIET_SUMMARY_HINGLISH = "Is message mein koi specific scam pattern nahi mila."
DETECTOR_ONLY_SUMMARY_HINGLISH = (
    "Hamare on-device detector ne is message ko flag kiya, lekin koi specific scam pattern match "
    "nahi hua. Official channel se sender ko check kiye bina link par click ya code na dein."
)


def route_lang(lang_detected: str, requested: str = "auto") -> str:
    """The language a request is answered in: "en" or "hinglish".

    ``lang_detected`` is detect_lang(raw text). A request of "en" or "hinglish" overrides it.
    Under "auto", only a hinglish detection routes to hinglish; hi_deva and other stay "en".
    """
    if requested not in LANG_REQUESTS:
        raise ValueError(f"lang must be one of {LANG_REQUESTS}, got {requested!r}")
    if requested != "auto":
        return requested
    return HINGLISH_LANG if lang_detected == "hinglish" else LANG


def template_gen(package: EvidencePackage, lang: str) -> dict[str, Any]:
    """The explainer output for hinglish template mode: no LLM call, the verifier's Hinglish
    templates. Keys match generate_slot_advisory, so the assembly in analyse() is shared."""
    output = json.dumps(template_only_advisory(package, lang), ensure_ascii=False)
    verdict = check_points(output, package, lang, package.risk)
    n = len(package.evidence)
    return {
        "raw_output": "",
        "output": output,
        "calls": [],
        "skipped": False,
        "point_checks": [[c.type, c.ok] for c in verdict.points],
        "point_problems": [list(c.problems) for c in verdict.points],
        "point_status": ["template"] * n,
        "point_sources": ["template"] * n,
        "g2_rejected": [],
        "cleanup": [],
        "final_verifier_ok": check_points(output, package, lang).ok,
    }


class Detector(Protocol):
    def score(self, text: str) -> float:  # probability that the SMS is a scam
        ...


class Generator(Protocol):
    def generate(
        self, prompt: str, max_new_tokens: int = 64, stop_when: Callable[[str], bool] | None = None
    ) -> tuple[str, dict[str, Any]]: ...


def collapse_by_type(package: EvidencePackage) -> EvidencePackage:
    """Fix D: the package the explainer sees, with one evidence item per type (the first of each).

    A message with several items of one type (row 4: six prize_lottery items) would otherwise get
    one explainer call and one point per item, and the same sentence repeats. The verifier needs
    exactly one point per item of the package it is given, so the advisory is built against this
    collapsed package. Risk, actions and summary are unchanged: they come from the full package,
    and the template summary lists the types in first-occurrence order either way.
    """
    first: dict[str, Any] = {}
    for e in package.evidence:
        first.setdefault(e.type, e)
    return dataclasses.replace(package, evidence=list(first.values()))


def ids_by_type(package: EvidencePackage) -> dict[str, list[str]]:
    """Fix D: every evidence id of the full package, grouped by type (first-occurrence order)."""
    out: dict[str, list[str]] = {}
    for e in package.evidence:
        out.setdefault(e.type, []).append(e.id)
    return out


def _ms(t0: float, t1: float) -> float:
    return round((t1 - t0) * 1000.0, 3)


class EdgeAnalyzer:
    def __init__(
        self,
        detector: Detector,
        llm: Generator,
        threshold: float,
        *,
        max_new_tokens: int = SLOT_MAX_NEW_TOKENS,
        clock: Callable[[], float] = time.perf_counter,
        detector_only_mode: str = DEFAULT_DETECTOR_ONLY_MODE,
        hinglish_mode: str = DEFAULT_HINGLISH_MODE,
        url_scorer: UrlScorer | None = None,
        url_threshold: float | None = None,
    ) -> None:
        """``url_scorer`` and ``url_threshold`` together turn the URL model on. With either one
        None (the registry has no threshold for url_student) the URL path is off and the
        decision is the same as before it existed. The URL model adds evidence in verify mode
        only: in quiet mode the scorer is never called (edge_decision.URL_EVIDENCE_MODE)."""
        if hinglish_mode not in HINGLISH_MODES:
            raise ValueError(
                f"hinglish_mode must be one of {HINGLISH_MODES}, got {hinglish_mode!r}"
            )
        self.detector = detector
        self.llm = llm
        self.threshold = float(threshold)
        self.max_new_tokens = max_new_tokens
        self.clock = clock
        self.detector_only_mode = check_mode(detector_only_mode)
        self.hinglish_mode = hinglish_mode
        self.url_scorer = url_scorer
        self.url_threshold = None if url_scorer is None else url_threshold
        self.url_active = url_model_active(self.detector_only_mode, self.url_threshold)

    def analyse(self, text: str, lang: str = "auto") -> dict[str, Any]:
        clock = self.clock
        t_start = clock()
        canon = canonicalize(text)
        t_canon = clock()
        p_scam = float(self.detector.score(text))
        t_detect = clock()
        hit = best_url_hit(canon.urls, self.url_scorer) if self.url_active else None
        evidence = add_url_evidence(
            run_rules(canon), canon, hit, self.url_threshold, self.detector_only_mode
        )
        dec = decide_edge(canon, evidence, p_scam, self.threshold, self.detector_only_mode)
        state, package, action = dec.state, dec.package, dec.action
        t_evidence = clock()
        lang_detected = detect_lang(text)
        lang_route = route_lang(lang_detected, lang)
        if dec.detector_only:
            return self._detector_only_result(
                p_scam,
                dec,
                (t_start, t_canon, t_detect, t_evidence),
                lang=lang_route,
                lang_detected=lang_detected,
            )
        remediation = remediation_for(package, action.value, lang_route)
        llm_package = collapse_by_type(package)
        template_hinglish = lang_route == HINGLISH_LANG and self.hinglish_mode == "template"
        if template_hinglish:
            gen = template_gen(llm_package, lang_route)
        else:
            gen = generate_slot_advisory(
                self.llm,
                llm_package,
                lang=lang_route,
                max_new_tokens=self.max_new_tokens,
                guards=True,
                variant=None if lang_route == HINGLISH_LANG else EDGE_PROMPT_VARIANT,
                cleanup=True,
            )
        t_llm_done = clock()
        calls = gen["calls"]
        llm_total = sum(float(c["wall_ms"] or 0.0) for c in calls)
        t_end = clock()
        verify_ms = max(_ms(t_evidence, t_llm_done) - llm_total, 0.0)
        final = json.loads(gen["output"])
        raw_points = json.loads(gen["raw_output"])["points"] if gen["raw_output"] else []
        cleanup = gen.get("cleanup") or []
        sources = gen["point_sources"]
        skipped = bool(gen["skipped"])
        points = []
        groups = ids_by_type(package)
        for i, e in enumerate(llm_package.evidence):
            check = {"ok": None, "problems": []}
            if not skipped:
                check = {
                    "ok": bool(gen["point_checks"][i][1]),
                    "problems": gen["point_problems"][i],
                }
            points.append(
                {
                    "evidence_id": e.id,
                    "evidence_ids": groups[e.type],
                    "count": len(groups[e.type]),
                    "type": e.type,
                    "text": final["points"][i],
                    "source": sources[i],
                    "llm_text": cleanup[i]["original"]
                    if cleanup
                    else (raw_points[i] if raw_points else None),
                    "cleanup": {
                        "echo_stripped": cleanup[i]["echo_stripped"],
                        "duplicate_of": cleanup[i]["duplicate_of"],
                    }
                    if cleanup
                    else None,
                    "check": check,
                }
            )
        if skipped or not any(p["source"] == "llm" for p in points):
            mode = "template"
        elif all(p["source"] == "llm" for p in points):
            mode = "llm"
        else:
            mode = "partial"
        g2 = [
            {
                "evidence_id": package.evidence[c["index"]].id,
                "sentence": c["sentence"],
                "problem": c["problem"],
            }
            for c in gen["g2_rejected"]
        ]
        return {
            "decision": {
                "risk": package.risk.value,
                "action": action.value,
                "score": round(p_scam, 4),
                "threshold": round(self.threshold, 4),
                "scam_state": state,
                "confidence_state": package.confidence_state,
                "evidence_count": len(package.evidence),
                "detector_only": False,
                "lang_detected": lang_detected,
            },
            "evidence": [
                {"id": e.id, "type": e.type, "weight": round(e.weight, 3), "text": e.text}
                for e in package.evidence
            ],
            "advisory": {
                "lang": lang_route,
                "mode": mode,
                "path": "low_template"
                if skipped
                else (PATH_TEMPLATE_HINGLISH if template_hinglish else "explainer"),
                "llm_skipped_low_risk": skipped,
                "summary": {
                    "text": final["summary"],
                    "source": "template",
                    "check": "template, not an LLM sentence",
                },
                "points": points,
                "evidence_items": len(package.evidence),
                "action": action.value,
            },
            "remediation": remediation,
            "verifier": {
                "final_contract_ok": bool(gen["final_verifier_ok"]),
                "g2_rejected": g2,
                "remediation_ok": not remediation_problems(remediation, package),
            },
            "llm_calls": [
                {
                    "evidence_id": c["evidence_id"],
                    "status": c["status"],
                    "prompt_tokens": c.get("prompt_tokens"),
                    "new_tokens": c.get("new_tokens"),
                    "ttft_ms": c.get("ttft_ms"),
                    "prefill_tokens": c.get("prefill_tokens"),
                    "prefix_reused": c.get("prefix_reused"),
                    "wall_ms": c.get("wall_ms"),
                    "stop_hit": c.get("stop_hit"),
                    "cached": bool(c.get("cached", False)),
                }
                for c in calls
            ],
            "timings_ms": {
                "canonicalise": _ms(t_start, t_canon),
                "detect": _ms(t_canon, t_detect),
                "evidence": _ms(t_detect, t_evidence),
                "llm_calls": [round(float(c["wall_ms"] or 0.0), 3) for c in calls],
                "llm_total": round(llm_total, 3),
                "llm_cached": sum(1 for c in calls if c.get("cached")),
                "verify": round(verify_ms, 3),
                "total": _ms(t_start, t_end),
            },
        }

    def _detector_only_result(
        self, p_scam: float, dec, stamps: tuple, *, lang: str = LANG, lang_detected: str = "en"
    ) -> dict[str, Any]:
        """Detector-only path (Fix B), no explainer call, no points.

        quiet: the decision is LOW and IGNORE (benign-style) and the summary is the neutral
        DETECTOR_ONLY_QUIET_SUMMARY. verify: the decision is UNCERTAIN and VERIFY_INDEPENDENTLY
        with DETECTOR_ONLY_SUMMARY (the notice shown before the demo decision).
        """
        t_start, t_canon, t_detect, t_evidence = stamps
        package, action, state = dec.package, dec.action, dec.state
        quiet = self.detector_only_mode == "quiet"
        if lang == HINGLISH_LANG:
            summary = (
                DETECTOR_ONLY_QUIET_SUMMARY_HINGLISH if quiet else DETECTOR_ONLY_SUMMARY_HINGLISH
            )
        else:
            summary = DETECTOR_ONLY_QUIET_SUMMARY if quiet else DETECTOR_ONLY_SUMMARY
        final = {"lang": lang, "summary": summary, "points": []}
        contract_ok = check_points(json.dumps(final, ensure_ascii=False), package, lang).ok
        remediation = remediation_for(package, action.value, lang)
        t_end = self.clock()
        return {
            "decision": {
                "risk": package.risk.value,
                "action": action.value,
                "score": round(p_scam, 4),
                "threshold": round(self.threshold, 4),
                "scam_state": state,
                "confidence_state": package.confidence_state,
                "evidence_count": 0,
                "detector_only": True,
                "lang_detected": lang_detected,
            },
            "evidence": [],
            "advisory": {
                "lang": lang,
                "mode": "template",
                "path": "detector_only_quiet" if quiet else "detector_only",
                "llm_skipped_low_risk": False,
                "summary": {
                    "text": summary,
                    "source": "template",
                    "check": (
                        "quiet detector-only template (no warning), not an LLM sentence"
                        if quiet
                        else "detector-only template (Fix B), not an LLM sentence"
                    ),
                },
                "points": [],
                "action": action.value,
            },
            "remediation": remediation,
            "verifier": {
                "final_contract_ok": bool(contract_ok),
                "g2_rejected": [],
                "remediation_ok": not remediation_problems(remediation, package),
            },
            "llm_calls": [],
            "timings_ms": {
                "canonicalise": _ms(t_start, t_canon),
                "detect": _ms(t_canon, t_detect),
                "evidence": _ms(t_detect, t_evidence),
                "llm_calls": [],
                "llm_total": 0.0,
                "llm_cached": 0,
                "verify": 0.0,
                "total": _ms(t_start, t_end),
            },
        }


# ----------------------------------------------------------------- POST /analyze_url
# Rule types a bare URL can trigger. The SMS-only rules are not run on a link check.
URL_RULE_TYPES = ("brand_domain_mismatch", "shortened_url", "suspicious_url")


def check_url(
    url_text: str,
    scorer: UrlScorer | None,
    threshold: float | None,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """POST /analyze_url: the URL score, the domain rules, a verdict and remediation for one link.

    ``verdict`` is "phishing" when the URL model is at or above its threshold, or a brand-domain
    mismatch fired; "suspicious" when only the shortened or suspicious-URL rules fired; otherwise
    "no_signal". The state is never "scam" (there is no SMS student here), so the model alone
    gives UNCERTAIN; a brand mismatch (weight 0.9) gives HIGH. No explainer runs.
    """
    active = scorer is not None and threshold is not None
    t0 = clock()
    canon = canonicalize(url_text)
    t_canon = clock()
    hit = best_url_hit(canon.urls, scorer) if active else None
    rules = [e for e in run_rules(canon) if e.type in URL_RULE_TYPES]
    evidence = add_url_model_evidence(rules, canon, hit, threshold if active else None)
    t_evidence = clock()
    state = "uncertain" if evidence else "benign"
    package = build_package(canon, evidence, state)
    action = choose_action(package)
    remediation = remediation_for(package, action.value, LANG)
    model_fired = any(e.type == URL_MODEL_TYPE for e in evidence)
    brand = any(e.type == "brand_domain_mismatch" for e in evidence)
    if model_fired or brand:
        verdict = "phishing"
    elif evidence:
        verdict = "suspicious"
    else:
        verdict = "no_signal"
    t_end = clock()
    return {
        "verdict": verdict,
        "risk": package.risk.value,
        "action": action.value,
        "url_count": len(canon.urls),
        "url_model": {
            "enabled": active,
            "threshold": round(float(threshold), 4) if active else None,
            "score": round(hit.score, 4) if hit is not None else None,
            "domain": (registered_domain(hit.url) or None) if hit is not None else None,
            "fired": model_fired,
        },
        "evidence": [
            {"id": e.id, "type": e.type, "weight": round(e.weight, 3), "text": e.text}
            for e in evidence
        ],
        "remediation": remediation,
        "remediation_ok": not remediation_problems(remediation, package),
        "timings_ms": {
            "canonicalise": _ms(t0, t_canon),
            "url_and_rules": _ms(t_canon, t_evidence),
            "total": _ms(t0, t_end),
        },
    }
