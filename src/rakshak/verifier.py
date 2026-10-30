"""Deterministic verifier for LLM explanations, plus the template fallback.

The LLM must return a JSON object ``{"summary": str, "reasons": [E*], "action": str}``.
``verify`` accepts it only if every claim is grounded in the evidence package:

* reasons are evidence ids from the package, and non-empty for HIGH and MEDIUM risk;
* action is one of the package's allowed actions;
* summary is at most 280 characters;
* every URL / domain / phone number in the summary appears in the package;
* the summary never calls the message safe or legitimate when risk is HIGH;
* the summary contains no positive instruction to click, call, pay or share, and no
  prompt-injection phrasing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rakshak.evidence import Action, EvidencePackage, Risk

MAX_SUMMARY = 280

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_URL_RE = re.compile(r"(?i)\b(?:https?|hxxps?)://\S+")
_DOMAIN_RE = re.compile(r"(?i)\b(?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+[a-z]{2,24}\b")
_PHONE_RE = re.compile(r"\+?\d[\d \-]{8,}\d")
_SAFE_RE = re.compile(r"(?i)\b(safe|legitimate|genuine|authentic|trusted|harmless|no risk)\b")
_ACTION_VERB_RE = re.compile(
    r"(?i)\b(click|tap|call|dial|pay|send|share|reply|download|install|transfer|"
    r"open|visit|forward|enter|provide|approve|accept)\b"
)
_INJECTION_RE = re.compile(
    r"(?i)(ignore (all |the |any )?(previous|above|prior|earlier) (instructions|rules|prompt)|"
    r"disregard (all |the )?(previous|above|prior)|system prompt|you are now|"
    r"new instructions|as an ai|developer mode|jailbreak)"
)
_NEGATION_BEFORE = re.compile(
    r"(?i)\b(not|never|don'?t|dont|do not|nahi|nahin|mat|kabhi|na|no)\b"
)
_NEGATION_AFTER = re.compile(r"(?i)^\W*(?:nahi|nahin|mat|na|not|never)\b")


@dataclass
class VerifyResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    parsed: dict[str, Any] | None = None


def _extract_json(raw: str) -> dict[str, Any] | None:
    candidates: list[str] = []
    fence = _FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1))
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    candidates.append(raw)
    for c in candidates:
        try:
            obj = json.loads(c)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _is_negated(text: str, start: int, end: int) -> bool:
    """True when a negation word sits just before or just after the matched span."""
    before = text[max(0, start - 14) : start]
    after = text[end : end + 14]
    if _NEGATION_BEFORE.search(before):
        return True
    return bool(_NEGATION_AFTER.search(after))


def _positive_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    hits = []
    for m in pattern.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            hits.append(m.group(0))
    return hits


def _grounded_domain(candidate: str, package: EvidencePackage) -> bool:
    cand = candidate.lower().rstrip(".")
    allowed = package.allowed_domain_names()
    if cand in allowed:
        return True
    return any(cand in u.lower() for u in package.urls)


def verify(llm_output: str, package: EvidencePackage) -> VerifyResult:
    """Check an LLM explanation against the evidence package.

    Returns ``VerifyResult(ok, reasons, parsed)``. ``reasons`` lists every rejected check
    (empty when ``ok``). ``parsed`` is the decoded JSON object when parsing succeeded.
    """
    problems: list[str] = []

    obj = _extract_json(llm_output or "")
    if obj is None:
        return VerifyResult(False, ["output is not a JSON object"], None)

    for key in ("summary", "reasons", "action"):
        if key not in obj:
            problems.append(f"missing key: {key}")
    if problems:
        return VerifyResult(False, problems, obj)

    summary, reasons, action = obj["summary"], obj["reasons"], obj["action"]

    # --- reasons -----------------------------------------------------------
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        problems.append("reasons must be a list of strings")
    else:
        unknown = [r for r in reasons if r not in package.evidence_ids]
        if unknown:
            problems.append(f"reasons cite unknown evidence ids: {unknown}")
        if package.risk in (Risk.HIGH, Risk.MEDIUM) and not reasons:
            problems.append("reasons must be non-empty for HIGH/MEDIUM risk")

    # --- action ------------------------------------------------------------
    allowed_values = {a.value for a in package.allowed_actions}
    if not isinstance(action, str) or action not in allowed_values:
        problems.append(f"action {action!r} is not in allowed actions {sorted(allowed_values)}")

    # --- summary -----------------------------------------------------------
    if not isinstance(summary, str):
        problems.append("summary must be a string")
        return VerifyResult(False, problems, obj)
    if len(summary) > MAX_SUMMARY:
        problems.append(f"summary is {len(summary)} chars, limit is {MAX_SUMMARY}")

    for m in _URL_RE.finditer(summary):
        if m.group(0).lower() not in " ".join(package.urls).lower():
            problems.append(f"summary contains URL not in package: {m.group(0)!r}")
    url_spans = [m.span() for m in _URL_RE.finditer(summary)]
    for m in _DOMAIN_RE.finditer(summary):
        if any(s <= m.start() < e for s, e in url_spans):
            continue
        if not _grounded_domain(m.group(0), package):
            problems.append(f"summary mentions domain not in package: {m.group(0)!r}")

    for m in _PHONE_RE.finditer(summary):
        digits = re.sub(r"\D", "", m.group(0))
        if digits not in package.phones:
            problems.append(f"summary contains phone number not in package: {m.group(0)!r}")

    if package.risk is Risk.HIGH and _positive_matches(_SAFE_RE, summary):
        problems.append("summary calls the message safe or legitimate while risk is HIGH")

    for verb in _positive_matches(_ACTION_VERB_RE, summary):
        problems.append(f"summary gives a positive instruction: {verb!r}")

    if _INJECTION_RE.search(summary):
        problems.append("summary contains prompt-injection phrasing")

    return VerifyResult(ok=not problems, reasons=problems, parsed=obj)


# ---------------------------------------------------------------------------
# Deterministic template fallback
# ---------------------------------------------------------------------------

_TYPE_PHRASE = {
    "en": {
        "credential_request": "asks for an OTP, PIN or password",
        "upi_collect_or_pay": "uses a UPI collect or payment request",
        "kyc_block_urgency": "uses KYC urgency or a block threat",
        "account_suspension": "claims an account or card is suspended",
        "prize_lottery": "claims a prize or lottery",
        "apk_install": "asks for an app or APK installation",
        "remote_access_app": "asks for a remote-access app",
        "electricity_disconnect": "threatens electricity disconnection",
        "parcel_customs": "asks for a parcel or customs fee",
        "job_task_scam": "offers a task or job with easy earnings",
        "investment_returns": "promises guaranteed investment returns",
        "digital_arrest": "claims to be a police or government agency",
        "brand_domain_mismatch": "links to a domain that does not match the brand named",
        "shortened_url": "contains a shortened link",
        "suspicious_url": "contains a suspicious link",
    },
    "hinglish": {
        "credential_request": "OTP, PIN ya password maangta hai",
        "upi_collect_or_pay": "UPI collect ya payment request bhejta hai",
        "kyc_block_urgency": "KYC ke naam par block ka dar dikhata hai",
        "account_suspension": "account ya card band hone ka dawa karta hai",
        "prize_lottery": "inaam ya lottery ka dawa karta hai",
        "apk_install": "app ya APK installation maangta hai",
        "remote_access_app": "remote access app maangta hai",
        "electricity_disconnect": "bijli katne ki dhamki deta hai",
        "parcel_customs": "parcel ya customs fee maangta hai",
        "job_task_scam": "asaan kamai ka task ya job offer deta hai",
        "investment_returns": "guaranteed investment returns ka vaada karta hai",
        "digital_arrest": "police ya sarkari agency hone ka dawa karta hai",
        "brand_domain_mismatch": "brand ka naam le kar galat domain ka link deta hai",
        "shortened_url": "chhota (shortened) link deta hai",
        "suspicious_url": "shaq wala link deta hai",
    },
}

_ACTION_PRIORITY = [
    Action.DO_NOT_SHARE_OTP,
    Action.DO_NOT_CLICK,
    Action.REPORT_1930_OR_CHAKSHU,
    Action.CONTACT_BANK_OFFICIAL,
    Action.OPEN_OFFICIAL_APP,
    Action.BLOCK_SENDER,
    Action.VERIFY_INDEPENDENTLY,
    Action.IGNORE,
]

_SAFE_TEXT = {
    "en": {
        "high": (
            "Do not share OTP or PIN, and do not click links. "
            "Verify through the official app or the number on your card."
        ),
        "medium": "Verify this independently before acting. Do not share OTP or PIN.",
        "uncertain": (
            "Could not confirm this. "
            "Verify through the official app or the number on your card."
        ),
        "low": "No scam signals were found by the rules. Stay cautious with unexpected requests.",
    },
    "hinglish": {
        "high": (
            "OTP ya PIN share na karein, aur link par click na karein. "
            "Official app ya card ke number se verify karein."
        ),
        "medium": "Action lene se pehle ise khud verify karein. OTP ya PIN share na karein.",
        "uncertain": "Ise confirm nahi kar paye. Official app ya card ke number se verify karein.",
        "low": "Rules ne koi scam sanket nahi paaya. Anjaan requests par dhyan rakhein.",
    },
}


def template_explanation(package: EvidencePackage, lang: str = "en") -> dict[str, Any]:
    """Deterministic explanation that always passes ``verify`` for the same package."""
    if lang not in _TYPE_PHRASE:
        raise ValueError(f"lang must be 'en' or 'hinglish', got {lang!r}")
    phrases = _TYPE_PHRASE[lang]
    risk_key = package.risk.value.lower()
    base = _SAFE_TEXT[lang][risk_key]

    seen_types: list[str] = []
    for e in package.evidence:
        if e.type not in seen_types:
            seen_types.append(e.type)
    if seen_types:
        described = "; ".join(phrases[t] for t in seen_types[:3])
        prefix = (
            f"This message {described}."
            if lang == "en"
            else f"Is message mein: {described}."
        )
        summary = f"{prefix} {base}"
    else:
        summary = base

    summary = summary[:MAX_SUMMARY]

    action = next(
        (a for a in _ACTION_PRIORITY if a in package.allowed_actions),
        package.allowed_actions[0],
    )
    reasons = [e.id for e in package.evidence]
    return {"summary": summary, "reasons": reasons, "action": action.value}
