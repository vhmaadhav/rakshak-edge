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
from functools import lru_cache
from importlib import resources
from typing import Any

from rakshak.evidence import Action, EvidencePackage, Risk, choose_action
from rakshak.nids.evidence import NETWORK_DOMAIN, NetAction

# Deliberately looser than the prompt tells the model. The prompt asks for "at most 200
# characters" (rakshak.prompts, POINTS_SYSTEM) while this accepts 280, and MAX_POINT is 160 in both.
# Instructing conservatively and verifying generously avoids throwing away a sound explanation for
# a length overrun the user would not notice; the limit exists to stop a rambling output, not to
# enforce the exact number in the instructions. Do not "fix" the mismatch by tightening this.
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
    if package.domain == NETWORK_DOMAIN and isinstance(action, str):
        sms_only = {a.value for a in Action} - {a.value for a in NetAction}
        if action in sms_only:
            problems.append(f"action {action!r} is an SMS-only code, not valid for network")

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
        "urgency": "uses time pressure (turant, abhi, 24 hours)",
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
        "urgency": "jaldi karne ka dabav dalta hai (turant, abhi, 24 ghante)",
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
        # LOW means the rules found nothing, NOT that the message is safe: on the campaign test
        # split 83.8% of labelled scams produce no evidence at all, and this string is what the
        # user sees for every one of them. The old wording ("No scam signals were found by the
        # rules") was accurate about the mechanism but read as a clean bill of health, which is
        # the worst failure mode a scam detector can have. It now states the limit of what was
        # checked and keeps the caution. Do not weaken this back into a reassurance.
        "low": (
            "These checks found no known scam pattern, which does not make the message safe. "
            "Never share an OTP, PIN or password, and verify unexpected requests yourself."
        ),
    },
    "hinglish": {
        "high": (
            "OTP ya PIN share na karein, aur link par click na karein. "
            "Official app ya card ke number se verify karein."
        ),
        "medium": "Action lene se pehle ise khud verify karein. OTP ya PIN share na karein.",
        "uncertain": "Ise confirm nahi kar paye. Official app ya card ke number se verify karein.",
        # Same reasoning as the English LOW string above.
        "low": (
            "In jaanch mein koi jaana-pehchana scam pattern nahi mila, lekin isse message safe "
            "nahi hota. OTP, PIN ya password kabhi share na karein, aur anjaan requests khud "
            "verify karein."
        ),
    },
}


def _template_summary(package: EvidencePackage, lang: str) -> str:
    """Deterministic summary sentence(s) for a package, at most MAX_SUMMARY characters."""
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
    return summary[:MAX_SUMMARY]


def template_explanation(package: EvidencePackage, lang: str = "en") -> dict[str, Any]:
    """Deterministic explanation that always passes ``verify`` for the same package."""
    summary = _template_summary(package, lang)

    action = next(
        (a for a in _ACTION_PRIORITY if a in package.allowed_actions),
        package.allowed_actions[0],
    )
    reasons = [e.id for e in package.evidence]
    return {"summary": summary, "reasons": reasons, "action": action.value}


# ---------------------------------------------------------------------------
# ELA verifier v2: the points contract. ``verify`` above stays as the v1 contract that the
# network path (rakshak.nids.pi_monitor) still uses.
#
# The explainer returns {"summary": str, "points": [str, ...]} (and an optional "lang"), with one
# point per evidence item in package order. The system attaches the ids and the action, so the
# output never carries them. ``check_points`` says what passes. ``render_advisory`` swaps each
# failing point, and the summary when it fails, for the deterministic template sentence.
# ---------------------------------------------------------------------------

TYPE_PHRASE = _TYPE_PHRASE  # public name used by rakshak.prompts
MAX_POINT = 160
V2_LANGS = ("en", "hinglish")
_V2_KEYS = frozenset({"summary", "points", "lang"})
_LEXICON_FILE = "data/ela_lexicon.json"

# Hinglish imperatives that instruct the reader. Bare "karein" is not listed: "verify karein" is
# advice to check, not an action request. "do" is not listed because "Do not" is common.
_HINGLISH_POSITIVE_RE = re.compile(
    r"(?i)\b(bhejo|bhejein|bhejiye|bhejen|batao|bataiye|bataye|dijiye|dijie|dein|daalo|dalo|"
    r"kholo|kholein|dabao|dabayein|click karo|click karein|call karo|call karein|"
    r"share karo|share karein|pay karo|pay karein|install karo|install karein|"
    r"download karo|download karein)\b"
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_CURRENCY_BEFORE_RE = re.compile(r"(?i)(?:rs\.?|inr|₹|rupees?)\s*$")


@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, dict[str, re.Pattern[str]]]:
    """Compiled per-language, per-type keyword patterns from src/rakshak/data/ela_lexicon.json.

    A keyword matches at a word start (case-insensitive), so "block" also matches "blocked".
    """
    raw = json.loads(
        resources.files("rakshak").joinpath(_LEXICON_FILE).read_text(encoding="utf-8")
    )
    compiled: dict[str, dict[str, re.Pattern[str]]] = {}
    for lang in V2_LANGS:
        table: dict[str, re.Pattern[str]] = {}
        for etype, words in raw[lang].items():
            alts = [r"\s+".join(re.escape(p) for p in w.split()) for w in words]
            table[etype] = re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + ")", re.IGNORECASE)
        compiled[lang] = table
    return compiled


def has_lexicon_keyword(text: str, lang: str, etype: str) -> bool:
    pattern = load_lexicon()[lang].get(etype)
    return bool(pattern is not None and pattern.search(text))


def _object_span(raw: str) -> tuple[int, int] | None:
    """Span of the first balanced {...} object, skipping braces inside JSON strings."""
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def json_complete(raw: str) -> bool:
    """True once ``raw`` contains a complete top-level JSON object (used as a stop rule)."""
    return _object_span(raw) is not None


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """The first balanced JSON object in ``raw`` as a dict, or None. Text around it is ignored."""
    span = _object_span(raw or "")
    if span is None:
        return None
    try:
        obj = json.loads(raw[span[0] : span[1]])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _blank(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for s, e in spans:
        for k in range(s, e):
            chars[k] = " "
    return "".join(chars)


def _slot_numbers(package: EvidencePackage) -> set[str]:
    """Digit strings that the package itself contains (evidence text, URLs, domains, phones)."""
    brand = [d for doms in package.brand_domains.values() for d in doms]
    vocab = " ".join(
        [*(e.text for e in package.evidence), *package.urls, *package.domains, *brand,
         *package.phones]
    )
    nums = {re.sub(r"\D", "", t) for t in _NUMBER_RE.findall(vocab)}
    nums.update(package.phones)
    return nums


def ungrounded_claims(text: str, package: EvidencePackage) -> list[str]:
    """URLs, domains, phone numbers and amounts in ``text`` that the package does not contain.

    A number is checked when it has three or more digits or follows a currency word. Small
    numbers such as "24 hours" are allowed.
    """
    problems: list[str] = []
    package_urls = " ".join(package.urls).lower()
    url_spans: list[tuple[int, int]] = []
    for m in _URL_RE.finditer(text):
        url_spans.append(m.span())
        if m.group(0).lower() not in package_urls:
            problems.append(f"URL not in package: {m.group(0)!r}")
    spans = list(url_spans)
    for m in _DOMAIN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in url_spans):
            continue
        spans.append(m.span())
        if not _grounded_domain(m.group(0), package):
            problems.append(f"domain not in package: {m.group(0)!r}")
    bare = _blank(text, spans)
    phones = set(package.phones)
    phone_spans: list[tuple[int, int]] = []
    for m in _PHONE_RE.finditer(bare):
        phone_spans.append(m.span())
        if re.sub(r"\D", "", m.group(0)) not in phones:
            problems.append(f"phone number not in package: {m.group(0)!r}")
    bare = _blank(bare, phone_spans)
    slots = _slot_numbers(package)
    for m in _NUMBER_RE.finditer(bare):
        digits = re.sub(r"\D", "", m.group(0))
        currency = bool(_CURRENCY_BEFORE_RE.search(bare[max(0, m.start() - 10) : m.start()]))
        if (len(digits) >= 3 or currency) and digits not in slots:
            problems.append(f"number not in package: {m.group(0)!r}")
    return problems


def _text_problems(
    text: Any,
    package: EvidencePackage,
    lang: str,
    role: str,
    etype: str | None = None,
    risk: Risk | None = None,
) -> list[str]:
    """Checks shared by points and the summary. ``role`` is "point" or "summary".

    ``risk`` is the opt-in attribution guard (G2, see ``attribution_problems``). None keeps the
    checks that existed before the guard.
    """
    if not isinstance(text, str):
        return [f"{role} is not a string"]
    body = text.strip()
    problems: list[str] = []
    if not body:
        problems.append(f"{role} is empty")
    limit = MAX_POINT if role == "point" else MAX_SUMMARY
    if len(body) > limit:
        problems.append(f"{role} is {len(body)} chars, limit {limit}")
    problems.extend(ungrounded_claims(body, package))
    if package.risk in (Risk.HIGH, Risk.MEDIUM):
        for word in _positive_matches(_SAFE_RE, body):
            problems.append(
                f"{role} calls the message {word.lower()} while risk is {package.risk.value}"
            )
    for verb in _positive_matches(_ACTION_VERB_RE, body) + _positive_matches(
        _HINGLISH_POSITIVE_RE, body
    ):
        problems.append(f"{role} gives a positive instruction: {verb!r}")
    if _INJECTION_RE.search(body):
        problems.append(f"{role} has prompt-injection phrasing")
    if etype is not None and not has_lexicon_keyword(body, lang, etype):
        problems.append(f"no {lang} lexicon keyword for {etype}")
    problems.extend(attribution_problems(body, risk))
    return problems


@dataclass
class PointCheck:
    index: int
    type: str
    ok: bool
    problems: list[str] = field(default_factory=list)


@dataclass
class PointsVerdict:
    """``ok`` is the full pass: schema, every point and the summary pass."""

    ok: bool
    schema_ok: bool
    schema_problems: list[str]
    summary_ok: bool
    summary_problems: list[str]
    points: list[PointCheck]
    parsed: dict[str, Any] | None
    lang: str


def _schema_failed(
    problems: list[str], types: list[str], parsed: dict[str, Any] | None, lang: str
) -> PointsVerdict:
    checks = [PointCheck(i, t, False, ["schema failed"]) for i, t in enumerate(types)]
    return PointsVerdict(
        ok=False,
        schema_ok=False,
        schema_problems=problems,
        summary_ok=False,
        summary_problems=["schema failed"],
        points=checks,
        parsed=parsed,
        lang=lang,
    )


def check_points(
    llm_output: str, package: EvidencePackage, lang: str = "en", risk: Risk | None = None
) -> PointsVerdict:
    """Check an explainer output against the points contract for ``package``.

    ``risk`` is the opt-in attribution guard (G2). None (the default) is the original contract,
    used by the Kaggle jobs. When given it must equal ``package.risk``; the guard then applies to
    points and summary (see ``attribution_problems``).

    Schema: a JSON object with exactly the keys summary and points (lang is optional and must be
    en or hinglish when present), and points is a list of exactly N strings, N being the number
    of evidence items. Any schema problem fails every point.
    Points and summary: see ``_text_problems``. Point i must contain a lexicon keyword for the
    type of evidence item i in ``lang``.
    """
    if lang not in V2_LANGS:
        raise ValueError(f"lang must be 'en' or 'hinglish', got {lang!r}")
    _check_risk_arg(risk, package)
    types = [e.type for e in package.evidence]
    n = len(types)
    obj = extract_json_object(llm_output or "")
    if obj is None:
        return _schema_failed(["output is not a JSON object"], types, None, lang)

    problems: list[str] = []
    extra = sorted(set(obj) - _V2_KEYS)
    if extra:
        problems.append(f"unexpected keys: {extra}")
    for key in ("summary", "points"):
        if key not in obj:
            problems.append(f"missing key: {key}")
    if "lang" in obj and obj["lang"] not in V2_LANGS:
        problems.append(f"lang {obj['lang']!r} is not en or hinglish")
    if "summary" in obj and not isinstance(obj["summary"], str):
        problems.append("summary must be a string")
    if "points" in obj:
        points = obj["points"]
        if not isinstance(points, list):
            problems.append("points must be a list")
        elif len(points) != n:
            problems.append(f"points has {len(points)} items, expected {n}")
        elif not all(isinstance(p, str) for p in points):
            problems.append("points must be strings")
    if problems:
        return _schema_failed(problems, types, obj, lang)

    checks = []
    for i, (etype, text) in enumerate(zip(types, obj["points"], strict=True)):
        probs = _text_problems(text, package, lang, "point", etype, risk)
        checks.append(PointCheck(i, etype, not probs, probs))
    summary_probs = _text_problems(obj["summary"], package, lang, "summary", None, risk)
    return PointsVerdict(
        ok=all(c.ok for c in checks) and not summary_probs,
        schema_ok=True,
        schema_problems=[],
        summary_ok=not summary_probs,
        summary_problems=summary_probs,
        points=checks,
        parsed=obj,
        lang=lang,
    )


def template_point(etype: str, lang: str = "en") -> str:
    """Deterministic one-sentence point for an evidence type. Passes ``check_points`` (tested)."""
    if lang not in V2_LANGS:
        raise ValueError(f"lang must be 'en' or 'hinglish', got {lang!r}")
    prefix = "This message " if lang == "en" else "Yeh message "
    return f"{prefix}{_TYPE_PHRASE[lang][etype]}."


def render_advisory(
    llm_output: str, package: EvidencePackage, lang: str = "en", risk: Risk | None = None
) -> dict[str, Any]:
    """The advisory a user sees: the system's action and ids, and each point from the explainer
    where it passed or from the template where it failed.

    ``mode`` is "llm" (full pass), "partial" (some points or the summary fell back) or
    "template" (the schema failed, so every sentence is the template).
    """
    verdict = check_points(llm_output, package, lang, risk)
    points_out: list[dict[str, Any]] = []
    for i, e in enumerate(package.evidence):
        if verdict.schema_ok and verdict.points[i].ok:
            text, source = verdict.parsed["points"][i].strip(), "llm"
        else:
            text, source = template_point(e.type, lang), "template"
        points_out.append({"evidence_id": e.id, "type": e.type, "text": text, "source": source})
    if verdict.summary_ok:
        summary, summary_source = verdict.parsed["summary"].strip(), "llm"
    else:
        summary, summary_source = _template_summary(package, lang), "template"
    if verdict.ok:
        mode = "llm"
    elif not verdict.schema_ok:
        mode = "template"
    else:
        mode = "partial"
    return {
        "lang": lang,
        "mode": mode,
        "action": choose_action(package).value,
        "reasons": [e.id for e in package.evidence],
        "summary": summary,
        "summary_source": summary_source,
        "points": points_out,
    }


# ---------------------------------------------------------------------------
# Edge guards (G1 and G2). Opt-in: nothing above changes unless a caller passes them.
#
# G1 (``skip_llm_for_risk``): a LOW-risk decision gets no explainer text. The advisory is the
# template (``template_only_advisory``) and the explainer is not called.
#
# G2 (``attribution_problems``): a point or summary that attributes scam intent to the message
# ("is a scam", "scammers", "a common tactic used by scammers", "trying to rush you") is rejected
# at LOW and MEDIUM risk. HIGH is not checked. At MEDIUM and UNCERTAIN only a hedged form is
# allowed: "may|could|might be a scam|fraud|phishing|fake", or "<scam|fraud|dhoka|dhokha|thagi>
# ho sakta|sakti|sakte hai". The match is case-insensitive, on word boundaries, and a negation
# is NOT an exemption: "not a scam" is rejected too, because a negated mention is still a claim
# about the sender. Templates are deterministic and are not run through G2 by the final checks.
# ---------------------------------------------------------------------------

_ATTRIBUTION_WORDS = (
    # scam / fraud / phishing family
    "scam", "scams", "scammer", "scammers", "scammed", "scamming", "scammy",
    "fraud", "frauds", "fraudster", "fraudsters", "fraudulent", "defraud", "defrauds",
    "defrauding", "phish", "phishing", "phisher", "phishers", "spoof", "spoofing",
    "conman", "conmen", "con artist", "con artists", "con man", "swindle", "swindler",
    "swindlers",
    # deception, theft and intent
    "trick", "tricks", "tricked", "tricking", "trickery", "cheat", "cheats", "cheated",
    "cheating", "steal", "steals", "stealing", "stole", "stolen", "theft", "thief", "thieves",
    "lure", "lures", "lured", "luring", "bait", "impersonate", "impersonates",
    "impersonating", "impersonation", "pretend", "pretends", "pretending", "hack", "hacker",
    "hackers", "hacking", "malicious", "bogus", "fake", "exploit", "exploits", "exploiting",
    "tactic", "tactics", "victim", "victims",
    # Hinglish (romanised) equivalents of the same words
    "dhoka", "dhokha", "dhokhaa", "dhokebaaz", "dhokebaz", "dhokebaazi", "thag", "thagi",
    "thagna", "thagne", "thagon", "thagega", "jaalsaaz", "jalsaaz", "chor", "chori",
    "lootera", "luteraa", "lutere", "loot", "nakli", "farzi",
)
_ATTRIBUTION_PHRASES = (
    r"trying to (?:rush|scare|pressure|trick|fool|cheat|steal|extract|dupe|deceive|defraud|con"
    r"|scam|hack|rob)",
    r"(?:rush|rushes|rushing|pressure|pressuring|scare|scaring|fool|fooling|dupe|duping|deceive"
    r"|deceiving|rob|robbing) you",
    r"take advantage",
    r"designed to (?:steal|trick|scam|cheat|defraud|rob|extract)",
    r"intends? to (?:steal|trick|scam|cheat|defraud|rob|extract)",
)


def _attribution_alternatives() -> list[str]:
    alts = [r"\s+".join(re.escape(p) for p in w.split()) for w in _ATTRIBUTION_WORDS]
    alts += list(_ATTRIBUTION_PHRASES)
    return alts


ATTRIBUTION_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:" + "|".join(_attribution_alternatives()) + r")(?![a-z0-9])"
)
# Hedged forms allowed at MEDIUM and UNCERTAIN. The whole phrase is the span; the attribution
# match must sit inside it.
HEDGE_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:(?:may|could|might)\s+be\s+(?:an?\s+)?(?:scam|fraud|phishing|fake)"
    r"|(?:scam|fraud|dhoka|dhokha|thagi)\s+ho\s+(?:sakta|sakti|sakte)\s+hai)(?![a-z0-9])"
)
# Added 2026-11-06 (Fix A): "is likely a scam|fraud|phishing" is a hedge too, at MEDIUM and
# UNCERTAIN only. The attribution guard still rejects any other attribution term in the same
# sentence. LOW stays strict, HIGH unchecked.
LIKELY_HEDGE_RE = re.compile(
    r"(?i)(?<![a-z0-9])likely\s+(?:an?\s+)?(?:scam|fraud|phishing)(?![a-z0-9])"
)
_HEDGED_RISKS = (Risk.MEDIUM, Risk.UNCERTAIN)


def attribution_problems(text: str, risk: Risk | None) -> list[str]:
    """G2 problems for one point or summary at ``risk``. Empty when the guard is off (None) or
    the risk is HIGH, and empty when every attribution term sits inside a hedge at MEDIUM or
    UNCERTAIN (HEDGE_RE, or LIKELY_HEDGE_RE). Each problem starts with "attribution of scam intent"
    so callers can count them."""
    if risk is None or risk is Risk.HIGH or not isinstance(text, str):
        return []
    hedges = [m.span() for m in HEDGE_RE.finditer(text)]
    if risk in _HEDGED_RISKS:
        hedges += [m.span() for m in LIKELY_HEDGE_RE.finditer(text)]
    hits: list[str] = []
    for m in ATTRIBUTION_RE.finditer(text):
        if risk in _HEDGED_RISKS and any(s <= m.start() and m.end() <= e for s, e in hedges):
            continue
        hits.append(m.group(0).lower())
    if not hits:
        return []
    return [f"attribution of scam intent at {risk.value}: {', '.join(sorted(set(hits)))}"]


def _check_risk_arg(risk: Risk | None, package: EvidencePackage) -> None:
    if risk is not None and risk is not package.risk:
        raise ValueError(
            f"risk={risk.value} does not match the package risk {package.risk.value}"
        )


def skip_llm_for_risk(package: EvidencePackage) -> bool:
    """G1: True when the advisory must be template-only because the decision is LOW risk."""
    return package.risk is Risk.LOW


def template_only_advisory(package: EvidencePackage, lang: str = "en") -> dict[str, Any]:
    """The points-contract advisory built only from templates (the G1 output)."""
    if lang not in V2_LANGS:
        raise ValueError(f"lang must be 'en' or 'hinglish', got {lang!r}")
    return {
        "lang": lang,
        "summary": _template_summary(package, lang),
        "points": [template_point(e.type, lang) for e in package.evidence],
    }


# ---------------------------------------------------------------------------
# Rank arm: K deterministic candidate sentences per evidence type. The explainer does not write
# them. It only scores each candidate by log-likelihood (see the job), and the best one is kept.
# Every candidate is a template, so every candidate passes ``check_points`` (tested).
# ---------------------------------------------------------------------------

RANK_K = 4
_RANK_FRAMES = {
    "en": (
        "This message {p}.",
        "Be careful: this message {p}.",
        "Note that this message {p}.",
        "Warning sign: this message {p}.",
    ),
    "hinglish": (
        "Yeh message {p}.",
        "Dhyan rakhein: yeh message {p}.",
        "Chetavni: yeh message {p}.",
        "Saavdhan rahein: yeh message {p}.",
    ),
}


def rank_candidates(etype: str, lang: str = "en") -> list[str]:
    """The K candidate sentences for one evidence type, the first being ``template_point``."""
    if lang not in V2_LANGS:
        raise ValueError(f"lang must be 'en' or 'hinglish', got {lang!r}")
    phrase = _TYPE_PHRASE[lang][etype]
    return [frame.format(p=phrase) for frame in _RANK_FRAMES[lang]]
