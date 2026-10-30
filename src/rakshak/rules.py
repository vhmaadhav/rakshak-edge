"""Evidence rules over a canonicalised message.

Rules are deterministic and offline. Each rule fires zero or more times and yields
``(type, span, matched_text, weight)``. ``run_rules`` assigns stable ids E1..En in emission
order. Keywords cover English, romanised Hindi / Hinglish, and Devanagari.

Rules run on ``canon.normalized`` (homoglyphs, spacing and leet already folded), except for
URL rules which run on ``canon.urls``.

``urgency`` is a weak, generic signal (turant, abhi, aaj hi, 24 ghante, तुरंत). It never sets an
action on its own (``ACTION_MAP`` has no entry for it), and it cannot raise the risk above
medium, so benign "abhi aa rahe hain" does not become a scam verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import regex

from rakshak.canonicalize import Canon
from rakshak.url_features import (
    SHORTENERS,
    brand_domain_mismatch,
    registered_domain,
    url_features,
)

EVIDENCE_TYPES = [
    "credential_request",
    "upi_collect_or_pay",
    "kyc_block_urgency",
    "account_suspension",
    "prize_lottery",
    "apk_install",
    "remote_access_app",
    "electricity_disconnect",
    "parcel_customs",
    "job_task_scam",
    "investment_returns",
    "digital_arrest",
    "brand_domain_mismatch",
    "shortened_url",
    "suspicious_url",
    "urgency",
]


@dataclass(frozen=True)
class Evidence:
    id: str
    type: str
    span: tuple[int, int]
    text: str
    weight: float


RawHit = tuple[str, tuple[int, int], str, float]
RuleFn = Callable[[Canon], Iterator[RawHit]]

_FLAGS = regex.IGNORECASE | regex.VERSION1


def _pattern_rule(
    etype: str, pattern: str, weight: float
) -> RuleFn:
    rx = regex.compile(pattern, _FLAGS)

    def _rule(canon: Canon) -> Iterator[RawHit]:
        for m in rx.finditer(canon.normalized):
            yield etype, m.span(), m.group(0), weight

    return _rule


# ---------------------------------------------------------------------------
# Credential request (needs negation handling so "Do not share your OTP" does not fire)
# ---------------------------------------------------------------------------

_CRED_TERMS = (
    r"(?:otp|o\.t\.p|one[ -]time password|mpin|upi pin|atm pin|pin|cvv|cvc|password|"
    r"passcode|pass code|verification code|"
    r"पासवर्ड|ओटीपी|पिन|सीवीवी|वेरिफिकेशन कोड)"
)
_CRED_VERBS = (
    r"(?:share|send|give|tell|provide|submit|reply|forward|enter|bata\w*|btao|bhej\w*|"
    r"dij\w*|dein|bolo|share kar\w*|"
    r"बताएं|बताइए|बताइये|बताओ|बताएँ|भेजें|भेजो|भेजिए|भेजिये|दें|दीजिए|दीजिये|शेयर|"
    r"शेयर करें|शेयर करो|भेजना|बताना)"
)
_CRED_PAIR = regex.compile(
    rf"(?:\b{_CRED_VERBS}\b.{{0,30}}\b{_CRED_TERMS}\b)|(?:\b{_CRED_TERMS}\b.{{0,30}}\b{_CRED_VERBS}\b)",
    _FLAGS,
)
_NEGATION = regex.compile(
    r"\b(?:not|never|don'?t|dont|do not|nahi|nahin|nhi|mat|kabhi|kisi ko na|na)\b|"
    r"(?<![\p{L}\p{M}])(?:नहीं|न|मत|कभी)(?![\p{L}\p{M}])",
    _FLAGS,
)


_CRED_VERB_ONLY = regex.compile(rf"\b{_CRED_VERBS}\b", _FLAGS)


def _credential_request(canon: Canon) -> Iterator[RawHit]:
    text = canon.normalized
    for m in _CRED_PAIR.finditer(text):
        # Every verb in the match is a candidate. A verb negated in its own sentence
        # ("Do not share", "mat batao", "न बताएं") does not count, so "Your OTP is 1234. Do not
        # share it" stays benign while "share your OTP" fires.
        for verb in _CRED_VERB_ONLY.finditer(m.group(0)):
            verb_pos = m.start() + verb.start()
            sentence_start = max(
                text.rfind(ch, 0, verb_pos) for ch in ".!?\n|।"
            ) + 1
            lead = text[sentence_start:verb_pos][-30:]
            if _NEGATION.search(lead):
                continue
            yield "credential_request", m.span(), m.group(0), 0.95
            break


# ---------------------------------------------------------------------------
# Pattern rules
# ---------------------------------------------------------------------------

_RULES: list[RuleFn] = [
    _credential_request,
    _pattern_rule(
        "upi_collect_or_pay",
        r"(?:\bupi (?:collect|collection|request)\b|\bcollect request\b|"
        r"\b(?:approve|accept|allow|authori[sz]e) (?:the |this |a )?"
        r"(?:payment|request|collect\w*|mandate)\b|"
        r"\b(?:payment|request|collect|mandate) (?:ko |ki |ka )?(?:approve|accept|allow)\w*\b|"
        r"\b(?:approve|accept|allow)\w* (?:karo|kar do|kijiye|kariye|karein|karen|kare)\b|"
        r"\bscan (?:the |this |any )?(?:qr|code)\b.{0,30}\b(?:receive|get|refund|cashback|money)\b|"
        r"\b(?:receive|get) (?:money|refund|cashback|payment)\b.{0,40}\b(?:upi|pin|scan)\b|"
        r"\bupi ?pin\b.{0,40}\b(?:daal\w*|dal\w*|enter|dale|dalen|dalein|dalo)\b|"
        r"\bpaise (?:lene|paane|pane|receive|prapt|milne)\b.{0,40}\b(?:upi|pin|scan|qr|collect)\b|"
        r"\b(?:upi|pin|scan|qr)\b.{0,40}\bpaise (?:lene|paane|pane|receive|milenge|milega)\b|"
        r"\bpay\s*(?:rs\.?|₹)?\s*\d+\b.{0,30}\b(?:verify|activate|unlock|confirm)\b|"
        r"पेमेंट.{0,20}अप्रूव|यूपीआई.{0,20}(?:कलेक्ट|रिक्वेस्ट)|"
        r"यूपीआई\s*पिन.{0,30}(?:डालें|डालिए|डालो|दर्ज)|"
        r"पैसे\s+(?:लेने|प्राप्त|पाने).{0,30}(?:यूपीआई|पिन|स्कैन|क्यूआर))",
        0.85,
    ),
    _pattern_rule(
        "kyc_block_urgency",
        r"(?:\bkyc\b.{0,80}\b(?:block\w*|suspen[sd]\w*|expir\w*|deactivat\w*|pending|urgent\w*|"
        r"immediately|today|within \d+ ?(?:hours?|days?)|last date|freez\w*|hold)\b|"
        r"\b(?:block\w*|suspen[sd]\w*|expir\w*|deactivat\w*|pending|urgent\w*|immediately|"
        r"within \d+ ?(?:hours?|days?)|last date|freez\w*)\b.{0,80}\bkyc\b|"
        r"केवाईसी.{0,60}(?:बंद|ब्लॉक|निलंबित|समाप्त|आज|तुरंत|जल्द|जल्दी|वरना|नहीं तो)|"
        r"(?:बंद|ब्लॉक|निलंबित|समाप्त|तुरंत|जल्द|जल्दी).{0,60}केवाईसी|"
        r"\bkyc\b.{0,40}\b(?:band|block|jald|jaldi|turant|aaj|bandh)\w*|"
        r"\bkyc\b.{0,60}\b(?:warna|varna|vrna|nahi to|nahi toh|jayega|jaayega|jaega|hoga|hogi|"
        r"ho jayega|ho jaayega)\b|"
        r"\b(?:warna|varna|nahi to|nahi toh)\b.{0,60}\bkyc\b)",
        0.8,
    ),
    _pattern_rule(
        "account_suspension",
        r"(?:\b(?:a/?c|acc|account|khata|debit card|credit card|card|sim|wallet|upi id)\b"
        r".{0,40}\b(?:suspen[sd]\w*|block\w*|deactivat\w*|frozen|freez\w*|locked|on hold|"
        r"will be closed|band ho\w*|block ho\w*|band kar\w*|band hoga|band hogi|"
        r"ruk ja\w*|bandh\w*|lock ho\w*|freeze ho\w*)\b|"
        r"\b(?:suspen[sd]\w*|block\w*|deactivat\w*|frozen|freez\w*|locked|band ho\w*|"
        r"block ho\w*|band hoga|band hogi|lock ho\w*)\b.{0,40}"
        r"\b(?:a/?c|acc|account|khata|card|sim|wallet)\b|"
        r"खाता.{0,30}(?:बंद|ब्लॉक|निलंबित|फ्रीज)|(?:बंद|ब्लॉक|निलंबित|फ्रीज).{0,30}खाता)",
        0.75,
    ),
    _pattern_rule(
        "prize_lottery",
        r"(?:\blottery\b|\blucky draw\b|\bjackpot\b|\byou (?:have )?won\b|\bwinner\b|"
        r"\bcongratulations?\b.{0,40}\b(?:prize|won|reward|gift|cash)\b|"
        r"\bprize\b|\b(?:kbc|kaun banega)\b|\binam\b|\binaam\b|\bjeet\w*\b|\bjeeta\b|"
        r"लॉटरी|इनाम|जीते|जीता|जीती|जीतें|जीतिए|जीतेंगे|पुरस्कार|बधाई.{0,20}जीत)",
        0.8,
    ),
    _pattern_rule(
        "apk_install",
        r"(?:\.apk\b|\bapk\b|\binstall (?:the |this |our )?(?:app|application|apk|file)\b|"
        r"\bdownload (?:the |this )?(?:app|application|apk)\b|"
        r"\bapp (?:download|install)\w* (?:karo|kare\w*|karein)\b|"
        r"\bइंस्टॉल\b|\bडाउनलोड\b)",
        0.9,
    ),
    _pattern_rule(
        "remote_access_app",
        r"(?:\banydesk\b|\bteam ?viewer\b|\bquick ?support\b|\brustdesk\b|\bultraviewer\b|"
        r"\bscreen ?share\b|\bremote (?:access|support|control)\b|\bany ?desk\b)",
        0.9,
    ),
    _pattern_rule(
        "electricity_disconnect",
        r"(?:\b(?:electricity|power|bijli|bijlee|meter|connection)\b.{0,60}\b(?:disconnect\w*|cut|"
        r"cut off|band|kat\w*|kaat\w*|suspen[sd]\w*|block\w*|unpaid|overdue)\b|"
        r"\b(?:disconnect\w*|cut|band|kat\w*|kaat\w*|suspen[sd]\w*)\b.{0,40}"
        r"\b(?:electricity|power|bijli|bijlee|meter|connection)\b|"
        r"बिजली.{0,40}(?:काट|बंद|कट|डिस्कनेक्ट)|"
        r"कनेक्शन.{0,40}(?:काट|कट|बंद|डिस्कनेक्ट))",
        0.8,
    ),
    _pattern_rule(
        "parcel_customs",
        r"(?:\b(?:parcel|package|courier|shipment|consignment|customs|duty|saman)\b.{0,50}"
        r"\b(?:pay\w*|fee|charges?|customs|held|hold|pending|clear\w*|release|re-?deliver\w*|"
        r"rs\.?|₹|\d+|shulk\w*|kastam|kasam|phas\w*|ruka|ruka|rok\w*|jama)\b|"
        r"\b(?:pay\w*|fee|charges?|customs|held|clear\w*|release|shulk\w*|kastam|jama)\b.{0,50}"
        r"\b(?:parcel|package|courier|shipment|consignment|saman)\b|"
        r"\b(?:shulk\w*|kastam)\b.{0,50}\b(?:jama\w*|pay\w*|bharo|bhejo|karo|karna|de\w*)\b|"
        r"पार्सल.{0,40}(?:शुल्क|पेमेंट|रोक|रुका|रुक|कस्टम)|"
        r"(?:शुल्क|कस्टम).{0,40}(?:जमा|पेमेंट|भुगतान))",
        0.7,
    ),
    _pattern_rule(
        "job_task_scam",
        r"(?:\bpart[- ]?time\b.{0,30}\b(?:job|work|income|earn\w*)\b|"
        r"\bwork from home\b.{0,40}\b(?:earn\w*|income|salary|per day|daily)\b|"
        r"\b(?:like|subscribe)\b.{0,30}\b(?:youtube|video|channel|task|review)\b|"
        r"\b(?:hotel|product)\s+rating\b|\bearn\s*(?:rs\.?|₹)?\s*\d+\b.{0,30}\b(?:per|a|each|daily|/)\s*(?:task|day|hour)\b|"
        r"\btask\b.{0,30}\b(?:commission|earn\w*|payment|reward)\b|"
        r"\bjob offer\b|\bghar baithe\b.{0,30}\b(?:kamao|kamayi|income|earn\w*)\b|"
        r"\b(?:roz|daily|rozana|ghar baithe|ghar se)\b.{0,30}\b(?:kamao|kamayein|kamaye|"
        r"kamaiye|kamai|kamayi|earn\w*|income)\b|"
        r"\b(?:kamao|kamayein|kamaye|kamaiye|kamayi|kamai)\b.{0,40}"
        r"\b(?:roz|daily|rozana|ghante|ghanta|paise|paisa|rupaye|rs|₹)\b|"
        r"\bonline (?:kaam|job|task)\b|"
        r"घर बैठे.{0,30}(?:कमाए|कमाई|कमाएं)|पार्ट.?टाइम.{0,20}(?:नौकरी|काम))",
        0.8,
    ),
    _pattern_rule(
        "investment_returns",
        r"(?:\b(?:guaranteed|assured|fixed)\b.{0,20}\b(?:return|profit|income|returns)\b|"
        r"\bdouble your\b|\b\d+\s?%\s*(?:daily|per day|monthly|per month|weekly|return)\b|"
        r"\b(?:crypto|bitcoin|usdt|forex)\b.{0,40}\b(?:profit|return|invest\w*|trading|tips?|group)\b|"
        r"\btrading (?:group|tips?|signals?|course)\b|\bstock tips?\b|"
        r"\binvest\w*\b.{0,30}\b(?:return|profit|%)\b|"
        r"\b(?:invest\w*|nivesh\w*|share market|trading|crypto|stock)\b.{0,50}"
        r"\b(?:munafa|munaafa|laabh|profit|dugna|double|guarantee\w*)\b|"
        r"\b(?:munafa|munaafa|laabh|profit|dugna)\b.{0,40}"
        r"\b(?:invest\w*|nivesh\w*|share|trading|group|guarantee\w*|daily|roz)\b|"
        r"\b(?:paisa|paise|rupaye|money)\s+double\b|\bdouble\s+(?:paisa|paise|rupaye|money)\b|"
        r"निवेश.{0,30}(?:गारंटी|मुनाफा|रिटर्न|दोगुना|डबल)|"
        r"(?:दोगुना|डबल).{0,30}(?:पैसा|पैसे|रुपये))",
        0.8,
    ),
    _pattern_rule(
        "digital_arrest",
        r"(?:\bdigital arrest\b|\b(?:cbi|police|ed officer|enforcement directorate|"
        r"narcotics|cyber ?crime|cyber ?cell|crime branch|trai officer|customs officer|"
        r"income tax (?:department|officer))\b.{0,60}\b(?:arrest\w*|warrant|case|fir|jail|"
        r"court|seiz\w*|freez\w*|video call|verify|money laundering|illegal|complaint|"
        r"notice|summon\w*|karwai|karvai|khilaf|darj\w*|kanoon\w*)\b|"
        r"\b(?:arrest\w*|warrant|video call|money laundering|illegal sim)\b.{0,60}"
        r"\b(?:cbi|police|ed|narcotics|cyber|trai|customs|crime branch)\b|"
        r"\b(?:aapke|aapka|aapko|aap)\b.{0,40}\b(?:khilaf|case darj|warrant|giraftar\w*|gunah)\b|"
        r"गिरफ्तार|डिजिटल अरेस्ट|(?:क्राइम ब्रांच|साइबर क्राइम).{0,60}"
        r"(?:वारंट|केस|कार्रवाई|वीडियो कॉल|वेरिफाई|गिरफ्तार)|"
        r"(?:खिलाफ|वारंट).{0,40}(?:केस|शिकायत|दर्ज|गिरफ्तार))",
        0.9,
    ),
    # Bare authority mentions are weak on their own; news and benign reports mention police.
    _pattern_rule("digital_arrest", r"(?:\bcbi\b|\bpolice\b|\bcrime branch\b|सीबीआई|पुलिस)", 0.5),
    # Generic urgency: weak on its own and never mapped to an action.
    _pattern_rule(
        "urgency",
        r"(?:\b(?:turant|turnt|tatkal|jaldi|jaldee|abhi|aaj hi|aaj hee|urgent(?:ly)?|"
        r"immediately|immediate action|right now|last chance|today only|expires? today|"
        r"last date)\b|"
        r"\b\d{1,2} ?(?:hours?|hrs?|ghante|ghanta|ghantey)\b|"
        r"(?<![\p{L}\p{M}])(?:तुरंत|तुरन्त|जल्दी|जल्द|आज ही|\d{1,2} घंटे)(?![\p{L}\p{M}])|"
        r"अभी\s+(?:करें|करो|कीजिए|जुड़ें|भेजें|क्लिक|रजिस्टर|कॉल|संपर्क|अपडेट|वेरिफ\w*|कराएं|करवाएं))",
        0.4,
    ),
]


def _domain_rules(canon: Canon) -> Iterator[RawHit]:
    text = canon.normalized
    for url in canon.urls:
        span = _locate(text, url)
        reg = registered_domain(url)
        if brand_domain_mismatch(text, url):
            yield "brand_domain_mismatch", span, url, 0.9
        if reg in SHORTENERS:
            yield "shortened_url", span, url, 0.6
        f = url_features(url)
        if (
            f["tld_risky"]
            or f["punycode"]
            or f["ip_literal"]
            or f["has_at_sign"]
            or f["brand_host_mismatch"]
            or f["suspicious_token_count"] >= 2
            or f["n_subdomains"] >= 3
        ):
            weight = 0.9 if f["brand_host_mismatch"] or f["punycode"] or f["ip_literal"] else 0.7
            yield "suspicious_url", span, url, weight


def _locate(text: str, needle: str) -> tuple[int, int]:
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return (0, 0)
    return (idx, idx + len(needle))


def run_rules(canon: Canon) -> list[Evidence]:
    """Run every rule over the canonicalised message and return ids E1..En in order."""
    raw: list[RawHit] = []
    for rule in _RULES:
        raw.extend(rule(canon))
    raw.extend(_domain_rules(canon))
    return [
        Evidence(id=f"E{i}", type=etype, span=span, text=text, weight=weight)
        for i, (etype, span, text, weight) in enumerate(raw, start=1)
    ]
