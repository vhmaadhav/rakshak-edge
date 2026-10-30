"""Template keys: canonical text with URLs, phone numbers, amounts and numbers masked.

Two messages that differ only in the link, the number or the amount share a template key,
which is what the near-duplicate clustering and campaign assignment operate on. Month and
weekday names are kept on purpose: they are part of many lures ("expires on Friday").
"""

from __future__ import annotations

import re

_URL = re.compile(
    r"(?:https?://|hxxps?://|www\.)\S+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|in|net|org|co|io|me|ly|gl|is|to|cc|tk|xyz|top|site|online"
    r"|info|biz|app|link|click|live|gov|edu|bank|shop|store|ru|cn|uk|us|de|fr|br|buzz|pw"
    r"|ml|ga|cf|gq|su|ws|one|sbs|bond|club|digital|sbi|pro|tech|ai|ph|jp|kr|au|ca|eu|ng"
    r"|cfd|icu|vip|work|rest|fun|cyou|xin|mobi|ac)(?:/\S*)?\b"
)
_PHONE = re.compile(r"\+?\d(?:[\s-]?\d){8,}")
_AMOUNT = re.compile(
    r"(?:rs\.?|inr|usd|eur|euro|gbp|rupees?|rupaye|₹|\$|€|£)\s*\d[\d.,]*"
    r"(?:\s*(?:lakh|lac|crore|k)\b)?"
    r"|\d[\d.,]*\s*(?:rs|rupees|rupaye|inr|euro|usd|lakh|lac)\b"
)
_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
_WS = re.compile(r"\s+")


def template_key(text_canon: str) -> str:
    """Mask URLs -> <URL>, phones -> <PHONE>, amounts -> <AMT>, other numbers -> <NUM>."""
    out = text_canon.lower()
    out = _URL.sub(" <URL> ", out)
    out = _PHONE.sub(" <PHONE> ", out)
    out = _AMOUNT.sub(" <AMT> ", out)
    out = _NUMBER.sub(" <NUM> ", out)
    return _WS.sub(" ", out).strip()
