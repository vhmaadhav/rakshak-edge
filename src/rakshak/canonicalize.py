"""Security canonicaliser: NFKC, invisible-char strip, homoglyph skeleton, spacing and
leetspeak folding, and URL extraction (including obfuscated forms).

Devanagari is left intact: the homoglyph map only touches Latin lookalikes.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import regex

# ---------------------------------------------------------------------------
# Invisible / bidi / control characters
# ---------------------------------------------------------------------------

_ZERO_WIDTH = {
    "​",  # zero width space
    "‌",  # zero width non-joiner
    "‍",  # zero width joiner
    "⁠",  # word joiner
    "⁡", "⁢", "⁣", "⁤",
    "﻿",  # BOM / ZWNBSP
    "᠎",  # mongolian vowel separator
    "­",  # soft hyphen
    "͏",  # combining grapheme joiner
}
_BIDI = set("‪‫‬‭‮⁦⁧⁨⁩‎‏؜")

# ---------------------------------------------------------------------------
# Homoglyph skeleton: Cyrillic / Greek / fullwidth / mathematical lookalikes -> ASCII
# ---------------------------------------------------------------------------

_HOMOGLYPH = {
    # Cyrillic
    "А": "A", "а": "a", "В": "B", "Е": "E", "е": "e", "К": "K", "М": "M",
    "Н": "H", "О": "O", "о": "o", "Р": "P", "р": "p", "С": "C", "с": "c",
    "Т": "T", "Х": "X", "х": "x", "У": "Y", "у": "y", "Ѕ": "S", "ѕ": "s",
    "І": "I", "і": "i", "Ј": "J", "ј": "j", "Ԁ": "D", "ԁ": "d", "Ԛ": "Q",
    "ԛ": "q", "Ԝ": "W", "ԝ": "w", "Ү": "Y", "ү": "y", "Ӏ": "l",
    # Greek
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "ο": "o", "ν": "v", "α": "a", "ρ": "p", "ι": "i", "κ": "k", "τ": "t",
    "ϲ": "c",
}


def _build_fullwidth_and_math() -> dict[str, str]:
    table: dict[str, str] = {}
    # Fullwidth ASCII block U+FF01..U+FF5E maps to U+0021..U+007E.
    for cp in range(0xFF01, 0xFF5F):
        table[chr(cp)] = chr(cp - 0xFEE0)
    # Mathematical alphanumeric block: fold any letter whose NFKC form is ASCII letter.
    for cp in range(0x1D400, 0x1D7C4):
        ch = chr(cp)
        name = unicodedata.name(ch, "")
        if not name.startswith("MATHEMATICAL"):
            continue
        folded = unicodedata.normalize("NFKC", ch)
        if len(folded) == 1 and folded.isascii() and folded.isalpha():
            table[ch] = folded
    return table


_HOMOGLYPH.update(_build_fullwidth_and_math())

_HOMOGLYPH_TRANS = str.maketrans({k: v for k, v in _HOMOGLYPH.items() if len(k) == 1})

# ---------------------------------------------------------------------------
# Leetspeak folding (only inside alphabetic tokens)
# ---------------------------------------------------------------------------

_LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)
_TOKEN_RE = regex.compile(r"[\p{L}\p{N}@$]+")
_HAS_LETTER = regex.compile(r"\p{L}")
_HAS_LATIN_LETTER = regex.compile(r"[A-Za-z]")

# ---------------------------------------------------------------------------
# Spacing fold: "K Y C" -> "KYC"  (runs of single letters separated by single spaces)
# ---------------------------------------------------------------------------

# "K Y C": three or more single letters separated by single spaces only.
_SPACED_RUN = regex.compile(r"(?<![\p{L}\p{N}.\-_*|])\p{L}(?: \p{L}){2,}(?![\p{L}\p{N}])")
# "S.B.I" style: letters joined by dots / hyphens / underscores / pipes (3+ letters).
_DOTTED_RUN = regex.compile(r"(?<![\p{L}\p{N}])\p{L}(?:[.\-_*|]\p{L}){2,}(?![\p{L}\p{N}])")

# Scheme-less host candidates: word.word with a plausible TLD, optionally obfuscated.
_BARE_HOST_RE = regex.compile(
    r"(?<![\w@/.\-])"
    r"((?:[a-z0-9\-]+(?:\[\.\]|\(dot\)|\.))+[a-z]{2,24})"
    r"((?::\d{2,5})?(?:/[^\s<>\"']*)?)",
    regex.IGNORECASE,
)

_KNOWN_TLDS = {
    "com", "in", "net", "org", "co", "io", "me", "ly", "gl", "is", "to", "cc", "tk",
    "xyz", "top", "site", "online", "info", "biz", "app", "link", "click", "live",
    "gov", "edu", "ac", "bank", "shop", "store", "cfd", "icu", "vip", "work", "ru", "cn", "uk",
    "us", "de", "fr", "br", "xin", "buzz", "cyou", "rest", "fun",
    "pw", "ml", "ga", "cf", "gq", "su", "ws", "one", "sbs", "bond", "club", "digital",
    "sbi", "pro", "mobi", "tech", "ai", "ph", "jp", "kr", "au", "ca", "eu", "ng",
}
# Only used to decide whether a bare word.word is a host; obviously-not-hosts are filtered.
_FILE_LIKE_TLDS = {"txt", "pdf", "jpg", "png", "exe", "apk", "doc", "docx", "xls", "csv", "zip"}

_ZW_CHARS = _ZERO_WIDTH | _BIDI


@dataclass
class Flags:
    zero_width_count: int = 0
    homoglyph_count: int = 0
    spacing_fold: int = 0
    leet_fold: int = 0
    mixed_script: bool = False


@dataclass
class Canon:
    original: str
    normalized: str
    urls: list[str] = field(default_factory=list)
    flags: Flags = field(default_factory=Flags)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_invisible(text: str, flags: Flags) -> str:
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in _ZW_CHARS or (cat == "Cc" and ch not in "\n\t\r"):
            flags.zero_width_count += 1
            continue
        out.append(ch)
    return "".join(out)


def _homoglyph_fold(text: str, flags: Flags) -> str:
    count = 0
    out = []
    for ch in text:
        if ch in _HOMOGLYPH and len(_HOMOGLYPH[ch]) == 1:
            out.append(_HOMOGLYPH[ch])
            count += 1
        else:
            out.append(ch)
    flags.homoglyph_count += count
    return "".join(out)


def _has_mixed_script(text: str) -> bool:
    """Latin letters mixed with Cyrillic or Greek letters inside one word."""
    for tok in regex.findall(r"\p{L}+", text):
        scripts = set()
        for ch in tok:
            if "CYRILLIC" in unicodedata.name(ch, ""):
                scripts.add("cyr")
            elif "GREEK" in unicodedata.name(ch, ""):
                scripts.add("grk")
            elif ch.isascii() and ch.isalpha():
                scripts.add("lat")
        if "lat" in scripts and len(scripts) > 1:
            return True
    return False


def _collapse_spacing(text: str, flags: Flags) -> str:
    def _join_letters(m: regex.Match[str]) -> str:
        letters = regex.findall(r"\p{L}", m.group(0))
        flags.spacing_fold += 1
        return "".join(letters)

    text = _SPACED_RUN.sub(_join_letters, text)

    def _join_dotted(m: regex.Match[str]) -> str:
        letters = regex.findall(r"\p{L}", m.group(0))
        flags.spacing_fold += 1
        return "".join(letters)

    text = _DOTTED_RUN.sub(_join_dotted, text)
    return text


_DIGIT_RUN = regex.compile(r"\d{2,}")


def _leet_fold(text: str, flags: Flags) -> str:
    def _fold_token(m: regex.Match[str]) -> str:
        tok = m.group(0)
        # Only fold alphabetic-looking tokens. Pure digit tokens (OTPs, amounts) and tokens
        # with a run of 2+ digits (Rs500, 482913, 1800) are left untouched.
        if not _HAS_LATIN_LETTER.search(tok):
            return tok
        if _DIGIT_RUN.search(tok):
            return tok
        folded = tok.translate(_LEET)
        if folded != tok:
            flags.leet_fold += sum(1 for a, b in zip(tok, folded, strict=True) if a != b)
        return folded

    return _TOKEN_RE.sub(_fold_token, text)


def _unobfuscate_url_text(text: str) -> str:
    """Rewrite hxxp(s):// and [.] / (dot) into plain form for URL parsing."""
    text = regex.sub(r"(?i)\bhxxp(s?)://", lambda m: f"http{m.group(1).lower()}://", text)
    text = regex.sub(r"(?i)\bhxxp(s?):", lambda m: f"http{m.group(1).lower()}:", text)
    text = regex.sub(r"\s*[\[(]\s*\.\s*[\])]\s*", ".", text)
    text = regex.sub(r"(?i)\s*[\[(]\s*dot\s*[\])]\s*", ".", text)
    return text


def extract_urls(text: str) -> list[str]:
    """Extract URL candidates from text, including scheme-less and obfuscated forms.

    Returns normalised URLs (obfuscation removed, scheme added only where absent in the
    original text is kept as-is). Order follows first appearance; duplicates removed.
    """
    work = _unobfuscate_url_text(text)
    found: list[tuple[int, str]] = []
    seen: set[str] = set()

    for m in re.finditer(r"(?i)\b(?:https?|ftp)://[^\s<>\"']+", work):
        url = m.group(0).rstrip(".,;:!?)\"'")
        if url.lower() not in seen:
            seen.add(url.lower())
            found.append((m.start(), url))

    for m in _BARE_HOST_RE.finditer(work):
        host = m.group(1)
        rest = m.group(2) or ""
        last = host.rstrip(".").rsplit(".", 1)[-1].lower()
        if last in _FILE_LIKE_TLDS or last not in _KNOWN_TLDS:
            continue
        # Skip if this host is already covered by an explicit-scheme URL.
        candidate = (host + rest).rstrip(".,;:!?)\"'")
        if any(candidate.lower() in u.lower() for _, u in found):
            continue
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        found.append((m.start(), candidate))

    found.sort(key=lambda x: x[0])
    return [u for _, u in found]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def canonicalize(text: str) -> Canon:
    """Canonicalise a message. The original text is preserved verbatim in `original`."""
    flags = Flags()
    flags.mixed_script = _has_mixed_script(text)

    work = unicodedata.normalize("NFKC", text)
    work = _strip_invisible(work, flags)
    work = _homoglyph_fold(work, flags)
    work = _unobfuscate_url_text(work)

    # URLs are extracted after invisible/homoglyph folding but before spacing/leet folding,
    # so "sbi-kyc.in/upd" keeps its literal host while "sЅbi" style lookalikes still resolve.
    urls = extract_urls(work)

    work = _collapse_spacing(work, flags)
    work = _leet_fold(work, flags)

    # Collapse runs of whitespace introduced by stripping.
    work = regex.sub(r"[ \t]+", " ", work).strip()

    return Canon(original=text, normalized=work, urls=urls, flags=flags)
