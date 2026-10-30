"""Semantics-preserving attack families used to stress-test the detector.

Each family is a pure function ``f(text, rng) -> str`` where ``rng`` is a ``random.Random``.
``perturb(text, family, seed)`` is the public entry point; ``FAMILIES`` is the registry.

The intent is that a scam message perturbed by any family still means the same thing to a
human, so a detector that canonicalises properly should still fire the same rules.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

from rakshak.canonicalize import _KNOWN_TLDS

Perturber = Callable[[str, random.Random], str]

ZERO_WIDTH_CHARS = ["​", "‌", "‍", "⁠", "﻿"]

# ASCII letter -> Cyrillic / Greek lookalike (inverse of the canonicaliser's skeleton map).
HOMOGLYPH_FOR = {
    "a": "а", "c": "с", "e": "е", "i": "і", "j": "ј",
    "o": "о", "p": "р", "s": "ѕ", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "I": "І", "K": "К", "M": "М", "O": "О", "P": "Р",
    "S": "Ѕ", "T": "Т", "X": "Х", "Y": "У",
}

LEET_FOR = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

PUNCT_SEPARATORS = [".", "-", "*", "_", "|"]

# Brands and the misspelled/split forms used by brand_misspell.
BRANDS = ["SBI", "HDFC", "ICICI", "Paytm", "PhonePe", "Amazon", "Flipkart", "IRCTC", "Airtel"]

BENIGN_SENTENCES = [
    "Thanks for your patience.",
    "Happy Diwali from all of us!",
    "Your order is on the way.",
    "Have a great weekend.",
    "Please ignore if already done.",
    "Team lunch is at 1 pm tomorrow.",
    "Good morning, hope you are well.",
]

# Phrase -> semantically equivalent alternatives. Every alternative is covered by the
# urgency vocabulary in rules.py so paraphrasing does not hide the signal.
URGENCY_PARAPHRASES: dict[str, list[str]] = {
    "immediately": ["right away", "without delay", "urgently", "at once"],
    "urgent": ["time-sensitive", "immediate action required", "urgent action"],
    "within 24 hours": ["within a day", "today itself", "in the next 24 hours"],
    "blocked": ["suspended", "deactivated", "put on hold"],
    "last warning": ["final notice", "final warning"],
    "expire": ["expiry", "get deactivated"],
}

_TLD_ALTERNATION = "|".join(sorted(_KNOWN_TLDS, key=len, reverse=True))
_URL_RE = re.compile(
    rf"(?i)\b((?:https?|hxxps?)://)?((?:[a-z0-9][a-z0-9\-]*\.)+(?:{_TLD_ALTERNATION}))\b(/\S*)?"
)
_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _outside_urls(text: str, fn: Callable[[str], str]) -> str:
    """Apply a word-level transform to everything except URL spans.

    Word-level families must not split or leet-replace characters inside a URL, otherwise
    the URL itself stops being a URL. Obfuscating the URL is the job of url_obfuscate.
    """
    parts: list[str] = []
    pos = 0
    for m in _URL_RE.finditer(text):
        parts.append(fn(text[pos : m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(fn(text[pos:]))
    return "".join(parts)


def zero_width_insert(text: str, rng: random.Random) -> str:
    """Insert zero-width characters inside alphabetic words."""
    out = []
    for ch in text:
        out.append(ch)
        if ch.isalpha() and rng.random() < 0.25:
            out.append(rng.choice(ZERO_WIDTH_CHARS))
    return "".join(out)


def homoglyph_swap(text: str, rng: random.Random) -> str:
    """Replace some Latin letters with visually identical Cyrillic letters."""
    out = []
    for ch in text:
        if ch in HOMOGLYPH_FOR and rng.random() < 0.4:
            out.append(HOMOGLYPH_FOR[ch])
        else:
            out.append(ch)
    return "".join(out)


def char_spacing(text: str, rng: random.Random) -> str:
    """Space out the letters of some words: KYC -> K Y C.

    Two neighbouring words are never both spaced: "y o u r S B I" would be indistinguishable
    from one run of letters, so word boundaries would be lost.
    """
    def _space(s: str) -> str:
        state = {"last_end": -10}

        def _one(m: re.Match[str]) -> str:
            word = m.group(0)
            adjacent = m.start() - state["last_end"] <= 1
            if adjacent or rng.random() < 0.5:
                return word
            state["last_end"] = m.end()
            return " ".join(word)

        return _WORD_RE.sub(_one, s)

    return _outside_urls(text, _space)


def punctuation_split(text: str, rng: random.Random) -> str:
    """Insert a separator between the letters of some words: OTP -> O.T.P."""

    def _split(m: re.Match[str]) -> str:
        word = m.group(0)
        if rng.random() < 0.5:
            return word
        sep = rng.choice(PUNCT_SEPARATORS)
        return sep.join(word)

    return _outside_urls(text, lambda s: _WORD_RE.sub(_split, s))


def leetspeak(text: str, rng: random.Random) -> str:
    """Replace some letters with leet digits in words of at least four letters.

    Substitutions are never adjacent and never exceed a third of the word, so the folded
    token stays letter-dominant and the canonicaliser can map it back.
    """

    def _leet_word(m: re.Match[str]) -> str:
        word = m.group(0)
        if len(word) < 4 or rng.random() < 0.4:
            return word
        budget = len(word) // 3
        chars = list(word)
        last_replaced = -2
        used = 0
        for i, ch in enumerate(chars):
            if used >= budget or i - last_replaced < 2:
                continue
            key = ch.lower()
            if key in LEET_FOR and rng.random() < 0.5:
                chars[i] = LEET_FOR[key]
                last_replaced = i
                used += 1
        return "".join(chars)

    return _outside_urls(text, lambda s: _WORD_RE.sub(_leet_word, s))


def brand_misspell(text: str, rng: random.Random) -> str:
    """Split or dot brand names: SBI -> S.B.I, Paytm -> Payt m, Paytm -> Pay.tm."""
    out = text
    for brand in BRANDS:
        pattern = re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE)
        if not pattern.search(out):
            continue

        def _mangle(m: re.Match[str], brand: str = brand) -> str:
            word = m.group(0)
            if len(word) < 4:
                # Too short for a split inside the word. The match is case-insensitive, so
                # "sbi" or "Sbi" reach here as well as "SBI". Dot every letter instead.
                return ".".join(word)
            cut = rng.randint(2, len(word) - 2)
            style = rng.choice(["space", "dot"])
            return word[:cut] + (" " if style == "space" else ".") + word[cut:]

        out = pattern.sub(_mangle, out, count=1)
    return out


def benign_sentence_insert(text: str, rng: random.Random) -> str:
    """Prepend or append a benign sentence; the original content is kept verbatim."""
    filler = rng.choice(BENIGN_SENTENCES)
    if rng.random() < 0.5:
        return f"{filler} {text}"
    return f"{text} {filler}"


def urgency_paraphrase(text: str, rng: random.Random) -> str:
    """Replace urgency phrases with equivalent template alternatives."""
    out = text
    for phrase, alternatives in URGENCY_PARAPHRASES.items():
        pattern = re.compile(rf"(?i)\b{re.escape(phrase)}\b")

        def _swap(m: re.Match[str], alts: list[str] = alternatives) -> str:
            return rng.choice(alts)

        out = pattern.sub(_swap, out)
    return out


def _punycode_label(label: str) -> str:
    try:
        return "xn--" + label.encode("punycode").decode("ascii")
    except UnicodeError:
        return label


def url_obfuscate(text: str, rng: random.Random) -> str:
    """Obfuscate URLs: hxxp scheme, [.] defanging, subdomain padding, punycode lookalikes."""

    def _obf(m: re.Match[str]) -> str:
        scheme, host, path = m.group(1) or "", m.group(2), m.group(3) or ""
        choice = rng.choice(["defang", "pad", "punycode", "hxxp"])
        if choice == "defang":
            host = host.replace(".", "[.]", 1)
            return f"hxxps://{host}{path}" if scheme else f"{host}{path}"
        if choice == "pad":
            # Prepend brand-looking labels; the registered domain is unchanged.
            return f"{scheme}sbi.co.in.{host}{path}"
        if choice == "punycode":
            labels = host.split(".")
            # Lookalike: encode the first label as punycode, which hides it from naive matching.
            labels[0] = _punycode_label(labels[0] + "і")
            return f"{scheme}{'.'.join(labels)}{path}"
        return f"hxxp://{host}{path}"

    return _URL_RE.sub(_obf, text)


FAMILIES: dict[str, Perturber] = {
    "zero_width_insert": zero_width_insert,
    "homoglyph_swap": homoglyph_swap,
    "char_spacing": char_spacing,
    "punctuation_split": punctuation_split,
    "leetspeak": leetspeak,
    "brand_misspell": brand_misspell,
    "benign_sentence_insert": benign_sentence_insert,
    "urgency_paraphrase": urgency_paraphrase,
    "url_obfuscate": url_obfuscate,
}


def perturb(text: str, family: str, seed: int) -> str:
    """Apply one attack family to ``text`` with a deterministic seed."""
    if family not in FAMILIES:
        raise KeyError(f"unknown perturbation family: {family!r}")
    return FAMILIES[family](text, random.Random(seed))
