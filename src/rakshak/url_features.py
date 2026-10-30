"""Offline lexical and structural features for a single URL, plus brand-domain mismatch.

No network access: the public-suffix list comes only from tldextract's bundled snapshot
(``suffix_list_urls=()``) and the cache is disabled.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from importlib import resources
from urllib.parse import urlsplit

import tldextract

_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, fallback_to_snapshot=True)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly",
    "shorturl.at", "tiny.cc", "rebrand.ly", "bit.do", "t.ly", "v.gd", "s.id", "lnkd.in",
    "buff.ly", "tiny.one", "clck.ru", "urlz.fr",
}

SUSPICIOUS_TOKENS = [
    "login", "signin", "verify", "verification", "kyc", "update", "secure", "security",
    "account", "bank", "otp", "refund", "reward", "lottery", "claim", "unlock", "suspend",
    "blocked", "billing", "payment", "wallet", "invoice", "support", "helpdesk", "activate",
    "confirm", "validate", "reactivate", "upi", "pay",
]
_TOKEN_RE = re.compile(r"[a-z]+")

RISKY_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "cc", "click", "link", "icu", "buzz",
    "cyou", "rest", "sbs", "cfd", "shop", "online", "site", "live", "work", "fun", "vip",
    "support", "monster", "zip", "mov", "quest", "su", "pw", "ws", "bond", "cam",
}
COMMON_TLDS = {"com", "in", "org", "net", "edu", "co.in", "gov.in", "ac.in", "nic.in", "io"}

_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_DELIMS = "-_./?=&@%:~+#"


@lru_cache(maxsize=1)
def load_brands() -> dict[str, dict]:
    """Curated brand table from src/rakshak/data/brands.json."""
    raw = resources.files("rakshak").joinpath("data/brands.json").read_text(encoding="utf-8")
    return json.loads(raw)["brands"]


@lru_cache(maxsize=1)
def _compiled_brand_patterns() -> dict[str, re.Pattern[str]]:
    patterns = {}
    for name, info in load_brands().items():
        alts = sorted({re.escape(a) for a in info["aliases"]}, key=len, reverse=True)
        patterns[name] = re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])", re.I)
    return patterns


def _split(url: str) -> tuple[str, str, str, str]:
    """Return (scheme, host, path, query) with the host lower-cased and port/userinfo removed."""
    candidate = url if "://" in url else "http://" + url
    parts = urlsplit(candidate)
    host = (parts.hostname or "").lower().rstrip(".")
    return parts.scheme.lower(), host, parts.path or "", parts.query or ""


def registered_domain(url: str) -> str:
    """eTLD+1 of the URL's host using the bundled suffix snapshot ('' when unknown)."""
    _, host, _, _ = _split(url)
    if not host:
        return ""
    ext = _EXTRACTOR(host)
    return ext.top_domain_under_public_suffix or ""


def _subdomain(host: str) -> str:
    ext = _EXTRACTOR(host)
    return ext.subdomain


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _tld_of(host: str) -> str:
    ext = _EXTRACTOR(host)
    return ext.suffix or ""


def _brand_in_host_not_registered(host: str, reg: str) -> float:
    if not host:
        return 0.0
    compact = host.replace(".", " ").replace("-", " ")
    labels = set(compact.split())
    for info in load_brands().values():
        if reg in info["domains"]:
            continue
        for alias in info["host_aliases"]:
            if len(alias) <= 3:
                if alias in labels:
                    return 1.0
            elif alias in host:
                return 1.0
    return 0.0


def url_features(url: str) -> dict[str, float]:
    """Structural and lexical features for one URL. All values are floats."""
    scheme, host, path, query = _split(url)
    full = url.lower()
    reg = registered_domain(url)
    sub = _subdomain(host) if host else ""
    tld = _tld_of(host) if host else ""
    labels = host.split(".") if host else []

    tokens = set(_TOKEN_RE.findall(full))
    suspicious_hits = [t for t in SUSPICIOUS_TOKENS if any(t in tok for tok in tokens)]

    feats: dict[str, float] = {
        "url_len": float(len(url)),
        "host_len": float(len(host)),
        "path_len": float(len(path)),
        "query_len": float(len(query)),
        "n_subdomains": float(len(sub.split(".")) if sub else 0),
        "n_host_labels": float(len(labels)),
        "host_digit_ratio": (sum(c.isdigit() for c in host) / len(host)) if host else 0.0,
        "host_entropy": _entropy(host),
        "host_hyphens": float(host.count("-")),
        "n_delims": float(sum(url.count(d) for d in _DELIMS)),
        "path_segments": float(len([p for p in path.split("/") if p])),
        "has_https": 1.0 if scheme == "https" else 0.0,
        "has_scheme": 1.0 if "://" in url else 0.0,
        "has_at_sign": 1.0 if "@" in url else 0.0,
        "punycode": 1.0 if "xn--" in host else 0.0,
        "ip_literal": 1.0 if (_IPV4_RE.match(host) or host.startswith("[")) else 0.0,
        "shortener": 1.0 if reg in SHORTENERS else 0.0,
        "tld_common": 1.0 if tld in COMMON_TLDS else 0.0,
        "tld_risky": 1.0 if tld in RISKY_TLDS else 0.0,
        "tld_other": 0.0,
        "suspicious_token_count": float(len(suspicious_hits)),
        "tok_login": 1.0 if any("login" in t or "signin" in t for t in tokens) else 0.0,
        "tok_verify": 1.0 if any("verif" in t or "validat" in t for t in tokens) else 0.0,
        "tok_kyc": 1.0 if "kyc" in tokens or any("kyc" in t for t in tokens) else 0.0,
        "tok_update": 1.0 if any(t.startswith("update") for t in tokens) else 0.0,
        "tok_secure": 1.0 if any("secur" in t for t in tokens) else 0.0,
        "brand_host_mismatch": _brand_in_host_not_registered(host, reg),
        "registered_domain_missing": 1.0 if not reg else 0.0,
    }
    feats["tld_other"] = 0.0 if (feats["tld_common"] or feats["tld_risky"]) else 1.0
    return feats


def brands_in_text(text: str) -> list[str]:
    """Names of curated brands whose alias appears in ``text``."""
    return [name for name, pat in _compiled_brand_patterns().items() if pat.search(text)]


def brand_domain_mismatch(text: str, url: str) -> bool:
    """True when the message names a curated brand but the URL is not on one of its domains.

    A URL on an unknown registered domain counts as a mismatch when the message names a
    brand, which is the main phishing signal here.
    """
    named = brands_in_text(text)
    if not named:
        return False
    reg = registered_domain(url)
    brands = load_brands()
    for name in named:
        if reg and reg in brands[name]["domains"]:
            return False
    return True


def mentioned_brand_domains(text: str) -> dict[str, list[str]]:
    """Map each brand named in text to its official domains (for prompt-side evidence)."""
    brands = load_brands()
    return {name: list(brands[name]["domains"]) for name in brands_in_text(text)}
