"""Offline URL Domain Intelligence and Hybrid URL Evidence Aggregator.

Combines three independent offline layers without requiring live DNS or network access:
1. **Domain Intelligence (`DomainIntelScorer`)**: scheme-agnostic lexical/structural features
   (brand impersonation, leetspeak/homograph spoofing, free-hosting platforms, risky TLDs,
   shorteners) + CRC32-hashed char(3-5) n-grams trained strictly domain-disjoint across
   multi-source URL corpora (~68 KB float16 NPZ bundle, pure Python + NumPy, <0.4 ms/URL).
2. **Lightweight ONNX URL Model (`FrozenURLExpert`)**: neural phishing classifier trained on
   PhreshPhish.
3. **Independent Evidence Aggregation (`HybridUrlScorer`)**: combines ONNX and domain-intel
   logits without double-counting correlated lexical rules, optionally consulting the offline
   Tranco reputation guard, and emits structured domain signals for three-outcome verdicts
   (``no_signal`` / ``suspicious`` / ``phishing``).
"""

from __future__ import annotations

import json
import math
import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from rakshak.url_features import (
    RISKY_TLDS,
    SHORTENERS,
    _registered,
    _split,
    _subdomain,
    _tld_of,
    load_brands,
    url_features,
)
from rakshak.url_model import UrlScorer
from rakshak.url_reputation import ReputationGuard, default_guard

DEFAULT_MODEL_PATH = Path(__file__).parent / "data" / "url_domain_intel.npz"

HOSTING_SUFFIXES = (
    "web.app",
    "firebaseapp.com",
    "pages.dev",
    "workers.dev",
    "netlify.app",
    "vercel.app",
    "github.io",
    "gitlab.io",
    "blogspot.com",
    "wordpress.com",
    "weebly.com",
    "wixsite.com",
    "herokuapp.com",
    "000webhostapp.com",
    "duckdns.org",
    "ddns.net",
    "no-ip.org",
    "ngrok.io",
    "ngrok-free.app",
    "trycloudflare.com",
    "amazonaws.com",
    "azurewebsites.net",
    "appspot.com",
    "repl.co",
    "glitch.me",
    "surge.sh",
    "r2.dev",
    "ipfs.io",
    "dweb.link",
    "godaddysites.com",
    "myshopify.com",
    "square.site",
    "webflow.io",
)
_VOWELS = set("aeiou")
_DROP_FEATS = {"has_https"}
_LEET_L = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
_LEET_I = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
_DIGIT_RUN_RE = re.compile(r"\d+")


@lru_cache(maxsize=1)
def _brand_aliases() -> tuple[tuple[frozenset[str], tuple[str, ...]], ...]:
    brands = load_brands()
    return tuple(
        (frozenset(info["domains"]), tuple(info["host_aliases"])) for info in brands.values()
    )


_GLOBAL_SPOOF_BRANDS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (frozenset({"paypal.com"}), ("paypal",)),
    (frozenset({"microsoft.com", "live.com", "office.com"}), ("microsoft", "outlook", "office365")),
    (frozenset({"apple.com", "icloud.com"}), ("apple", "icloud")),
    (frozenset({"netflix.com"}), ("netflix",)),
    (frozenset({"facebook.com", "fb.com"}), ("facebook",)),
    (frozenset({"instagram.com"}), ("instagram",)),
    (frozenset({"whatsapp.com"}), ("whatsapp",)),
    (frozenset({"linkedin.com"}), ("linkedin",)),
    (frozenset({"dhl.com"}), ("dhl",)),
    (frozenset({"fedex.com"}), ("fedex",)),
)


def _leetspeak_brand_spoof(host: str, reg: str) -> float:
    """True (1.0) when digit/symbol substitutions in ``host`` spell a brand off-domain."""
    if not host or not any(ch in host for ch in "013457@"):
        return 0.0
    norm_l = host.translate(_LEET_L)
    norm_i = host.translate(_LEET_I)
    for doms, aliases in (*_brand_aliases(), *_GLOBAL_SPOOF_BRANDS):
        if reg in doms:
            continue
        for alias in aliases:
            if len(alias) < 4:
                continue
            if alias not in host and (alias in norm_l or alias in norm_i):
                return 1.0
    return 0.0


def extract_domain_features(url: str) -> dict[str, float]:
    """Scheme-agnostic lexical, structural, homograph, and brand-impersonation features."""
    f = url_features(url)
    for k in _DROP_FEATS:
        f.pop(k, None)
    _scheme, host, path, query, _ = _split(url)
    reg = _registered(host)
    sub = _subdomain(host) if host else ""
    dom_stem = reg.split(".")[0] if reg else ""
    is_hosting = 1.0 if any(host == s or host.endswith("." + s) for s in HOSTING_SUFFIXES) else 0.0
    has_non_ascii_host = 1.0 if any(ord(c) > 127 for c in host) else 0.0
    dom_letters = sum(c.isalpha() for c in dom_stem)
    dom_vowels = sum(c in _VOWELS for c in dom_stem)
    dom_digits = sum(c.isdigit() for c in dom_stem)
    pq = (path + "?" + query).lower()
    brand_pq = 0.0
    for doms, aliases in _brand_aliases():
        if reg in doms:
            continue
        for a in aliases:
            if len(a) >= 4 and a in pq:
                brand_pq = 1.0
                break
        if brand_pq:
            break
    digit_runs = _DIGIT_RUN_RE.findall(host)
    f.update(
        {
            "is_hosting": is_hosting,
            "has_non_ascii_host": has_non_ascii_host,
            "leetspeak_brand_spoof": _leetspeak_brand_spoof(host, reg),
            "dom_len": float(len(dom_stem)),
            "dom_vowel_ratio": dom_vowels / max(dom_letters, 1),
            "dom_digit_ratio": dom_digits / max(len(dom_stem), 1),
            "dom_hyphens": float(dom_stem.count("-")),
            "brand_in_pq": brand_pq,
            "max_digit_run": float(max((len(r) for r in digit_runs), default=0)),
            "sub_len": float(len(sub)),
            "has_sub_non_www": 1.0 if (sub and sub != "www") else 0.0,
        }
    )
    return f


@lru_cache(maxsize=1)
def _official_brand_domains() -> frozenset[str]:
    brands = load_brands()
    return frozenset(d.lower() for info in brands.values() for d in info["domains"])


def is_official_brand_domain(url: str) -> bool:
    """True when ``url`` resolves to a curated official brand domain with no obfuscation."""
    _scheme, host, _path, _query, _ = _split(url)
    reg = _registered(host)
    if not reg or reg not in _official_brand_domains():
        return False
    if "xn--" in host or any(ord(c) > 127 for c in host):
        return False
    authority = re.split(r"[/?#]", url.split("//", 1)[-1], maxsplit=1)[0]
    return "@" not in authority


def domain_signals(url: str) -> list[str]:
    """Human-readable offline domain-intelligence flags for a single URL."""
    f = extract_domain_features(url)
    _scheme, host, _path, _query, _ = _split(url)
    reg = _registered(host)
    tld = _tld_of(host) if host else ""
    official = is_official_brand_domain(url)
    out: list[str] = []
    if f.get("brand_host_mismatch", 0.0) > 0:
        out.append("brand_in_host_mismatch")
    if f.get("leetspeak_brand_spoof", 0.0) > 0:
        out.append("homograph_brand_spoof")
    if f.get("punycode", 0.0) > 0 or f.get("has_non_ascii_host", 0.0) > 0:
        out.append("idn_homograph")
    if f.get("brand_in_pq", 0.0) > 0:
        out.append("brand_in_path_mismatch")
    if f.get("ip_literal", 0.0) > 0:
        out.append("ip_literal")
    if f.get("has_at_sign", 0.0) > 0:
        out.append("userinfo_at")
    if reg in SHORTENERS or f.get("shortener", 0.0) > 0:
        out.append("shortener")
    if f.get("is_hosting", 0.0) > 0:
        out.append("free_hosting_platform")
    if tld in RISKY_TLDS or f.get("tld_risky", 0.0) > 0:
        out.append("risky_tld")
    if not official and f.get("suspicious_token_count", 0.0) >= 2:
        out.append("multi_phishing_tokens")
    return out


def _hash_url_ngrams(url: str, n_buckets: int) -> dict[int, float]:
    _scheme, host, path, query, _ = _split(url)
    h = host[4:] if host.startswith("www.") else host
    h_str = f"<{h}>"
    u_str = f"^{h}${path[:80]}"
    if query:
        u_str += f"?{query[:40]}"
    raw_counts: dict[int, float] = {}
    for prefix, text in ((b"H", h_str), (b"U", u_str)):
        length = len(text)
        for n in (3, 4, 5):
            for i in range(length - n + 1):
                hv = zlib.crc32(prefix + text[i : i + n].encode("utf-8"))
                b = hv & (n_buckets - 1)
                sign = 1.0 if (hv & n_buckets) else -1.0
                raw_counts[b] = raw_counts.get(b, 0.0) + sign
    sub: dict[int, float] = {}
    sq = 0.0
    for b, v in raw_counts.items():
        if v == 0.0:
            continue
        sv = math.copysign(1.0 + math.log(abs(v)), v)
        sub[b] = sv
        sq += sv * sv
    norm = math.sqrt(sq) or 1.0
    return {b: v / norm for b, v in sub.items()}


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit(p: float, eps: float = 1e-6) -> float:
    q = min(max(float(p), eps), 1.0 - eps)
    return math.log(q / (1.0 - q))


class DomainIntelScorer:
    """Pure-Python/NumPy offline domain-intelligence URL scorer (~68 KB weights)."""

    def __init__(self, path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as bundle:
            self.w_hash = bundle["w_hash"].astype(np.float32)
            self.w_lex = bundle["w_lex"].astype(np.float32)
            self.bias = float(bundle["bias"][0])
            self.mean = bundle["mean"].astype(np.float32)
            self.scale = bundle["scale"].astype(np.float32)
            self.meta: dict[str, Any] = json.loads(str(bundle["meta"][0]))
        self.n_buckets = int(self.meta["n_buckets"])
        self.feat_names: tuple[str, ...] = tuple(self.meta["feat_names"])

    def logit_one(self, url: str) -> float:
        feats = extract_domain_features(url)
        x = np.asarray([feats[k] for k in self.feat_names], dtype=np.float32)
        z_lex = float(np.dot((x - self.mean) / self.scale, self.w_lex))
        ng = _hash_url_ngrams(url, self.n_buckets)
        z_hash = sum(float(self.w_hash[b]) * v for b, v in ng.items())
        return z_hash + z_lex + self.bias

    def score_one(self, url: str) -> float:
        return _sigmoid(self.logit_one(url))

    def score_urls(self, urls: Sequence[str]) -> list[float]:
        return [self.score_one(u) for u in urls]


@lru_cache(maxsize=1)
def default_domain_intel() -> DomainIntelScorer | None:
    """Bundled DomainIntelScorer, or None if the weight archive is absent."""
    return DomainIntelScorer() if DEFAULT_MODEL_PATH.exists() else None


@dataclass(frozen=True)
class HybridUrlAssessment:
    url: str
    hybrid_score: float
    onnx_score: float | None
    domain_score: float
    reputation: str | None  # "popular", "not_popular", "shortener", ... or None
    signals: tuple[str, ...]


class HybridUrlScorer:
    """Combines ONNX URL classifier + offline DomainIntelScorer via calibrated logit fusion.

    Implements ``UrlScorer`` (``score_urls``) so it drops directly into ``EdgeAnalyzer`` and
    ``check_url``. Logit averaging ensures that a single-source domain-shift false positive
    (e.g., ONNX spiking on an unseen benign root domain where domain-intel is strongly negative)
    is damped without collapsing recall when both independent models see phishing evidence.
    """

    def __init__(
        self,
        onnx_scorer: UrlScorer | None,
        domain_scorer: DomainIntelScorer | None = None,
        *,
        onnx_weight: float = 0.5,
        domain_weight: float = 0.5,
        onnx_logit_clip: float = 4.0,
        reputation_guard: ReputationGuard | None = None,
        popular_logit_penalty: float = 0.0,
    ) -> None:
        self.onnx_scorer = onnx_scorer
        self.domain_scorer = domain_scorer if domain_scorer is not None else default_domain_intel()
        if self.onnx_scorer is None and self.domain_scorer is None:
            raise ValueError(
                "HybridUrlScorer requires at least one of onnx_scorer or domain_scorer"
            )
        self.onnx_weight = float(onnx_weight)
        self.domain_weight = float(domain_weight)
        self.onnx_logit_clip = float(onnx_logit_clip)
        self.reputation_guard = reputation_guard
        self.popular_logit_penalty = float(popular_logit_penalty)

    def _combine(
        self,
        url: str,
        p_onnx: float | None,
        z_domain: float | None,
        *,
        use_reputation: bool = True,
    ) -> tuple[float, str | None]:
        rep_reason: str | None = None
        penalty = 0.0
        guard = self.reputation_guard
        if use_reputation and guard is not None:
            rep = guard.check(url)
            rep_reason = rep.reason
            if rep.established and self.popular_logit_penalty > 0.0:
                penalty = self.popular_logit_penalty

        if p_onnx is not None and z_domain is not None:
            lo = max(-self.onnx_logit_clip, min(self.onnx_logit_clip, _logit(p_onnx)))
            z = self.onnx_weight * lo + self.domain_weight * z_domain - penalty
            return _sigmoid(z), rep_reason
        if z_domain is not None:
            return _sigmoid(z_domain - penalty), rep_reason
        assert p_onnx is not None
        lo = max(-self.onnx_logit_clip, min(self.onnx_logit_clip, _logit(p_onnx)))
        return _sigmoid(lo - penalty), rep_reason

    def score_urls(self, urls: Sequence[str]) -> list[float]:
        urls_list = list(urls)
        if not urls_list:
            return []
        onnx_scores = (
            self.onnx_scorer.score_urls(urls_list) if self.onnx_scorer is not None else None
        )
        out: list[float] = []
        for i, url in enumerate(urls_list):
            p_onnx = None if onnx_scores is None else float(onnx_scores[i])
            z_dom = self.domain_scorer.logit_one(url) if self.domain_scorer is not None else None
            score, _ = self._combine(url, p_onnx, z_dom)
            out.append(score)
        return out

    def assess(self, url: str) -> HybridUrlAssessment:
        p_onnx = (
            float(self.onnx_scorer.score_urls([url])[0]) if self.onnx_scorer is not None else None
        )
        z_dom = self.domain_scorer.logit_one(url) if self.domain_scorer is not None else None
        p_dom = _sigmoid(z_dom) if z_dom is not None else 0.0
        guard = self.reputation_guard if self.reputation_guard is not None else default_guard()
        rep_reason = guard.check(url).reason if guard is not None else None
        p_hyb, _ = self._combine(url, p_onnx, z_dom)
        return HybridUrlAssessment(
            url=url,
            hybrid_score=p_hyb,
            onnx_score=p_onnx,
            domain_score=p_dom,
            reputation=rep_reason,
            signals=tuple(domain_signals(url)),
        )
