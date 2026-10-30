"""Shared constants and small helpers for the dataset pipeline."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

from rakshak.url_features import SHORTENERS, registered_domain

SEED = 13
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Near-duplicate clustering on template keys (see docs/DATACARD.md for the justification).
NUM_PERM = 128
SHINGLE_K = 5
NEAR_DUP_THRESHOLD = 0.7
# A registered domain shared by more than this many scam messages is treated as a shared
# platform/CDN and does not join campaigns (it would otherwise chain unrelated clusters).
DOMAIN_FANOUT_CAP = 200

# Hosting, messaging and big-platform domains that many unrelated senders share.
PLATFORM_DOMAINS = frozenset(
    {
        "google.com",
        "goo.gl",
        "forms.gle",
        "youtube.com",
        "youtu.be",
        "facebook.com",
        "fb.com",
        "fb.me",
        "instagram.com",
        "whatsapp.com",
        "wa.me",
        "t.me",
        "telegram.me",
        "twitter.com",
        "x.com",
        "linkedin.com",
        "apple.com",
        "icloud.com",
        "microsoft.com",
        "amazon.com",
        "amazon.in",
        "paypal.com",
        "shrtco.de",
        "bitly.ws",
        "short.gy",
        "blogspot.com",
        "wixsite.com",
        "weebly.com",
        "godaddysites.com",
        "web.app",
        "firebaseapp.com",
        "netlify.app",
        "vercel.app",
        "pages.dev",
        "herokuapp.com",
        "github.io",
        "workers.dev",
        "sites.google.com",
        "wordpress.com",
        "medium.com",
        "notion.site",
        "carrd.co",
        "glitch.me",
        "ngrok.io",
        "replit.dev",
    }
)
NON_CAMPAIGN_DOMAINS = frozenset(SHORTENERS) | PLATFORM_DOMAINS

_WS = re.compile(r"\s+")


def stable_id(*parts: str) -> str:
    """Deterministic 20-hex id from the source name and the row key (run-independent)."""
    payload = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:20]


def dedup_key(text_canon: str) -> str:
    """Key for exact duplicates: text_canon, lower-cased with whitespace collapsed."""
    return _WS.sub(" ", text_canon).strip().lower()


@lru_cache(maxsize=500_000)
def cached_registered_domain(url: str) -> str:
    """registered_domain() with a per-process cache (tldextract is slow on cold hosts).

    Malformed URLs scraped from message text (e.g. invalid IPv6 brackets) yield ''.
    """
    try:
        return registered_domain(url)
    except ValueError:
        return ""
