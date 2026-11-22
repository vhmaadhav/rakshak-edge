"""Offline reputation guard for URL-model evidence (pre-registered in docs/RECALL_V2_PLAN.md §2).

A URL is *established* when its registrable domain (Public Suffix List **including the private
section**, so ``x.github.io`` or ``x.web.app`` is its own registrable domain) is in a frozen
popularity list (Tranco top-100k, list V354N), and the host is not a shortener, not a known
user-content host on a popular parent, not a raw IP, and the URL has no userinfo part.

Quiet mode shows URL-model evidence only for URLs that are *not* established. The list is a plain
text file (one domain per line, rank order) loaded once; lookups are set membership. Nothing here
touches the network: tldextract uses its bundled suffix snapshot.
"""

from __future__ import annotations

import hashlib
import ipaddress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import tldextract

from rakshak.url_features import SHORTENERS, _split

# PSL private section on: hosting platforms split into per-tenant registrable domains.
_PRIVATE_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=(), cache_dir=None, fallback_to_snapshot=True,
    include_psl_private_domains=True,
)

# Fixed in docs/RECALL_V2_PLAN.md §2 item 3 before scoring. Do not tune.
USER_CONTENT_HOSTS = (
    "sites.google.com", "docs.google.com", "forms.gle", "drive.google.com",
    "storage.googleapis.com", "firebasestorage.googleapis.com", "dropbox.com",
    "dl.dropboxusercontent.com", "1drv.ms", "onedrive.live.com", "sharepoint.com", "ipfs.io",
    "cloudflare-ipfs.com", "dweb.link", "wa.me", "t.me", "telegram.me", "linktr.ee", "bit.ly",
    "tinyurl.com", "rebrand.ly", "cutt.ly", "is.gd",
)

TRANCO_LIST_ID = "V354N"
TRANCO_TOP_N = 100_000
DEFAULT_LIST_PATH = (
    Path(__file__).parent / "data" / f"tranco_{TRANCO_LIST_ID}_top{TRANCO_TOP_N}.txt"
)


def _host_matches(host: str, entries) -> bool:
    return any(host == e or host.endswith("." + e) for e in entries)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def private_registrable(host: str) -> str:
    """Registrable domain with the PSL private section ('' when unknown)."""
    if not host:
        return ""
    try:
        ext = _PRIVATE_EXTRACTOR(host)
    except (ValueError, UnicodeError):
        return ""
    return (ext.top_domain_under_public_suffix or "").lower()


@dataclass(frozen=True)
class Reputation:
    established: bool
    # "popular", "not_popular", "shortener", "user_content", "ip", "userinfo", "no_host"
    reason: str
    registrable: str


class ReputationGuard:
    """Set lookup over a frozen popularity list."""

    def __init__(self, domains: set[str], list_sha256: str | None = None) -> None:
        self.domains = domains
        self.list_sha256 = list_sha256

    @classmethod
    def from_file(cls, path: Path | str = DEFAULT_LIST_PATH, top_n: int = TRANCO_TOP_N):
        """Load a Tranco CSV (``rank,domain``) or a plain one-domain-per-line file."""
        raw = Path(path).read_bytes()
        domains: list[str] = []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            dom = line.split(",", 1)[1] if "," in line else line
            domains.append(dom.strip().lower().rstrip("."))
            if len(domains) >= top_n:
                break
        return cls(set(domains), hashlib.sha256(raw).hexdigest())

    def check(self, url: str) -> Reputation:
        _scheme, host, _path, _query, _broken = _split(url)
        host = (host or "").lower().rstrip(".")
        if not host:
            return Reputation(False, "no_host", "")
        if _is_ip(host):
            return Reputation(False, "ip", "")
        if "@" in url.split("//", 1)[-1].split("/", 1)[0]:
            return Reputation(False, "userinfo", private_registrable(host))
        reg = private_registrable(host)
        if reg in SHORTENERS or _host_matches(host, SHORTENERS):
            return Reputation(False, "shortener", reg)
        if _host_matches(host, USER_CONTENT_HOSTS):
            return Reputation(False, "user_content", reg)
        if reg and reg in self.domains:
            return Reputation(True, "popular", reg)
        return Reputation(False, "not_popular", reg)

    def established(self, url: str) -> bool:
        return self.check(url).established


@lru_cache(maxsize=1)
def default_guard() -> ReputationGuard | None:
    """The guard from the bundled list, or None when the list file is absent."""
    return ReputationGuard.from_file() if DEFAULT_LIST_PATH.exists() else None
