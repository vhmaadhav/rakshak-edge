#!/usr/bin/env python3
"""Export the data files the Android app reads, from the rakshak Python package.

Usage (on the build VM, repo on PYTHONPATH):
    PYTHONPATH=<repo>/src python android/tools/export_assets.py <repo> <out_dir>

Writes into <out_dir>:
  brands.json            copy of src/rakshak/data/brands.json (brand table for the domain rules)
  ela_lexicon.json       copy of src/rakshak/data/ela_lexicon.json (English point keywords)
  llama3_2_LICENSE.txt   copy of the Llama licence (shipped in the About screen)
  public_suffix_icann.txt  ICANN section of tldextract's bundled snapshot, one rule per line
  lang_lexicon.json      detect_lang word sets and constants (datasets/lang.py)
  suffix_check.json      result of checking the Python reference suffix algorithm (the Kotlin
                         port copies it) against tldextract. The script exits non-zero on any
                         mismatch, so a bad export cannot be committed silently.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys

import tldextract

from rakshak.datasets import lang as lang_mod

SUFFIX_HOSTS = [
    "www.example.co.in", "foo.bar.unknowntld", "192.168.1.1", "sbi-kyc.in", "a.b.github.io",
    "sbi.co.in.evil.xyz", "www.ck", "a.www.ck", "x.nom.br", "a.b.ck", "b.bd", "co.uk",
    "example.com", "paytm-kyc.co.in", "localhost", "bit.ly", "t.co", "x.y.kawasaki.jp",
    "city.kawasaki.jp", "a.b.c.d.example.org", "xn--e1afmkfd.xn--p1ai", "user.dyndns.org",
    "shop.amazon.in", "secure-login.sbi.co.in", "verify.kyc.top", "1.2.3.4", "a", "",
]


def load_icann_rules(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    begin, end = "===BEGIN ICANN DOMAINS===", "===END ICANN DOMAINS==="
    if begin in text:
        text = text.split(begin, 1)[1].split(end, 1)[0]
    rules: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("//"):
            rules.append(to_ascii_rule(s.lower()))
    return rules


def to_ascii_rule(rule: str) -> str:
    """IDN rules are stored in Unicode in the snapshot; tldextract matches punycode hosts."""
    prefix = ""
    if rule.startswith("!"):
        prefix, rule = "!", rule[1:]
    labels = [lbl if lbl.isascii() else lbl.encode("idna").decode("ascii")
              for lbl in rule.split(".")]
    return prefix + ".".join(labels)


IPV4 = re.compile(r"\d{1,3}(\.\d{1,3}){3}")


def suffix_index(labels: list[str], rules: set[str]) -> int:
    """Reference for the Kotlin port (UrlFeatures.kt). Labels are lower-case, leftmost first."""
    n = len(labels)
    for i in range(n):
        cand = ".".join(labels[i:])
        if "!" + cand in rules:
            return i + 1
        if i + 1 < n and "*." + ".".join(labels[i + 1:]) in rules:
            return i
        if cand in rules:
            return i
    return n


def reference_split(host: str, rules: set[str]) -> dict[str, str]:
    labels = host.lower().rstrip(".").split(".") if host else []
    if not labels:
        return {"subdomain": "", "domain": "", "suffix": "", "top": ""}
    if IPV4.fullmatch(host):
        return {"subdomain": "", "domain": host, "suffix": "", "top": ""}
    idx = suffix_index(labels, rules)
    suffix = ".".join(labels[idx:])
    domain = labels[idx - 1] if idx >= 1 else ""
    subdomain = ".".join(labels[: idx - 1]) if idx >= 2 else ""
    top = f"{domain}.{suffix}" if domain and suffix else ""
    return {"subdomain": subdomain, "domain": domain, "suffix": suffix, "top": top}


def main() -> int:
    repo, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    data = repo / "src" / "rakshak" / "data"
    for name in ("brands.json", "ela_lexicon.json"):
        shutil.copyfile(data / name, out / name)
    shutil.copyfile(data / "llama3_2_LICENSE.txt", out / "llama3_2_LICENSE.txt")

    snapshot = pathlib.Path(tldextract.__file__).parent / ".tld_set_snapshot"
    rules_list = load_icann_rules(snapshot)
    (out / "public_suffix_icann.txt").write_text("\n".join(rules_list) + "\n", encoding="utf-8")
    rules = set(rules_list)

    extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, fallback_to_snapshot=True)
    mismatches = []
    for host in SUFFIX_HOSTS:
        ext = extractor(host)
        ref = reference_split(host, rules)
        got = {
            "subdomain": ext.subdomain or "",
            "domain": ext.domain or "",
            "suffix": ext.suffix or "",
            "top": ext.top_domain_under_public_suffix or "",
        }
        if got != ref:
            mismatches.append({"host": host, "tldextract": got, "reference": ref})
    (out / "suffix_check.json").write_text(
        json.dumps({"hosts": len(SUFFIX_HOSTS), "mismatches": mismatches}, indent=1),
        encoding="utf-8",
    )
    if mismatches:
        print("suffix reference mismatches:", json.dumps(mismatches, indent=1), file=sys.stderr)
        return 1

    lang = {
        "hinglish_strong": sorted(lang_mod.HINGLISH_LEXICON),
        "hinglish_weak": sorted(lang_mod.HINGLISH_WEAK),
        "english": sorted(lang_mod.ENGLISH_LEXICON),
        "min_hinglish_hits": lang_mod.MIN_HINGLISH_HITS,
        "hinglish_ratio": lang_mod.HINGLISH_RATIO,
    }
    (out / "lang_lexicon.json").write_text(json.dumps(lang, ensure_ascii=False, indent=0),
                                           encoding="utf-8")
    print("exported to", out, "rules:", len(rules_list), "suffix hosts checked:", len(SUFFIX_HOSTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
