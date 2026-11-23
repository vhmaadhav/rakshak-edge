"""Tests for offline URL domain intelligence and hybrid URL evidence aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from edge_pipeline import check_url  # noqa: E402

from rakshak.url_domain_intel import (  # noqa: E402
    HybridUrlScorer,
    default_domain_intel,
    domain_signals,
)


class _ConstScorer:
    def __init__(self, score: float) -> None:
        self.score = score

    def score_urls(self, urls):
        return [self.score for _ in urls]


def test_domain_signals_catch_homograph_and_brand_impersonation():
    sigs_leet = domain_signals("https://paypa1-secure-login.xyz/verify")
    assert "homograph_brand_spoof" in sigs_leet
    assert "risky_tld" in sigs_leet

    sigs_idn = domain_signals("https://xn--pypal-4ve.com/signin")
    assert "idn_homograph" in sigs_idn

    sigs_host = domain_signals("https://my-bank-kyc.web.app/login/verify")
    assert "free_hosting_platform" in sigs_host
    assert "multi_phishing_tokens" in sigs_host

    sigs_clean = domain_signals("https://www.wikipedia.org/wiki/Python")
    assert sigs_clean == []


def test_bundled_domain_intel_scorer_separates_benign_root_from_phishing():
    dom = default_domain_intel()
    assert dom is not None
    assert dom.meta["domain_disjoint"] is True

    benign = [
        "https://w3schools.com",
        "https://www.wikipedia.org",
        "https://owasp.org/www-project-top-ten/",
    ]
    phish = [
        "https://allegrolokalnie.pl-oferta961287365.cfd/oferta/item",
        "http://paypa1-secure-verify.xyz/login/account",
        "https://sbi-kyc-update-alert.web.app/verify",
    ]
    b_scores = dom.score_urls(benign)
    p_scores = dom.score_urls(phish)
    assert all(s < 0.25 for s in b_scores), b_scores
    assert all(s > 0.75 for s in p_scores), p_scores


def test_hybrid_scorer_damps_single_source_ood_false_positive_and_keeps_phishing():
    dom = default_domain_intel()
    assert dom is not None

    # Simulate frozen ONNX spiking (0.90) on an unseen benign root domain:
    # DomainIntelScorer disagrees, damping hybrid score well below the 0.6308 shipped threshold.
    hybrid_fp_onnx = HybridUrlScorer(_ConstScorer(0.90), dom)
    benign_score = hybrid_fp_onnx.score_urls(["https://w3schools.com"])[0]
    assert benign_score < 0.63

    # When both ONNX (0.95) and DomainIntel agree on a real phishing URL, hybrid stays high.
    hybrid_tp = HybridUrlScorer(_ConstScorer(0.95), dom)
    phish_score = hybrid_tp.score_urls(["https://allegrolokalnie.pl-oferta961287365.cfd/oferta/x"])[0]
    assert phish_score > 0.90


def test_check_url_with_hybrid_scorer_attaches_domain_intel_and_homograph_verdict():
    dom = default_domain_intel()
    assert dom is not None
    hybrid = HybridUrlScorer(_ConstScorer(0.20), dom)

    # Homograph brand spoof promotes to phishing even if ONNX score is low
    res = check_url("https://paypa1-verify.xyz/login", hybrid, 0.63, high_threshold=0.85)
    assert res["verdict"] == "phishing"
    assert "domain_intel" in res
    assert "homograph_brand_spoof" in res["domain_intel"]["signals"]

    # Clean benign root URL stays no_signal even with noisy ONNX
    noisy_hybrid = HybridUrlScorer(_ConstScorer(0.90), dom)
    clean = check_url("https://w3schools.com", noisy_hybrid, 0.63, high_threshold=0.85)
    assert clean["verdict"] == "no_signal"
    assert clean["domain_intel"]["onnx_score"] == 0.9
    assert clean["domain_intel"]["domain_score"] < 0.2
