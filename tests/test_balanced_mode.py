"""Tests for the 3-outcome balanced detector-only mode (quiet / balanced / verify)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
BENCH = REPO / "deploy" / "pi" / "bench"
for p in (SERVICE, REPO / "src", BENCH, REPO / "deploy" / "pi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import edge_decision  # noqa: E402
import edge_workers  # noqa: E402
from edge_pipeline import EdgeAnalyzer  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import Action, Risk  # noqa: E402
from rakshak.url_model import UrlHit  # noqa: E402


class _FixedDetector:
    def __init__(self, p: float) -> None:
        self.p = p

    def score(self, text: str) -> float:
        return self.p


class _FixedUrlScorer:
    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def score_urls(self, urls: list[str]) -> list[float]:
        self.calls += 1
        return [self.score for _ in urls]


class _StubLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, max_new_tokens: int = 64, stop_when=None):
        self.calls += 1
        return " mentions a suspicious link.", {"wall_ms": 1.0, "stop_hit": True}


def test_balanced_mode_three_outcomes_for_detector_only_messages():
    canon = canonicalize("Hello, the meeting moved to 4pm")
    # Below threshold -> benign / LOW / IGNORE
    low = edge_decision.decide_edge(canon, [], 0.10, 0.35, mode="balanced", verify_threshold=0.999)
    assert low.state == "benign"
    assert low.package.risk == Risk.LOW and low.action == Action.IGNORE

    # Between threshold and verify_threshold -> detector_only stays quiet (LOW / IGNORE)
    mid = edge_decision.decide_edge(canon, [], 0.85, 0.35, mode="balanced", verify_threshold=0.999)
    assert mid.state == "scam" and mid.detector_only is True
    assert mid.package.risk == Risk.LOW and mid.action == Action.IGNORE

    # At or above verify_threshold -> escalates to UNCERTAIN / VERIFY_INDEPENDENTLY
    high = edge_decision.decide_edge(
        canon, [], 0.9995, 0.35, mode="balanced", verify_threshold=0.999
    )
    assert high.state == "scam" and high.detector_only is True
    assert high.package.risk == Risk.UNCERTAIN and high.action == Action.VERIFY_INDEPENDENTLY


def test_balanced_mode_corroborates_url_evidence_only_when_student_or_rule_agrees():
    canon = canonicalize("Check your account at http://example-verify.test/portal")
    hit = UrlHit(url="http://example-verify.test/portal", score=0.95)

    # Student below threshold + no rule -> URL model alone is NOT corroborated in balanced mode
    ev_uncorr = edge_decision.add_url_evidence(
        [], canon, hit, 0.63, mode="balanced", p_scam=0.10, p_threshold=0.35
    )
    assert ev_uncorr == []

    # Student at/above threshold -> URL evidence IS corroborated in balanced mode
    ev_corr = edge_decision.add_url_evidence(
        [], canon, hit, 0.63, mode="balanced", p_scam=0.60, p_threshold=0.35
    )
    assert [e.type for e in ev_corr] == ["phishing_url_model"]

    # Quiet mode still suppresses URL evidence by default
    ev_quiet = edge_decision.add_url_evidence(
        [], canon, hit, 0.63, mode="quiet", p_scam=0.99, p_threshold=0.35
    )
    assert ev_quiet == []


def test_balanced_pipeline_detector_only_paths_and_corroborated_url():
    plain = "Hello, the meeting moved to 4pm"
    mid = EdgeAnalyzer(
        _FixedDetector(0.80),
        _StubLLM(),
        0.35,
        detector_only_mode="balanced",
        verify_threshold=0.999,
    ).analyse(plain)
    assert mid["decision"]["risk"] == "LOW"
    assert mid["decision"]["action"] == "IGNORE"
    assert mid["advisory"]["path"] == "detector_only_quiet"

    high = EdgeAnalyzer(
        _FixedDetector(0.9995),
        _StubLLM(),
        0.35,
        detector_only_mode="balanced",
        verify_threshold=0.999,
    ).analyse(plain)
    assert high["decision"]["risk"] == "UNCERTAIN"
    assert high["decision"]["action"] == "VERIFY_INDEPENDENTLY"
    assert high["advisory"]["path"] == "detector_only"

    link_sms = "Open http://unknown-portal.example/status for details"
    scorer = _FixedUrlScorer(0.95)
    out = EdgeAnalyzer(
        _FixedDetector(0.60),
        _StubLLM(),
        0.35,
        detector_only_mode="balanced",
        verify_threshold=0.999,
        url_scorer=scorer,
        url_threshold=0.63,
    ).analyse(link_sms)
    assert scorer.calls == 1
    assert any(e["type"] == "phishing_url_model" for e in out["evidence"])
    assert out["decision"]["action"] != "IGNORE"


def test_cfg_from_args_carries_verify_threshold():
    class Args:
        detector_only_mode = "balanced"
        verify_threshold = 0.995
        llm_cpus = None

        def __getattr__(self, name):
            return None

    cfg = edge_workers.cfg_from_args(Args(), Path("/tmp/root"))
    assert cfg["detector_only_mode"] == "balanced"
    assert cfg["verify_threshold"] == pytest.approx(0.995)


def test_check_url_three_tier_verdict_with_high_threshold():
    from edge_pipeline import check_url

    url = "https://portal-status.example/account"
    low = check_url(url, _FixedUrlScorer(0.20), 0.63, high_threshold=0.99)
    assert low["verdict"] == "no_signal"
    assert low["url_model"]["high_threshold"] == pytest.approx(0.99)

    mid = check_url(url, _FixedUrlScorer(0.85), 0.63, high_threshold=0.99)
    assert mid["verdict"] == "suspicious"
    assert mid["url_model"]["fired"] is True

    high = check_url(url, _FixedUrlScorer(0.995), 0.63, high_threshold=0.99)
    assert high["verdict"] == "phishing"
    assert high["url_model"]["fired"] is True


def test_balanced_mode_transactional_aware_two_tier_thresholds():
    # General non-transactional message at p=0.97 escalates (default verify_threshold=0.9566)
    general = canonicalize("Congratulations you have been shortlisted for our special offer today")
    assert not edge_decision.is_transactional_notice(general.normalized)
    dec_gen = edge_decision.decide_edge(general, [], 0.97, 0.35, mode="balanced")
    assert dec_gen.package.risk == Risk.UNCERTAIN
    assert dec_gen.action == Action.VERIFY_INDEPENDENTLY

    # Transactional bank/UPI/OTP notice at p=0.97 stays quiet (default tx_verify_threshold=0.999)
    tx = canonicalize("Rs. 450 debited from A/c XX1234 via UPI. Available balance Rs. 12400.")
    assert edge_decision.is_transactional_notice(tx.normalized)
    dec_tx_mid = edge_decision.decide_edge(tx, [], 0.97, 0.35, mode="balanced")
    assert dec_tx_mid.package.risk == Risk.LOW
    assert dec_tx_mid.action == Action.IGNORE

    # Transactional notice at extreme confidence (p=0.9995) still escalates to verify
    dec_tx_hi = edge_decision.decide_edge(tx, [], 0.9995, 0.35, mode="balanced")
    assert dec_tx_hi.package.risk == Risk.UNCERTAIN
    assert dec_tx_hi.action == Action.VERIFY_INDEPENDENTLY
