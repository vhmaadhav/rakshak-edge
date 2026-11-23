"""Tests for P0 (transactional scam vs legit), P2 (DNS low-and-slow boundaries),
P3 (URL hybrid adversarial & unseen domains), and P4 (Indian SMS / field validation harness)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "scripts", REPO / "deploy" / "pi" / "service", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import edge_decision  # noqa: E402
import eval_validation_suite as V  # noqa: E402
from edge_pipeline import check_url  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import Action, Risk  # noqa: E402
from rakshak.nids import host_rules as R  # noqa: E402
from rakshak.url_domain_intel import HybridUrlScorer, default_domain_intel  # noqa: E402


class _ConstScorer:
    def __init__(self, score: float) -> None:
        self.score = score

    def score_urls(self, urls):
        return [self.score for _ in urls]


# --- P0: Transactional scam vs. legitimate passive transactional notice ---


def test_p0_passive_transactional_vs_action_soliciting_scam():
    # Legitimate passive notices (including Myntra refund & TATA Power UPI VPA handle) stay passive
    passive_samples = [
        "SBI: A/c XX4821 debited by Rs.1,250.00 via UPI Ref 632189451022. Avl Bal Rs.34,820.50.",
        "Myntra: Your return for order 412890 has been processed. Refund of INR 1,299 initiated.",
        (
            "TATA Power: Bill for November is INR 1,420. "
            "Due: 2026-12-05. Pay via UPI: 9812345670@oktatapower"
        ),
        "849201 is your SBI OTP for online payment. Valid for 5 mins. Do not share with anyone.",
    ]
    for text in passive_samples:
        c = canonicalize(text)
        assert edge_decision.is_passive_transactional_notice(c.normalized), text
        dec = edge_decision.decide_edge(c, [], 0.97, 0.35, mode="balanced")
        assert dec.package.risk == Risk.LOW and dec.action == Action.IGNORE, text

    # Transactional-vocabulary scams soliciting an action (call/unblock/disconnect/claim) escalate!
    scam_samples = [
        "Dear SBI customer, your A/c XX4821 is on hold. Call 9876543210 immediately to unblock.",
        (
            "BESCOM Alert: Your electricity bill is overdue and will be disconnected tonight. "
            "Contact 8812345678."
        ),
        "Axis Bank: Your account is frozen. Dial 9345678901 within 1 hour to reactivate.",
    ]
    for text in scam_samples:
        c = canonicalize(text)
        assert edge_decision.is_transactional_notice(c.normalized), text
        assert edge_decision.is_action_solicitation(c.normalized), text
        assert not edge_decision.is_passive_transactional_notice(c.normalized), text
        dec = edge_decision.decide_edge(c, [], 0.97, 0.35, mode="balanced")
        assert dec.package.risk == Risk.UNCERTAIN
        assert dec.action == Action.VERIFY_INDEPENDENTLY


# --- P2: DNS low-and-slow boundary matrix ---


def test_p2_dns_low_and_slow_multi_horizon_matrix():
    res = V.eval_p2_dns_boundaries()
    by_id = {r["id"]: r for r in res["cases"]}

    # Fast 5-query tunnel at 15 s spacing (span 60 s exact) caught by all configs
    assert by_id["tunnel_5q_every_15s"]["baseline_d1"] is True
    assert by_id["tunnel_5q_every_15s"]["hardened_d3"] is True
    assert by_id["tunnel_5q_every_15s"]["multi_horizon_d3_300s"] is True

    # Low-and-slow tunnels (6 distinct queries spaced 20s..55s over 100s..275s):
    # Evade 60 s window, CAUGHT by 300 s multi-horizon window!
    for cid in (
        "low_slow_6q_every_20s",
        "low_slow_6q_every_30s",
        "low_slow_6q_every_45s",
        "low_slow_6q_every_55s",
    ):
        assert by_id[cid]["hardened_d3"] is False, cid
        assert by_id[cid]["multi_horizon_d3_300s"] is True, cid

    # Documented evasion floor (> 60 s per query -> span > 300 s; or low-entropy hex24)
    assert by_id["low_slow_6q_every_65s"]["multi_horizon_d3_300s"] is False
    assert by_id["evasion_hex24_low_entropy"]["multi_horizon_d3_300s"] is False

    # Repeated single-hostname CDN lookalike (even 30 queries over 5 min!) produces ZERO FPs
    assert by_id["distinct_1_of_6_in_30s"]["baseline_d1"] is True
    assert by_id["distinct_1_of_6_in_30s"]["hardened_d3"] is False
    assert by_id["distinct_1_of_6_in_30s"]["multi_horizon_d3_300s"] is False
    assert by_id["benign_cdn_30q_5min_single_host"]["multi_horizon_d3_300s"] is False

    # Boundary length 39 vs 40 (low entropy)
    assert by_id["boundary_len39_low_entropy"]["multi_horizon_d3_300s"] is False
    assert by_id["boundary_len40_low_entropy"]["multi_horizon_d3_300s"] is True

    # Pre-registered default evaluate(flows) does NOT enable slow window unless requested
    slow_flows = next(
        c["flows"] for c in V.build_dns_boundary_cases() if c["id"] == "low_slow_6q_every_30s"
    )
    assert R.evaluate(slow_flows) == []


# --- P3: URL hybrid adversarial & unseen-domain checks ---


def test_p3_official_bank_clean_and_single_source_onnx_capped():
    dom = default_domain_intel()
    assert dom is not None

    # Official HDFC / ICICI URLs with low model scores must NOT be flagged suspicious
    clean_hyb = HybridUrlScorer(_ConstScorer(0.01), dom)
    for url in (
        "https://www.hdfcbank.com/personal/pay/cards",
        "https://www.icicibank.com/personal-banking/accounts",
    ):
        res = check_url(url, clean_hyb, 0.6308, high_threshold=0.75)
        assert res["verdict"] == "no_signal", (url, res)

    # Unseen clean root where ONNX spikes (0.996) but DomainIntel is clean (<0.30, 0 signals)
    # must NOT escalate to hard "phishing" (capped at "suspicious" or "no_signal")
    spiky_hyb = HybridUrlScorer(_ConstScorer(0.996), dom)
    res_z = check_url("https://zerodha.com", spiky_hyb, 0.6308, high_threshold=0.75)
    assert res_z["verdict"] != "phishing", res_z


def test_p3_offline_domain_intel_adversarial_suite_high_detection():
    res = V.eval_p3_url_adversarial(None, url_threshold=0.6308, high_threshold=0.75)
    totals = res["totals"]
    assert totals["benign_hybrid_phishing_fp"] == 0
    assert totals["attack_hybrid_any_detected"] >= totals["attack_n"] - 1


# --- P4: PII redaction, DLT header regex, and blinded field-validation harness ---


def test_p4_pii_redaction_and_field_sms_harness(tmp_path):
    red = V.redact_pii(
        "SBI A/c XX4821 paid Rs.500 to 9876543210@okhdfcbank. Call 9811223344 Ref 632189451022"
    )
    assert "9876543210@okhdfcbank" not in red
    assert "9811223344" not in red
    assert "XX4821" not in red
    assert "632189451022" not in red
    assert "[UPI_OR_EMAIL]" in red and "[PHONE]" in red and "[ACCT]" in red

    assert V.DLT_SENDER_RE.match("VM-SBIINB")
    assert V.DLT_SENDER_RE.match("AD-HDFCBK-S")
    assert not V.DLT_SENDER_RE.match("+919876543210")

    # End-to-end field validation harness test on temporary JSONL
    sample_jsonl = tmp_path / "field_sample.jsonl"
    rows = [
        {
            "sender": "VM-SBIINB",
            "label": 0,
            "text": "SBI: A/c XX4821 debited Rs.500 via UPI Ref 632189451022.",
        },
        {
            "sender": "+919876543210",
            "label": 1,
            "text": "URGENT: Your SBI KYC blocked! Update http://sbi-kyc.xyz/login",
        },
    ]
    sample_jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = V.run_field_sms_validation(
        sample_jsonl,
        lambda ts: [0.20, 0.99],
        0.35,
        None,
    )
    assert out["overall"]["tp"] == 1 and out["overall"]["tn"] == 1
    assert out["dlt_alphanumeric_count"] == 1
    assert out["gates"]["recall_ge_85pct"] is True
    assert out["gates"]["fpr_le_3pct"] is True
    assert out["gates"]["min_sample_n_ge_200"] is False  # small sample gate properly flags n < 200
