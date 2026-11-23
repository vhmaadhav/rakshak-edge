#!/usr/bin/env python3
# ruff: noqa: E501
"""Comprehensive offline validation suite for Rakshak-Edge (P0, P2, P3, P4).

Evaluates four critical deployment-readiness dimensions on real Arm hardware or laptop:
- P0: Transactional scam vs. legitimate transactional-message separation (corpus + paired Indian
      adversarial stress bank, comparing quiet / balanced_v2 / passive-tx-aware balanced_v3).
- P2: DNS tunneling low-and-slow boundary matrix (spacing, distinct qnames, label length/entropy,
      burst vs. trickle, repeated CDN) across baseline, distinct-qname, and multi-horizon configs.
- P3: URL hybrid adversarial & unseen-domain stress testing (10 adversarial evasion families +
      6 unseen Indian/global benign domain categories + optional full PhreshPhish/url_domain OOD).
- P4: Real-world Indian SMS corpus stratification (by source, language including Hinglish and
      Devanagari Hindi, Indian banking/UPI/KYC/telecom strata, DLT header patterns) plus a
      privacy-preserving blinded field-validation harness (--field-sms, --field-pcaps) with PII
      redaction and explicit deployment gates.

No network access and no live packet capture; pure Python + NumPy + Pandas (zero sklearn).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (
    REPO / "deploy" / "pi" / "service",
    REPO / "src",
    REPO / "deploy" / "pi" / "bench",
    REPO / "deploy" / "pi",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import edge_decision  # noqa: E402
from edge_pipeline import check_url  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.nids import host_rules  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402
from rakshak.url_domain_intel import HybridUrlScorer, default_domain_intel  # noqa: E402
from rakshak.url_model import UrlScorer, best_url_hit  # noqa: E402

# ---------------------------------------------------------------------------
# P0: Paired Indian Transactional Legitimate vs. Transactional Scam Stress Bank
# (Clearly marked as curated adversarial test bank, never conflated with real inbox captures.)
# ---------------------------------------------------------------------------
PAIRED_TRANSACTIONAL_BANK: tuple[tuple[str, int, str, str], ...] = (
    # --- Legitimate passive transactional notices (label=0) ---
    (
        "legit_bank_debit",
        0,
        "en",
        "SBI: A/c XX4821 debited by Rs.1,250.00 on 22-Nov-26 via UPI Ref 632189451022. Avl Bal Rs.34,820.50.",
    ),
    (
        "legit_bank_credit",
        0,
        "en",
        "HDFC Bank: Rs.15,000.00 credited to A/c XX9014 by NEFT Ref HDFC0009182736. Available balance Rs.62,410.00.",
    ),
    (
        "legit_upi_paid",
        0,
        "en",
        "ICICI Bank Acct XX7712 debited for Rs 499.00 on 22-Nov-26; Zomato credited. UPI:632612345678.",
    ),
    (
        "legit_otp_protective",
        0,
        "en",
        "849201 is your SBI OTP for online payment of Rs.2,400. Valid for 5 mins. Do not share with anyone.",
    ),
    (
        "legit_otp_hinglish",
        0,
        "hinglish",
        "HDFC Bank login ka OTP 391048 hai. 5 minute tak valid. Kisi ke saath share mat karein.",
    ),
    (
        "legit_order_refund",
        0,
        "en",
        "Myntra: Your return for order 412890 has been processed. Refund of INR 1,299 initiated to original payment mode.",
    ),
    (
        "legit_utility_bill",
        0,
        "en",
        "TATA Power: Bill for November is INR 1,420. Due date: 2026-12-05. Pay via UPI: 9812345670@oktatapower",
    ),
    (
        "legit_parcel_status",
        0,
        "en",
        "DTDC: Your shipment D88421094 has reached Bengaluru hub and is out for delivery today. No payment due.",
    ),
    (
        "legit_telecom_pack",
        0,
        "en",
        "Jio: Recharge of Rs.299 is successful for 98XXXXXX12. Plan validity 28 days. Enjoy 1.5GB/day data.",
    ),
    (
        "legit_emi_notice",
        0,
        "en",
        "Axis Bank: Auto-debit of EMI Rs.8,450 for Loan A/c XX3319 completed on 20-Nov-26. Statement available in app.",
    ),
    (
        "legit_irctc_pnr",
        0,
        "en",
        "IRCTC: PNR 4528190342 confirmed. Train 12951 Rajdhani Exp on 28-Nov-26, Coach B2 Seat 14.",
    ),
    (
        "legit_credit_card",
        0,
        "en",
        "Kotak Bank: Thank you for payment of INR 12,300 towards Credit Card XX6102. Ref KT9812345.",
    ),
    (
        "legit_hinglish_debit",
        0,
        "hinglish",
        "Aapke SBI A/c XX5510 se Rs.750 UPI se debit hue. Ref 632198765432. Bacha balance Rs.18,200.",
    ),
    (
        "legit_hinglish_parcel",
        0,
        "hinglish",
        "BlueDart parcel 7718293045 deliver ho gaya hai. Yeh confirmation hai, koi payment nahi maanga gaya.",
    ),
    (
        "legit_epfo_passbook",
        0,
        "en",
        "EPFO: Monthly contribution of Rs.3,600 credited to UAN 1009XXXX4512 for Oct-2026. Balance updated.",
    ),
    (
        "legit_fastag_toll",
        0,
        "en",
        "ICICI FASTag: Rs.115 deducted at Kherki Daula Toll Plaza on 22-Nov-26. Remaining balance Rs.840.",
    ),
    (
        "legit_gas_booking",
        0,
        "en",
        "Indane: LPG refill booking Ref 20938172 confirmed. Delivery scheduled within 3 working days.",
    ),
    (
        "legit_salary_credit",
        0,
        "en",
        "Canara Bank: A/c XX2201 credited with Rs.68,500.00 towards Salary for Nov 2026. Avl Bal Rs.91,240.00.",
    ),
    (
        "legit_amazon_delivery",
        0,
        "en",
        "Amazon: Package with tracking ID 3029184756 was delivered at front desk. Thank you for shopping.",
    ),
    (
        "legit_mutual_fund",
        0,
        "en",
        "Zerodha Coin: SIP order for Nifty 50 Index Fund Rs.5,000 processed. Units allotted in demat A/c.",
    ),
    # --- Transactional-vocabulary scams (label=1, many intentionally without URLs) ---
    (
        "scam_tx_bank_call",
        1,
        "en",
        "Dear SBI customer, your A/c XX4821 has been put on temporary hold due to incomplete PAN KYC. Call 9876543210 immediately to unblock.",
    ),
    (
        "scam_tx_elec_cut",
        1,
        "en",
        "BESCOM Alert: Your electricity bill for last month is overdue. Power connection will be disconnected tonight at 9:30 PM. Contact officer 8812345678.",
    ),
    (
        "scam_tx_upi_collect",
        1,
        "en",
        "You have received a UPI cashback reward of Rs.4,999 in your bank account. Approve the collect request and enter UPI PIN to claim immediately.",
    ),
    (
        "scam_tx_kyc_link",
        1,
        "en",
        "HDFC Bank: Your debit card and netbanking A/c will expire today. Update PAN KYC now at http://hdfcb4nk-kyc-alert.in/update",
    ),
    (
        "scam_tx_parcel_fee",
        1,
        "en",
        "India Post: Your parcel shipment IN982341 is held at warehouse due to incomplete address. Pay Rs.25 redelivery fee at http://indiapost-parcel-address-update.com/track",
    ),
    (
        "scam_tx_trai_sim",
        1,
        "en",
        "TRAI Notice: Your mobile SIM telecom service will be suspended within 2 hours due to unverified KYC. Call helpdesk 7765432109 immediately.",
    ),
    (
        "scam_tx_challan",
        1,
        "en",
        "Traffic Police e-Challan: Pending fine of Rs.2,000 on your vehicle. Pay now to avoid court notice: http://parivahan-challan-pay.xyz/fine",
    ),
    (
        "scam_tx_credit_limit",
        1,
        "en",
        "ICICI Bank: Your credit card reward points worth Rs.9,850 will expire tonight. Contact 9123456780 or reply YES to claim cash in A/c.",
    ),
    (
        "scam_tx_hinglish_kyc",
        1,
        "hinglish",
        "Priya grahak, aapka SBI bank account KYC ke bina aaj band ho jayega. Turant 9811223344 par call karein aur verify karein.",
    ),
    (
        "scam_tx_hinglish_bill",
        1,
        "hinglish",
        "Aapka bijli bill update nahi hua hai, aaj raat 10 baje electricity connection disconnect ho jayega. Turant 8765432109 par contact karein.",
    ),
    (
        "scam_tx_hinglish_upi",
        1,
        "hinglish",
        "Congratulations! Aapko Paytm UPI cashback Rs.5,000 mila hai. Claim karne ke liye turant http://ph0nepe-cashback-claim.com/upi par click karein.",
    ),
    (
        "scam_tx_otp_social",
        1,
        "en",
        "HDFC Bank security alert: Unauthorized debit of Rs.48,900 initiated on A/c XX9014. To cancel this transaction, call 9988776655 and confirm OTP immediately.",
    ),
    (
        "scam_tx_loan_approve",
        1,
        "en",
        "Instant Personal Loan of Rs.5,00,000 pre-approved for your bank account at 0% interest. WhatsApp 7890123456 today to claim.",
    ),
    (
        "scam_tx_epfo_claim",
        1,
        "en",
        "EPFO Alert: Your PF withdrawal claim of Rs.72,400 is pending bank verification. Update A/c details today at http://epfo-pf-claim-online.in/withdraw",
    ),
    (
        "scam_tx_income_tax",
        1,
        "en",
        "Income Tax Dept: Tax credit of Rs.18,450 approved for your PAN. Confirm bank account number immediately at http://incometax-refund-verify.xyz/claim",
    ),
    (
        "scam_tx_hinglish_parcel",
        1,
        "hinglish",
        "Aapka courier parcel address galat hone ke karan hold par hai. Delivery confirm karne ke liye 9012345678 par call karein.",
    ),
    (
        "scam_tx_wrong_upi",
        1,
        "en",
        "Mistakenly sent Rs.12,000 to your UPI account from my bank A/c. Urgent medical emergency, please reply and transfer back to 8123456789 today.",
    ),
    (
        "scam_tx_fastag_block",
        1,
        "en",
        "FASTag KYC Alert: Your vehicle FASTag will be blacklisted today. Update RC details immediately at http://fastag-kyc-update.cfd/verify",
    ),
    (
        "scam_tx_axis_unblock",
        1,
        "en",
        "Axis Bank: Your account netbanking is frozen due to suspicious activity. Dial 9345678901 within 1 hour to reactivate.",
    ),
    (
        "scam_tx_hinglish_lottery",
        1,
        "hinglish",
        "Aapka mobile number Jio KBC lottery Rs.25 lakh jeet chuka hai! Bank A/c mein paise claim karne ke liye 9871234560 par WhatsApp karein.",
    ),
)

# ---------------------------------------------------------------------------
# P3: Adversarial & Unseen-Domain URL Stress Bank (48 URLs, 16 categories)
# ---------------------------------------------------------------------------
ADVERSARIAL_URL_BANK: tuple[tuple[str, int, str], ...] = (
    # Unseen benign Indian & global domains (label=0)
    ("benign_indian_gov", 0, "https://www.incometax.gov.in/iec/foportal/"),
    ("benign_indian_gov", 0, "https://uidai.gov.in/en/"),
    ("benign_indian_gov", 0, "https://parivahan.gov.in/parivahan/"),
    ("benign_indian_gov", 0, "https://www.indiapost.gov.in"),
    ("benign_indian_bank", 0, "https://www.onlinesbi.sbi/"),
    ("benign_indian_bank", 0, "https://www.hdfcbank.com/personal/pay/cards"),
    ("benign_indian_bank", 0, "https://www.icicibank.com/personal-banking/accounts"),
    ("benign_indian_bank", 0, "https://www.npci.org.in/what-we-do/upi/product-overview"),
    ("benign_indian_fintech", 0, "https://www.irctc.co.in"),
    ("benign_indian_fintech", 0, "https://zerodha.com"),
    ("benign_indian_fintech", 0, "https://razorpay.com"),
    ("benign_indian_fintech", 0, "https://groww.in/mutual-funds"),
    ("benign_modern_tld", 0, "https://anthropic.com"),
    ("benign_modern_tld", 0, "https://linear.app/features"),
    ("benign_modern_tld", 0, "https://vercel.com/docs"),
    ("benign_modern_tld", 0, "https://huggingface.co/models"),
    ("benign_modern_tld", 0, "https://svelte.dev/docs"),
    (
        "benign_cdn_ecom",
        0,
        "https://cdn.shopify.com/s/files/1/0533/2089/files/asset-9f8a7b6c5d4e.png",
    ),
    ("benign_cdn_ecom", 0, "https://www.flipkart.com/samsung-galaxy-s24-5g/p/itm1234567890abcdef"),
    # Adversarial phishing evasions & newly minted Indian scams (label=1)
    (
        "adv_path_padding",
        1,
        "https://secure-account-check.xyz/blog/2026/11/how-to-improve-productivity-at-work/",
    ),
    (
        "adv_path_padding",
        1,
        "https://allegro-oferta-98123.cfd/articles/2026/top-10-healthy-recipes-for-families/",
    ),
    (
        "adv_path_padding",
        1,
        "https://kyc-bank-portal.web.app/news/technology/annual-conference-highlights/",
    ),
    ("adv_sub_stuffing", 1, "https://sbi.co.in.verify-account-india.com/login"),
    ("adv_sub_stuffing", 1, "https://hdfcbank.com.kyc-update-alert.net/confirm"),
    ("adv_sub_stuffing", 1, "https://amazon.in.order-refund-claim.org/redeem"),
    ("adv_homograph_idn", 1, "https://xn--pypal-4ve.com/signin"),
    ("adv_homograph_idn", 1, "https://xn--sbi-kyc-9ua.in/verify"),
    ("adv_homograph_cyrillic", 1, "https://\u0430mazon.in/account/verify"),
    ("adv_homograph_cyrillic", 1, "https://\u0440aytm.in/kyc/update"),
    ("adv_leetspeak", 1, "https://paypa1-secure-verify.xyz/login"),
    ("adv_leetspeak", 1, "https://hdfcb4nk-kyc-alert.in/update"),
    ("adv_leetspeak", 1, "https://ph0nepe-cashback-claim.com/upi"),
    ("adv_leetspeak", 1, "https://g00gle-account-unlock.site/signin"),
    ("adv_leetspeak", 1, "https://icic1-netbanking-check.com/login"),
    ("adv_free_hosting", 1, "https://fast-app-9281.web.app/sbi/kyc/login"),
    ("adv_free_hosting", 1, "https://portal-2026.pages.dev/?brand=hdfc&action=verify"),
    ("adv_free_hosting", 1, "https://claim-reward-99.firebaseapp.com/upi"),
    ("adv_userinfo_ip", 1, "https://www.sbi.co.in@evil-portal.com/login"),
    ("adv_userinfo_ip", 1, "http://185.199.108.153:8080/hdfc/kyc/verify"),
    ("adv_unseen_indian_com", 1, "https://sbi-kyc-helpdesk-india.com/update"),
    ("adv_unseen_indian_com", 1, "https://bescom-electricity-bill-pay.in/disconnect"),
    ("adv_unseen_indian_com", 1, "https://epfo-pf-claim-online.in/withdraw"),
    ("adv_unseen_indian_com", 1, "https://trai-sim-kyc-verification.org/confirm"),
    ("adv_unseen_indian_com", 1, "https://indiapost-parcel-address-update.com/track"),
    ("adv_case_obf", 1, "HTTPS://SBI-KYC-VERIFY.XYZ//Login/Account"),
    ("adv_case_obf", 1, "https://EVIL-OFFER-9912.CFD./verify"),
    ("adv_shortener", 1, "https://bit.ly/sbi-kyc-2026"),
    ("adv_shortener", 1, "https://tinyurl.com/hdfc-otp-verify"),
)

# ---------------------------------------------------------------------------
# P4: Indian SMS Domain Strata & Privacy-Preserving PII Redaction
# ---------------------------------------------------------------------------
INDIAN_STRATA: dict[str, re.Pattern[str]] = {
    "indian_bank_upi": re.compile(
        r"\b(?:sbi|hdfc|icici|axis|kotak|pnb|bob|canara|upi|vpa|paytm|phonepe|gpay|bhim|imps|neft|rtgs|ifsc)\b",
        re.I,
    ),
    "indian_kyc_gov": re.compile(
        r"\b(?:kyc|aadhaar|aadhar|pan card|epfo|trai|challan|fastag|rbi|npci|irctc|income tax|uidai|digilocker)\b",
        re.I,
    ),
    "indian_currency_tel": re.compile(
        r"\b(?:rs\.?|inr|rupees?|lakh|crore|jio|airtel|bsnl|bescom|msedcl|tatapower|indane|bharatgas|indiapost|dtdc|bluedart|delhivery)\b",
        re.I,
    ),
}
DLT_SENDER_RE = re.compile(r"^[A-Za-z]{2}-[A-Za-z0-9]{3,9}(?:-[SPGT])?$")

_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b"), "[UPI_OR_EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b(?:XX|xx|\*{2,})\d{3,6}\b"), "[ACCT]"),
    (re.compile(r"\b\d{6,16}\b"), "[NUM]"),
)


def redact_pii(text: str) -> str:
    """Mask phone numbers, UPI/email handles, account tails, and long numeric IDs before logging."""
    out = str(text)
    for rx, repl in _PII_PATTERNS:
        out = rx.sub(repl, out)
    return out


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "n": int(len(yt)),
        "pos": int((yt == 1).sum()),
        "neg": int((yt == 0).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "fpr": round(fpr, 4),
        "f1": round(f1, 4),
    }


def vcgencmd_snapshot() -> dict[str, str]:
    out: dict[str, str] = {}
    for cmd in ("measure_temp", "get_throttled", "measure_clock arm"):
        try:
            res = subprocess.run(
                ["vcgencmd", *cmd.split()], capture_output=True, text=True, timeout=3, check=False
            )
            if res.returncode == 0:
                out[cmd] = res.stdout.strip()
        except OSError:
            break
    return out


# ---------------------------------------------------------------------------
# P2: DNS Low-and-Slow Boundary Matrix (pure flow-level, zero live capture)
# ---------------------------------------------------------------------------
def _make_dns_flow(ts_ms: float, name: str, src: str = "10.20.30.99") -> dict[str, Any]:
    return {
        "src_ip": src,
        "dst_ip": "192.168.1.1",
        "src_port": 53000,
        "dst_port": 53,
        "protocol": 17,
        "bidirectional_first_seen_ms": ts_ms,
        "bidirectional_duration_ms": 2.0,
        "bidirectional_bytes": 140,
        "bidirectional_syn_packets": 0,
        "dst2src_packets": 1,
        "requested_server_name": name,
    }


def build_dns_boundary_cases() -> list[dict[str, Any]]:
    """Deterministic DNS tunneling boundary cases covering spacing, distinctness, length & entropy."""
    t0 = 1_700_000_000_000.0
    cdn_single = "assets-prod-eu-west-1-7f3a9c2e1b0d4a6f8e5c.cdn.example.net"
    cases: list[dict[str, Any]] = []

    def distinct_b32(k: int, spacing_s: float) -> list[dict[str, Any]]:
        alphabet = "abcdefghijklmnopqrstuvwxyz234567"
        out = []
        for i in range(k):
            # 44-char high-entropy distinct base32 subdomain label per query
            label = "".join(alphabet[(i * 7 + j * 13 + j * j) % len(alphabet)] for j in range(44))
            out.append(_make_dns_flow(t0 + i * spacing_s * 1000.0, f"{label}.tun.example.net"))
        return out

    # 1. Spacing sweep (5 queries, distinct high-entropy base32)
    for sp in (10.0, 12.0, 15.0, 16.0, 20.0):
        cases.append(
            {
                "id": f"tunnel_5q_every_{int(sp)}s",
                "kind": "attack",
                "span_s": round(4 * sp, 1),
                "flows": distinct_b32(5, sp),
            }
        )

    # 2. Low-and-slow sweep (6–8 distinct queries over 2–6 minutes)
    for k, sp in ((6, 20.0), (6, 30.0), (6, 45.0), (6, 55.0), (6, 65.0), (6, 90.0)):
        cases.append(
            {
                "id": f"low_slow_{k}q_every_{int(sp)}s",
                "kind": "attack_low_slow",
                "span_s": round((k - 1) * sp, 1),
                "flows": distinct_b32(k, sp),
            }
        )

    # 3. Distinct-qname sweep (6 queries in 30 s with 1, 2, 3 distinct suspicious names)
    for d in (1, 2, 3):
        names = [
            f"cdn-edge-shard-{m}-7f3a9c2e1b0d4a6f8e5c9a1b2c3d4e5f.cdn.example.net" for m in range(d)
        ]
        flows = [_make_dns_flow(t0 + i * 5000.0, names[i % d]) for i in range(6)]
        cases.append(
            {
                "id": f"distinct_{d}_of_6_in_30s",
                "kind": "benign_cdn" if d == 1 else "borderline_distinct",
                "span_s": 25.0,
                "flows": flows,
            }
        )

    # 4. Sustained benign CDN lookalike over 5 minutes (30 repeated queries of 1 hostname)
    cases.append(
        {
            "id": "benign_cdn_30q_5min_single_host",
            "kind": "benign_cdn",
            "span_s": 290.0,
            "flows": [_make_dns_flow(t0 + i * 10_000.0, cdn_single) for i in range(30)],
        }
    )

    # 5. Label length & entropy evasion floor (24-char low-entropy hex vs 39-char vs 40-char)
    hex24 = [
        _make_dns_flow(t0 + i * 2000.0, f"{i:04x}a1b2c3a1b2c3a1b2c3a1.hex.example.net")
        for i in range(8)
    ]
    cases.append(
        {"id": "evasion_hex24_low_entropy", "kind": "evasion_floor", "span_s": 14.0, "flows": hex24}
    )

    len39_low = [
        _make_dns_flow(t0 + i * 2000.0, f"{'a' * 37}{i:02d}.low.example.net") for i in range(6)
    ]
    cases.append(
        {
            "id": "boundary_len39_low_entropy",
            "kind": "boundary_below",
            "span_s": 10.0,
            "flows": len39_low,
        }
    )

    len40_low = [
        _make_dns_flow(t0 + i * 2000.0, f"{'a' * 38}{i:02d}.low.example.net") for i in range(6)
    ]
    cases.append(
        {"id": "boundary_len40_low_entropy", "kind": "attack", "span_s": 10.0, "flows": len40_low}
    )

    return cases


def eval_p2_dns_boundaries() -> dict[str, Any]:
    """Run the DNS low-and-slow boundary matrix across three rule configurations."""
    configs = {
        "baseline_d1": {"dns_min_distinct": 1, "dns_slow_window_ms": None},
        "hardened_d3": {"dns_min_distinct": 3, "dns_slow_window_ms": None},
        "multi_horizon_d3_300s": {
            "dns_min_distinct": 3,
            "dns_slow_window_ms": host_rules.DNS_SLOW_WINDOW_MS,
        },
    }
    cases = build_dns_boundary_cases()
    rows = []
    for case in cases:
        verdicts = {}
        for cname, kwargs in configs.items():
            hits = host_rules.evaluate(case["flows"], **kwargs)
            verdicts[cname] = any(h.type == "dns_tunnel" for h in hits)
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "queries": len(case["flows"]),
                "span_s": case["span_s"],
                **verdicts,
            }
        )
    summary = {}
    for cname in configs:
        benign_fp = sum(1 for r in rows if r["kind"] == "benign_cdn" and r[cname])
        benign_n = sum(1 for r in rows if r["kind"] == "benign_cdn")
        fast_det = sum(1 for r in rows if r["kind"] == "attack" and r[cname])
        fast_n = sum(1 for r in rows if r["kind"] == "attack")
        slow_det = sum(1 for r in rows if r["kind"] == "attack_low_slow" and r[cname])
        slow_n = sum(1 for r in rows if r["kind"] == "attack_low_slow")
        summary[cname] = {
            "benign_cdn_fp": f"{benign_fp}/{benign_n}",
            "fast_attack_detected": f"{fast_det}/{fast_n}",
            "low_slow_detected": f"{slow_det}/{slow_n}",
        }
    return {"summary": summary, "cases": rows}


# ---------------------------------------------------------------------------
# P3: Adversarial & Unseen-Domain URL Evaluation
# ---------------------------------------------------------------------------
def eval_p3_url_adversarial(
    onnx_scorer: UrlScorer | None,
    *,
    url_threshold: float = 0.6308,
    high_threshold: float = 0.75,
) -> dict[str, Any]:
    dom = default_domain_intel()
    assert dom is not None, "Bundled url_domain_intel.npz not found"
    hybrid = HybridUrlScorer(onnx_scorer, dom)

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for cat, label, url in ADVERSARIAL_URL_BANK:
        res_hyb = check_url(url, hybrid, url_threshold, high_threshold=high_threshold)
        res_onnx = check_url(url, onnx_scorer, url_threshold) if onnx_scorer is not None else None
        by_cat.setdefault(cat, []).append(
            {
                "label": label,
                "url": url,
                "hybrid_verdict": res_hyb["verdict"],
                "hybrid_score": res_hyb["url_model"]["score"],
                "domain_intel": res_hyb.get("domain_intel"),
                "onnx_verdict": res_onnx["verdict"] if res_onnx is not None else None,
            }
        )

    cat_summary: dict[str, Any] = {}
    benign_n = benign_phish = benign_any = onnx_benign_any = 0
    attack_n = attack_phish = attack_any = onnx_attack_any = 0
    for cat, items in by_cat.items():
        y0 = items[0]["label"]
        n = len(items)
        n_phish = sum(1 for x in items if x["hybrid_verdict"] == "phishing")
        n_any = sum(1 for x in items if x["hybrid_verdict"] != "no_signal")
        n_onnx = sum(1 for x in items if x["onnx_verdict"] and x["onnx_verdict"] != "no_signal")
        cat_summary[cat] = {
            "label": y0,
            "n": n,
            "hybrid_phishing": n_phish,
            "hybrid_any_flag": n_any,
            "onnx_only_flag": n_onnx if onnx_scorer is not None else None,
            "flagged_benign": [
                x["url"] for x in items if y0 == 0 and x["hybrid_verdict"] != "no_signal"
            ],
            "missed_attacks": [
                x["url"] for x in items if y0 == 1 and x["hybrid_verdict"] == "no_signal"
            ],
        }
        if y0 == 0:
            benign_n += n
            benign_phish += n_phish
            benign_any += n_any
            onnx_benign_any += n_onnx
        else:
            attack_n += n
            attack_phish += n_phish
            attack_any += n_any
            onnx_attack_any += n_onnx

    return {
        "totals": {
            "benign_n": benign_n,
            "benign_hybrid_phishing_fp": benign_phish,
            "benign_hybrid_any_flag": benign_any,
            "benign_onnx_only_flag": onnx_benign_any if onnx_scorer is not None else None,
            "attack_n": attack_n,
            "attack_hybrid_phishing": attack_phish,
            "attack_hybrid_any_detected": attack_any,
            "attack_onnx_only_detected": onnx_attack_any if onnx_scorer is not None else None,
        },
        "by_category": cat_summary,
    }


# ---------------------------------------------------------------------------
# P0 & P4: SMS Policy Evaluation (Paired Adversarial + Real Corpus + Field Harness)
# ---------------------------------------------------------------------------
def score_sms_policy(
    texts: Sequence[str],
    probs: Sequence[float],
    sms_thr: float,
    url_scorer: UrlScorer | None,
    *,
    mode: str = "balanced",
    verify_threshold: float = edge_decision.DEFAULT_VERIFY_THRESHOLD,
    tx_verify_threshold: float = edge_decision.DEFAULT_TX_VERIFY_THRESHOLD,
    url_threshold: float = 0.6308,
    canons: Sequence[Any] | None = None,
    rules_list: Sequence[list[Any]] | None = None,
) -> tuple[np.ndarray, list[bool], list[bool]]:
    """Run canonicalize -> rules -> corroborated URL -> decide_edge identically to the service."""
    canons = list(canons) if canons is not None else [canonicalize(t) for t in texts]
    rules_list = list(rules_list) if rules_list is not None else [run_rules(c) for c in canons]
    url_map: dict[str, float] = {}
    if url_scorer is not None and mode != "quiet":
        all_urls = sorted({u for c in canons for u in c.urls})
        if all_urls:
            url_map = dict(zip(all_urls, url_scorer.score_urls(all_urls), strict=True))

    class _MapScorer:
        def score_urls(self, us):
            return [url_map[u] for u in us]

    ms = _MapScorer() if url_map else None
    preds = np.zeros(len(texts), dtype=np.int32)
    is_tx: list[bool] = []
    no_ev: list[bool] = []
    for i, c in enumerate(canons):
        p = float(probs[i])
        base_r = rules_list[i]
        hit = best_url_hit(c.urls, ms) if (ms is not None and c.urls) else None
        ev = edge_decision.add_url_evidence(
            base_r, c, hit, url_threshold, mode, p_scam=p, p_threshold=sms_thr
        )
        dec = edge_decision.decide_edge(
            c,
            ev,
            p,
            sms_thr,
            mode,
            verify_threshold=verify_threshold,
            tx_verify_threshold=tx_verify_threshold,
        )
        preds[i] = 0 if dec.action.value == "IGNORE" else 1
        is_tx.append(edge_decision.is_transactional_notice(c.normalized))
        no_ev.append(len(ev) == 0)
    return preds, is_tx, no_ev


def run_field_sms_validation(
    path: Path,
    score_fn,
    sms_thr: float,
    url_scorer: UrlScorer | None,
) -> dict[str, Any]:
    """Blinded field-validation runner for real Indian SMS JSONL/CSV exports with PII redaction."""
    import pandas as pd

    if path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Field SMS file must contain 'text' and 'label' (0/1) columns")
    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    senders = (
        df["sender"].fillna("").astype(str).tolist()
        if "sender" in df.columns
        else [""] * len(texts)
    )
    probs = score_fn(texts)
    preds, is_tx, _ = score_sms_policy(texts, probs, sms_thr, url_scorer, mode="balanced")
    overall = binary_metrics(labels, preds)
    dlt_mask = [bool(DLT_SENDER_RE.match(s.strip())) for s in senders]
    fps_redacted = [
        {
            "idx": i,
            "prob": round(float(probs[i]), 4),
            "sender_dlt": dlt_mask[i],
            "text": redact_pii(texts[i])[:140],
        }
        for i in range(len(texts))
        if labels[i] == 0 and preds[i] == 1
    ]
    fns_redacted = [
        {
            "idx": i,
            "prob": round(float(probs[i]), 4),
            "sender_dlt": dlt_mask[i],
            "text": redact_pii(texts[i])[:140],
        }
        for i in range(len(texts))
        if labels[i] == 1 and preds[i] == 0
    ]
    # Explicit deployment readiness gates
    gates = {
        "recall_ge_85pct": overall["recall"] >= 0.85,
        "fpr_le_3pct": overall["fpr"] <= 0.03,
        "min_sample_n_ge_200": overall["n"] >= 200,
    }
    return {
        "file": str(path),
        "overall": overall,
        "transactional_subset": binary_metrics(
            [labels[i] for i, t in enumerate(is_tx) if t],
            [int(preds[i]) for i, t in enumerate(is_tx) if t],
        )
        if any(is_tx)
        else None,
        "dlt_alphanumeric_count": sum(dlt_mask),
        "gates": gates,
        "pass_all_gates": all(gates.values()),
        "sample_redacted_fps": fps_redacted[:10],
        "sample_redacted_fns": fns_redacted[:10],
    }


def run_field_pcap_validation(pcap_dir: Path, nids_analyzer) -> dict[str, Any]:
    """Blinded field-validation runner for real network PCAP directories (session-level metrics)."""
    pcaps = sorted(pcap_dir.glob("*.pcap"))
    rows = []
    benign_total = benign_alerted = 0
    for p in pcaps:
        res = nids_analyzer.analyse_body(p.read_bytes())
        is_benign = "benign" in p.stem.lower()
        if is_benign:
            benign_total += 1
            if res["decision"]["alerts"] > 0:
                benign_alerted += 1
        rows.append(
            {
                "pcap": p.name,
                "flows": res["flows_processed"],
                "alerts": res["decision"]["alerts"],
                "model_only_hosts": res["decision"]["model_only_hosts"],
                "family": res["decision"]["family"],
                "flow_source": res["flow_source"],
            }
        )
    return {
        "pcap_dir": str(pcap_dir),
        "pcap_count": len(pcaps),
        "benign_session_fpr": round(benign_alerted / max(benign_total, 1), 4)
        if benign_total
        else None,
        "captures": rows,
    }


# ---------------------------------------------------------------------------
# Full Hardware Runner (Pi 5 or local repo with models)
# ---------------------------------------------------------------------------
def run_full_hardware_suite(root: Path, out_path: Path | None = None) -> dict[str, Any]:
    import pandas as pd
    from nids_pipeline import NidsAnalyzer
    from rakshak_edge_service import _DetectorAdapter, load_nids
    from rbench.datasets import _read_split_ids, load_rows
    from rbench.registry import load_registry
    from rbench.runners.onnx import SmsRunner

    from rakshak.fusion.url_expert import FrozenURLExpert

    t0 = time.perf_counter()
    report: dict[str, Any] = {
        "root": str(root),
        "vcgencmd_start": vcgencmd_snapshot(),
    }

    reg = load_registry(root / "deploy" / "pi" / "bench" / "registry.toml")
    sms_model = reg.model("sms_student")
    url_model = reg.model("url_student")
    sms_thr = float(sms_model.threshold)

    sms_runner = SmsRunner()
    sms_runner.load(root / sms_model.path, 2, {})
    adapter = _DetectorAdapter(sms_runner, batch_onnx=True)
    url_expert = FrozenURLExpert((root / url_model.path).parent)
    dom_scorer = default_domain_intel()
    assert dom_scorer is not None
    hybrid_url = HybridUrlScorer(url_expert, dom_scorer)

    def score_texts(texts: Sequence[str]) -> np.ndarray:
        canons = [canonicalize(t) for t in texts]
        rules = [run_rules(c) for c in canons]
        probs = np.empty(len(texts), dtype=np.float64)
        for i in range(0, len(texts), 128):
            probs[i : i + 128] = adapter.score_prepared(
                list(texts[i : i + 128]), canons[i : i + 128], rules[i : i + 128]
            )
        return probs

    # --- P0: Paired Adversarial Indian Transactional Bank + Real Corpus Breakdown ---
    paired_texts = [x[3] for x in PAIRED_TRANSACTIONAL_BANK]
    paired_y = [x[1] for x in PAIRED_TRANSACTIONAL_BANK]
    paired_probs = score_texts(paired_texts)
    p0_paired: dict[str, Any] = {}
    for mode_name, mode_arg in (
        ("quiet", "quiet"),
        ("balanced_v3_passive_tx", "balanced"),
        ("verify", "verify"),
    ):
        preds, _, _ = score_sms_policy(
            paired_texts, paired_probs, sms_thr, hybrid_url, mode=mode_arg
        )
        m = binary_metrics(paired_y, preds)
        m["missed_scams"] = [
            PAIRED_TRANSACTIONAL_BANK[i][0]
            for i in range(len(paired_y))
            if paired_y[i] == 1 and preds[i] == 0
        ]
        m["false_alarms"] = [
            PAIRED_TRANSACTIONAL_BANK[i][0]
            for i in range(len(paired_y))
            if paired_y[i] == 0 and preds[i] == 1
        ]
        p0_paired[mode_name] = m

    # Real corpora P0 + P4 breakdown
    sms_df = pd.read_parquet(root / "data/processed/sms.parquet")
    test_ids = set(_read_split_ids(root / "data/processed/splits/campaign.json", "test"))
    eval_ok = sms_df["eval_ok"].fillna(False).astype(bool) & (
        ~sms_df["synthetic"].fillna(False).astype(bool)
    )
    df_test = sms_df[sms_df["id"].astype(str).isin(test_ids) & eval_ok].reset_index(drop=True)

    glm_df = pd.read_parquet(root / "data/processed/external/external_groundlm.parquet")
    glm_neg = glm_df[
        glm_df["policy_negative"].fillna(False).astype(bool)
        & glm_df["text_available"].fillna(False).astype(bool)
        & glm_df["eval_ok"].fillna(False).astype(bool)
        & (glm_df["label_binary"] == 0)
    ].reset_index(drop=True)
    glm_pos = glm_df[
        glm_df["text_available"].fillna(False).astype(bool)
        & glm_df["eval_ok"].fillna(False).astype(bool)
        & (glm_df["label_binary"] == 1)
    ].reset_index(drop=True)

    smx_df = pd.read_parquet(root / "data/processed/external/external_smishx.parquet")
    smx_df = smx_df[smx_df["eval_ok"].fillna(False).astype(bool)].reset_index(drop=True)

    p0_corpus: dict[str, Any] = {}
    p4_strata: dict[str, Any] = {}
    for dname, sub in (
        ("campaign_test", df_test),
        ("groundlm_neg", glm_neg),
        ("groundlm_pos", glm_pos),
        ("smishx_all", smx_df),
    ):
        texts = sub["text"].fillna("").astype(str).tolist()
        y = sub["label_binary"].astype(int).to_numpy()
        probs = score_texts(texts)
        preds_q, is_tx, no_ev = score_sms_policy(texts, probs, sms_thr, hybrid_url, mode="quiet")
        preds_b3, _, _ = score_sms_policy(texts, probs, sms_thr, hybrid_url, mode="balanced")
        is_tx_a = np.asarray(is_tx)
        no_ev_a = np.asarray(no_ev)
        entry: dict[str, Any] = {
            "quiet_all": binary_metrics(y, preds_q),
            "balanced_v3_all": binary_metrics(y, preds_b3),
        }
        if is_tx_a.any():
            entry["balanced_v3_tx_all"] = binary_metrics(y[is_tx_a], preds_b3[is_tx_a])
        tx_no_ev = is_tx_a & no_ev_a
        if tx_no_ev.any():
            entry["balanced_v3_tx_no_rule"] = binary_metrics(y[tx_no_ev], preds_b3[tx_no_ev])
        p0_corpus[dname] = entry

        if dname == "campaign_test":
            by_source: dict[str, Any] = {}
            for src, grp_idx in sub.groupby(sub["source"].astype(str)).groups.items():
                idx = np.asarray(list(grp_idx))
                by_source[str(src)] = {
                    "quiet": binary_metrics(y[idx], preds_q[idx]),
                    "balanced_v3": binary_metrics(y[idx], preds_b3[idx]),
                }
            by_lang: dict[str, Any] = {}
            for lg, grp_idx in sub.groupby(sub["lang"].astype(str)).groups.items():
                idx = np.asarray(list(grp_idx))
                by_lang[str(lg)] = {
                    "quiet": binary_metrics(y[idx], preds_q[idx]),
                    "balanced_v3": binary_metrics(y[idx], preds_b3[idx]),
                }
            by_indian: dict[str, Any] = {}
            for sname, srx in INDIAN_STRATA.items():
                mask = np.asarray([bool(srx.search(t)) for t in texts])
                if mask.any():
                    by_indian[sname] = {
                        "quiet": binary_metrics(y[mask], preds_q[mask]),
                        "balanced_v3": binary_metrics(y[mask], preds_b3[mask]),
                    }
            p4_strata = {
                "by_source": by_source,
                "by_lang": by_lang,
                "by_indian_domain_strata": by_indian,
            }

    report["p0_transactional"] = {"paired_indian_bank": p0_paired, "corpora": p0_corpus}

    # --- P2: DNS Low-and-Slow Boundaries + 26-pcap holdout ---
    p2_res = eval_p2_dns_boundaries()
    matches = list(Path("/home/hermoine/rakshak-bench").rglob("ho_benign_cdn_lookalike_6.pcap"))
    if matches:
        hdir = matches[0].parent
        nids_scorer, _ = load_nids(root)
        holdout_multi = NidsAnalyzer(
            nids_scorer,
            scratch=root / "run" / "nids_uploads",
            flow_source="auto",
            model_mode="quiet",
            dns_min_distinct=3,
            dns_slow_window_ms=host_rules.DNS_SLOW_WINDOW_MS,
        )
        p2_res["holdout_26_pcaps_multi_horizon"] = run_field_pcap_validation(hdir, holdout_multi)
    report["p2_dns_low_and_slow"] = p2_res

    # --- P3: URL Hybrid Adversarial & Unseen-Domain + PhreshPhish/url_domain regression check ---
    p3_res = eval_p3_url_adversarial(url_expert, url_threshold=0.6308, high_threshold=0.75)
    pp_rows = load_rows(root, reg.dataset("phreshphish_test"), limit=4_000, seed=42).items
    ud_rows = load_rows(root, reg.dataset("url_domain_test"), limit=4_000, seed=42).items
    for tag, rows in (("phreshphish_4k", pp_rows), ("url_domain_ood_4k", ud_rows)):
        urls = [r["text"] for r in rows]
        y_u = [int(r["label"]) for r in rows]
        scores = hybrid_url.score_urls(urls)
        p3_res[tag] = {
            "at_0.6308_suspicious": binary_metrics(y_u, [1 if s >= 0.6308 else 0 for s in scores]),
            "at_0.7500_phishing": binary_metrics(y_u, [1 if s >= 0.75 else 0 for s in scores]),
        }
    report["p3_url_adversarial"] = p3_res

    # --- P4: Indian SMS Corpus Stratification ---
    report["p4_indian_sms_strata"] = p4_strata
    report["vcgencmd_end"] = vcgencmd_snapshot()
    report["elapsed_s"] = round(time.perf_counter() - t0, 2)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rakshak-Edge P0/P2/P3/P4 validation suite")
    p.add_argument("--root", default=str(REPO), help="Bench root with models/ and data/")
    p.add_argument("--suite", choices=("all", "p2", "p3_offline", "field"), default="all")
    p.add_argument("--field-sms", default=None, help="Blinded real SMS CSV/JSONL/Parquet file")
    p.add_argument("--field-pcaps", default=None, help="Directory of real network PCAP files")
    p.add_argument("--out", default=None, help="Write JSON report to this path")
    args = p.parse_args(argv)

    out_path = Path(args.out) if args.out else None
    if args.suite == "p2":
        res = eval_p2_dns_boundaries()
    elif args.suite == "p3_offline":
        res = eval_p3_url_adversarial(None)
    elif args.suite == "field":
        root = Path(args.root)
        res = {}
        if args.field_sms:
            from rakshak_edge_service import _DetectorAdapter
            from rbench.registry import load_registry
            from rbench.runners.onnx import SmsRunner

            from rakshak.fusion.url_expert import FrozenURLExpert

            reg = load_registry(root / "deploy" / "pi" / "bench" / "registry.toml")
            sm = reg.model("sms_student")
            um = reg.model("url_student")
            sr = SmsRunner()
            sr.load(root / sm.path, 2, {})
            ad = _DetectorAdapter(sr, batch_onnx=True)
            hyb = HybridUrlScorer(FrozenURLExpert((root / um.path).parent), default_domain_intel())
            res["field_sms"] = run_field_sms_validation(
                Path(args.field_sms),
                lambda ts: ad.score_many(list(ts)),
                float(sm.threshold),
                hyb,
            )
        if args.field_pcaps:
            from nids_pipeline import NidsAnalyzer
            from rakshak_edge_service import load_nids

            ns, _ = load_nids(root)
            na = NidsAnalyzer(
                ns,
                scratch=root / "run" / "nids_uploads",
                flow_source="auto",
                model_mode="quiet",
                dns_min_distinct=3,
                dns_slow_window_ms=host_rules.DNS_SLOW_WINDOW_MS,
            )
            res["field_pcaps"] = run_field_pcap_validation(Path(args.field_pcaps), na)
    else:
        res = run_full_hardware_suite(Path(args.root), out_path=out_path)

    if out_path is not None and args.suite != "all":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
