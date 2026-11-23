# Adversarial, Boundary & Indian-Context Validation Report (P0, P2, P3, P4)

Date: **2026-11-23**. Measured on **Raspberry Pi 5** (4 GB, 4x Cortex-A76, `rakshak-thermald` active, start `57.1 °C` → end `69.7 °C`, `throttled=0x0`, total suite runtime `128.13 s`) using `scripts/eval_validation_suite.py --suite all`.

> **Scientific honesty note:** Every dataset below is explicitly labeled as either (a) an existing repo benchmark split, (b) a curated adversarial stress bank, or (c) synthetic RFC 1918 PCAP traffic. Real carrier-delivered Indian SMS inbox captures and live home-router LAN traffic are **not** claimed here; Section 4 provides the blinded, PII-redacting field-validation harness (`--suite field`) to execute those checks once consented field captures are collected.

---

## 1. P0 — Transactional Scam vs. Legitimate Message Evaluation

### Problem audited
The initial transactional-aware `balanced` policy (`balanced_v2`) applied a strict `tx_verify_threshold = 0.999` to every detector-only message matching transactional vocabulary (`bank|upi|otp|delivery|bill|balance|rs|inr|emi|loan|...`). While that kept GroundLM legitimate-message FPR under `1%`, an attacker crafting a **transactional-style scam** (fake bank hold, electricity bill disconnection, UPI collect/cashback lure, fake courier fee) with no URL or rule match could exploit that higher threshold.

### Hardening (`balanced_v3`, passive-transactional gating)
`deploy/pi/service/edge_decision.py` now distinguishes **passive transactional notices** (`is_passive_transactional_notice`) from **action-soliciting transactional messages** (`is_action_solicitation`: fixed domain semantics for callback/click/verify/unblock/disconnect/claim cues and phone numbers, explicitly excluding passive terms like `"refund"` and UPI VPA handles such as `9812345670@oktatapower`):
- Passive transactional notices (debit/credit alerts, protective OTPs, passive order/refund notices, utility due notices) retain `tx_verify_threshold = 0.999`.
- Action-soliciting transactional messages use `verify_threshold = 0.9566`.

### 1a. Real corpus evaluation on Pi 5 ONNX (`sms_student.onnx` + `HybridUrlScorer`)

| Corpus / Slice | n (`pos` / `neg`) | `quiet` (V1 default) | **`balanced_v3` (passive-tx-aware)** |
|---|---:|---|---|
| **Campaign-disjoint test — all** | 22,145 (`11,193` / `10,952`) | Rec 16.20%, FPR **0.61%**, F1 0.2774 | **Rec 88.60% (+72.4 pp)**, **FPR 0.83%** (`91`), **F1 0.9355** |
| ↳ **Transactional subset** (`is_tx`) | 4,375 (`4,207` / `168`) | — | **Rec 89.23%**, Prec 99.44%, **F1 0.9406** |
| ↳ **No-rule transactional subset** (`is_tx & no_ev`) | 3,391 (`3,245` / `146`) | — | **Rec 86.07%**, **FPR 0.68%** (`1/146`), Prec **99.96%**, **F1 0.9250** |
| **GroundLM policy negatives** (legit SMS) | 1,133 (`0` / `1,133`) | FPR **0.18%** (`2`) | **FPR 1.50%** (`17 / 1,133`) |
| **GroundLM positive scams** (external) | 91 (`91` / `0`) | Rec 12.09% (`11`) | **Rec 53.85%** (`49 / 91`) · **TX scams: `25 / 25` (`100.0%`)**, **no-rule TX: `23 / 23` (`100.0%`)** |
| **SmishX `eval_ok`** (independent) | 1,107 (`501` / `606`) | Rec 18.76%, FPR **3.63%**, F1 0.3047 | **Rec 79.64%**, FPR **6.77%**, **F1 0.8480** · TX rec `66.27%` (no-rule TX `51.72%`) |

### 1b. Paired adversarial Indian transactional stress bank (`40` EN + Hinglish messages)

Curated 20-vs-20 paired bank (`PAIRED_TRANSACTIONAL_BANK` in `scripts/eval_validation_suite.py`) pairing genuine Indian passive notices (SBI/HDFC/ICICI/Axis/Kotak/UPI/OTP/Myntra/TATA Power/DTDC/Jio/IRCTC/EPFO/FASTag/Indane/Zerodha) against transactional-vocabulary scams mimicking the same institutions (many intentionally without URLs):

| Mode | Scam Recall (`n=20`) | Legit FPR (`n=20`) | F1 | Failure Analysis |
|---|---:|---:|---:|---|
| `quiet` (V1 default) | 90.0% (`18/20`) | **10.0%** (`2/20`) | 0.9000 | Missed `scam_tx_credit_limit`, `scam_tx_loan_approve`; FPs: `legit_hinglish_parcel`, `legit_amazon_delivery` (parcel rule). |
| **`balanced_v3` (passive-tx)** | **100.0% (`20/20`)** | 15.0% (`3/20`) | **0.9302** | **0 missed scams**; 1 extra FP (`legit_mutual_fund`, SIP/demat notice outside bank/UPI/OTP vocabulary). |
| `verify` (all detector flags) | 100.0% (`20/20`) | **100.0% (`20/20`)** | 0.6667 | Every legitimate Indian transactional notice triggers a false warning. |

---

## 2. P2 — DNS Low-and-Slow Boundary Testing

### Multi-horizon rule (`--nids-dns-min-distinct 3 --nids-dns-slow-window-s 300`)
`src/rakshak/nids/host_rules.py` retains the pre-registered 2026-11-07 rule as default (`dns_min_distinct=1`, `dns_slow_window_ms=None`). Enabling `--nids-dns-slow-window-s 300` adds a secondary 5-minute horizon requiring `>= 6` distinct suspicious DNS query names, catching throttled beacons without triggering on repeated single-hostname CDN lookups.

### 2a. 18-case boundary matrix (exact detection envelope vs. evasion floor)

| Scenario | Kind | Queries / Span | `baseline_d1` | `hardened_d3` | **`multi_horizon_d3_300s`** |
|---|---|---:|:---:|:---:|:---:|
| `tunnel_5q_every_10s` / `12s` / `15s` | Fast attack | 5 / 40–60 s | **Fire** | **Fire** | **Fire** |
| `tunnel_5q_every_16s` / `20s` | Fast edge miss (5 q > 60 s) | 5 / 64–80 s | Miss | Miss | Miss |
| `low_slow_6q_every_20s` | Low-and-slow | 6 / 100 s | Miss | Miss | **Fire** |
| `low_slow_6q_every_30s` | Low-and-slow | 6 / 150 s | Miss | Miss | **Fire** |
| `low_slow_6q_every_45s` | Low-and-slow | 6 / 225 s | Miss | Miss | **Fire** |
| `low_slow_6q_every_55s` | Low-and-slow | 6 / 275 s | Miss | Miss | **Fire** |
| `low_slow_6q_every_65s` / `90s` | **Evasion floor** (> 300 s) | 6 / 325–450 s | Miss | Miss | Miss |
| `distinct_1_of_6_in_30s` | Benign CDN lookalike | 6 / 25 s | **FP** | Clean | **Clean** |
| `benign_cdn_30q_5min_single_host` | Sustained benign CDN | 30 / 290 s | **FP** | Clean | **Clean** |
| `distinct_2_of_6_in_30s` / `distinct_3_of_6_in_30s` | Distinctness edge | 6 / 25 s | Fire / Fire | Clean / **Fire** | Clean / **Fire** |
| `boundary_len39_low_entropy` / `len40` | Label-length edge | 6 / 10 s | Clean / **Fire** | Clean / **Fire** | Clean / **Fire** |
| `evasion_hex24_low_entropy` | **Evasion floor** (<40 ch, <3.8 bits) | 8 / 14 s | Miss | Miss | Miss |

### 2b. Full 26-pcap holdout on Pi 5 (`pcap_nfstream`, `multi_horizon_d3_300s`)
- **Benign capture session FPR:** **`0 / 6` (`0.0%`)** (`ho_benign_cdn_lookalike_6.pcap` cleanly suppressed).
- **Attack detection:** **`11 / 11` (`100%`)** with **`11 / 11` exact family labels** and **`9 / 9` boundary captures exact**.
- **Documented evasion floor:** beacons slower than 1 distinct query per 60 s (`< 6` distinct queries per 5 min) or short low-entropy hex labels (`< 40` chars and `< 3.8` bits/char, e.g. `ho_dns_hex24.pcap`) remain outside the stateless per-request rule envelope.

---

## 3. P3 — URL Hybrid Adversarial & Unseen-Domain Testing

### Hardening applied after initial adversarial audit
1. **Official curated brand domain guard (`is_official_brand_domain`):** stopped lexical-only `suspicious_url` / `multi_phishing_tokens` false positives on official bank URLs (`hdfcbank.com/personal/pay/cards`, `icicibank.com/personal-banking/accounts`).
2. **ONNX logit clipping (`onnx_logit_clip=4.0`) + independent-evidence cap:** prevents an uncalibrated `0.996` ONNX spike on a bare benign root domain (where `DomainIntel < 0.30` and zero structural signals fired) from escalating alone into hard `"phishing"`.

### 3a. 48-URL adversarial & unseen-domain stress bank (Pi 5 ONNX + `DomainIntelScorer`)

| Category | Label | n | Hybrid `phishing` | Hybrid any flag (`phishing`+`suspicious`) | ONNX-only any flag | Notes |
|---|---|---:|---:|---:|---:|---|
| `benign_indian_gov` (`.gov.in`) | Benign | 4 | **0** | **0** | 0 | Clean |
| `benign_indian_bank` (SBI/HDFC/ICICI/NPCI) | Benign | 4 | **0** | **0** | 3 | Fixed from `3/4` ONNX FPs → `0/4` |
| `benign_cdn_ecom` (Shopify CDN, Flipkart) | Benign | 2 | **0** | **0** | 0 | Clean |
| `benign_modern_tld` (`.app`/`.dev`/`.co`/`.com`) | Benign | 5 | **0** | 1 | 1 | `anthropic.com` capped at `suspicious` |
| `benign_indian_fintech` (IRCTC/Zerodha/Razorpay/Groww) | Benign | 4 | 1 | 3 | 4 | `zerodha.com` capped at `suspicious`; `razorpay.com` still `phishing` (`pay` substring without Tranco list) |
| **All 10 adversarial attack categories** | **Attack** | **29** | **27 / 29** | **29 / 29 (`100%`)** | 29 / 29 | Path padding (3/3), subdomain stuffing (3/3), IDN (2/2), Cyrillic homograph (2/2), leetspeak (5/5), free hosting (3/3), `@`/IP (2/2), unseen Indian `.com`/`.in` scams (5/5), case obfuscation (2/2), shorteners (2/2) |

### 3b. OOD regression check with `onnx_logit_clip=4.0` (Pi 5, `4,000`-row seeded samples)
- **`url_domain` OOD test (4k):** @ `0.75` (`phishing`) → **Recall `93.05%`**, **FPR `1.37%`**, **Precision `99.50%`**, **F1 `0.9617`** (@ `0.6308` `suspicious`: Rec `95.03%`, FPR `4.60%`, F1 `0.9667`).
- **PhreshPhish `official.test` (4k):** @ `0.75` (`phishing`) → **Recall `87.09%`**, **FPR `0.84%`**, **Precision `98.91%`**, **F1 `0.9262`** (@ `0.6308` `suspicious`: Rec `90.41%`, FPR `1.41%`, F1 `0.9417`).

---

## 4. P4 — Real-World Indian SMS Stratification & Field-Validation Readiness

### 4a. Indian-context & multilingual strata inside `campaign_test` (`22,145` rows, Pi 5 ONNX)

| Stratum | n (`pos` / `neg`) | `quiet` (V1) | **`balanced_v3`** |
|---|---:|---|---|
| **Domain: `indian_bank_upi`** | 598 (`589` / `9`) | Rec 32.94%, FPR 0.00%, F1 0.4955 | **Rec 94.06%**, **FPR 0.00%**, **F1 0.9694** |
| **Domain: `indian_kyc_gov`** | 228 (`227` / `1`) | Rec 69.60%, FPR 0.00%, F1 0.8208 | **Rec 97.36%**, **FPR 0.00%**, **F1 0.9866** |
| **Domain: `indian_currency_tel`** | 2,014 (`1,969` / `45`) | Rec 14.78%, FPR 11.11%, F1 0.2570 | **Rec 91.47%**, FPR 11.11%, **F1 0.9542** |
| **Language: `en`** | 16,907 (`9,134` / `7,773`) | Rec 18.36%, FPR 0.71%, F1 0.3087 | **Rec 87.99%**, **FPR 0.98%**, **F1 0.9320** |
| **Language: `hinglish`** | 398 (`320` / `78`) | Rec 27.81%, FPR 2.56%, F1 0.4331 | **Rec 88.44%**, **FPR 2.56%**, **F1 0.9355** |
| **Language: `hi_deva` (Devanagari Hindi)** | 1,279 (`37` / `1,242`) | Rec 43.24%, FPR 0.56%, F1 0.5333 | **Rec 70.27%**, **FPR 0.64%**, **F1 0.7324** |
| **Source: `scamshield`** | 15,977 (`6,123` / `9,854`) | Rec 17.30%, FPR 0.67%, F1 0.2922 | **Rec 87.07%**, **FPR 0.91%**, **F1 0.9236** |
| **Source: `imc25_fishing_for_smishing`** | 4,947 (`4,947` / `0`) | Rec 14.68%, F1 0.2559 | **Rec 91.19%**, **F1 0.9539** |
| **Source: `mishra_soni_smsphishing` (India)** | 189 (`68` / `121`) | Rec 33.82%, FPR 0.00%, F1 0.5055 | **Rec 75.00%**, **FPR 0.00%**, **F1 0.8571** |
| **Source: `uci_sms_spam` (legacy UK)** | 1,032 (`55` / `977`) | Rec 9.09%, FPR 0.10%, F1 0.1639 | Rec 43.64%, FPR 0.10%, F1 0.6000 |

### 4b. What remains before live production claims (blinded field validation)
1. **Real carrier Indian SMS inbox validation:** Consented live inbox exports (covering Indian TRAI DLT alphanumeric headers `VM-SBIINB`, `AD-HDFCBK-S`, regional languages beyond Devanagari/Hinglish, and temporal drift) must be evaluated via:
   ```bash
   python3 scripts/eval_validation_suite.py --root ~/rakshak-bench --suite field \
       --field-sms /path/to/consented_indian_sms.jsonl --out field_sms_report.json
   ```
   The harness automatically redacts phone numbers, UPI/email handles, account tails, and long numeric references (`redact_pii`) before writing any failure samples, tracks DLT alphanumeric sender coverage, and enforces explicit go/no-go gates (`recall >= 85%`, `FPR <= 3%`, `n >= 200`).
2. **Real home/SOHO network capture validation:** Unlabeled or labeled real LAN/router PCAPs (smart TVs, phones, IoT beacons, real CDN traffic, DoH/mDNS) must be evaluated via:
   ```bash
   python3 scripts/eval_validation_suite.py --root ~/rakshak-bench --suite field \
       --field-pcaps /path/to/real_lan_pcaps/ --out field_nids_report.json
   ```
   Until real-network benign session FPR is measured on live router captures, NIDS remains strictly rules-first and validated on synthetic captures only.
