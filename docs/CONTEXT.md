# Rakshak-Edge — project context

Bharat AI-SoC Challenge 2026–27, **Problem Statement 3** (on-device cyber safety).
Source: Maadhav's deep-research report (2026-10-29). Claims below marked *unverified* still need a primary source — see `docs/RESEARCH_NOTES.md`.

## Hard constraints (from the challenge page)
- Everything runs locally on Arm: Android phone, Raspberry Pi 5 (CPU, or AI HAT+/HAT+ 2), or other Arm Linux.
- **Llama-family model + ExecuTorch are mandatory** for the advisory/reasoning part. Hailo optional; if used, keep its pipeline separate from ExecuTorch/Llama and document it.
- Deliverables: GitHub repo, source, app, demo video, technical report. Team 1–3, one PS only.
- Dates: Phase 1 due **23 Nov 2026**; shortlist ~mid-Dec; Phase 2 due **5 Feb 2027**; finals 15–16 Feb 2027.
- No public numeric rubric. Mentor video: detection quality, on-device performance, user effectiveness.

## The bet
SMS scam detection is the primary use case; an embedded URL is one evidence source, not a second PS.
Thesis: a tiny multi-expert detector decides; calibrated uncertainty lets it abstain; Llama 3.2 1B (INT4, ExecuTorch/XNNPACK) only **verbalises detector-produced evidence IDs**, and a deterministic verifier rejects anything else ("proof-carrying explanation").

Do **not** claim novelty for any single piece (char models, conformal, fusion, LLM explanations). The claim is the architecture + evaluation regime:
> offline, on-device; heterogeneous experts; calibrated abstention; Llama verbalises only verifier-approved evidence/actions; evaluated under campaign shift, code-mixed text and realistic obfuscation, not random-split accuracy.

## Pipeline
1. Security canonicaliser — NFKC, zero-width strip, homoglyph skeleton, URL extraction, spacing/leetspeak folding.
2. Experts — message (char TF-IDF+LR → fastText-style → char-CNN; transformer only if it wins on hard splits after quantisation), URL (lexical/structural features + char model, no network), rules (OTP/PIN/CVV ask, UPI collect, KYC urgency, APK install, brand↔domain mismatch).
3. Fusion — LR / small GBM over expert scores + evidence flags; calibrate on held-out campaigns; three states: benign / uncertain / scam.
4. Evidence package (E1..En + allowed action codes) → Llama → constrained JSON `{summary, reasons:[E*], action}` → verifier → else deterministic template.
5. Llama never sees raw attacker text by default (prompt-injection defence).

## Evaluation (must-have)
Splits: random (legacy only), **campaign-disjoint**, temporal, English→Hinglish/Hindi, clean→obfuscated.
Ablations: no URL, no message, rules only, classifier only, no canonicaliser, no calibration, free-form vs constrained Llama.
Metrics: P/R/F1, PR-AUC, FPR, recall@low-FPR, ECE/Brier, coverage-vs-error, per-family. Explanations: evidence precision, unsupported-claim rate, action correctness.
Device: model size, peak RSS, p50/p95/p99 latency, TTFT, prefill/decode tok/s, thermals; airplane mode; measured energy only.

Four headline figures: unseen-campaign detection · robustness after obfuscation · FPR/calibration · explanation faithfulness.

## Data (to verify)
- SMS: UCI SMS Spam (5,574, CC BY 4.0, baseline only); IEEE S&P 2024 "On SMS Phishing Tactics and Infrastructure" (67,991 msgs, campaigns); HF Indian/Hinglish sets (`scam-ham-india`, `scamshield-dataset`, …) — augmentation only, audit provenance.
- URL: PhreshPhish (main), LegitPhish (independent val), PhishTank/OpenPhish feeds for post-freeze temporal hold-out.
- Synthetic data = training/robustness only, never the headline test.

## Targets
Android phone first (share-to-scan / paste UX; Play restricts READ_SMS). Pi 5 CPU as reproducibility target. No Hailo for Phase 1.
Fallback submission: char n-gram SMS LR + lexical URL model + calibrated LR fusion + Llama evidence explainer.

## Working rules
- Commit very rarely (batch large milestones).
- Opus orchestrates; Haiku subagents do the work.
- Check `sensors | grep Tctl` before local compute; ≥85 °C → stop heavy work. Never poll.

## Ambition: SOTA (Maadhav, 2026-10-30)
Goal is state of the art and the best, not just a working demo.
- Every reported benchmark is compared against its best published number under the *same* protocol (docs/SOTA_TARGETS.md), plus our harder campaign/temporal/adversarial splits where most published numbers don't exist.
- Strategy: strongest teacher (fine-tuned multilingual/Indic encoders, ensembles, HPO) → distill into tiny on-device students; report both, ship the student.
- All training, HPO, distillation and benchmark evals run on Kaggle kernels via the kaggle CLI (`kaggle/` launcher, ntfy completion event, no polling). Laptop = code, git, unit tests.
- Cross-device benchmarking: Qualcomm AI Hub (cloud-hosted real Snapdragon devices) for the specialist models — CPU-only profiles are the headline, NPU numbers a bonus. Final submission numbers still come from our own physical device. Plan: docs/QAI_HUB_PLAN.md.

## Decisions from deep-research report 2 (2026-10-30; claims being verified in docs/RESEARCH_NOTES.md)
- Rename "proof-carrying explanation" → **Evidence-Locked Advisory (ELA)**; no "proof" or blanket "SOTA" wording unless reproduced under identical protocol.
- Primary safety objective: **low FPR on legitimate Indian transactional SMS** (bank/UPI/OTP/delivery). External test: GroundLM 2026 Indian Financial SMS benchmark (target FPR ≤1%).
- **Spam ≠ scam**: re-adjudicate labels into benign_transactional / benign_personal / promotion / scam / unclear; fraud claims use scam vs not-scam, promotions are not fraud.
- Four frozen test partitions: T1 source-disjoint, T2 campaign-disjoint, T3 temporal (only where dates exist), T4 paired adversarial (only delivery-surviving attacks; bidi excluded).
- **Dual-view student**: raw/byte view + canonical view; canonicalisation is non-destructive and emits obfuscation evidence.
- Teacher bake-off adds mmBERT; mandatory baselines fastText/char-CNN/rules. Student <10 MB distilled on fraud + teacher KL + intent + evidence + attack-consistency.
- Outputs SAFE / UNCERTAIN / SCAM with risk–coverage curves; no conformal "guarantee under drift" claims.
- Name the Llama quantisation exactly (8da4w / SpinQuant), never just "INT4".
- Freeze benchmark before chasing models; predeclared go/kill thresholds.

## Phase 2 option: Android AppFunctions (noted 2026-10-30)
Expose `scanMessage(text) -> {verdict, evidence_ids, action_id}` as an AppFunction (androidx.appfunctions, Android 16 / API 36+, experimental) so on-device agents can call the offline detector. Not for Phase 1: experimental API, Gemini caller access is private preview, caller agent may not be offline. Demo via `adb shell cmd app_function` or a tiny caller app. Source: developer.android.com/ai/appfunctions.

## Scope addition: network-intrusion module (Maadhav, 2026-10-30)
PS3 lists network-traffic monitoring and malware behaviour. Added a flow-based home/SME/IoT network IDS as a coverage module (SMS + URL stay the headline research claim):
- Runs on Raspberry Pi 5 (Scapy/libpcap flow capture → features → small detector); phone stays the SMS/URL device; Llama via ExecuTorch on both.
- Same Evidence-Locked Advisory: network evidence IDs (e.g. PORT_SCAN, SYN_FLOOD, BRUTE_FORCE, BEACONING, DNS_TUNNEL, NEW_DEVICE_RARE_PORT) + remediation action codes.
- Evaluation rigour as for SMS: no random splits as headline; time-split, unseen-attack-family and cross-dataset (e.g. train CIC-IDS2017 → test CSE-CIC-IDS2018) tests; corrected dataset versions where label errors are documented.
- Plan and targets: docs/NETWORK_IDS_PLAN.md.
