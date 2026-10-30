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
