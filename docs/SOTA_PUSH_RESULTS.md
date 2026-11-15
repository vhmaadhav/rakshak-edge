# SOTA push, 2026-11-12: what was tried and what it established

Plan: `docs/SOTA_PUSH_PLAN.md`. Every arm was pre-registered (gates and threshold rules written before any test
scoring) and run on Kaggle (free tier) or on the Pi under `thermal_run.py`. No Modal. Nothing committed.

## Outcome table

| Workstream | What | Result | Doc |
|---|---|---|---|
| W1a | mmBERT-base teacher, full suite, cal-split threshold | KILLED (679-725 GroundLM FP) | `TEACHER_SUITE_RESULTS.md` |
| W1a (post-hoc) | same model, threshold calibrated on half of GroundLM legit | 6 / 1 133 FP, 86 % SmishX, 64 % GroundLM positives (not a gated result) | same |
| W1b | mmBERT-small folds (3 seeds) + suite (seed 0) | KILLED; weaker ranking than base | same |
| W1c | Encoder export | fp16 storage passes parity (small 479 MB, base 1.01 GB); dynamic INT8 broken (0.171 / 0.829) | `ENCODER_PI_RESULTS.md` |
| W1d | Pi benchmark (duty-cycled, no cooler) | small fp16 p95 **115 ms** ✔; base cannot finish (heat) | same |
| W1e | Encoder + licensed CloveAI/FinEE exposure | KILLED (same-source calibration far too low); ranking up (GroundLM positives AUC 0.944 → 0.99) | `ENCODER_EXPOSURE_RESULTS.md` |
| W1f | Threshold on consented inbox legit | KILLED (scrubbed inbox looks scam-like; near-zero recall) | `INBOX_CALIBRATION_RESULTS.md` |
| W1g | Cross-fitted GroundLM calibration, fresh seed | KILLED (FP 10 / 1 133 ✔; recall, campaign ✘) | `CROSSFIT_CONFIRM_RESULTS.md` |
| W1h | Same protocol, 3-seed ensemble | KILLED (FP 22 / 1 133) → SMS model work stops for Phase 1 | `ENSEMBLE_CONFIRM_RESULTS.md` |
| W2 | URL student on PhreshPhish + url_domain | KILLED (PP AP at 1 % ✘; worse on domain-novel URLs) | `URL_GENERAL_RESULTS.md` |
| URL quiet | Corroborated URL evidence in quiet mode | KILLED on U2 (5 vs ≤ 4); SmishX visible 53 → 143 | `URL_SERVICE_RESULTS.md` |
| W3 | NIDS rules on real CIC-IDS2017 + DNS-tunnel captures | running (Kaggle CPU) | `NIDS_REAL_PLAN.md` |
| Explainer | Prompt injection on the shipped INT4 `.pte` (Pi) | 0 / 150 harmful flips, 0 / 300 unsupported; **structural**: the payload never reaches the explainer | `PTE_INJECTION_RESULTS.md` |

## What we can claim (defensible)

1. **System-level Indian legit FPR on device:** quiet mode shows 2 / 1 133 = 0.18 % visible warnings on GroundLM's
   policy negatives (`docs/GROUNDLM_EDGE_FPR.md`), against zero-shot GPT-4o 14.6 % and Gemini 2.5 Flash 12.1 % on the
   same 1 294 negatives (paper Table 2). The cost is SmishX visible recall of 23.5 %; that trade-off is stated with
   the number.
2. **On-device explainer:** Pi 5, no cooler: p50 0.96 s, 3.1 J per request with prefix reuse
   (`docs/EDGE_PIPELINE.md`). Injection in the message body cannot reach the explainer by construction (150/150
   identical prompts). No published on-device scam explainer reports these metrics.
3. **A measured negative finding worth reporting:** pretrained multilingual encoders rank Indian legit SMS against real
   scams almost perfectly (AUC 0.99), but the ≤ 1 % FPR operating point sits in a score tail whose position is not
   stable across seeds or calibration sources. Without a clean, real, licensed Indian legit set, no threshold we
   could place passed confirmation (W1a-W1h). This explains why GroundLM's 0.39 % is an in-distribution result.
4. **NIDS:** rules-first 11 / 11 on the synthetic holdout (Pi = laptop). The real-traffic result is pending (W3).

## What would move the SMS detector further (not doable before 23 Nov)

- A real, unscrubbed, consented Indian legit SMS set from several people, for calibration only (the scrub step is what
  broke W1f).
- The cooler and 27 W supply, to run mmBERT-base on the Pi.
