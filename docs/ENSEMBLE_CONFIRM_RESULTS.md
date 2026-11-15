# W1h: 3-seed ensemble under cross-fitted GroundLM calibration, results (2026-11-12)

Plan: `docs/ENSEMBLE_CONFIRM_PLAN.md` (pre-registered after W1g, before seeds 4-6 existed). Kaggle T4 kernel
`maadhavvh/rakshak-sms-crossfit-confirm-w1h`, 4 934 s, ok. Score = mean logit margin of seeds 4, 5, 6 (W1e recipe).
τ_0 = 14.73, τ_1 = 21.38 (margins).

## Verdict: KILLED (C1, C3, C4 fail). Per the plan, SMS model work for Phase 1 stops here.

| Gate | Value | Limit | Pass |
|---|---|---|---|
| C1 pooled held-out GroundLM FP | 22 / 1 133 | ≤ 11 | ✘ |
| C2 SmishX smishing TP | τ_0: 210, τ_1: 170 | ≥ 157 at both | ✔ |
| C3 GroundLM positives recall | τ_0: 0.780, τ_1: 0.703 | ≥ 0.90 | ✘ |
| C4 campaign clean F1 | τ_0: 0.846, τ_1: 0.698 | ≥ 0.9713 | ✘ |

Per seed (not gated): held-out FP 77 / 15 / 10; SmishX 151-66 / 214-163 / 161-156; GroundLM positives 64-59 / 89-60 / 63-63
of 91; campaign F1 0.49-0.87.

## What the SMS encoder line established (W1a-W1h)

1. A pretrained multilingual encoder ranks Indian legit SMS against real scams far better than the byte student (AUC
   0.990-0.998 vs 0.71), with or without licensed same-family exposure. This holds on every seed.
2. A ≤ 1 % Indian-legit FPR operating point lives in the extreme tail of the score. Where that tail falls moves a lot
   between seeds and between calibration sources (held-out FP 6-77 / 1 133 at the same rule), and recall at the point
   moves with it (SmishX 66-224 / 226).
3. No calibration source we hold places it reliably: the model's own cal split, same-source licensed legit, and the
   scrubbed inbox all fail, and cross-fitted GroundLM calibration passes the false-positive gate on one fresh model
   (W1g) but not on a 3-seed ensemble (W1h), and never with recall ≥ 0.90 on GroundLM's positives.
4. Deployment cost: the mmBERT-base fp16 export (1.01 GB, parity pass) cannot complete a duty-cycled benchmark on the
   cooler-less Pi; mmBERT-small (p95 115 ms) ranks noticeably worse.

The shipped SMS detector stays v0 + rules + quiet mode (`docs/SYSTEM_V1.md`). The encoder results go in the report as
measured negative results, with the ranking finding stated plainly.
