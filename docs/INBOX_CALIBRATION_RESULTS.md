# W1f: encoder threshold calibrated on the consented inbox, results (2026-11-12)

Plan: `docs/INBOX_CALIBRATION_PLAN.md` (pre-registered). Kaggle T4 kernel `maadhavvh/rakshak-sms-inbox-calibration`,
700 s, ok. Models: W1e seeds 0-2 (primary) and the base teacher (reference). No training. τ = smallest logit margin with
≤ 1 % FPR on the train-donor (phones 2+3) legit rows; primary arm = filled text, secondary = raw scrubbed text.
No inbox text or sender is in any output.

## Verdict: KILLED

| Model | Arm | τ (margin) | G1 GroundLM FP / 1 133 | G2 SmishX / 226 | G3 GroundLM positives / 91 | G4 campaign clean F1 | G5 phone-1 legit FPR |
|---|---|---|---|---|---|---|---|
| W1e s0 | filled | 29.14 | 0 | 1 | 0 | 0.086 | 3 / 2 976 |
| W1e s1 | filled | 30.56 | 0 | 6 | 2 | 0.155 | 3 / 2 976 |
| W1e s2 | filled | 24.69 | 0 | 4 | 0 | 0.116 | 6 / 2 976 |
| teacher | filled | 24.01 | 0 | 12 | 1 | 0.172 | 22 / 2 976 |
| (scrubbed arms) | raw | 28.98-35.45 | 0 | 0-1 | 0 | 0.025-0.082 | 4-18 / 2 976 |

## Why

The encoder scores the inbox legit rows (scrubbed, or scrubbed then surrogate-filled) as strongly scam-like: putting
≤ 1 % of them over the line takes a margin of 24-35, above almost every real scam. That matches the earlier finding
that the scrub step (every digit and most names removed, typed placeholders) moves real messages far from anything in
training (`docs/INBOX_FILLED_RESULTS.md`, `docs/SURROGATE_D0_RESULTS.md`). The inbox is unusable as a calibration set
in its exported form.

## Where this leaves the SMS detector

| Threshold source | Outcome |
|---|---|
| Model's own cal split (W1a) | far too low: 679-911 GroundLM FP |
| Held-out same-source licensed legit (W1e) | far too low: 620-773 FP |
| Consented inbox, scrubbed/filled (W1f) | far too high: near-zero recall |
| **Half of GroundLM's legit SMS, tested on the template-disjoint other half** (post-hoc, Amendment 2) | **12-16 / 1 133 at 86-99 % SmishX, 63-88 / 91 GroundLM positives (W1e)** |

The only real, unscrubbed Indian legit SMS available is GroundLM. Cross-fitted calibration on it is the standard way
to use such a set honestly (calibrate on one half, test on the disjoint other). The next step is a **fresh,
pre-registered confirmation** of that protocol on a model never evaluated before: `docs/CROSSFIT_CONFIRM_PLAN.md`.
