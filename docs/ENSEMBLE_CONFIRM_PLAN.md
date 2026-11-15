# W1h: 3-seed ensemble under cross-fitted GroundLM calibration (pre-registration)

Written 2026-11-12, after W1g failed (`docs/CROSSFIT_CONFIRM_RESULTS.md`) and before any seed 4-6 model exists. Approved
by Claude under Maadhav's delegation (2026-11-11); not reviewed line by line by him.

Hypothesis: averaging seeds stabilises the extreme-tail ranking that the ≤ 1 % operating point depends on.

- Models: the W1e recipe (identical to W1g), **seeds 4, 5, 6**, none seen before. Score = **mean logit margin** of the
  three. No seed is dropped or reweighted after scoring.
- Protocol, gates C1-C4, labelling and reporting: **identical to `docs/CROSSFIT_CONFIRM_PLAN.md`**, applied to the
  ensemble score. Per-seed results are also reported (not gated).
- Decision: pass → the ensemble is the SMS detector candidate (export of three base encoders: Pi feasibility needs the
  cooler, see `docs/ENCODER_PI_RESULTS.md`). Fail → SMS model work for Phase 1 stops; v0 + rules + quiet mode ships.
- This is the last SMS-model confirmation run before the Phase 1 deliverables. No further variants are tried on these gates.
