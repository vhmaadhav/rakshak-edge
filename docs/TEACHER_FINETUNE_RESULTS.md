# Teacher fine-tune T0 / T1: results (fetched 2026-11-12)

Plan: `docs/TEACHER_FINETUNE_PLAN.md` (pre-registered 2026-11-11). Kernels `maadhavvh/rakshak-sms-teacher-finetune-t0`
and `-t1`, version 1, status `ok`. They completed on 2026-11-11 and were fetched on 2026-11-12, result files only
(`results.json`, `metrics.json`, `report.md`, `run.log`, `thresholds.json` in `kaggle/out/sms_teacher_finetune-{t0,t1}/output/`).
The 1.26 GB checkpoints stay in the kernel outputs on Kaggle and were not downloaded.

Model `jhu-clsp/mmBERT-base`, seed 0, one seed per arm (as the plan sets out for the provisional step).

| Arm | Training rows | Best epoch | τ (rule, before test) | G1 GroundLM FP / 1 133 (≤ 11) | G2 SmishX TP / 226 (≥ 157) | G3 campaign F1, filled (≥ 0.9713) | G5 test-donor legit FPR (≤ 1 %) | Provisional pass |
|---|---|---|---|---|---|---|---|---|
| T0 (campaign only) | 76 591 | 1 | 0.9999996 | 0 ✔ | 4 ✘ | 0.182 ✘ | 0.07 % ✔ | no |
| T1 (+ inbox train donor, filled) | 78 627 (2 036 inbox) | 0 | 0.9974 | 5 ✔ | 57 ✘ | 0.908 ✘ | 1.14 % ✘ | no |

Hypotheses: T0 is reported only. T1: **H7 false** (G5 1.14 % > 1 %), **H8 false**, **H9 false**.
**Stop rule 3 fires:** the inbox negatives are not learnable from one donor even by the encoder. Report and stop;
more donors is the only remaining route.

Ranking measures (AUC, all rows):

| Measure | T0 | T1 |
|---|---|---|
| SmishX label 1 vs GroundLM legit (A1) | 0.818 | 0.945 |
| SmishX label 1 vs test-donor legit, filled (A2) | 0.288 | 0.823 |
| SmishX label 1 vs SmishX ham | 0.922 | 0.958 |
| Campaign test, filled | 0.9975 | 0.9973 |

Reading (not a claim): one donor's inbox raises the teacher's ranking against that donor's held-out phone from 0.29
to 0.82 AUC. But the frozen threshold rule (≤ 1 % FPR on the real legit validation set) still puts τ near 1, so
recall collapses, as it did in every student arm. The threshold rule and the single-donor data are binding together.
The next step needs a second donor (Sachin), as the plan says. Fp32 ONNX export of the earlier teacher checkpoint
works (`kaggle/out/sms_teacher_export/`), but INT8 dynamic quantisation is not usable (argmax agreement 0.906,
Pearson 0.912 against fp32).
