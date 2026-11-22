# Rakshak-Edge: where things stand (2026-11-15)

Fast-inference work (verdict-first, explanation table, Pi numbers): see docs/PERF_FAST_INFERENCE.md.

Pulled from the local repo, GitHub, Kaggle, HF and agent-wire. Nothing was changed while gathering this.
**Phase 1 is due 2026-11-23 (8 days).** Phase 2 is 2027-02-05; finals are 2027-02-15/16.

## 1. What ships (frozen in docs/SYSTEM_V1.md, 2026-11-12)
Offline scam SMS + URL detector on a Pi 5 (4 GB, no cooler, USB PSU), plus a rules-first NIDS path (`/analyze_flows`).
- **The student detector decides** (SMS student v0, tau 0.35127). Rules add evidence. Quiet mode is the default; `--detector-only-mode verify` is available too.
- **Llama 3.2 1B** (SpinQuant INT4, ExecuTorch 1.5.1) only writes sentences for verifier-approved evidence IDs. Guards: G1 (no LLM at LOW) and G2 (attribution). A template fallback always produces output.
- **URL evidence on /analyze is verify-only** (gate U2 failed, 5/1,133 vs a limit of 4). `/analyze_url` works in every mode.
- **Model work for Phase 1 has stopped.** The 11-12 SOTA push (W1a/b/e/f/g/h, W2, INT8 encoder) was killed on pre-registered gates (docs/ENSEMBLE_CONFIRM_RESULTS.md).

## 2. Headline numbers
| Area | Number | Source |
|---|---|---|
| SMS campaign-disjoint | F1 0.9813 [0.9794, 0.9830], PR-AUC 0.9977 | docs/BASELINE.md |
| SMS temporal | precision 0.276 (77 scams) | docs/BASELINE.md |
| GroundLM FPR | 41.57% raw; **0.18% visible in quiet mode** (2/1,133) | BASELINE, SYSTEM_V1 §2b |
| SmishX recall | 69.47% raw; 23.45% visible in quiet mode | same |
| URL PhreshPhish test | FPR 0.92%, recall 92.0%; Pi parity 300/300 | docs/URL_SERVICE_RESULTS.md |
| URL url_domain | FPR 69.4% (not gated, which is why URL quiet mode is off) | same |
| NIDS rules (synthetic pcaps) | 11/11 attacks detected, 1/6 benign alerting | docs/NIDS_EDGE.md |
| NIDS cross-dataset | F1 0.297 / 0.332 / 0.718 (near chance on 2 pairs) | docs/NIDS_EDGE.md |
| Explainer raw pass | 26–27/33; 12.9% unterminated; 14.64 tok/s | docs/BASELINE_PI.md, EDGE_PIPELINE.md |
| Pi latency (direct path) | p50 989 ms, p95 1,887 ms; soak p95 2,152 ms; RSS 1.84 GB | SYSTEM_V1 §2c |
| Pi energy / thermal | 3.52 J/req (PMIC estimate, uncalibrated); 70–73 °C | EDGE_PIPELINE, SYSTEM_V1 §4 |

Caveats in SYSTEM_V1 §4: GroundLM and SmishX overlap the training text by 5–6%, and the GroundLM rules were designed on the same rows.
Still unresolved: campaign F1 is reported as both 0.9813 and 0.9749; the explainer pass is reported as both 26/33 and 27/33; the Pi .pte hash was not re-checked.

## 3. Phase 1 deliverables (repo, source, app, demo video, technical report)
- **Done:** repo, source, Pi edge service and dashboard, offline proof (netns, lo only), measured numbers, system card.
- **Missing:**
  1. **Technical report.** docs/SYSTEM_V1.md is its source.
  2. **Demo video.**
  3. **Android app.** Not built; the ExecuTorch AAR 1.5.1 path is unconfirmed.
  4. **Final Pi numbers with the Active Cooler and a 27 W PSU.**
  5. Repo readability for judges, and repo visibility.
  6. Native-speaker Hinglish review.
  7. U5 re-run under `thermal_run --kill 65` (the earlier run breached the 65 °C start rule).
  8. Decision on the HIGH-tier wording ("is a scam").

## 4. Repo / git state
- Local HEAD = origin/main = `90809f2` (pushed 2026-11-12). Only branch: `main`. No extra worktrees.
- **The URL path, fusion lazy imports, encoder bench, NIDS holdout scripts, Kaggle jobs and SOTA-push docs (including SYSTEM_V1.md) are not committed.** That is 26 modified files (+1,789/−92) and about 50 untracked files.
- Stashes: `stash@{0}` (STATUS backup), `stash@{1}` autostash (22 files, +1,718, not inspected), and `stash@{2}` (ci/ela-qlora STATUS).
- docs/STATUS.md is 1,809 lines and mixes in an old ELA/QLoRA log. Needs a cleanup.

## 5. GitHub
- Issue #1 is open: "Handoff to @Sachin0496" (body dated 11-01, partly stale). It lists 7 tasks: GroundLM FPR, rank-arm audit + ELA QLoRA, ScamShield confound, spam-vs-scam relabel, Pi benchmarks, Android app, report + demo.
- PRs #2–#14 were all merged on 11-06. No PRs are open. The issue body still says #2 is awaiting review, which is out of date.

## 6. Kaggle
- About 37 `maadhavvh/rakshak-*` kernels exist. The last run was `rakshak-sms-crossfit-confirm-w1h` on 11-12, and the 5 most recent all show COMPLETE. Nothing has run since.
- Datasets: `rakshak-data` (11-07, 161 MB), `rakshak-inbox`, `rakshak-teacher-logits` (10-31), plus dated snapshots (url-e0, multilingual-m3, template-cal, clean-kd, fusion-v1, ela-synthetic).
- Outputs are in `kaggle/out/` (about 50 dirs). `latest_launch.json` = sms_crossfit_confirm/w1h.
- The NIDS real-pcaps (W3) run was "still running" on 11-12 and **its result is not in the repo.**

## 7. Hugging Face: blocked
`highestintheroom/rakshak-data` and `highestintheroom/rakshak-models` both return RepositoryNotFound, while whoami shows org membership. The current token most likely has no org-repo scope. **Swap in an org-scoped read/write token** before anything that needs HF, including the model-card cleanup.

## 8. Other
- There are no rakshak threads on agent-wire.
- Inbox-data stop rule: a second donor (Sachin) is needed (docs/TEACHER_FINETUNE_RESULTS.md).
- Rules still in force:
  - Kaggle-first compute; launching without asking is OK.
  - No Modal without an OK.
  - Commit only when told, squashed.
  - git identity is vhmaadhav only.
  - Never poll.
  - Check CPU temperature before local compute.
