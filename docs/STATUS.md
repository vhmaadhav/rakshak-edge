# Status — 2026-10-30

## Done
- Repo + initial commit `8005d18` (vhmaadhav), pushed. Everything after it is uncommitted.
- Core package, data pipeline (dedup, campaigns, splits), eval (campaign-clustered bootstrap, paired tests, `rakshak.eval.recompute`), Kaggle launcher with ntfy completion.
- Data rebuilt with better Hinglish lang-ID and Hinglish/Devanagari rules. docs/DATACARD.md numbers are stale; regenerate.
- Processed data on Kaggle as `maadhavvh/rakshak-data` (the older build; re-push with `kaggle/push_data.sh` after the rebuild).
- URL baselines: char-LR F1 0.991, recall 0.851 @ cal FPR ≤0.01% on url_domain (75% phish test, not realistic base rate).
- Research verified: docs/RESEARCH_NOTES.md, docs/SOTA_TARGETS.md (PhreshPhish v2 numbers), docs/CONTEXT.md decisions.
- SmishX (1,200, MIT) downloaded; GroundLM Indian Financial SMS release (Zenodo 20480151) downloading into data/raw.
- HF repos moved to org `highestintheroom` on 2026-10-31 (`rakshak-data`, `rakshak-models`, private). Docs point at the new names; HF access needs an org-scoped token.

## In flight when the session stopped
- Kaggle: `sms_baselines`, `sms_teacher` pilot (XLM-R, 1 epoch). Fetch with `kaggle/fetch.sh <job>`.
- Agents (may have finished or stopped): chunked labelling tool (`src/rakshak/llm/`, `docs/BUDGET.md`), PhreshPhish extract job, student models (`src/rakshak/models/`, `kaggle/jobs/sms_student.py`), Llama ELA eval + ExecuTorch export jobs.

## Overnight 2026-10-30 → resume here
Running on Kaggle (remote; no local listeners are running):
- `sms_teacher`: mmBERT on campaign + source_holdout with `--transfer-texts` (distillation logits).
- `nids_baselines`: T1/T2/T4 + T5 LODO, home-baseline calibration, anomaly/hybrid, ONNX student.
- `sms_teacher-b`: mmBERT on lang_shift + temporal with `--transfer-texts` (launched; fetch with `kaggle/fetch.sh sms_teacher-b`).
On resume: `kaggle/fetch.sh sms_teacher`, `kaggle/fetch.sh nids_baselines` (and `sms_teacher-b`), review, then:
1. Distil SMS students from mmBERT transfer logits (3 seeds) — fixes Hinglish collapse.
2. GroundLM FPR test (target ≤1%), URL CNN 3 seeds, NIDS review (check hybrid MCC, bot_iot caveat).
3. Llama jobs after Maadhav accepts the Llama 3.2 licence on kaggle.com/models/metaresearch/llama-3.2.
Colab free tier reclaimed a detached run after ~45 min; use Colab only for short jobs. No Colab sessions open.
Uncommitted: NIDS module + teacher/Colab changes (next milestone commit).

## Paused 2026-10-30 (laptop off)
- Still running on Kaggle (remote, unaffected): `sms_teacher` bake-off (MuRIL, IndicBERTv2, mDeBERTa, XLM-R, mmBERT), `sms_student`, `sms_baselines`. On return: `kaggle/fetch.sh <job>` once each (no waiters running).
- Not launched (Kaggle 2-GPU limit): `kaggle/launch.sh url_student --gpu t4 --internet --datasets maadhavvh/rakshak-data --job-args="--kd --budget-h 9"`.
- Blocked on Maadhav: accept Llama 3.2 licence at kaggle.com/models/metaresearch/llama-3.2, then launch `llama_ela_eval` and `llama_export` (commands in docs/EXECUTORCH_PLAN.md).
- Done since last note: data re-pushed (PhreshPhish full, SmishX + GroundLM external tests, label_policy/label_scam, clustered CIs); XLM-R pilot F1 0.981 campaign split, canonicaliser cuts attack success 4.1%→0.4%; PhreshPhish URL-only baselines AP 0.436/0.737/0.909 at 0.05/1/5% (paper's GTE uses URL+HTML: 0.605/0.876/0.938, reference only).

## Next
1. Kaggle jobs pass `cluster_ids` (campaign_id / reg_domain) so CIs are campaign-clustered; re-push data.
2. Add SmishX + GroundLM as external test sets; spam-vs-scam label taxonomy.
3. Teacher bake-off (MuRIL, IndicBERTv2, mDeBERTa, XLM-R, mmBERT) once the pilot passes.
4. Student + distillation; Llama ELA eval; ExecuTorch export (accept Llama licence on Kaggle/HF first).

## Rules
Kaggle for all compute; no Modal without Maadhav's explicit per-run OK; cloud-credit labelling parked; commit very rarely as vhmaadhav.
- New --run-name tests (tests/test_kaggle_run_name.py) not yet run: CPU was 92–94 °C from a non-project process.
