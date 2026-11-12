# GroundLM full-paper review (external research, pasted by Maadhav 2026-11-07)

Source: Maadhav's OpenResearch run on Gupta 2026, ACL Anthology 2026.groundlm-1.79 (PDF read in full, tables
checked), with exact counts recomputed from the supplement (Zenodo 10.5281/zenodo.22732001,
`predictions/label_policy_base_predictions.csv`, `xlmr_operating_points.json`) and training code
(Zenodo 10.5281/zenodo.20480151). The review's own artifact files (indian-transactional-sms-evidence-review.md,
gupta-table4.png, gupta-exact-counts.csv, gupta-prompt-counts.csv) were NOT copied here; they live with the
OpenResearch run.

## The paper

- Model: FacebookAI/xlm-roberta-base (270M), binary fine-tune. Default threshold tau = 0.50 (checkpoint
  picked by validation F1; no FPR-constrained threshold on validation; the threshold sweep is on test).
- Training mix: CloveAI English-India, vinit9638 Hindi, IIITD Hinglish, flatmars mixed + 30,000 global-English
  (SMS and email). Fingerprint dedup; 80/10/10 language-and-label stratified random split, seed 42; AdamW
  lr 2e-5, wd 0.01, batch 16, 3 epochs, class-weighted CE. Post-dedup train/val counts not stated.
- Labels: only Indian source positives are re-labelled. GPT-4o + Gemini 2.5 Flash both "fraud" -> positive;
  both "marketing/non-fraud" -> safe; else dropped; 18 native-speaker prefix overrides (31 rows).
  7,234 positives -> 5,896 fraud / 794 non-fraud / 544 unclear. Source negatives untouched.
- Test: 1,393 = 99 policy-positive + 1,294 policy-negative (1,222 untouched source negatives + 72 converted
  positives).

| System (minimal prompt) | FP / 1,294 | FPR | TP / 99 | Recall |
|---|---:|---:|---:|---:|
| Policy-aligned XLM-R | 5 | 0.39% (printed 0.4%) | 97 | 0.980 |
| Gemini 2.5 Flash | 157 | 12.1% | 99 | 1.000 |
| GPT-4o | 189 | 14.6% | 99 | 1.000 |
| Claude Haiku 4.5 | 247 | 19.1% | 98 | 0.990 |
| English-only XLM-R (same backbone, no Indian data) | 284 | 21.9% (printed 22.0%) | 96 | 0.970 |
| Llama-3.3-70B | 302 | 23.3% | 99 | 1.000 |
| GPT-4o-mini | 440 | 34.0% | 99 | 1.000 |
| Qwen-2.5-7B | 496 | 38.3% | 99 | 1.000 |

- Threshold sweep (XLM-R): tau .30 -> 7 FP / 99 TP; .40 -> 5 / 99; .50 -> 5 / 97; .60-.70 -> 3 / 97;
  .80 -> 2 / 95; .90 -> 2 / 92. Bank, UPI, EMI, marketing FPR zero throughout.
- English-only control: Bank/UPI FPR 57.1% / 48.6% vs 0% / 0% for the policy-aligned model.
- Untouched negatives only: 1/1,222 = 0.08% FPR (APIs 10.9-35.2%).
- Structural dedup: 1,393 rows -> 1,219 clusters; bank/UPI still 0/104 and 0/124 (does NOT remove
  train/test template overlap).
- Human check on majority-safe subsets: bank 1/33, UPI 0/8 false flags (GPT-4o-mini 21/33, 7/8).
- Per-template buckets are message-body regex families, not DLT template IDs. OTP n=1, Aadhaar/KYC n=1,
  reward n=2 negatives: too small for claims. Telecom: not evaluated separately. UPI negatives are synthetic.
- NOT evaluated: sender ID, sender reputation, DLT suffix, registered template ID; no template-held-out,
  institution-held-out, temporal, allowlist or minimal-pair ablation.
- Recall is policy-label recall (LLM judges helped build positives), not independent gold.

## Related work (author-reported, not rerun)

- SMS-Shield 2026 (HF ganmoor-ai-labs/sms-shield): sender+body, sender-flip counterfactual pairs; 488/488
  safe kept safe, dangerous recall 859/863 on a 1,390 synthetic eval set; baseline comparison not matched.
- Silver Guard 2026: MobileBERT header+body; DLT-shaped senders auto-labelled safe; claims FPR ~40% -> <1%,
  no denominator.
- SMS_Spam_Model 2026: hard negatives (OTP/payment/delivery), grouped split by source message; 0/60 FP,
  57/60 TP on a 60+60 real-India set (also used for model selection).
- SPAM//SCAN 2026: legit-alert rules + official-domain allowlist overridden by fraud signals; probe
  correctness 127/167 -> 159/167 (not FPR; rules built on the probes).
- SMISDroid 2026: MobileBERT + app rules; 1/585 = 0.17% FPR, 760/763 recall (random stratified split).
- COPS 2024: FPR 1.5%, recall 0.962. Johnny 2026: minority-vote Top3 FPR 2.86% / recall 99.6% on 5k+5k.
- DLT suffixes (-P promotional/service-explicit, -S service-implicit, -T transactional, -G government)
  appended from 6 May 2025 (MSG91 docs). No published measured ablation of suffix features or of template
  allowlists on independent Indian transactional traffic was found.

## Licensed corpora with Indian transactional messages

| Resource | Size, licence | Note |
|---|---|---|
| CloveAI/india-spam-sms (HF) | 20,010, MIT | bank/UPI/telecom/utility style; synthetic institutional subsets |
| Indian Payment SMS Corpus (skyaara) | 172 (58 seeds + 114 variants), CC BY 4.0 | synthetic/redacted; `variant_of` for grouped splits |
| FinEE (HF Ranjit0034) | 152,519, Apache-2.0 | 2,419 real anonymised ICICI + synthetic; extraction task |
| Indian Telecom SMS Spam Collection (junioralive) | 2,267 (1,522 ham / 745 spam), MIT | mixed corpus |
| SMS-Shield eval set | 1,390, Apache-2.0 (declared) | synthetic sender+body |
| Ridham115 audited synthetic Indian scam SMS | 1,580 (797 scam / 783 safe), CC BY 4.0 | AI-audited, partial human check |
| gandharvbakshi OTP/intent/phishing | 73,470, MIT-tagged, gated | check commercial terms |

IIITD is request-access without an explicit redistribution licence; vinit9638 and flatmars licences unstated.

## Implications for Rakshak (orchestrator notes)

1. GroundLM's 0.4% is an in-distribution result: its train and test come from the same Indian sources
   (random stratified split, template overlap). Its own English-only control (no Indian exposure) gets 21.9%
   global and 57% / 49% bank/UPI FPR. Our student's 41.6% is the "no exposure" regime, so the main lever is
   Indian transactional training exposure (data), not a bigger model.
2. A fair comparison needs a matched protocol: either retrain our small student on a licensed version of
   their mix and test on the same 1,294 (report 1,133 scoreable + the 161 hash-only gap), or present ours
   as a cross-source result next to their English-only control.
3. Their predictions and code are public (Zenodo 22732001, 20480151), so per-row comparison on the shared
   test rows is possible.
4. The open gaps the paper leaves (no sender/DLT feature, no template-held-out evaluation, no telecom bucket,
   synthetic UPI negatives) are where Rakshak can add something measurable, if our data has sender IDs.
5. Open question for Maadhav: is "Sachin Gupta" the teammate Sachin (sachin0496)? Affects teacher choice
   and conflict-of-interest wording.
