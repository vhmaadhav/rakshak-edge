# SOTA ideas for Rakshak-Edge, 2026-11-12 (read-only research)

Scope: build on docs/SOTA_PUSH_PLAN.md, SYSTEM_V1.md, SOTA_TARGETS.md, TEACHER_XSOURCE_RESULTS.md, SMS_RETRAIN_RESULTS.md, INBOX_FILLED_RESULTS.md, TEACHER_FINETUNE_RESULTS.md, URL_SERVICE_RESULTS.md, NIDS_EDGE.md (holdout), and research_notes/Rakshak Edge SOTA benchmarks/*.md. Nothing here is re-stated from those files unless it changes a decision.

Conventions
- VERIFIED = title, authors and arXiv id confirmed with the reference lookup tool in this session (2026-11-12), or read in a prior repo note whose read is cited there.
- SNIPPET = seen only in a search-result snippet or aggregator. Numbers marked SNIPPET are UNVERIFIED; do not cite them as results.
- REPO = our own result file (already measured).
- Lookup tools for references are partly unavailable after a session restart (the hosted research-desk lookup server disconnected). Entries marked VERIFIED were confirmed before the disconnect.

Status of this pass: searches done for all 8 workstreams; lookups done for 7 IDs. Gaps are listed per section and in section 4.

---

## 0. What the repo evidence already says (decision inputs)

- Teacher fold (mmBERT-base, no GroundLM/SmishX in training, seed 0): AUC 0.995 SmishX label-1 vs GroundLM legit; at "keep 70%" recall (88.8% of SmishX label-1) GroundLM FPR is 1 of 1,133 (REPO, TEACHER_XSOURCE_RESULTS.md). Against the inbox donor it is 1,336 of 2,976 (44.9%) at the same operating point (REPO).
- Frozen rule failure: every student arm sets tau near 1.0 because the rule asks for <=1% FPR on a validation set that is harder than the test (REPO, SMS_RETRAIN_RESULTS.md, INBOX_FILLED_RESULTS.md). The teacher ranks well; the threshold rule and the single-donor data bind together (REPO, TEACHER_FINETUNE_RESULTS.md).
- Dynamic INT8 export of mmBERT-base fails: argmax agreement 0.906 vs fp32 (REPO, TEACHER_FINETUNE_RESULTS.md). Per-channel weight-only INT8 is the next candidate (REPO, SOTA_PUSH_PLAN.md W1c).
- URL: tau 0.6308 gives PhreshPhish test FPR 0.92% but url_domain FPR 69.4% (REPO, URL_SERVICE_RESULTS.md).
- NIDS holdout: rules-first 11/11 attack instances with right family; 5 of 6 benign files silent; model-only alerts on 5 of 6 benign (REPO, NIDS_EDGE.md holdout section).
- Explainer: Pi p50 0.96 s, verifier-gated (REPO, SOTA_PUSH_PLAN.md).

The most important implication: the SMS problem is mostly an operating-point and data-diversity problem, not a model-class problem. The teacher already separates GroundLM at ~1 FP per 1,133 when the threshold is set by recall, not by a 1% FPR rule on a harder set.

---

## 1. Top-10 ranked table

Rank = expected impact x feasibility in 10 days on free Kaggle T4/CPU, no paid APIs. Effort in analyst-days (one person). Gate values are proposals to pre-register, not results.

| # | Idea | Workstream | Expected gain and evidence | Effort (days) | Risk | First experiment (Kaggle job sketch and gate) |
|---|---|---|---|---|---|---|
| 1 | Pre-registered two-band operating point: a recall-floor threshold from the fold's own cal split plus a "verify" band (abstain), replacing the <=1% FPR rule. | SMS (eval/threshold) | Teacher already gives 1/1,133 FP at 88.8% kept (REPO). Selective and conformal risk control give a distribution-free route to a threshold with a stated guarantee (Angelopoulos et al., arXiv 2208.02814, VERIFIED; ICLR 2024 per lookup snippet). | 1-2 | Low-medium: the cal split may be unrepresentative; the verify band may hide misses. | Score the existing teacher fold (W1a). Set tau on the fold's cal split for recall >= 0.85 on SmishX label-1. Gate: GroundLM FP <= 11 AND test-donor FPR reported (not gated) AND verify-band share <= 15%. Pre-register before scoring. |
| 2 | mmBERT-small (140M, ModernBERT arch) fold student for the Pi, per-channel weight-only INT8 export with embeddings and head fp32. | SMS (encoder + ARM) | Small encoder fits beside a 1B LLM on a 4 GB board if RSS and latency fit (estimate). mmBERT is ModernBERT-architecture, 1,800+ languages, 22 layers (arXiv 2509.06888, VERIFIED title/authors/year). Repo shows dynamic INT8 fails (0.906 argmax) so the export recipe is the risk. | 3-4 | Medium: small-model ranking may be worse than base; INT8 parity is unproven on ARM for encoders (see section 2.1). | W1b/W1c as planned in SOTA_PUSH_PLAN.md. Gate: argmax agreement >= 0.995 and max abs dp <= 0.02 vs PyTorch on 2,000 rows; GroundLM FP <= 11 at the fold's tau. |
| 3 | Second donor's real inbox (data), then the same two-band rule. | SMS (data) | One donor moves teacher AUC on that donor from 0.29 to 0.82 (REPO, TEACHER_FINETUNE_RESULTS.md). More donors is the only route the repo has not tried. | 1 (if donor is available) | High: depends on people and consent; scrub and review cost. | Run the T1 recipe with donor 2 held out. Gate: test-donor-1 legit FPR <= 1% at the two-band tau. |
| 4 | Just Train Twice (JTT) on the teacher to upweight teacher-error examples, with the group label taken from source. | SMS (shortcut) | JTT closes ~75% of worst-group gap vs group DRO without group labels (Liu et al., arXiv 2107.09044, VERIFIED title/authors/year; the 75% figure is SNIPPET). Source leakage is the known shortcut (REPO: source probe accuracy 0.887 scrubbed / 0.871 filled). | 2 | Medium: no spam or phishing-specific gain found in our search. | Kaggle T4, one seed: stage 1 ERM 3 epochs, stage 2 upweight errors x20. Gate: SmishX recall at teacher tau not lower by >2 points; GroundLM FP not higher. |
| 5 | LEACE (closed-form linear concept erasure) to remove source-identity linearly decodable information from the teacher embedding before the head. | SMS (shortcut) | LEACE gives perfect linear erasure in closed form (Belrose et al., arXiv 2306.03819, VERIFIED title/authors/year; code concept-erasure). Source probe on student was 0.87 accuracy (REPO). | 1 | Low-medium: only linear; a nonlinear probe may still decode source. | Fit LEACE on teacher CLS embeddings with source as concept (train split only), retrain linear head. Gate: source probe accuracy <= 0.55 on held-out sources AND SmishX AUC drop <= 0.01. |
| 6 | URL: domain-grouped training plus a Tranco benign table as a cross-dataset test, judged by registered-domain-grouped FPR. | URL | Feature-based URL detectors are dataset-dependent (Mia et al., arXiv 2411.09813, VERIFIED; lookup gives year 2024, venue arXiv, DOI 10.1145/3704522.3704532). Our 69.4% url_domain FPR is the same failure (REPO). | 2-3 | Medium: our existing W2 plan already covers this; the new part is an independent benign set. | W2 from SOTA_PUSH_PLAN.md plus a second benign source (PhishBench-External, Mendeley 2ctjcnm9kn, 1,000 URLs, Tranco benign; SNIPPET for contents). Gate: url_domain FPR <= 5% at PhreshPhish-test FPR <= 1.5%. |
| 7 | NIDS: run the rules-first pipeline on labelled CIC captures with corrected labels (Engelen 2021 / Liu 2022 corrections), report per-family detection and benign alerts per host-window. | NIDS | Cross-dataset collapse is measured: XGBoost UNSW to CIC F1 0.8116 but CIC to UNSW 0.0730 (Atif and Ali, arXiv 2609.13511, VERIFIED read in prior note). CIC-IDS2017 has documented label errors (Liu et al., IEEE CNS 2022, DOI 10.1109/CNS56114.2022.9947235; SNIPPET). | 2-3 | Medium: label versions must be stated; capture size on laptop limits; Kaggle CPU needed. | W3 from SOTA_PUSH_PLAN.md. Gate: rules-first recall per family reported; no retuning; benign alert rate per host-hour reported. |
| 8 | Adaptive-attack test for the explainer and the gate (search-based attacker, not static strings). | LLM (safety) | "The Attacker Moves Second" bypasses 12 recent defences, most with >90% ASR, under adaptive search and RL (Nasr, Carlini et al., arXiv 2510.09023, VERIFIED title/authors/year; per-defence numbers SNIPPET). Our 0/150 flips needs an adaptive arm. | 2 | Low cost, but a positive result would be a real finding against us. | Kaggle or laptop-free: GCG-style or random-search attacks on the 1B explainer's prompt, 150 seeds. Report Clopper-Pearson upper bound, not a point zero. Gate: upper 95% bound on ASR <= 2%. |
| 9 | Grounding verifier for explanations (small checker, MiniCheck-style). | LLM | MiniCheck (Tang, Laban, Durrett, arXiv 2404.10774, VERIFIED; EMNLP 2024 per repo note) reports GPT-4-level grounding at small cost. | 2 | Medium: our verifier already checks structure; a learned checker needs labelled faithfulness data we do not have. | Score 300 explanations with a small checker; human-audit 100. Gate: agreement with audit >= 0.85. |
| 10 | Encoder-to-byte-student KD (W4). | SMS | Agentic KD reports 94.31% accuracy for a 0.5B student (arXiv 2602.10869, VERIFIED existence; numbers SNIPPET) but validation is teacher-generated synthetic, so weak evidence for our setting. | 3 | High: the student may keep the same source shortcut. | Only after items 1-2 pass. Distil soft labels from the fold teacher on the filled training split. Gate: same as item 1 gates. |

Ranks 1-2 are the highest-value items: the teacher already has the signal; what fails is the threshold rule and the deployable size.

---

## 2. Per-topic evidence

### 2.1 Small multilingual encoders and ARM quantisation (workstream 1)

Model candidates and sizes:
- mmBERT (arXiv 2509.06888, Marone, Weller, Fleshman, Yang, Lawrie, Van Durme, JHU; VERIFIED title/authors/year). ModernBERT architecture, 22 layers, intermediate 1152, 3T tokens, 1,800+ languages (paper abstract and intro, SNIPPET for layer count). HF sizes: jhu-clsp/mmBERT-small about 140M total / 42M non-embedding; mmBERT-base 307M / 110M (HF card snippet; UNVERIFIED on card).
- Multilingual-MiniLM-L12-H384 (arXiv 2002.10957 MiniLM, VERIFIED existence; the released checkpoint card lists 21M transformer and 96M embedding parameters, SNIPPET). multilingual-e5-small is initialised from it (model card, SNIPPET); the e5 training scale is inconsistent across sources (UNVERIFIED).
- IndicBERTv2 (278M, MIT) and MuRIL base (Apache-2.0) from SOTA_TARGETS.md (card reads, prior note). Both are larger than the Pi budget before vocabulary pruning.
- Vocabulary pruning arithmetic (estimate): a 250K-vocab embedding at 384 dims is about 96M params; pruning to 30K tokens gives 11.5M. This is why the 140M mmBERT-small has to be exported with embeddings in fp32 or pruned.

Latency evidence on ARM:
- No primary source found for BERT-class INT8 accuracy on ARM CPUs with ORT QDQ, per-channel weights and LayerNorm excluded (search returned x86 and CNN results only; GAP).
- I-BERT (arXiv 2101.01321, Kim et al., ICML 2021; SNIPPET) notes that low-precision GELU, Softmax and LayerNorm cause large accuracy loss, which is why many quantisers keep them in fp32. This supports excluding those ops from INT8 in our export.
- Intel validated list (x86): BERT-base MRPC and BERT-large SQuAD INT8 lose about 0.6% relative accuracy vs fp32 (SNIPPET). Not ARM; use only as a sanity bound.
- Arm KleidiAI page claims 10x on Cortex-A76 with unspecified model (vendor headline, UNVERIFIED). Do not cite.

Recommendation for the export recipe (based on the above and our repo result):
1. Weight-only per-channel INT8 on the encoder MatMuls (Q/K/V/O and FFN); keep embeddings, LayerNorm, softmax and classifier head in fp32. This matches SOTA_PUSH_PLAN.md W1c.
2. Do not use dynamic activation INT8 on the first run (repo: argmax 0.906).
3. Gate with parity, not accuracy alone: argmax agreement and max abs dp on 2,000 rows (already in W1c).
4. For the Pi, benchmark ORT on aarch64 with the same thermal rule as the rest of the project (thermal_run.py --kill 65).

Distillation evidence:
- MiniLM (arXiv 2002.10957) distils only the last layer's self-attention for multilingual models, because soft-label distillation over a 250K vocabulary is expensive (SNIPPET from the paper summary). Relevant if item 10 is attempted.
- Agentic KD (arXiv 2602.10869, ElZemity, Sylvester, Arief, De Lemos, VERIFIED existence): teacher choice matters a lot; best student 94.31% acc, 96.25% recall on synthetic validation (SNIPPET). Weak as evidence for a real-traffic claim.

Whether small encoders keep cross-domain robustness: not measured in any source found. Our own repo (teacher fold AUC 0.995 on GroundLM, 0.72 on test donor) is the only cross-source evidence and it is for the 24M-row, 110M-param base encoder, not the small one. A small model must be scored on the same two-band protocol before it is trusted.

### 2.2 Shortcut and spurious-feature mitigation (workstream 2)

- JTT (Liu, Haghgoo, Chen, Raghunathan, Koh, Sagawa, Liang, Finn; arXiv 2107.09044; VERIFIED title/authors; ICML 2021 per snippet). Method: train ERM for a few epochs, upweight the examples it misclassifies, retrain. Reported to close ~75% of the worst-group gap of group DRO without group labels, on Waterbirds, CelebA, MultiNLI (SNIPPET for the 75% figure). Hyperparameters need a small labelled validation set. Applies here with source as the group.
- Outlier-aware training follow-up (SNIPPET from search listing): removes unlearnable examples before reweighting, claims competitive or better than JTT. Not read; UNVERIFIED.
- LEACE (Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman; arXiv 2306.03819; VERIFIED). Closed-form least-squares erasure making all linear classifiers fail on the concept with minimal embedding change. Demonstrated on gender bias in BERT embeddings (SNIPPET). Only linear: a nonlinear probe still decodes source. Use with a probe-based check, not as a proof.
- Counterfactual augmentation: Kaushik et al. 2019, Wang and Culotta (arXiv 2012.10040, SNIPPET), Feder et al. (arXiv 2310.12803, NeurIPS 2023, SNIPPET), Bansal and Sharma (arXiv 2305.16863, SNIPPET). None on spam or phishing. For SMS, counterfactual pairs should be generated from our own templates (minimal pairs, already tried in arm B: cost SmishX recall; REPO).
- UNMASK (arXiv 2608.09209, aggregator only): automated discovery and causal verification of spurious shortcuts in text classifiers. Date and venue UNVERIFIED; verify on arXiv before citing.
- Anonymisation tags (<URL>, <NAMED_ENTITY>) taught shortcuts (repo, IMC25 source). Mitigation options: mask at train and test identically (the repo's fill approach: INBOX_FILLED_RESULTS.md showed fill did not restore recall, so masking alone is not enough). Source balancing (SOURCE_BALANCE_RESULTS.md exists in repo; not re-read this pass).

Evidence on spam or phishing: none found in this pass. The Mia et al. result (2411.09813) is the closest: URL features do not transfer across datasets. That is a finding about shortcuts, not a fix.

### 2.3 Threshold and operating-point selection under shift (workstream 3)

- Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster; arXiv 2208.02814; VERIFIED; ICLR 2024 per lookup snippet). Controls expected monotone loss (e.g. FNR, F1 loss) on a calibration set, with distribution-free guarantee if calibration and test are exchangeable. Our target set is not exchangeable with GroundLM (different source), so the guarantee does not transfer across sources. Use it per source, or report the guarantee as in-distribution only.
- Selective prediction under domain shift: a 2026 multilingual uncertainty study (arXiv 2603.07330, SNIPPET) reports softmax-based confidence loses reliability under domain shift. Consistent with our repo: the score ranks but the probability is not calibrated across sources. Rank-based thresholds (quantile of cal split) are safer than probability thresholds.
- Mondrian (class-conditional) conformal: valid per-class coverage but conservative with small per-class calibration sets (SNIPPET from SSBC docs and summary; the SSBC package is single-author, UNVERIFIED). Clustered conformal (Ding et al.) proposed for small data (SNIPPET).
- Small-sample FPR bound: for n legit calibration rows and zero observed false positives, the one-sided Clopper-Pearson upper bound on FPR is about 3/n (rule of three). Our target set (1,133 GroundLM negatives) can support a 95% upper bound of 0.26% at zero FP, and about 0.6% at 5 FP. This is standard statistics, not a published method; use it to report the uncertainty of any FPR we claim.
- The frozen rule in our repo asks the calibration FPR to be <= 1% on a validation set of 11,882 rows that is harder than the test (REPO). A better rule: fix the recall floor (SmishX label-1 kept >= 85%) on the fold's own cal split, then report GroundLM FPR as a held-out test with its Clopper-Pearson bound. This is item 1 in the table.
- Per-group thresholds: no published evidence found in this pass for spam or SMS.

### 2.4 Synthetic legitimate SMS generation (workstream 4)

- TRAI and DLT structure (multiple vendor documentation pages; SNIPPET only, conflicting). Template variables use {#var#}; each template needs approved entity (PE), header (sender ID) and template ID; header format is described as 6 characters, or up to 11 alphanumeric for transactional, or a 9-character format in different sources (CONFLICT). Enforcement dates differ (PE ID from 1 Nov 2020 and template ID from 1 Feb 2021 in one source; 1 Apr 2021 in another). Use the regulator's current text before citing; for generation we only need the shape, not the exact header length.
- Repo evidence: synthetic licensed legit (FinEE, arm A) and minimal pairs (arm B) moved GroundLM FPR from 41.6% only to 24-33% (REPO, SMS_RETRAIN_RESULTS.md). Same-source CloveAI moved it to 4-10%, which is not a cross-source result (REPO).
- Conclusion from our own evidence: synthetic legit does not transfer cross-source. Generating more legit with an open LLM (Llama 3.x 8B or Qwen2.5-7B on a T4) under DLT templates is unlikely to close the gap, and we have no published evidence that it does. Rank low.
- Agentic KD (arXiv 2602.10869) generated synthetic SMS with four frontier teachers; it reports the teacher choice drives results (SNIPPET). It is a synthetic-only validation, so no transfer claim.
- StructTransform (arXiv 2502.11853, SNIPPET): LLM-generated smishing is diverse enough to evade fine-tuned classifiers. That is a red-team result; use it as a reason to test synthetic-generated scams against our system, not as training data.
- Use case we can defend: LLM-generated paraphrases of the test donor's real messages, scrubbed, as a robustness test (not as training negatives). Not evaluated in any source.

### 2.5 URL phishing generalisation (workstream 5)

- PhreshPhish (arXiv 2507.10854 v2; VERIFIED in prior read, SOTA_TARGETS.md). Temporal split, leakage filter, base-rate APs. Our repo uses PhreshPhish-train for URL student and gets AP 0.484 to 0.941 across base rates (REPO).
- URL-only comparators: none published (SOTA_TARGETS.md). Do not claim beating GTE with URL-only features.
- Cross-dataset URL features: Mia et al. (arXiv 2411.09813, VERIFIED; lookup: 2024, DOI 10.1145/3704522.3704532) conclude features are often dataset-dependent and an adversary can evade by changing the domain name. Recommended: drop features that encode dataset-specific artefacts (length, certain character n-grams) and keep registered-domain features; test by leave-one-dataset-out.
- Rashid et al. (Computer Networks 2024; SNIPPET): cross-dataset F1 gain of about 0.06 from unsupervised domain adaptation over three datasets. Not read.
- Yi, Omotosho, Balogun (Security and Privacy 2026, doi 10.1002/spy2.70175; SNIPPET): accuracy above 94% across datasets with the same feature set, interpretability framing. Not read.
- XF-PhishBERT (PMC12669641, SNIPPET): ModernBERT plus URL features with a "186% performance retention" claim. Unusual; do not cite without the paper.
- PhishBench-External (Mendeley 2ctjcnm9kn, Oct 2025, SNIPPET): 1,000 URLs, phishing from PhishTank, benign from Tranco, 27 features including domain age and registrar. Use as an independent benign and phish test. Licence and balance UNVERIFIED.
- Homoglyph and typosquatting: no evaluation found in this pass. Johnny (arXiv 2609.01171, REPO-read via notes) shows homoglyphs drop BERT to 41.3% accuracy, full adversarial training restores to 99.5% (SMS, not URL). For URLs, add a Unicode normalisation step before the char-CNN and test on a homoglyph-perturbed copy of the PhreshPhish test (our own test; no published URL number).
- Benign sampling: Tranco top-N plus a long-tail sample (sites that are not in Tranco) should both be in the benign test. This is our protocol choice; no published source cited for the exact mix.

### 2.6 NIDS (workstream 6)

- Atif and Ali (arXiv 2609.13511, VERIFIED read in prior note): NF-UNSW-NB15-v2 to NF-CSE-CIC-IDS2018-v2, XGBoost F1 0.8116 (UNSW to CIC) vs 0.0730 (CIC to UNSW); RoBERTa-LoRA 0.6582 and 0.0599; CIC to UNSW balanced accuracy near chance (0.505). Report both directions.
- Cantone, Marrocco, Bria (arXiv 2402.10974; IEEE Access 12, DOI 10.1109/ACCESS.2024.3472907 per arXiv record in prior note): cross-dataset accuracy near random chance for most pairs; best train-to-test F1 39.12% (CIC-IDS2017 to CSE-CIC-IDS2018, LDA). Metric definition (binary or macro) unverified.
- Label errors: Liu et al. (IEEE CNS 2022, DOI 10.1109/CNS56114.2022.9947235; SNIPPET for content) and Engelen, Rimmer, Joosen (IEEE SPW 2021, DOI 10.1109/SPW53761.2021.00009; SNIPPET: "more than 20% of traces reconstructed or relabelled"). Dube 2023 (J. Computer Virology and Hacking Techniques, DOI 10.1007/s11416-023-00509-7; VERIFIED in prior note) on faulty use of CIC-IDS2017. Any headline must state the label version.
- Zeek SSH detection: Zeek's detect-bruteforcing script raises a password-guessing notice above a failure limit (docs.zeek.org v8.2.0, SNIPPET). We found no published detection rate on labelled real captures (GAP). Slips' brute-force module uses a default threshold of 9 attempts (tool docs, SNIPPET).
- Rules-first on real captures is therefore not benchmarked against any published Zeek number. Our holdout (rules 11/11, benign 5/6 silent; REPO) is synthetic and was tuned by reading the rule constants (NIDS_EDGE.md says so). The W3 plan (real labelled captures, no retuning) is the only credible next step.
- Flow-based cross-dataset with LLMs: Atif and Ali found LLMs did not beat classical ML under shift (REPO-read note). Rehman et al. (arXiv 2510.17883, SNIPPET): zero-shot LLM macro F1 0.78 (7B) on UNSW-NB15 only.
- Domain-adversarial NIDS: DI-NIDS (Layeghy, Baktashmotlagh, Portmann; arXiv 2210.08252; VERIFIED abstract in prior note; no numbers read). Our feature set would need NFv2 features to try it.
- What PS3 judges are likely to expect: real or public captures, per-family detection, benign alert rate, and an honest statement of label quality. This is an inference, not a source.

### 2.7 On-device LLM explanation: quality, safety, Arm (workstream 7)

- PhishXplain (arXiv 2505.06836; read in prior note): web phishing, Llama 3.2 3B Q4 on a laptop CPU; composite reliability 9.1 (3B) vs 7.4 (1B); live median 5.2 s. Directly relevant to our 1B choice: 1B is weaker on explanation reliability in their own measurement. Use this as the caveat in our report.
- The Attacker Moves Second (arXiv 2510.09023, Nasr, Carlini, Sitawarin, Schulhoff, Hayes, Ilie, Pluto, Song, Chaudhari, Shumailov, Thakurta, Xiao, Terzis, Tramèr; VERIFIED title/authors/year). 12 defences bypassed, most with >90% ASR under adaptive search (SNIPPET from abstract and secondary write-ups). Use its adaptive protocol for item 8.
- Adaptive attacks paper (arXiv 2503.00061, SNIPPET): LLM-based detector at 0% ASR without adaptation, 31.25% under multi-objective GCG (figures unverified).
- MiniCheck (arXiv 2404.10774, Tang, Laban, Durrett; VERIFIED). Small grounding checker, LLM-AggreFact benchmark; margin over AlignScore reported as 4-10 points (SNIPPET). Use the released model, not a retrained one, to avoid another training cycle.
- Prompt Guard 2 (Meta, 86M mDeBERTa and 22M DeBERTa-xsmall; model card, SNIPPET). Third-party claims of high false positives (Semalith, arXiv 2607.22545, competitor, UNVERIFIED). Not recommended for our benign SMS path without a test.
- Pi 5 and ExecuTorch Llama 1B: the Arm and PyTorch figures (decode 50.2 tok/s on OnePlus 12 with SpinQuant; SOTA notes, REPO-read) are phone numbers. A developer's April 2026 blog reports ExecuTorch 1.2.0 XNNPACK rejecting fused int4 subgraphs on aarch64 (per_op_mode workaround), about 0.87 tok/s for Gemma 4 on Pi 5 vs 6.71 for llama.cpp (single blog, anecdote, UNVERIFIED). This is a concrete risk to check: confirm whether our 1B export is fused or per-op, and measure both.
- Arm KleidiAI int4 kernels: the Arm blog claims int4 matmul uses I8MM (Armv8.6) in one post and SDOT in another (SNIPPET). Cortex-A76 is Armv8.2 (dot product, no I8MM; general knowledge, not checked against a source). Measure, do not assume.
- Pi 5 llama.cpp Llama 3.2 1B Q4_K_M: about 1.2 s first token and 286-328 tok/s prefill (SpecPicks April 2026; SNIPPET). Use as a rough llama.cpp reference only.
- Energy reporting: Arm LP method (PMIC rails, J per token with and without idle; SNIPPET from the Arm overview) is the boundary to state. Our 7-12 J/request needs the output token count to be comparable.
- Faithfulness: no SMS-explanation faithfulness metric in any source found (GAP). Build one from the verifier's evidence IDs: fraction of cited facts traceable to the evidence set (automatic) plus 100-item human audit.

### 2.8 Defensible SOTA claim and protocol (workstream 8)

What we can claim now, with evidence:
- A pretrained multilingual encoder trained without Indian legit data gets 1 false positive on 1,133 GroundLM negatives at 88.8% recall on SmishX label-1 (REPO, one fold, one seed). This is a cross-source result, the first in our repo. It is not a GroundLM-recall result (GroundLM positives not scored).
- A rules-first NIDS and a URL student with measured out-of-domain FPR (69.4%) are honest negative results.

What a credible claim needs (protocol):
1. Pre-register each gate and threshold rule before scoring (as the repo already does).
2. Paired comparison with GroundLM's released XLM-R predictions on the same 1,294 policy-negatives (Zenodo 20480151, Apache-2.0; ACL 2026.groundlm-1.79 for the headline 0.4% FPR and 0.980 recall; SOTA_TARGETS.md, prior note). Recall denominator not stated in the summary read: get the PDF before quoting recall.
3. Report FPR with Clopper-Pearson 95% intervals (section 2.3).
4. Score on at least two independent legit donors and on GroundLM; report each separately.
5. Use campaign-grouped or template-held-out splits; report the leakage filter used.
6. Do not combine numbers across the ACL version and the Zenodo EMNLP-submission version: their titles and label sets differ (prior note).
7. Report the 2-band verify rate as part of the system result; a single FPR without the abstain rate is incomplete.

What is not claimable: beating UCI or Mishra and Soni (saturated, random split, overlap; SOTA_TARGETS.md). Any URL-only win over PhreshPhish GTE row (which uses HTML). Any NIDS cross-dataset result without both directions.

---

## 3. Things to avoid

Methods or claims with negative or weak evidence in our repo or in the sources:
- Synthetic legit as the fix. Arms A and B (FinEE, minimal pairs) moved GroundLM FPR only to 24-33% and cost SmishX recall (REPO). Do not retry without a new source of real negatives.
- Same-source data as evidence. CloveAI (same source as GroundLM) gave 4-10% FPR; that is not cross-source and must not be reported as a generalisation result.
- Dynamic INT8 for mmBERT-base export (argmax 0.906, REPO).
- Tag-based shortcut fixes alone: filling tags did not restore recall (INBOX_FILLED_RESULTS.md, REPO).
- Probability thresholds across sources: softmax confidence loses reliability under domain shift (arXiv 2603.07330, SNIPPET). Use rank-based thresholds.
- Frozen <=1% FPR rule on a harder validation set (REPO, the reason all student arms failed).
- Claiming robustness from a fixed attack set (0/150 flips). Adaptive attacks break many defences (arXiv 2510.09023, VERIFIED title; figures SNIPPET).
- Using LLM-generated smishing as training data without a human audit. StructTransform (arXiv 2502.11853, SNIPPET) shows synthetic smishing can evade classifiers; it is a red-team source, and training on it would only teach our model that style.
- Zero-shot LLM classifiers for SMS or NIDS. GroundLM reports six LLM APIs over the 1% FPR gate on Bank and UPI (prior note, ACL); Atif and Ali found LLMs did not beat classical ML under shift.
- Citing SNIPPET-only numbers: Samsung IEEE Access (~0.99 at 127 kB), COPS, PhishLang, Prompt Guard 2 throughput, XF-PhishBERT 186%, Semalith FP rates, Cau et al. 2507.07916 (not found).
- Phone benchmarks as Pi targets (OnePlus 12, S24+). Pi 5 decode is about 4-5x slower (prior note).

Requires paid APIs or unlicensable data (do not plan on it):
- IEEE DataPort Hindi spam SMS (subscription required, prior note).
- SMS-Shield (LinkedIn-only, synthetic-teacher labels, not a benchmark).
- Nahapetyan et al. dataset: release status unverified; campaign IDs would be the grouping key if released.
- Any GPT-4o or other API used as a labeller or as a deployed comparator: the project's no-paid-API rule applies to training data.
- Meta Prompt Guard 2 in the benign SMS path: no validated false-positive number on our data.

Protocol mistakes to avoid:
- Reporting AUC on all 578 label-1 rows as if it were the 226 eval_ok smishing rows (REPO notes differ on which set).
- Mixing PhreshPhish v1 and v2 numbers (SOTA_TARGETS.md).
- Mixing CIC-IDS2017 label versions.

---

## 4. Gaps (what this pass could not establish)

- No ARM INT8 accuracy evidence for BERT-class encoders (ORT QDQ or weight-only, per-channel). Need our own bench.
- No Pi 5 ExecuTorch Llama 1B TTFT or decode with KleidiAI (only phone numbers and one anecdote).
- No published Zeek SSH detection rate on labelled real captures (2024-2026).
- No URL cross-dataset numbers with Tranco benign sets that we could read (Rashid 2024 and Yi 2026 are snippets).
- No spam or phishing-specific shortcut-mitigation result (JTT, LEACE, counterfactuals) found.
- No conformal or selective-classification paper for text under domain shift with small calibration sets, read in full.
- No SMS explanation faithfulness metric.
- mmBERT-small benchmark numbers on GLUE or MTEB are not read from the paper.
- TRAI DLT header and enforcement details conflict across sources; check the regulator text.
- Hosted lookup tool disconnected mid-pass; IDs marked VERIFIED were confirmed before it did. Re-verify the rest before citing.

---

## 5. Suggested order for the 10 days

Day 1-2: item 1 (two-band rule, score existing teacher fold, pre-registered). Item 5 (LEACE on teacher embeddings, CPU).
Day 2-4: item 2 (mmBERT-small fold, per-channel INT8 parity, Pi bench under thermal rule). Item 4 (JTT, one seed, T4).
Day 3-5: item 6 (URL domain-grouped retrain plus PhishBench-External test). Item 8 (adaptive attack on explainer, run in parallel).
Day 5-7: item 7 (NIDS real captures, Kaggle CPU; no retuning).
Day 7-10: item 3 only if donor 2 exists; otherwise write up the negative result with Clopper-Pearson bounds.

Every result gets its own docs/*_RESULTS.md. No commits or pushes from this research.
