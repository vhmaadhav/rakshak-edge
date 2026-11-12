# Reducing false positives of small SMS scam classifiers on legitimate Indian transactional SMS

Research date: 2026-11-06. Search-based pass (about 15 tool calls). Most sources are search snippets, not full-text reads. Items marked UNVERIFIED were not confirmed against the primary document.

## Key question 1: What does GroundLM (and related 2025-2026 Indian SMS fraud work) do to get low FPR?

### Takeaway
I could not find any public source named "GroundLM", and the numbers in the project brief (270M XLM-R at 0.4% FPR, 0.980 recall, 1,133 scoreable negatives) are not verifiable from public sources. The closest public artifact is an anonymized EMNLP 2026 Industry submission whose public record reports no FPR or recall numbers at all. Treat the GroundLM figures as the project's own unpublished numbers until a citable source is located.

### Cited Findings
- Closest public match: Zenodo record v4 (May 31, 2026), an anonymized EMNLP 2026 Industry Track submission titled "Default LLM APIs Over Flag Indian Financial SMS: A Cleaned Label Benchmark and Template Level Deployment Audit". Under anonymous double-blind review, so not peer-reviewed. — [Zenodo](https://zenodo.org/records/20480151)
- Benchmark contents: cleaned multilingual test set of 1,393 rows (Hindi Devanagari, Hinglish, Indian English), with a per-row `prob_scam` score. Source material includes IIITD SMSAssassin (Yadav et al., HotMobile 2011), redistributed hash-only in v4. — [Zenodo](https://zenodo.org/records/20480151)
- Label quality: dual-LLM-judge label cleanup, then 3-annotator agreement on a 276-row stratified subset, Fleiss kappa = 0.82, Krippendorff alpha = 0.82. — [Zenodo](https://zenodo.org/records/20480151)
- Systems evaluated: cleaned XLM-R (about 270M parameters), English-only XLM-R control, GPT-4o, GPT-4o mini, Claude Haiku 4.5, Gemini 2.5 Flash, Llama 3.3 70B and Qwen 2.5 7B (both via Together AI). Per-sample predictions are released, but the page gives no FPR or recall figures for any system. — [Zenodo](https://zenodo.org/records/20480151)
- Template-level audit method: a frozen regex template taxonomy, a 9-bucket institutional partition for per-template FPR, and an operating-point sweep over tau = 0.30 to 0.90 reporting global and per-template FPR. The page gives no audit results. — [Zenodo](https://zenodo.org/records/20480151)
- License: Apache-2.0 for code, scripts, and redistributable datasets. — [Zenodo](https://zenodo.org/records/20480151)
- Related, self-reported and UNVERIFIED: SMS-Shield (Qwen3-4B-Instruct + LoRA, about 17k synthetic Indian SMS labeled by Gemma3-27B, including "hard negatives such as real bank alerts and bill reminders"; a second model re-judged every sample and disagreements were dropped; claims zero genuine OTPs or bank alerts flagged on a 1,390-item test set; 2.4 GB GGUF). The post is the only source (LinkedIn profile, HTTP 999 on fetch, so only the search snippet was read). The HF repo is named `ganmoor-ai-labs/sms-shield` and the Ollama tag is `santosh07401/sms-shield`; the two namespaces differ and ownership should be checked. — [LinkedIn post](https://in.linkedin.com/in/draina); [HF repo named in snippet, not fetched](https://in.linkedin.com/in/draina)
- Related UNVERIFIED benchmark-style claim: Google paper on UPI scam detection with LLMs (arXiv 2410.19845, Oct 2024). Reports Gemini Ultra at 93.33% accuracy on curated UPI transaction data. Not SMS, and no FPR reported. — [arXiv 2410.19845](https://arxiv.org/abs/2410.19845)
- Public Indian scam/OTP data (see Key question 5 for licences): 2025 synthetic OTP/intent/phishing dataset of 73,470 messages (gated on HF, MIT claimed); Hinglish/English scam dataset of about 3,787 rows, with only 35 hand-written Hinglish examples. — [HF OTP dataset](https://huggingface.co/datasets/gandharvbakshi/SMS-dataset-OTP-OTP_INTENT_Phishing); [HF Hinglish dataset](https://huggingface.co/bolewara/hinglish-scam-text-dataset)
- Over-blocking cost (reported in a search summary, not read in full): in a comparison of commercial anti-smishing apps, the two that blocked the most smishing also blocked 85-100% of benign messages. — [arXiv 2309.07447](https://arxiv.org/html/2309.07447v2)

### Inferences
- The strongest reusable item from the GroundLM-related work is the methodology, not a number: label cleanup with two judges plus agreement stats, and per-template FPR reporting. Both are cheap to adopt.
- The 1,393 public rows and the 1,133 "scoreable policy negatives" in the project brief may be related by filtering, but nothing public confirms it. Reconcile the counts before citing.
- Since the public benchmark releases per-sample scores for LLM systems but not for the student, the team's own student FPR analysis is still needed on this benchmark.

### Gaps
- No public GroundLM paper or artifact. Need the project's own source or a DOI before quoting "0.4% FPR, 0.980 recall".
- No source reports how bank, UPI, or OTP messages are labeled in the Zenodo set, so it is unclear whether those categories count as negatives.
- The SMS-Shield numbers rest on one unverified self-report.

## Key question 2: Which techniques address benign-template false positives?

### Takeaway
The most actionable levers are (a) DLT sender-header type suffixes as a feature or gate, (b) per-template FPR auditing with group-aware splits, and (c) label cleanup. I found no 2024-2026 paper that directly demonstrates hard-negative mining or focal loss reducing FPR on Indian transactional SMS. Those are mostly UNVERIFIED for this setting.

### Cited Findings
- DLT header format (since May 6, 2025): [Network][State]-[Brand]-[Type], with type suffix P (promotional), S (service), T (transactional), or G (government). Telcos append the suffix during DLT scrubbing; businesses do not add it. "Service Explicit" was reportedly eliminated, so service messages are reportedly defaulted to P. — [Storyboard18](https://www.storyboard18.com/how-it-works/why-your-sms-now-looks-like-a-code-trais-new-message-format-explained-70270.htm); [Zoho help](https://help.zoho.com/portal/en/kb/articles/sms-campaigns)
- The sources do not specify the length of the circle code or brand-name limits. Check with the telco or DLT platform. — [Storyboard18](https://www.storyboard18.com/how-it-works/why-your-sms-now-looks-like-a-code-trais-new-message-format-explained-70270.htm)
- Template-level FPR analysis as a method (per-template FPR across a tau sweep). — [Zenodo](https://zenodo.org/records/20480151)
- Hard negatives can be mislabeled false negatives in noisy data; positive-aware mining (removing negatives too similar to positives) was used in retrieval, not SMS classification. — [NV-Retriever arXiv 2407.15831](https://arxiv.org/html/2407.15831v2) (retrieval setting; not SMS)
- Hard-negative mining and LLM-generated hard negatives can violate relevance assumptions without filtering. — search summary of 2026 retrieval papers, no single verified URL; UNVERIFIED for classification.
- Mondrian (class- or group-conditional) conformal sets: per-class thresholds; can be overly conservative in many-class, low-data settings. — [arXiv 2306.09335](https://arxiv.org/pdf/2306.09335)
- Abstention/selective classification: a 2025 result gives likelihood-ratio abstention under covariate shift (general ML, not text-specific). — [arXiv 2505.15008](https://arxiv.org/abs/2505.15008v1). An older paper sets an explicit budget on wrongly flagged in-distribution samples (OOD setting, 2023). — [arXiv 2301.12386](https://arxiv.org/pdf/2301.12386)
- No public study found on focal loss or cost-sensitive weighting for SMS FPR. — gap.

### Inferences
- DLT suffix as a feature: it is visible sender metadata on device, not in the body, so the benchmark and training data must carry sender IDs for this to be testable. The Zenodo record does not state whether it does. Use the header as a feature or prior, not a hard allow, because headers can be impersonated (UNVERIFIED; no source checked).
- Two-stage design (allowlist of known DLT-registered T-type templates, then classifier) is consistent with the Zenodo regex-template precedent. Keep scam-indicator rules (links, OTP-request phrasing, mismatched sender) above the allowlist so a T-header cannot pass a phishing body.
- The earlier hard-negative augmentation arm failed (FPR still 7-37%). Augmentation alone is not sufficient. The likely missing piece is minimal pairs: a legit template and the same template with a scam edit, labeled per pair, plus group-aware splits so a template never appears in both train and test. This is a hypothesis, not sourced.

### Ranked fix list (impact vs effort, 2-week, free compute)
1. Group-aware split by template cluster, plus per-template FPR report (effort low, diagnostic value high). Prevents leakage that would inflate any FPR claim.
2. Label cleanup: two-judge disagreement filter on the training set, reporting kappa (effort low-medium; method from the Zenodo paper and SMS-Shield, both UNVERIFIED for effect size).
3. Legit minimal-pair hard negatives built from DLT-format bank, UPI, recharge, and OTP templates (effort medium; the augmentation arm already failed, so only pursue with minimal pairs and group splits).
4. Sender-header T/S/P feature or gate (effort medium; needs headers in data; impact unknown until tested).
5. Allowlist-plus-classifier two-stage pipeline with scam-indicator override (effort medium; deterministic, testable on device).
6. Abstention band (already in the demo as LOW) combined with per-source thresholds from Key question 4 (effort low; limits recall loss only if the model already separates classes).
7. Encoder swap (see Key question 3) (effort high on Kaggle; unclear gain).
8. Focal or cost-sensitive loss (effort low; no source found, so UNVERIFIED and should be tried as an ablation).

### Gaps
- No peer-reviewed 2024-2026 study of benign-template FPR on Indian SMS was located.
- No source tests whether DLT header features reduce FPR, or whether headers are spoofable in practice.

## Key question 3: Small multilingual encoders for on-device Indian SMS (accuracy/FPR and size on Arm)

### Takeaway
mmBERT-small (about 140M total parameters, but only about 42M non-embedding) is the most promising small multilingual option. Its size is dominated by a 256k-token vocabulary, so vocabulary pruning and int8 quantization would be needed to reach the roughly 10 MB ONNX target. I found no source reporting FPR for any of these encoders on Indian SMS, and no Arm latency numbers.

### Cited Findings
- mmBERT-small: about 140M total parameters, of which about 42M are non-embedding; hidden size 384; 22 layers; 3T tokens; over 1800 languages; 8,192-token context. Reported GLUE average 84.7 vs Multilingual MiniLM 78.3 (self-reported in the paper). — [arXiv 2509.06888](https://arxiv.org/html/2509.06888v1)
- A community fork truncated the vocabulary to about 69M total parameters. — [arXiv 2509.06888 search summary](https://arxiv.org/html/2509.06888v1); [emergentmind](https://www.emergentmind.com/papers/2509.06888)
- MuRIL base: 236M parameters, about 16B unique tokens, vocabulary 197,285. — [MuRIL HF README](https://huggingface.co/google/muril-base-cased/commit/5e1c4af0334fdb9f8080ab14efd7eeda566c70d8)
- IndicBERT v1: about 33M parameters. IndicBERT v2: parameter count not confirmed. A review says v2 has 7.5x the parameters of v1, which would be about 250M by my own arithmetic (UNVERIFIED). — [arXiv 2212.05409](https://arxiv.org/pdf/2212.05409)
- IndicXTREME: IndicBERT v2 generally outperforms mBERT, XLM-R, MuRIL, and IndicBERT v1 across nine tasks, though MuRIL sometimes wins on specific tasks. — [arXiv 2212.05409](https://arxiv.org/pdf/2212.05409)

### Inferences
- mmBERT-small with a pruned Indic-plus-English vocabulary and int8 quantization is the plausible route to a 10 MB ONNX. Size must be measured, not assumed.
- Encoder size alone does not fix FPR. The GroundLM-related XLM-R (270M) figure in the brief is not evidence that a 10 MB student can match it.

### Gaps
- No Indian SMS FPR or recall reported for mmBERT-small, MuRIL, IndicBERT v2, or MiniLM.
- No Arm (Raspberry Pi 5 or Android) latency or memory numbers found.

## Key question 4: Threshold and risk-control methods (LTT, conformal, Mondrian)

### Takeaway
Risk-control methods can give a finite-sample FPR guarantee for a chosen threshold, but they only help if the model's scores separate legit from scam at the operating point. The brief's observation that a threshold cannot fix this is consistent with that limit. Use these methods after the model's FPR drops, to certify it.

### Cited Findings
- Learn Then Test (LTT) recasts risk control as multiple hypothesis testing with fixed-sequence testing, gives finite-sample guarantees, and can handle non-monotone losses (FPR is not monotone in threshold in general). Identity of the source paper inferred from the arXiv ID returned in search. — [arXiv 2110.01052](https://arxiv.org/pdf/2110.01052v4) (search summary; primary theorem not checked)
- A fraud-detection paper applies LTT to control FPR while minimizing false negatives, and extends it to covariate shift. — [arXiv 2510 search result](https://research.feedzai.com/oshuzuxa/2025/10/high_probability_risk_control_under_covariate_shift.pdf) (feedzai, not a peer-reviewed venue)
- Mondrian conformal: split calibration by group and run split conformal per group; class-conditional coverage P(y in C(x) | y=c) >= 1 - alpha per class. In many-class, small-data settings, the classwise variant can be overly conservative. — [arXiv 2306.09335](https://arxiv.org/pdf/2306.09335)
- No source found applying conformal or Mondrian methods to control false positives on SMS. — gap.

### Inferences
- A per-source or per-template Mondrian threshold, with a calibration set made only of legit messages, could certify a per-source FPR. This is only useful if the legit-class scores are already well below the scam-class scores.
- The Zenodo per-template tau sweep (0.30 to 0.90) suggests the team should first find which templates drive FPR, then apply per-group thresholds only where needed.

### Gaps
- No source quantifies recall loss from LTT or Mondrian thresholds on SMS data.

## Key question 5: Public Indian transactional/legit SMS datasets and synthetic approaches (licence noted)

### Takeaway
Public legit Indian transactional SMS is scarce. The usable options are the Zenodo benchmark (Apache-2.0, cleaned, but hash-only for IIITD-sourced rows), a gated 2025 synthetic OTP/phishing set (MIT claimed), and a small Hinglish set. Synthetic bank and UPI templates are probably the main route, but I found no verified public bank-template corpus.

### Cited Findings
- Zenodo benchmark: Apache-2.0 for redistributable datasets; IIITD-sourced rows are hash-only in v4. — [Zenodo](https://zenodo.org/records/20480151)
- Synthetic Indian OTP/intent/phishing dataset: 73,470 messages, 2025, gated on HF, MIT claimed, free 10K preview. Labels are binary phishing-like, not a true smishing benchmark. — [HF dataset](https://huggingface.co/datasets/gandharvbakshi/SMS-dataset-OTP-OTP_INTENT_Phishing)
- Hinglish/English scam dataset: about 3,787 rows; 35 hand-written Hinglish scam examples; English from the public SMS Spam Collection. Licence not checked. — [HF dataset](https://huggingface.co/bolewara/hinglish-scam-text-dataset)
- Hindi spam SMS dataset: 3,894 messages, spam/ham labels, no smishing or transactional coverage stated, no date found. — [IEEE DataPort search result](https://ieee-dataport.org/taxonomy/term/9619)
- Synthetic generation with LLM labeling plus second-judge filtering (SMS-Shield, UNVERIFIED). — [LinkedIn post](https://in.linkedin.com/in/draina)

### Inferences
- Generate legit templates from DLT-format bank, UPI, recharge, and OTP patterns (the team can write these from known formats; no public corpus verified). Check licence-free status of any template source before use.
- Any synthetic generator must be filtered by a second judge and kept out of the test split, to avoid the benchmark-leakage problem above.

### Gaps
- No verified public corpus of Indian bank, UPI, or telecom transactional SMS with licence.
- No Hinglish or Indic-script legit-transaction dataset located.

## Source-date notes
- Zenodo record: May 31, 2026 (v4). EMNLP 2026 review is anonymous.
- mmBERT: arXiv 2509.06888, Sep 8, 2025.
- DLT suffix change: effective May 6, 2025.
- IndicBERT v2 / IndicXTREME: arXiv 2212.05409, 2022 (later versions exist).
- LTT: arXiv 2110.01052, 2021. Mondrian class-conditional: arXiv 2306.09335, 2023.
- Digital payments paper: arXiv 2410.19845, Oct 2024.
