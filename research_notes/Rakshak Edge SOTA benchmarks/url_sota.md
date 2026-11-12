# URL phishing detection: SOTA as of 2026-11-07

Scope: URL-only and URL+HTML results, protocols, base-rate metrics, model sizes. Builds on `docs/SOTA_TARGETS.md` sections 5-8 (compiled 2026-10-30). Search and fetch were done by a single agent with about 15 tool calls. Only arXiv abstracts or full HTML text were read unless marked otherwise. Secondary mirrors are marked UNVERIFIED.

## What changed vs SOTA_TARGETS.md (2026-10-30)

- PhreshPhish v2 Table 4 read from the arXiv v2 HTML. The P@R=0.90 rows for Linear and FFN are now filled (earlier file had AP only). Benchmark instance counts per base rate are new: 0.05% (586), 0.1% (296), 0.5% (59), 1% (29), 5% (5); 975 benchmark sets in total.
- Confirmed: PhreshPhish v2 has no URL-only baseline. All four baselines use URL+HTML (see section 1).
- PhishLang: the arXiv v3 Table 2/3 data is NOT PhreshPhish. It is PhishPedia-derived (30K phishing + 30K benign, HTML available), random 70/30 split, no dedup stated. MobileBERT has 25M parameters and 74 MB DRAM. The "7 KB" and "0.2 KB" values are input or artifact sizes, not model sizes. Earlier file's "74 MB memory" is correct as memory, not disk.
- New URL-only rows: PhishScout v2 (HF, MIT, self-reported), arXiv 2606.00889 (May 2026, PhiUSIIL, random ceiling).
- New 2026 preprint with a campaign-aware split: arXiv 2606.21690 (mirror only, UNVERIFIED).
- No later peer-reviewed paper reporting on PhreshPhish was found. Search-limited, not exhaustive.
- Correction: a secondary summary said PhreshPhish has 404 benchmark datasets. The paper says 975. Use 975.

## 1. PhreshPhish (arXiv 2507.10854 v2): per-base-rate numbers and URL-only status

Source: https://arxiv.org/html/2507.10854v2 (read directly; Table 4, "Benchmark results"). Versions: v1 submitted 14 Jul 2025; v2 revised 11 Feb 2026. Caption: mean +/- max abs deviation of 95% bootstrap CI. Test set base rate 46%; benchmark base rates 0.05, 0.10, 0.50, 1.0, 5.0, and full 24%.

| Model | Inputs (per paper) | Metric | 0.05% | 0.10% | 0.5% | 1.0% | 5.0% | 24% (full) |
|---|---|---|---|---|---|---|---|---|
| Linear (LinearSVC) | URL+HTML char n-grams + hand features (URL length, DOM node count) | AP | 0.2230 +/- 0.0043 | 0.3191 +/- 0.0047 | 0.5825 +/- 0.0050 | 0.6841 +/- 0.0045 | 0.8610 +/- 0.0047 | 0.9551 +/- 0.0028 |
| Linear | same | P@R=0.90 | 0.0149 +/- 0.0007 | 0.0298 +/- 0.0014 | 0.1223 +/- 0.0062 | 0.2191 +/- 0.0102 | 0.5950 +/- 0.0213 | 0.8957 +/- 0.0075 |
| FFN (one 32-ReLU hidden layer) | same features, binarized and frequency-selected | AP | 0.2511 +/- 0.0048 | 0.3368 +/- 0.0050 | 0.5798 +/- 0.0058 | 0.6813 +/- 0.0047 | 0.8640 +/- 0.0027 | 0.9607 +/- 0.0018 |
| FFN | same | P@R=0.90 | 0.0163 +/- 0.0007 | 0.0316 +/- 0.0012 | 0.1319 +/- 0.0044 | 0.2339 +/- 0.0068 | 0.6137 +/- 0.0111 | 0.9045 +/- 0.0038 |
| GTE (gte-large fine-tuned) | raw concatenated URL+HTML | AP | 0.6050 +/- 0.0051 | 0.6879 +/- 0.0044 | 0.8354 +/- 0.0035 | 0.8762 +/- 0.0032 | 0.9382 +/- 0.0016 | 0.9779 +/- 0.0020 |
| GTE | same | P@R=0.90 | 0.0891 +/- 0.0068 | 0.1563 +/- 0.0123 | 0.3736 +/- 0.0289 | 0.5426 +/- 0.0305 | 0.8576 +/- 0.0133 | 0.9746 +/- 0.0037 |
| Gemini-2.5-Pro zero-shot | URL + HTML (JS/CSS/comments removed) | AP | 0.1003 +/- 0.0012 | 0.1780 +/- 0.0020 | 0.4839 +/- 0.0038 | 0.6252 +/- 0.0044 | 0.8383 +/- 0.0030 | 0.9216 +/- 0.0034 |
| Gemini-2.5-Pro zero-shot | same | P@R=0.90 | 0.0390 +/- 0.0017 | 0.0722 +/- 0.0035 | 0.2606 +/- 0.0141 | 0.4147 +/- 0.0214 | 0.7861 +/- 0.0145 | 0.9352 +/- 0.0235 |

Source: arXiv 2507.10854 v2 HTML, Table 4 (all values as printed).

Findings:
- URL-only rows: none. The paper has no ablation separating URL from HTML. Linear and FFN use HTML-derived n-grams, and the LLM prompt includes HTML. A URL-only row must be trained and scored by us on the same split and base-rate subsamples.
- v1 vs v2: earlier file cites v1 GTE at 0.05% = 0.7771 AP. Not re-checked here. Do not mix v1 and v2 numbers. v2 is the reference.
- Split and dedup (v2 text, as recorded in SOTA_TARGETS): temporal split (test strictly later than train), 0.3 stratified test fraction, TF-IDF LSH leakage filter, canonical URL and near-HTML dedup, easy-phishing difficulty filter.
- Later papers: none found that report PhreshPhish numbers. The search returned only the paper and its mirrors. UNVERIFIED that none exist.

Gaps:
- No per-base-rate URL-only baseline from the authors.
- Whether the v2 HF dataset ships a URL-only baseline script was not checked.

## 2. Base-rate (prevalence) evaluation: how it is done and what a credible row looks like

Method (PhreshPhish protocol, per search summary of the paper's appendix; not re-read in full):
- Benchmarks are built by subsampling the test set to a target base rate (0.05% to 5%) while keeping the test data fixed. Subsampling is repeated (sampling without replacement) to get many instances, then AP and P@R=0.90 are averaged with bootstrap CIs.
- Precision rises with base rate when TPR and FPR are fixed.

Derivation (standard identity, not sourced; this is arithmetic):

  precision = TPR * pi / (TPR * pi + FPR * (1 - pi)), where pi = base rate.

Worked examples (our arithmetic):
- TPR 0.90, FPR 0.01% (1e-4). At pi = 0.05%: precision = 0.818. At pi = 1%: precision = 0.989.
- So the same FPR is fine at 1% prevalence and unusable at 0.05%.

Credible URL-only vs HTML comparison row, from the protocol above:
1. Same split and same base-rate subsample seeds as PhreshPhish v2. Report AP and P@R=0.90 at 0.05%, 0.10%, 0.5%, 1%, 5%, and full.
2. Mark the input column explicitly: URL-only, URL+HTML, or URL+HTML+DNS/TLS/WHOIS. Give model size on disk (INT8 and FP32), parameter count, and inference latency with named hardware.
3. Report TPR at FPR 0.01% and 0.1% (the browser-relevant points) in addition to the PhreshPhish metrics.
4. Show both random-split and time-split numbers, and state dedup rules.
5. Report confidence intervals from the same bootstrap method as PhreshPhish.

## 3. PhreshPhish-side numbers (our repo, repo-reported; NOT verified here)

Repo-reported, not re-run in this research:
- URL-only mean AP 0.4435 at 0.05%, and 0.5589 at 0.1% (three seeds).
- charcnn-wide AP 0.484 at 0.05% and 0.865 at 1%.
- url_domain test F1 0.999 (likely random split; treat as ceiling only).

Comparison against Table 4 (not like-for-like on inputs):
- At 0.05%: our URL-only charcnn-wide 0.484 is above the Linear (0.2230) and FFN (0.2511) URL+HTML rows, and below GTE URL+HTML (0.6050).
- At 1%: our 0.865 is above the Linear (0.6841) and FFN (0.6813) URL+HTML rows, and close to GTE (0.8762). This is only a reference point. Our number was not run on the same subsamples here.
- Any claim of beating GTE needs a URL-only GTE row on the same split.

## 4. Other URL benchmarks (best numbers, protocol, size)

### URLTran (MILCOM 2021; arXiv 2106.05256)
- TPR 86.80% at FPR 0.01% (URLTran_BERT); next-best baseline URLNet 71.20%; Texception 52.15%. Relative gain 21.9%.
- Source: arXiv abstract https://arxiv.org/abs/2106.05256, read 2026-11-07. Matches SOTA_TARGETS.
- Dataset, split protocol, and parameter count: NOT read from the abstract. UNVERIFIED. The full PDF was not fetched.
- Independent replication: none found.

### PhishLang (NDSS 2026; arXiv 2408.05667 v3)
- Source: arXiv v3 HTML https://arxiv.org/html/2408.05667v3 (read); NDSS listing https://www.ndss-symposium.org/ndss-paper/phishlang-a-real-time-fully-client-side-phishing-detection-framework-using-mobilebert. DOI 10.14722/ndss.2026.231037 (from SOTA_TARGETS).
- Model: MobileBERT, 25M params, 74 MB DRAM at inference (Table 2). HTML-based, not URL-only.
- Table 2 (model selection, test split): MobileBERT acc 0.96, P 0.95, R 0.96, F1 0.96, 0.39 s median. DistilBERT (66M) acc 0.94, 0.85 s, 502 MB. TinyBERT (15M) acc 0.85, 0.78 s, 495 MB. FastText (10M) acc 0.62. GPT-2 (117M) acc 0.68. Llama 2 (7B) acc 0.97, 33.71 s. T5-base acc 0.89.
- Table 3 (5K labeled samples): PhishLang acc 0.94, F1 0.94, 7 KB artifact (input size), 0.9 s. URLNet (their run) acc 0.73, F1 0.73, 0.2 KB, 0.7 s. PhishIntention acc 0.96. VisualPhishNet acc 0.85. GPT-4 (HTML) acc 0.92.
- Client-side timing (Table 9): median 0.44 s, 76 MB (Xeon W + A5000 GPUs, not a phone); 0.59 s / 93 MB (Core i5-13th gen, no GPU); 0.73 s / 89 MB (Celeron N4500, 2 GB RAM); 0.81 s / 102 MB (VM, 2 cores).
- Dataset: PhishPedia-derived, 22,419 phishing and about 26K benign with HTML, plus 411 BJE samples. Split: 70/30 with 5-fold CV. The paper does NOT say whether the split is random or temporal, stratified, or deduplicated.
- Inconsistencies: F1 0.96 in Table 2 vs 0.94 in Table 3 for PhishLang (different sets, not explained). Text vs Table 2 disagree on Bloom/T5 F1.
- Verdict: a HTML-based, random-split, unlabeled-protocol system. It is not a URL-only or temporal benchmark. Not a valid URL-only target. The paper's "7x less memory" claim is relative to unspecified baselines.

### PhishScout v2 (HF, saidutta69/PhishScout-v2; MIT; self-reported)
- Source: https://huggingface.co/saidutta69/PhishScout-v2 (read). Not peer-reviewed.
- Model: LightGBM (200 trees, depth 10, 63 leaves), ONNX, Platt-calibrated. 896.3 KB ONNX on the card, but the card also says 669 KB in one place. Use the on-disk file size when measured.
- Inputs: URL-only core model, 579 features (66 structured + 512 hashed char 3-5 grams). Says 66+512=578; minor inconsistency. An optional enrichment stage fetches page, TLS and RDAP data; the core model is URL-only.
- Data: PhishTrap, 20K URLs, balanced 50/50. Phish from Phishing.Database, OpenPhish, PhishStats; benign from Tranco top 10K. Split 70/15/15 (claimed disjoint, seed 42). Temporal split not stated.
- Test metrics (raw): acc 0.8995, P 0.9229, R 0.8717, F1 0.8966, ROC AUC 0.9615. Calibrated: acc 0.8981, F1 0.8957, AUC 0.9615. Test n = 2,994 (implied by confusion matrix TP 1252, FP 116, FN 245, TN 1381).
- Latency: about 24 us per URL median on CPU (card). Also claims "single-digit-microsecond"; conflicts. Hardware not named.
- Verdict: a credible small URL-only baseline, not a SOTA. Its AUC of 0.9615 on a hard, balanced, source-diverse benchmark is a useful lower reference. FPR at low operating points is not reported.

### arXiv 2606.00889 (May 2026; Emmanuel and Oghie)
- Source: abstract https://arxiv.org/abs/2606.00889 (read; full PDF not read).
- Inputs: 16 URL-string features, MLP. Framework includes a blacklist screen, so it is not pure URL-only.
- Dataset: PhiUSIIL (235,795 URLs). Protocol not stated.
- Headline MLP: acc 99.24%, P 98.74%, R 99.95%, F1 99.34%, ROC AUC 99.65%. Latency 1.2 ms/URL, 4,200 URL/s peak. Hardware not stated.
- Verdict: random-ceiling in-distribution. Not a headline. Same class as url_domain F1 0.999.

### PhiUSIIL (Comput. & Security 2023; DOI 10.1016/j.cose.2023.103545)
- 235,795 instances. Content-derived features (54), not URL-only. Random-ceiling. Downstream numbers UNVERIFIED (see SOTA_TARGETS).

### arXiv 2606.21690 (June 2026; UNVERIFIED, mirror only)
- Source: Emergent Mind mirror https://www.emergentmind.com/papers/2606.21690 (not the arXiv PDF). Snippet is partly garbled.
- Claims: campaign-aware split plus matched twins; clean held-out FPR 1.3%; 100% detection on fresh OpenPhish, 45.2% on subtle generic-domain phishing; bottleneck is cross-modality generalization, not classifier accuracy.
- Verdict: directionally important (campaign splits lower numbers). Do not cite until the PDF is read.

### Not re-checked in this pass (kept from SOTA_TARGETS, still UNVERIFIED)
- URLNet (arXiv 1802.03162): numbers not extracted.
- GramBeddings (DOI 10.1016/j.cose.2022.102964): 98.27% on 800K balanced, UNVERIFIED primary; live-PhishTank cases 75-76%.
- PhishStorm (DOI 10.1109/tnsm.2014.2377295): 94.91% / FPR 1.44%, snippet.
- ISCX-URL2016: random-ceiling, subsets vary.

## 5. 2025-2026 small URL models (char-CNN, distilled, byte-level) with size and latency

| System | Type | Size | Latency / hardware | Accuracy / AUC | Protocol | Status |
|---|---|---|---|---|---|---|
| PhishScout v2 | LightGBM + hashed char 3-5 grams (URL-only core) | 896 KB ONNX (card; 669 KB also stated) | 24 us median CPU | acc 0.8995, AUC 0.9615 | 70/15/15 on PhishTrap 20K, temporal not stated | HF card, MIT, self-reported |
| arXiv 2606.00889 MLP | 16 URL-string features, MLP | not stated | 1.2 ms/URL, hardware not stated | acc 99.24, AUC 99.65 | PhiUSIIL, protocol not stated | Abstract only; random ceiling |
| PhishLang (MobileBERT) | HTML + DOM text, 25M params | 74 MB DRAM | 0.39 s CPU; 0.44-0.81 s on client hardware | acc 0.96 (Table 2) | PhishPedia-derived, random 70/30 | HTML-based; not URL-only |
| Char-CNN + LightGBM ensemble (arXiv 2512.16717 family) | char CNN + engineered features | not stated | not stated | 99.8% reported (secondary, UNVERIFIED) | not stated | Not verified |
| Adversarial CNN-BiLSTM (Jan 2026 preprint) | char CNN-BiLSTM, adversarial training | not stated | not stated | not stated | 280K benign + 120K phish URLs | Preprint; not read |
| MDPI char-TCN with attention | char TCN | not stated | not stated | acc 97.54%, 86% precision on unseen phish (secondary) | not stated | UNVERIFIED |

Gaps:
- No 2025-2026 peer-reviewed char-CNN URL model with on-device latency, size, and low-FPR TPR was found.
- ByT5 and CANINE URL variants: none found in this pass.
- Samsung IEEE Access 2024 on-device smishing (about 127 KB, about 0.99 acc): still UNVERIFIED (see SOTA_TARGETS section 7).

## 6. Temporal and dedup protocols (PhishTank / OpenPhish)

- PhreshPhish v2 is the only URL+HTML benchmark found with a temporal split, leakage filter and dedup (section 1).
- Time-based evaluation literature: arXiv 2501.00356 (monthly test segmentation; degradation over time, lightweight models easier to retrain). Butnaru et al. (Future Internet 2021) reports stable performance on PhishTank attacks without retraining. Both not re-read.
- No 2026 benchmark directly comparing PhishTank and OpenPhish time splits was found.

## Recommendations for our notes and targets

1. Do not compare URL-only numbers to PhreshPhish GTE rows as if equal inputs. Label every row URL-only or URL+HTML.
2. Use Table 4 base rates and P@R=0.90 plus AP as the headline. Add TPR at FPR 0.01% (URLTran convention) because it is browser-relevant.
3. Treat PhiUSIIL, UCI, PhishStorm, url_domain F1 0.999 and 2606.00889 as random-ceiling sanity rows only.
4. PhishLang is not a URL target. Its 74 MB is DRAM, not disk. Its size claim needs the model file size.
5. Read the URLTran PDF, the 2606.21690 PDF, and the NDSS PDF before citing their protocols.

## Gaps (summary)

- PhreshPhish v1 numbers not re-read this pass.
- No URL-only row in PhreshPhish; none from other authors found.
- URLTran dataset and protocol: unverified.
- URLNet, GramBeddings, PhishStorm primary numbers: not re-checked.
- No peer-reviewed 2026 small char-CNN with on-device latency and size found.
- Search-limited: no complete survey of 2026 citers of PhreshPhish.
