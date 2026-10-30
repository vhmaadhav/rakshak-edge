# SOTA targets for Rakshak-Edge

Compiled 2026-10-30 by a research worker. Scope: published numbers we must beat, per benchmark, with protocol. Nothing here is a result of ours.

Conventions:
- **VERIFIED** = paper/record confirmed via OpenAlex/arXiv lookup (title and DOI/arXiv id agree). Author lists were often not returned by the lookup, so "partial" means title + year + id matched.
- **UNVERIFIED** = number or claim came from a search snippet, a secondary citation, a model card's self-report, or an abstract only. Do not cite as a target until checked against the primary PDF.
- Numbers are copied as printed. Nothing is inferred unless marked "estimate".

---

## 1. UCI SMS Spam Collection

- Origin: Almeida, Hidalgo, Yamakami, "Contributions to the study of SMS spam filtering", ACM DocEng 2011. DOI 10.1145/2034691.2034742 (VERIFIED). https://doi.org/10.1145/2034691.2034742
- Size and balance: 5,574 messages; spam share commonly reported around 13% (UNVERIFIED, UCI page not fetched).
- Known issues: near-duplicate and template messages. Most published numbers use random stratified splits with no dedup.
- Reported numbers:
  - BERT, 90/10 random split, F1 99.28%. arXiv 2206.02443 "Spam Detection Using BERT" (VERIFIED existence; number from snippet, UNVERIFIED). https://arxiv.org/pdf/2206.02443
  - RoBERTa (Uddin et al., 2024a), accuracy and F1 both 0.9984. Cited in the 2026 systematic review arXiv 2604.11429 (VERIFIED existence of review; primary Uddin paper UNVERIFIED). https://arxiv.org/pdf/2604.11429
  - Hugging Face `sms_spam` BERT cards: acc 0.9904, F1 0.9636 and acc 0.989, F1 0.9575. Self-reported, split undescribed (UNVERIFIED). Not a paper.
- Protocol verdict: not leakage-controlled. Treat as a legacy sanity check only. Ceiling is about 0.99+, so it is not a useful headline.

## 2. Mishra & Soni Mendeley SMS phishing dataset (ham / spam / smishing)

- Dataset: "SMS Phishing Dataset for Machine Learning and Pattern Recognition", Mendeley DOI 10.17632/f45bkkt8pr.1, v1 published 20 June 2022. https://data.mendeley.com/datasets/f45bkkt8pr/1
- Size: 5,971 messages: 4,844 ham, 489 spam, 638 smishing (from listing text).
- Overlap: a 2026 review calls it a partial clone of UCI, sharing 4,753 instances (4,199 ham, 554 spam). Dedup against UCI before any combined training (secondary, UNVERIFIED).
- Licence: not confirmed in our sources.
- Original papers (both VERIFIED by DOI):
  - Smishing Detector, Future Generation Computer Systems 2020. DOI 10.1016/j.future.2020.03.021 (https://doi.org/10.1016/j.future.2020.03.021)
  - DSmishSMS, Neural Computing and Applications 2021. DOI 10.1007/s00521-021-06305-y (https://doi.org/10.1007/s00521-021-06305-y)
- Reported numbers (both UNVERIFIED, from secondary citations):
  - DSmishSMS accuracy 97.9% for smishing and spam combined (per a 2025 preprint).
  - SN Computer Science 2022 follow-up, neural network version, accuracy 97.40%.
- Protocol: split and dedup not described in our sources. Treat as weak.

## 3. Nahapetyan et al., IEEE S&P 2024 smishing measurement dataset

- Paper: "On SMS Phishing Tactics and Infrastructure", Nahapetyan, Prasad, Childs, Oest, Ladwig, Kapravelos, Reaves. IEEE S&P 2024. DOI 10.1109/SP54263.2024.00169 (VERIFIED). https://doi.org/10.1109/SP54263.2024.00169
- Data: 67,991 phishing messages over 396 days, 2,011 disposable numbers, 35,128 campaigns, over 600 operations (per search summary of the paper).
- Classifier results: none found in the accessible text. Full-text PDF (https://par.nsf.gov/servlets/purl/10581077) failed to load twice (socket closed). Status: UNVERIFIED whether it reports any classifier metric. It is primarily a measurement paper.
- Dataset release: not confirmed. Contact authors or check NC State project page (robocall.science, cited in one news piece, UNVERIFIED).
- Use: campaign structure for grouped (campaign-disjoint) splits, if released. No numeric target available.

## 4. Indian / Hinglish / Hindi / code-mixed SMS spam and scam benchmarks

No peer-reviewed Hinglish SMS spam benchmark with a leakage-controlled protocol was found. Candidates:

- **HinSpam** (Hugging Face, Keshav0av/HinSpam). Binary Hinglish spam/ham. Self-reported F1 0.949, accuracy 0.953, MCC 0.906. Dataset size, split and provenance not described. Cites a 2026 paper we could not confirm. UNVERIFIED. https://huggingface.co/Keshav0av/HinSpam
- **Hindi Spam SMS dataset**, NIT Silchar, IEEE DataPort. 3,894 messages, spam/ham, mostly Devanagari. No accepted model numbers found. https://ieee-dataport.org/documents/spam-sms-hindi-language. Corrected 2026-10-30: count VERIFIED (DOI 10.21227/5y8x-n678); access requires an IEEE DataPort subscription; no licence shown. Not downloaded.
- **Multilingual SMS spam dataset + CNN-LSTM hybrid**, CHRIST University repository (2022). Metrics exist, values not extracted. https://archives.christuniversity.in/items/show/20174
- **RupeeBee technical report**, BERT with attention, 94% accuracy. Not peer-reviewed. Low weight. https://rupeebee.vercel.app/doc/rupeebee-publication.pdf
- **Policy-aligned Indian SMS benchmark (Gupta, GroundLM 2026)**. Corrected 2026-10-30, was UNVERIFIED. Paper: ACL Anthology 2026.groundlm-1.79 (workshop volume, not ACL main; https://aclanthology.org/2026.groundlm-1.79/). 1,393 messages (Hindi, Hinglish, English-India); primary endpoint is FPR on 1,294 policy-negatives. Fine-tuned 270M XLM-R: 0.4% global FPR, 0.980 recall (abstract, verified). Fleiss kappa 0.821 over 276 overlap rows (supplement results, verified). Positive labels partly from GPT-4o and Gemini, so recall is secondary. Release: Zenodo 20480151 v4 (Apache-2.0; IIITD-sourced rows hash-only); ACL supplement is text-free. Accuracy by model and per-template results: see RESEARCH_NOTES section for 2026-10-30.
- **SmishX dataset** (SOUPS 2025, Wang et al.; https://github.com/yizhu-joy/SmishX, MIT). 1,200 manually relabelled messages: 622 legitimate, 319 spam, 259 smishing (downloaded and counted, verified 2026-10-30). Reported overall accuracy 98.8% and user study N=175 (verified in paper). SUS 82.6 not found in paper text (appears only in a talk description). Local copy: `data/raw/smishx_soups2025/dataset.csv`.
- **Indian fraud conversations, not SMS:** ICFD-31k (IJCAI-ECAI 2026, proceedings paper 0786; HF `rishia2220/icfd-31k`, licence `other`). Synthetic English/Hinglish phone-call transcripts, so not a real-fraud benchmark. Not downloaded.
- **Code-mixed proxy (not spam):** arXiv 2405.20755, Hindi-English code-mixed hate speech. MuRIL F1 0.54 best; XLM-R F1 0.52. Useful only as a difficulty indicator. https://arxiv.org/pdf/2405.20755
- **Resource:** L3Cube-HingCorpus and HingBERT (arXiv 2204.08398), Hindi-English Roman-script corpus and BERT models. Pretraining resource, not a spam benchmark. https://arxiv.org/pdf/2204.08398

## 5. PhreshPhish (arXiv 2507.10854), URL and HTML benchmark

- Paper: "PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark", OpenText authors. arXiv 2507.10854 (VERIFIED by arXiv id and title). v1 July 2025, v2 Feb 2026. https://arxiv.org/abs/2507.10854 · v2 HTML https://arxiv.org/html/2507.10854v2
- Data: HF `phreshphish/phreshphish`. https://huggingface.co/datasets/phreshphish/phreshphish
- Collection: about 17 months (Jul 2024 to Dec 2025). Phishing from PhishTank, APWG eCrime eXchange and Netcraft. Benign from anonymised Webroot telemetry and Google search results for heavily targeted brands. Each sample has a URL and its HTML.
- Size: 616,993 initial; 613,785 after cleaning (247,584 phishing, 366,201 benign).
- Protocol (v2, VERIFIED from text):
  - Temporal split: test set strictly later than train. Test fraction 0.3, stratified.
  - Leakage filter: test points with TF-IDF cosine distance below threshold to any train point dropped (LSH).
  - Dedup of canonicalised URLs and near-similar HTML in test.
  - Difficulty filter removes some easy phishing points (score > 0.85), 10% random drop among them.
  - Test base rate 46% phishing. Benchmarks subsampled to base rates 0.05%, 0.10%, 0.5%, 1.0%, 5%, plus the full 24% set. 975 benchmark instances in total.
  - Metrics: average precision (AP) and precision at 90% recall (P@R=0.90). Values with 95% bootstrap error.
- Baselines (v2 table, VERIFIED from text; mean values):

| Model | Metric | 0.05% | 0.10% | 0.5% | 1.0% | 5.0% | 24% (full) |
|---|---|---|---|---|---|---|---|
| GTE (gte-large fine-tuned, URL+HTML) | AP | 0.6050 | 0.6879 | 0.8354 | 0.8762 | 0.9382 | 0.9779 |
| GTE | P@R=0.90 | 0.0891 | 0.1563 | 0.3736 | 0.5426 | 0.8576 | 0.9746 |
| FFN (32-unit, n-gram + hand features) | AP | 0.2511 | 0.3368 | 0.5798 | 0.6813 | 0.8640 | 0.9607 |
| Linear (LinearSVC) | AP | 0.2230 | 0.3191 | 0.5825 | 0.6841 | 0.8610 | 0.9551 |
| Gemini-2.5-Pro zero-shot (URL+pruned HTML) | AP | 0.1003 | 0.1780 | 0.4839 | 0.6252 | 0.8383 | 0.9216 |
| Gemini-2.5-Pro zero-shot | P@R=0.90 | 0.0390 | 0.0722 | 0.2606 | 0.4147 | 0.7861 | 0.9352 |

- Caveat: GTE and the LLM use HTML. Our URL-only system must be compared on a URL-only variant or reported as a separate row. Do not claim to beat GTE with URL-only features unless we run the same split.
- **URL-only numbers: none published** (checked 2026-10-30 against the full v2 text, read via WebFetch in two passes; a summariser, not the PDF tables). Table 2 and the appendix have four rows only: Linear (LinearSVC), FFN (32-unit), GTE and the Gemini-2.5-Pro zero-shot LLM. The paper says Linear and FFN take character n-grams of both URL and HTML, and the LLM prompt holds the URL plus pruned HTML. So there is no URL-only target to beat. Our URL-only students (kaggle/jobs/url_student.py) are reported against the GTE row for reference only, and always labelled URL-only, not as a like-for-like win.
- Caveat (corrected 2026-10-30): v1 numbers now checked, and they differ from v2. v1 GTE (arXiv 2507.10854v1, Table 2, read via WebFetch summary of the v1 HTML): 0.05% AP 0.7771 ± 0.0128, P@R=0.90 0.2222 ± 0.0296; 5.0% AP 0.9715; full test set (18.3% base rate in v1) AP 0.9918, P@R 0.9855. Do not mix v1 and v2 rows. v2 remains the reference.

## 6. Other URL benchmarks

- **URLTran**, Maneriker et al., MILCOM 2021. DOI 10.1109/milcom52596.2021.9653028 (VERIFIED). arXiv 2106.05256 (VERIFIED). https://arxiv.org/abs/2106.05256
  - Reported: TPR 86.80% at FPR 0.01%, vs URLNet 71.20% (own evaluation, from search summary; UNVERIFIED against paper).
  - Adversarial: homoglyph and compound-word attacks plus parameter reordering. Adversarial fine-tuning reportedly keeps low FPR. No independent replication.
- **URLNet**, Le, Pham, Sahoo, Hoi. arXiv 1802.03162 (VERIFIED). https://arxiv.org/abs/1802.03162 Reported numbers not extracted (UNVERIFIED). Dataset about 15M URLs (secondary).
- **ISCX-URL2016**, Mamun et al., LNCS 2016. DOI 10.1007/978-3-319-46298-1_30 (VERIFIED). https://doi.org/10.1007/978-3-319-46298-1_30
  - Subsets vary widely. Over 45,000 instances (35,378 legit, 9,965 phishing); balanced subsamples are common.
  - Random forest on 9 lexical features, 99.57% on 11,964 examples; deep model 98.95% accuracy, 95.10% recall. Both UNVERIFIED, not comparable.
  - A 2025 dataset paper calls it outdated and small.
- **PhishStorm**, Marchal et al., IEEE TNSM 2014. DOI 10.1109/tnsm.2014.2377295 (VERIFIED). https://doi.org/10.1109/tnsm.2014.2377295 Reported 94.91% correct classification, FPR 1.44%, on 96,018 URLs (from snippet, UNVERIFIED against paper). Feature-engineered, not deep.
- **Ebbu2017**: listed as GitHub repo `ebubekirbbr/pdd`, not Mendeley. Contributes about 25,874 active URLs to Mendeley "Phishing Websites Dataset" (DOI 10.17632/n96ncsr5g4.1, 30,000 phishing / 50,000 legit with HTML). No peer-reviewed SOTA numbers found. https://data.mendeley.com/datasets/n96ncsr5g4
- **GramBeddings**, Bozkır, Dalgıç, Aydos. Computers & Security 2022. DOI 10.1016/j.cose.2022.102964 (VERIFIED). https://doi.org/10.1016/j.cose.2022.102964
  - Reported accuracy 98.27% on an 800,000-URL balanced set (journal abstract via search; UNVERIFIED against primary).
  - Live-PhishTank case studies show poor generalisation: PMANet (arXiv 2311.12372) 20 URLs, GramBeddings 75%; TransURL (arXiv 2312.00508) 30 URLs, GramBeddings 76%, URLNet 86%. Small samples. https://arxiv.org/abs/2311.12372 · https://arxiv.org/abs/2312.00508
- **Mendeley phishing URL datasets (2024)**:
  - Almujahid, Haq, Alshehri, "Comparative evaluation of machine learning algorithms for phishing site detection", PeerJ Computer Science 2024. DOI 10.7717/peerj-cs.2131 (VERIFIED). CNN best on Mendeley and UCI datasets with SMOTE. Numbers not extracted (UNVERIFIED). https://doi.org/10.7717/peerj-cs.2131
  - Zieni dataset, Mendeley 10.17632/8mcz8jsgnb.1, balanced 5,000 phishing / 5,000 legit, 74 features. https://data.mendeley.com/datasets/8mcz8jsgnb/1
  - Tamal et al., Frontiers in Computer Science 2024, DOI 10.3389/fcomp.2024.1308634. Dataset 247,950 URLs (128,541 phishing, 119,409 legit) (dataset size from snippet, UNVERIFIED).
- **PhiUSIIL**, Prasad & Chandra, Computers & Security 2023. DOI 10.1016/j.cose.2023.103545 (VERIFIED). UCI lists 2024 (year discrepancy). https://doi.org/10.1016/j.cose.2023.103545 · https://archive.ics.uci.edu/dataset/967
  - 235,795 instances: 134,850 legit, 100,945 phishing. 54 features from page source and URL (content-derived, not URL-only).
  - The paper's own numbers were not extracted.
  - Downstream (secondary, UNVERIFIED): Acta Infologica Decision Tree 99.7%; BEEI Dec 2025 BiGRU 99.82% under 60:40 split (random, likely). https://beei.org/index.php/EEI/article/download/10099/4503
  - Cross-dataset: TCN trained on GramBeddings 800k, tested on PhiUSIIL, 81% recall (Electronics 2025, UNVERIFIED).
  - Verdict: near-ceiling in-distribution. Not a realistic target. Report as a sanity row only.
- **COPS**, arXiv 2402.04173, IEEE CCNC 2024. DOI 10.1109/ccnc51664.2024.10454635 (VERIFIED). Accuracy 98.15% smishing, 99.5% URL phishing (abstract; attribution by order). FNR 0.037 and FPR 0.015, task attribution unclear. Datasets not recorded. https://arxiv.org/abs/2402.04173
- Recent character-level work (arXiv 2512.16717, Dec 2025, char-CNN + LightGBM ensemble, 70-symbol vocab, max length 200). Numbers and protocol not extracted (UNVERIFIED). https://arxiv.org/abs/2512.16717

## 7. Robustness and adversarial evasion

- **"Johnny Still Receives Spam SMS: Assessing the Robustness of SMS Spam Detection"**, Salman, Islam, Ikram, Kaafar. arXiv 2609.01171, submitted 1 Sep 2026 (VERIFIED existence by arXiv id and title; preprint, not peer-reviewed). https://arxiv.org/abs/2609.01171 · HTML https://arxiv.org/html/2609.01171
  - Corrected 2026-10-30: EMNLP 2026 acceptance NOT FOUND (arXiv v1 only; no venue line). Do not describe as accepted.
  - Corrected 2026-10-30, delivery rule (Table 3): invisible-character and homoglyph perturbations survive SMS and RCS delivery; RTLO bidi reordering does not and is excluded from the threat model (Table 2 footnote). The abstract also evaluates commercial messaging apps and third-party anti-spam services (not only BERT).
  - Data: "Super SMS" dataset, 67,010 messages (40,837 benign, 26,181 spam). Sources include UCI SMS, NUS SMS, SpamHunter, X, ScamWatch, UK Action Fraud.
  - Test: 5,000 benign and 5,000 spam, random. Training is the rest. Random split, no temporal split stated.
  - Cross-dataset: Smishtank (913 after dedup, all spam) BERT recall 99.67%.
  - Attacks: perceptible (Spacing, Insert, Delete, Sub-Chars, Sub-Word, Swap) and imperceptible (Invisible, Homoglyphs, Reordering). Reordering excluded from threat model.
  - Key numbers (accuracy %, BERT not adversarially trained, Table 12): clean 99.0; Spacing 94.1; Insert 96.9; Delete 95.2; Sub-Chars 96.4; Sub-Word 97.2; Swap 95.9; Invisible 94.9; **Homoglyphs 41.3**; Reordering 86.5.
  - BERT, full adversarial training: Homoglyphs 99.5, Invisible 94.6, clean 97.9.
  - BERT, minimal AT (Insert + Invisible only): Homoglyphs 79.7, clean 99.7.
  - Open-weight HF spam models under Homoglyphs: 17.6% to 70.9%. (UNVERIFIED; not re-checked.)
  - Corrected 2026-10-30: the "RoBERTa 99.7% clean to 17.6%" claim is PARTIAL. Table 9 gives maria_roberta 17.6% under Invisible and Homoglyphs, but the text says "spacing and invisible characters" while Table 9 lists spacing 58.7%. The 99.7% clean value was not found in the text read.
  - Corrected 2026-10-30: BERT clean 99.33% acc / 0.30% FPR is Table 11; Table 12 uses 99.0. Minimal AT Table 15: homoglyph 79.7, invisible 98.7 (clean 99.7 not confirmed).
  - Inconsistency in the source: BERT clean is 99.33 in Table 11 and 99.0 in Table 12. SVM is 95.23 in Table 11 and 76.8 in Table 12. Use the full PDF to confirm before citing.
  - The paper reports accuracy under attack only, not attack success rate.
  - Section 7 (proposed ensemble) was truncated in our fetch. Its numbers are UNVERIFIED.
- **Samsung on-device smishing classifier**, IEEE Access, Jan 2024. DOI 10.1109/ACCESS.2024.3349577 (VERIFIED via DOI; authors not confirmed). https://doi.org/10.1109/ACCESS.2024.3349577 · https://research.samsung.com/research-papers/On-device-Smishing-Classifier-Resistant-to-Text-Evasion-Attack
  - Reported: about 0.99 accuracy, model about 127 kB; attack success rate 0.41 against variant smishing messages; EVA black-box generator; adversarial training improves robustness (all from abstract/secondary, UNVERIFIED).
  - Korean SMS data: about 250,000 smishing and 950,000 normal messages (UNVERIFIED).
  - Note: the claim that 98% of smishing messages are variants of earlier ones is from the secondary source (UNVERIFIED).
- **URLTran** adversarial results: see section 6. Not independently replicated.
- **Systematic review, arXiv 2604.11429**, "Short Message Service (SMS) Phishing Attacks and Defenses: A Systematic Review" (VERIFIED title and existence). Good index of datasets. Its comparison tables were not readable from the PDF in our tool. https://arxiv.org/pdf/2604.11429

## 8. Small and on-device models

| System | Size | Accuracy / metric | Source | Status |
|---|---|---|---|---|
| PhishLang (MobileBERT) | 74 MB **memory** (model size not stated) | acc 0.96, P 0.95, R 0.96, F1 0.96; 0.39 s median prediction | arXiv 2408.05667 (VERIFIED); NDSS 2026 DOI 10.14722/ndss.2026.231037 (VERIFIED) | Table from arXiv v3 via search; NDSS PDF unreadable. Abstract: 94.7% accuracy on evasive attacks across five threat families. Dataset and split not extracted. Code: github.com/uta-sprlab/phishlang |
| PhishLang DistilBERT | 502 MB memory | acc/P/R/F1 0.94; 0.85 s | same | UNVERIFIED beyond search summary |
| COPS | 3.46 MB | 98.15% smishing, 99.5% URL | arXiv 2402.04173 / CCNC 2024 | Abstract; dataset not recorded |
| EI-BERT (cross-distillation) | 1.91 MB (added 2026-10-30) | accuracy not extracted | arXiv 2507.04636; ACM DOI 10.1145/3711896.3737257 | Size VERIFIED; KDD 2025 venue PARTIAL (not matched on kdd.org) |
| Samsung IEEE Access | about 127 kB | about 0.99 accuracy | IEEE Access 2024 | Secondary; UNVERIFIED |
| Agentic KD (students) | Qwen2.5-0.5B; SmolLM2-135M | best 94.31% acc, 96.25% recall | arXiv 2602.10869 (VERIFIED title) | Validation set is teacher-generated synthetic, not real. Weak target |
| URLNet char-CNN | not stated | not extracted | arXiv 1802.03162 | UNVERIFIED |

Backbone sizes (from model cards we fetched):
- **mDeBERTa-v3-base** (microsoft/mdeberta-v3-base): 86M backbone + 190M embedding params; 12 layers; hidden 768; vocab 250K; MIT. https://huggingface.co/microsoft/mdeberta-v3-base
- **IndicBERTv2-MLM-only** (ai4bharat): card states 278M parameters; 23 Indic languages + English; IndicCorp v2, 20.9B tokens; MIT. Layers and hidden size not on card. https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only
- **MuRIL base cased** (google/muril-base-cased): card says BERT-base architecture, 17 languages including transliterated, Apache-2.0. Parameter count not on card. Secondary source gives about 236M params with vocab 197,285, which matches base architecture plus that vocab (our arithmetic: about 86M + 151M embedding). The secondary's "24 layers, hidden 1024" conflicts with "BERT base" and is likely wrong. https://huggingface.co/google/muril-base-cased
- **ModernBERT-base**: 149M params, 22 layers, 8,192 context, 2T tokens of English and code; limited non-English. https://huggingface.co/answerdotai/ModernBERT-base
- **mmBERT** (added 2026-10-30; PARTIAL): `jhu-clsp/mmBERT-small` 140M total / 42M non-embedding; `jhu-clsp/mmBERT-base` 307M total / 110M non-embedding; both MIT, ModernBERT architecture, 1,800+ languages (HF search snippets; the repo overview did not print sizes). Paper arXiv 2509.06888 (title verified). "ICML 2026" seen only as a note on the HF collection page. https://huggingface.co/collections/jhu-clsp/mmbert-a-modern-multilingual-encoder
- **multilingual-e5-small**: card fetch failed. Size UNVERIFIED.
- **Qwen2.5-0.5B, SmolLM2-135M**: names imply 0.5B and 135M (UNVERIFIED on cards).

Distillation arithmetic (estimate, not measured):
- Under 20 MB at INT8 is about 20M parameters total. Embeddings dominate; a 250K-vocab multilingual embedding alone is about 190M params, far over budget.
- Prune the vocabulary to the tokens observed in our corpus, e.g. 30K x 384 = 11.5M params, about 11.5 MB at INT8, before the encoder.
- Hashed char n-gram or fastText-style student: around 1 to 5 MB. Char-CNN URL student: around 1 to 3M params.
- Byte-level teachers (CANINE, ByT5) are too large to ship; use them as teachers only.

## Candidate model families (2026)

SMS and code-mixed text:
- Teacher: mDeBERTa-v3-base (278M, MIT), IndicBERTv2 (278M), MuRIL (about 236M, Apache-2.0), XLM-R base/large. Fine-tune on pooled deduplicated SMS corpus, 5 seeds.
- LLM teacher for pseudo-labels: Qwen2.5 or Llama instruct (7B-class), used only for labelling and hard-case mining, with a human-audited subset. Agentic KD (2602.10869) shows teacher choice matters a lot, so compare at least two teachers.
- Student: char n-gram LR (baseline), fastText-style hashed bigrams, char-CNN, 2-layer BiGRU. Distil with soft labels and temperature. Target below 5 MB INT8.
- Modern encoder ModernBERT: English and code only, so not a Hinglish candidate.

URLs:
- Student: URLNet-style char and word CNN (1 to 3M params), with lexical features and a calibrated LR/GBM fusion.
- Teacher: char-level transformer or gte-large in URL-only mode. Byte-level CANINE and ByT5 as teacher comparisons, not deployable.
- URLTran-style transformer as an upper-bound reference.

## Target to beat

| Benchmark | Metric | Target number | Source | Protocol / status |
|---|---|---|---|---|
| UCI SMS Spam | F1 / acc | 0.9928 F1 (BERT, 90/10); 0.9984 acc and F1 (RoBERTa) | arXiv 2206.02443; arXiv 2604.11429 (secondary) | Random, no dedup. Legacy only |
| Mishra & Soni Mendeley | acc | 97.9% (DSmishSMS); 97.40% (2022 NN) | NCA 2021 / SN CS 2022 (secondary) | Protocol unknown; overlaps UCI |
| Nahapetyan S&P 2024 | classifier metric | none found | DOI 10.1109/SP54263.2024.00169 | Measurement paper; full text unreadable here |
| Hinglish SMS spam | F1 / acc / MCC | 0.949 / 0.953 / 0.906 (HinSpam) | HF card, self-reported | UNVERIFIED; no peer-reviewed benchmark |
| Indian financial SMS, policy-aligned (Gupta, GroundLM 2026) | FPR / recall | 270M XLM-R: 0.4% global FPR, 0.980 recall, on 1,294 policy-negatives (added 2026-10-30) | ACL Anthology 2026.groundlm-1.79 (abstract verified) | Workshop paper; LLM-adjudicated positives, so recall secondary; FPR is the primary endpoint |
| PhreshPhish (URL+HTML) | AP / P@R0.90 | 0.05%: 0.6050 / 0.0891; 1%: 0.8762 / 0.5426; 5%: 0.9382 / 0.8576 (GTE, v2). Corrected 2026-10-30: v1 gives 0.7771 / 0.2222 at 0.05%, do not mix | arXiv 2507.10854 v2 (Table 2 checked against v2 HTML) | Temporal, dedup, leakage filter. HTML-based; URL-only needs own row |
| URLTran | TPR at FPR 0.01% | 86.80% | MILCOM 2021, DOI 10.1109/milcom52596.2021.9653028 | Authors' own; not replicated |
| GramBeddings | acc | 98.27% (800k balanced) | Comput. & Security 2022 (UNVERIFIED primary) | Random, in-distribution; live-URL tests 75-76% |
| PhishStorm | acc / FPR | 94.91% / 1.44% | IEEE TNSM 2014 (snippet, UNVERIFIED) | 96,018 URLs; older |
| PhiUSIIL | acc | 99.82% (BiGRU 60:40); 99.7% (DT, secondary) | Comput. & Security 2023 (DOI verified); downstream UNVERIFIED | Content features; random ceiling, not a headline |
| COPS | acc | 98.15% smishing; 99.5% URL | CCNC 2024, DOI 10.1109/ccnc51664.2024.10454635 | Abstract only; dataset unknown |
| Johnny (robustness) | acc under Homoglyph attack | BERT 41.3 (no AT) to 99.5 (full AT); mini AT 79.7 | arXiv 2609.01171 (preprint; EMNLP 2026 NOT FOUND, corrected 2026-10-30) | Random split; accuracy only; table inconsistencies |
| Samsung on-device | acc / size | about 0.99 at about 127 kB; ASR 0.41 | IEEE Access 2024 (secondary) | UNVERIFIED |
| PhishLang (small) | acc / F1 / memory | 0.96 / 0.96 at 74 MB memory (MobileBERT) | arXiv 2408.05667; NDSS 2026 | Dataset not extracted |
| Agentic KD (small) | acc / recall | 94.31% / 96.25% (Qwen2.5-0.5B student) | arXiv 2602.10869 | Synthetic validation; weak |

## Recommended experiment plan (teacher then student)

- Build leakage-controlled splits first: normalise text, exact and MinHash near-dup removal across UCI, Mishra & Soni, Smishtank and HinSpam; cluster into campaigns for grouped splits; add a temporal hold-out.
- Reproduce PhreshPhish on the HF release in URL-only mode at 0.05%, 0.5%, 1% and 5% base rates. Report AP and P@R0.90 against GTE (0.6050, 0.8354, 0.8762, 0.9382 AP). Label any HTML-using row separately.
- Pull the Nahapetyan dataset if released. Use its campaign IDs as the group key. If not released, state that clearly and do not claim a comparison.
- SMS teachers: fine-tune mDeBERTa-v3-base, IndicBERTv2 and MuRIL on the deduplicated pool, 5 seeds each. Report macro-F1, PR-AUC, FPR at 90/95% recall, ECE.
- Code-mixed set: author about 1,000 Hinglish and code-mixed messages, two annotators, report kappa. This is the held-out test for the English to Hinglish shift.
- LLM pseudo-labels: use one open 7B-class instruct model and one API model for hard cases. Audit a sample by hand. Compare the two as teachers, as in 2602.10869.
- Students: char n-gram LR, fastText-style hashed bigrams, char-CNN and 2-layer BiGRU. Distil from teacher soft labels with temperature. Prune vocab to the corpus. Keep INT8 size under 5 MB for the message expert and under 3 MB for the URL expert.
- URL expert: URLNet-style char and word CNN, distilled from a URL-only transformer teacher trained on PhreshPhish.
- Robustness: run the Johnny attack suite (spacing, insert, delete, substitute, swap, invisible, homoglyph) plus our own generator. Report accuracy and attack success rate with and without the canonicaliser. Target: homoglyph and invisible accuracy at or above the full-AT BERT (99.5, 94.6).
- Fusion: LR or small GBM over expert scores; calibrate on held-out campaigns (temperature or isotonic); report coverage-vs-error for the abstain state.
- Ablations as in CONTEXT.md, plus teacher size and distillation temperature.
- Report PhiUSIIL and UCI only as legacy sanity rows, never as headline.
- Run all training and evaluation on Kaggle per CONTEXT.md. Laptop only for code and unit tests.
- Device measurement later on Pi 5 CPU and Android (latency, RSS, energy). Record model size as the on-disk artifact size, not memory.
- Open verification items before any citation: Nahapetyan classifier results (retry full PDF); Samsung IEEE Access full text; PhishLang NDSS PDF table; Johnny Tables 11 vs 12 and Section 7; URLTran dataset details; GramBeddings primary numbers; multilingual-e5-small card; Mishra & Soni primary accuracy values.

## Source list (key)

- PhreshPhish: https://arxiv.org/abs/2507.10854 · https://huggingface.co/datasets/phreshphish/phreshphish
- Nahapetyan S&P 2024: https://doi.org/10.1109/SP54263.2024.00169 · https://par.nsf.gov/servlets/purl/10581077
- Almeida UCI (DocEng 2011): https://doi.org/10.1145/2034691.2034742
- Mishra & Soni: https://data.mendeley.com/datasets/f45bkkt8pr/1 · https://doi.org/10.1016/j.future.2020.03.021 · https://doi.org/10.1007/s00521-021-06305-y
- Johnny: https://arxiv.org/abs/2609.01171
- Samsung: https://doi.org/10.1109/ACCESS.2024.3349577
- COPS: https://arxiv.org/abs/2402.04173
- PhishLang: https://arxiv.org/abs/2408.05667 · https://www.ndss-symposium.org/ndss-paper/phishlang-a-real-time-fully-client-side-phishing-detection-framework-using-mobilebert
- URLTran: https://arxiv.org/abs/2106.05256
- URLNet: https://arxiv.org/abs/1802.03162
- PhishStorm: https://doi.org/10.1109/tnsm.2014.2377295
- ISCX-URL2016: https://doi.org/10.1007/978-3-319-46298-1_30
- GramBeddings: https://doi.org/10.1016/j.cose.2022.102964
- Almujahid PeerJ CS 2024: https://doi.org/10.7717/peerj-cs.2131
- PhiUSIIL: https://doi.org/10.1016/j.cose.2023.103545
- Agentic KD: https://arxiv.org/abs/2602.10869
- Systematic review: https://arxiv.org/pdf/2604.11429
- Hinglish: https://huggingface.co/Keshav0av/HinSpam · https://zenodo.org/records/20480151 · https://arxiv.org/pdf/2405.20755
- Model cards: https://huggingface.co/microsoft/mdeberta-v3-base · https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only · https://huggingface.co/google/muril-base-cased · https://huggingface.co/answerdotai/ModernBERT-base
