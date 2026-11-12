# Hinglish / Indic generalisation and independent evaluation for Rakshak Edge

Scope: research notes as of 2026-11-06. Searches were run with a small budget (about 15 tool calls), so this is a triage pass. Several claims are from background knowledge and are marked **[unverified]**. Where a paper was seen only through its abstract or a search snippet, that is stated.

## Q1. Small multilingual / Indic encoders and tokenisers for romanised Hindi

### Takeaway
mmBERT (2025, ICML 2026) is the most credible current small multilingual encoder found, and it is the natural teacher or student-init candidate. MuRIL and IndicBERT have Indic coverage but no shared code-mixed benchmark numbers were found. On-device size for each candidate is not verified, so the <50 MB target is still open.

### Cited Findings
- mmBERT: encoder-only, pretrained on 3T tokens across more than 1800 languages, built on ModernBERT (22 layers, intermediate dim 1152); reported XTREME average 72.8 for mmBERT base vs 70.4 for XLM-R — [arXiv 2509.06888](https://arxiv.org/html/2509.06888v1); accepted at ICML 2026 (PMLR 306) — [PMLR](https://proceedings.mlr.press/v306/marone26a.html)
- mmBERT base is described as up to 2-4x faster than earlier multilingual encoders — [Hugging Face blog](https://huggingface.co/blog/mmbert) (vendor/authors' own claim; numbers taken from search summary, not from the full paper)
- MuRIL was trained with translated and transliterated document pairs for Indian languages and evaluated on XTREME; the paper does not evaluate on code-mixed data — [arXiv 2103.10730](https://arxiv.org/abs/2103.10730v2)
- A LoRA-tuned MuRIL model card reports macro F1 0.7142 on 3-class code-mixed sentiment (about 20k held-out samples, Bengali-English and Hindi-English mixed); zero-shot MuRIL 0.3334 accuracy — [HF model card](https://huggingface.co/chaturvedipriyanshu17/muril-lora-codemixed-sentiment). Self-reported, not peer-reviewed, not a scam task.
- HingBERT-Mixed-v2 is MuRIL fine-tuned on L3Cube-HingCorpus, the most direct Hindi-English-specific MuRIL variant — [HF model card](https://huggingface.co/l3cube-pune/hing-mbert-mixed-v2); its baseline paper is arXiv 2204.08398 (not opened).
- arXiv 2509.02514 compares HingBERT/HingMBERT/HingRoBERTa against IndicBERT and MuRIL on Hindi-English NER (not classification; numbers not retrieved) — [arXiv 2509.02514](https://arxiv.org/html/2509.02514v1)
- GLUECoS includes English-Hindi code-mixed sentiment and NLI; it is the most established code-mixed classification benchmark found in search (numbers not retrieved) — cited in [search result list](https://www.wizwand.com/paper/68fa69b21e209131cea2384f) only as a pointer; primary source to confirm: GLUECoS (Khanuja et al., 2020) **[unverified, from background]**

### Inferences
- Ranked for the on-device constraint: (1) mmBERT base as teacher and as an init for a small student, since it covers Hindi and Romanised text through broad pretraining; (2) MuRIL/HingBERT as a Hinglish-specific alternative teacher for a second view in multi-teacher KD; (3) byte/char models (ByT5-small, CANINE) for robustness to homoglyph and leetspeak noise **[unverified, from background; not searched]**.
- Our current char/byte student is probably not weak on romanisation per se; the Hinglish drop (F1 about 0.41) is more likely from lexical coverage in training data. Encoder swaps alone are unlikely to fix it without data changes.
- Size: mmBERT's large multilingual vocabulary likely makes the embedding table the dominant size term. Quantised student size must be measured; no number was found **[unverified]**.

### Gaps
- No on-device (ONNX/INT8) size or latency numbers were found for mmBERT-small/base, MuRIL, IndicBERTv2, ByT5-small or CANINE.
- No IndicBERT or IndicBERTv2 code-mixed classification results were found.
- No peer-reviewed head-to-head of MuRIL vs mmBERT on romanised Hindi scam or spam text was found.

## Q2. Public Hinglish/Indic scam and spam SMS data, augmentation, and building an independent Indian test set

### Takeaway
There is no public, licensed, real-world Hinglish SMS scam corpus with clear provenance. The best new option is an anonymous 2026 EMNLP-submitted Zenodo artifact (1,393 Indian SMS test rows, Apache 2.0 code). Its labels come from an LLM-judge cleaning pipeline, so it is useful but not a fully independent gold standard. The fastest independent set is probably consent-based donation plus news-reported scam texts, with explicit labelling protocol and inter-annotator agreement.

### Cited Findings
- Zenodo record 20480151 (DOI 10.5281/zenodo.20480151, v4, published 31 May 2026): anonymous supplementary artifact for an EMNLP 2026 Industry Track submission; cleaned Indian SMS test set of 1,393 rows (Hindi Devanagari, Hinglish, English), with prob_scam scores; labels cleaned with a dual-LLM-judge pipeline; Fleiss and Krippendorff agreement 0.82 on a 276-row subset; code and redistributable data under Apache 2.0; IIITD SMSAssassin rows redistributed as hashes only — [Zenodo](https://zenodo.org/records/20480151)
- Hugging Face `bolewara/hinglish-scam-text-dataset`: 3,787 examples; Hinglish part is hand-written scam patterns (KYC expiry, Aadhaar blocking, OTP harvesting), English part drawn from the SMS Spam Collection — [HF](https://huggingface.co/bolewara/hinglish-scam-text-dataset). Synthetic in part; use only as augmentation or dev data, with a leakage check.
- "SpamShield" multilingual corpus, about 149k messages in 23 languages, including code-mixed and transliterated text. Source is a self-published blog post, not peer-reviewed; label scheme and licence unverified — [search snippet, blog.jmp/SpamShield listing](https://tech-for-dev.vercel.app/articles/3752102) (low trust)
- "Clues in Tweets" (arXiv 2204.01233): 21,918 SMS spam messages in 75 languages collected from Twitter — [arXiv 2204.01233](https://arxiv.org/pdf/2204.01233). Dataset availability and licence not checked.
- "An Overview of 7726 User Reports" (arXiv 2508.05276): of unique reported texts, 35.12% spam and 40.27% scam — [arXiv 2508.05276](https://arxiv.org/abs/2508.05276?context=cs). Useful for prevalence and taxonomy, not for a labelled corpus.
- IMC 2025 smishing study (Agarwal et al.): over 64.5k smishing image attachments and reports across 66 languages, mined from public forums — [UCL repository PDF](https://discovery.ucl.ac.uk/10214522/1/Agarwal_IMC_2025_Smishing.pdf). Report-based; text not parallel across languages.
- Aksharantar (AI4Bharat) licence is inconsistent across revisions. Current card: mined data CC0, manually transliterated data and benchmark CC-BY. An older revision lists CC BY-NC 4.0 in the YAML. Check the exact revision before any commercial use — [HF card](https://huggingface.co/datasets/ai4bharat/Aksharantar/blob/main/README.md?code=true), [AI4Bharat](https://indicnlp.ai4bharat.org/aksharantar/)
- Chakshu (Sanchar Saathi, DoT): a reporting channel for unsolicited commercial communication and spam via voice or SMS. The portal says it is not for financial fraud or cybercrime. Reports feed enforcement, and caller identity is shown from KYC data — [Sanchar Saathi portal](https://sancharsaathi.gov.in:443), [Deccan Herald](https://www.deccanherald.com/amp/story/india%2Fgovt-launches-portal-to-report-fraud-or-spam-callers-2921603). No official access, retention or data-sharing terms for individual reports were found; do not assume researcher access.

### Inferences
- Fastest credible independent Indian test set, in rank order: (1) donated SMS from consenting volunteers across several states and languages, labelled by two annotators with a written codebook and adjudication, reporting kappa; (2) SMS quoted in news reports on Indian scams, recorded with source and date; (3) the Zenodo 1,393-row set as a secondary check, not as the primary test set.
- Because the Zenodo set's labels are LLM-judge-cleaned, it is not independent of LLM labelling. Report it as a second opinion, and check overlap with our training data before use.
- Augmentation options (not validated in this session): transliteration with IndicXlit or Aksharantar for Devanagari-to-Roman, back-transliteration for noise, and LLM paraphrase of scam templates. LLM paraphrase raises template and synthetic-leakage risk; any synthetic train template must be excluded from the test campaigns by template-level split.

### Gaps
- No licensed real-world Hinglish SMS spam corpus with documented labelling protocol was found.
- The Zenodo artifact's full paper and its test-set overlap with IIITD or other training data were not read.
- IndicXlit licence not checked.
- No access route to Chakshu data was confirmed.

## Q3. Distillation that transfers across language shift

### Takeaway
Limited evidence in this session. The closest work supports multi-teacher or language-branch KD with teachers that cover the target language and with balanced data selection. No evidence was found for a sub-50 MB student on code-mixed SMS.

### Cited Findings
- Limisiewicz et al. (SIGTYP 2023): monolingual teachers plus balanced, sub-sampled data distilled into one multilingual student outperformed standard training on low-resource languages and kept high-resource performance — [ACL Anthology SIGTYP 2023 paper](https://preview.aclanthology.org/setup/2023.sigtyp-1.1) (cited from search summary: "You Can Have Your Data and Balance It Too", [arXiv listing](https://cs.paperswithcode.com/paper/you-can-have-your-data-and-balance-it-too))
- Liu and Shou, language-branch KD (arXiv 2010.14271): language-specific teachers jointly distill one multilingual student for cross-lingual reading comprehension — [arXiv 2010.14271](https://arxiv.org/pdf/2010.14271). Task is QA, not classification.
- A Zenodo report on multi-source vs single-source teacher ensembles for cross-lingual NER is autonomously generated; not relied on — [Zenodo 21280239](https://zenodo.org/record/21280239).

### Inferences
- Single-teacher KD with clean train-only targets lowers Hinglish F1 (as observed in our runs). A plausible mechanism: the student copies the teacher's English-centric decision boundary and never sees code-mixed inputs with soft labels. Fix to test: distil on unlabelled Hinglish and transliterated inputs (teacher-labelled), with multiple teachers (mmBERT plus a Hinglish-specific model) and per-language balancing.
- The hybrid's 35-39% false-positive rate on legitimate Indian transactional SMS points to missing hard negatives. Add OTP, bank debit/credit alerts, delivery and KYC-reminder SMS from consenting donors to the KD set, and report FPR on that set separately.
- Ranked options for the 2-week window: (1) add teacher-labelled transliterated and code-mixed unlabelled data plus hard negatives to the KD set; (2) multi-teacher KD (two teachers: mmBERT and a Hinglish-specific MuRIL variant) with per-language loss weighting; (3) intermediate-layer KD only if (1) and (2) fail, since it adds engineering cost **[unverified benefit, from background]**.

### Gaps
- No direct evidence found on intermediate-layer KD for code-mixed classification.
- No evidence found on KD data selection for classification under language shift with a model under 50 MB.

## Q4. Adversarial robustness evaluation for SMS: delivery-surviving perturbations and suites

### Takeaway
The September 2026 preprint "Johnny Still Receives Spam SMS" (arXiv 2609.01171) is the most relevant source. It restricts its perturbations to those that survive SMS or RCS delivery and reports that robustness transfers within a perturbation family but not across encoding-level attacks. Its full family list was not read; the abstract-level findings below need checking against the PDF before we rely on them.

### Cited Findings
- arXiv 2609.01171, "Johnny Still Receives Spam SMS: Assessing the Robustness of SMS Spam Detection" (Salman, Islam, Ikram, Kaafar; submitted 1 Sep 2026) — [arXiv](https://arxiv.org/abs/2609.01171):
  - Tests end-user anti-spam systems, third-party anti-spam services, and open-weight Hugging Face models under standard and adversarial conditions.
  - Includes only perturbations verified to survive real SMS or RCS delivery, not lab-only artifacts. The verification method is not described on the abstract page.
  - Robustness transfers well within a perturbation family but degrades sharply against structurally distinct, encoding-level attacks; adversarial training alone is insufficient.
  - Proposes a multi-model ensemble (adversarial training plus architecturally and tokenisation-diverse classifiers), with minority voting improving robustness.
  - The abstract page names neither the perturbation families nor the datasets. Those are in the full PDF, which was not read.
- ITU-T X.1238 (Dec 2025) gives guidance on spam over RCS messaging, a relevant standard for the delivery-survival argument — [ITU summary](https://www.itu.int/dms_pubrec/itu-t/rec/x/T-REC-X.1238-202512-I!!SUM-HTM-E.htm)
- Swiss NCSC reports that fraudsters use RCS to bypass SMS filters — [NCSC week 46 2024](https://www.ncsc.admin.ch/ncsc/en/home/aktuell/im-fokus/2024/wochenrueckblick_46.html)

### Inferences
- The suite should be organised by perturbation family and report a family-by-family matrix (train on one family, test on another), not a single averaged robustness number. The preprint's transfer finding means one average would hide the weak families.
- Include at least one encoding-level family (zero-width insertion, homoglyph substitution, leetspeak, URL shortener or percent-encoding) and one carrier-surviving family verified by a real send or a documented delivery pipeline. Note that canonicalisation specifically targets encoding-level attacks, so the suite should also test the canonicaliser's bypasses, not only the model.
- Judges will trust this more if the suite is pre-registered: list the families, the delivery-survival check, and the rule for including a perturbation before any results are seen.

### Gaps
- Full text of arXiv 2609.01171 not read; the perturbation list and delivery test are unknown.
- No TextAttack-style SMS-specific benchmark or homoglyph/leet/URL obfuscation benchmark with published numbers was found in this session.
- Carrier delivery testing requires real SIMs or a licensed SMS gateway; cost and feasibility not checked.

## Q5. Evaluation hygiene: splits, near-duplicate dedup, bootstrap CIs, and in-sample vs independent reporting

### Takeaway
Campaign- and template-disjoint splits with MinHash+LSH near-duplicate removal are standard and practical. Clustered bootstrap CIs and the statistics for in-sample vs independent reporting were not searched and rely on background knowledge.

### Cited Findings
- RETSim paper (arXiv 2311.17264): MinHash+LSH with 256 hash functions compared against SimHash and embedding-based methods on near-duplicate benchmarks; MinHash+LSH scored about 0.92 on one benchmark; MinHash stronger than the near-duplicate variant when one text is about 1.5x longer — [arXiv 2311.17264](https://arxiv.org/pdf/2311.17264)
- LessLeak-Bench (arXiv 2502.06215) used MinHash+LSH to detect overlap between pre-training data and benchmarks, a leakage-audit use — [arXiv 2502.06215](https://arxiv.org/pdf/2502.06215)
- The text-dedup repository provides MinHash and SimHash implementations and benchmarks — [text-dedup](https://socket.dev/pypi/package/text-dedup); datasketch tutorial on LSH index growth — [Yorko blog](https://yorko.github.io/2023/practical-near-dup-detection)

### Inferences
- Procedure to recommend: (1) cluster messages by MinHash+LSH at a fixed Jaccard threshold on normalised text (after canonicalisation, so obfuscated variants cluster together); (2) assign whole clusters to train, dev or test, and make campaign IDs the grouping unit where known; (3) report results with a cluster-level (campaign-clustered) bootstrap CI — standard practice **[unverified, from background; not searched]**; (4) label every result as in-sample (rules or thresholds designed on that data) or independent (no design contact), and report both.
- Because the Johnny preprint shows transfer within families only, the split should also be family-disjoint in one of the reported evaluations.

### Gaps
- No source found on clustered bootstrap CIs for text classification; statistical recommendation is background knowledge.
- No source on contamination checks specific to SMS classification.

## Ranked options and 2-week plan (synthesis)

Ranked by expected value per day of work, given free Kaggle compute and the on-device limit. Rankings are judgement, not measured results.

1. **Evaluation first (days 1-5):** build the independent Indian test set from consenting donors and news-reported scam SMS, with a codebook and kappa; add the Zenodo 1,393-row set as a secondary check after overlap checks; report in-sample and independent results separately; add MinHash+LSH campaign-clustered splits and clustered bootstrap CIs. Expected to matter most for judge trust.
2. **Adversarial suite (days 3-8):** family-by-family matrix including encoding-level and delivery-verified families, pre-registered; read arXiv 2609.01171 full text for its family list before finalising.
3. **Data for Hinglish (days 4-10):** teacher-labelled transliterated and code-mixed unlabelled data (IndicXlit or Aksharantar, check licence revision), plus hard negatives from legitimate Indian transactional SMS.
4. **KD fix (days 8-13):** two-teacher KD (mmBERT and a Hinglish-specific MuRIL variant) with per-language balancing; evaluate on the independent set and the adversarial matrix, and check on-device size by quantised ONNX export before committing.
5. **Lower priority:** intermediate-layer KD; byte/char encoders (ByT5, CANINE) as a robustness check if time remains **[unverified]**.

## Source dates and verification notes
- All dates are as shown in sources. Today is 2026-11-06.
- Johnny Still Receives Spam SMS: submitted 1 Sep 2026 (abstract page only).
- Zenodo 20480151: v4 published 31 May 2026 (anonymous, under review).
- mmBERT: arXiv 2025 (2509.06888), ICML 2026 proceedings.
- MuRIL: arXiv 2103.10730 (2021).
- Aksharantar licence: conflicting across revisions; verify the revision used.
