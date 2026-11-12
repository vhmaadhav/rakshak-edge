# SMS spam / smishing / Indian financial-SMS SOTA, as of 2026-11-07

Scope: verify and update /home/maadhav/Documents/Development/rakshak-edge/docs/SOTA_TARGETS.md (compiled 2026-10-30). Four key questions below. Method: this session read ACL, Zenodo, arXiv abstracts and the Johnny HTML through a summariser (WebFetch), not PDF tables, unless marked. Search-snippet-only items are marked UNVERIFIED.

Conventions: VERIFIED = read this session from the primary page (summariser output, not PDF-checked unless stated). UNVERIFIED = search snippet, secondary source, or carried from SOTA_TARGETS.md without re-check this session. CONFLICT = sources disagree.

## Q1. Legacy and public SMS/smishing benchmarks (UCI, Mishra & Soni, Nahapetyan, SmishX, synthetic sets)

### Best-result table (all benchmarks, verified or flagged)

| Benchmark | Best published number | Protocol (split / dedup / metric / model size) | Status |
|---|---|---|---|
| UCI SMS Spam (5,574 msgs) | F1 0.9928 (BERT, arXiv 2206.02443); acc 0.9984, F1 0.9984 (RoBERTa, Uddin et al. 2024a, cited in arXiv 2604.11429) | Random 90/10, no dedup. BERT-base size | UNVERIFIED (snippet). Legacy only, ceiling near 0.99 |
| Mishra & Soni Mendeley (5,971 msgs; 4,844 ham / 489 spam / 638 smishing) | 97.9% acc (DSmishSMS, NCA 2021); 97.40% (SN CS 2022) | Split and dedup not described. Overlaps UCI (4,753 instances shared, per a 2026 review) | UNVERIFIED (secondary). Weak |
| Nahapetyan et al., IEEE S&P 2024 (DOI 10.1109/SP54263.2024.00169) | No classifier metric found | Measurement paper: 67,991 phishing messages, 35,128 campaigns, 600+ operations; extracted from 200M+ SMS on 11 public gateways | VERIFIED metadata via search summary; classifier: none found. Dataset release: UNVERIFIED (footnote link not read) |
| SmishX (SOUPS 2025, github yizhu-joy/SmishX, MIT) | 98.8% overall accuracy; user study N=175 | 1,200 relabelled msgs: 622 legit / 319 spam / 259 smishing (counted locally in SOTA_TARGETS) | Carried from SOTA_TARGETS; not re-read this session |
| Samsung on-device smishing (IEEE Access, Jan 2024, DOI 10.1109/ACCESS.2024.3349577) | ~0.99 acc; model ~127 kB; attack success rate 0.41 vs variant smishing | Korean SMS: ~250,000 smishing / ~950,000 normal. Split not read | Metadata VERIFIED (DOI, venue); numbers UNVERIFIED (search summary of abstract). Author list "Jae Woo Seo, Jong Sung Lee et al." is from search only |
| Johnny Still Receives Spam SMS (arXiv 2609.01171) | Clean BERT ~99.0 to 99.33 (table-dependent); full-AT BERT: Homoglyph 99.5, Invisible 94.6; minimal-AT: Homoglyph 79.7 | Super SMS 67,010 (stated; see CONFLICT). Random split. Accuracy only | VERIFIED (HTML, summariser). Internally inconsistent tables (Q4) |

### Takeaway
No SMS spam or smishing benchmark has a leakage-controlled (dedup plus temporal or campaign-grouped) protocol with a published headline number. The legacy UCI/Mishra numbers are random-split and saturated at 0.99+. Nahapetyan is a measurement paper with no published classifier metric in the sources read. Our only defensible SMS comparison remains the Indian-financial benchmark (Q2) and the robustness suite (Q4).

### Cited Findings
- UCI origin: Almeida, Hidalgo, Yamakami, DocEng 2011, DOI 10.1145/2034691.2034742 (carried from SOTA_TARGETS; not re-read this session).
- Nahapetyan metadata (67,991 messages; 35,128 campaigns; 600+ operations; 200M+ SMS scanned on 11 gateways): [DOI 10.1109/SP54263.2024.00169](https://doi.org/10.1109/SP54263.2024.00169), via a search summary. Full PDF mirror [par.nsf.gov](https://par.nsf.gov/servlets/purl/10581077) was not read this session.
- Samsung abstract figures (0.99 accuracy, 127 kB, ASR 0.41, ~250k smishing / ~950k normal Korean SMS, claim that 98% of smishing messages are variants of earlier ones): [Samsung Research page](https://research.samsung.com/research-papers/On-device-Smishing-Classifier-Resistant-to-Text-Evasion-Attack); [DOI](https://doi.org/10.1109/ACCESS.2024.3349577). Source for numbers is a search summary (UNVERIFIED).
- Systematic review of SMS phishing (arXiv 2604.11429): [HTML](https://arxiv.org/html/2604.11429). The dataset and classifier comparison tables (Section 6) were not reached in our read; the fetch covered only the first 100k of 387k characters. CONFLICT: none found yet. Gap, see below.
- Synthetic conversational smishing dataset, 3,201 conversations (arXiv 2604.11752): [abs](https://arxiv.org/abs/2604.11752). Synthetic, so not a real-fraud benchmark. Its successor COVA-X (arXiv 2606.06879, 10,985 conversations, eight elder-scam categories): [papers.cool](https://papers.cool/arxiv/2606.06879). Not verified beyond the search listing.
- Gravity Falls (arXiv 2603.03270, March 2026): semi-synthetic C2 domains derived from SMS messages collected 2022 to 2025; benchmarks DGA detectors, not SMS classification. [HTML](https://arxiv.org/html/2603.03270v1). Zenodo doi:10.5281/zenodo.17624554 (from search).

### Inferences
- Any claim to beat UCI or Mishra & Soni is not a meaningful claim. Report them only as legacy sanity rows.
- Samsung and Nahapetyan are the closest real-SMS operational datasets. Neither has a public split protocol we can reproduce from our reads.

### Gaps
- Mishra & Soni primary paper accuracies (DSmishSMS, 2022 NN): not read; UNVERIFIED.
- Nahapetyan: classifier results and dataset release status unconfirmed. Next step: read the NSF PDF footnote, or contact the authors.
- Samsung IEEE Access full text not read; split and dedup unknown.
- 2604.11429 Section 6 comparison tables not read (need pages 100k to 387k).
- SmishX: not re-read this session.

---

## Q2. Indian financial SMS: GroundLM 2026 (Gupta) and related benchmarks

### Takeaway
The ACL paper is verified as a sole-author workshop paper (GroundLM 2026, Budapest, pp. 1124–1136). Its primary endpoint is FPR on 1,294 policy-negatives, and the fine-tuned XLM-R result (0.4% global FPR, recall 0.980) is confirmed as a claim. The Zenodo record is a different, anonymised EMNLP 2026 Industry supplement with a different title and no headline metrics. No other Indian SMS fraud benchmark with numbers and a documented protocol was found.

### Cited Findings
- ACL Anthology page (read via summariser): title "Default Zero-Shot LLM APIs Over-Flag Indian Financial SMS: A Policy-Aligned Benchmark and Template-Level Pre-Deployment Audit"; sole author Sachin Gupta; GroundLM 2026 (1st Workshop on Grounding Language Models), Budapest, Oct 2026, pp. 1124–1136; DOI 10.18653/2026.groundlm-1.79. [ACL Anthology](https://aclanthology.org/2026.groundlm-1.79/) · [DOI](https://doi.org/10.18653/2026.groundlm-1.79)
- Dataset: 1,393 messages (Hindi, Hinglish, English-India). Primary endpoint: FPR on 1,294 policy-negatives = 1,222 untouched source negatives + 72 source positives converted to "safe". Arithmetic check: 1,222 + 72 = 1,294. [ACL](https://aclanthology.org/2026.groundlm-1.79/)
- Fine-tuned 270M XLM-R: 0.0% FPR on Bank and UPI templates, 0.4% global FPR, 0.980 recall. [ACL](https://aclanthology.org/2026.groundlm-1.79/). 0.4% of 1,294 is about 5 false positives (5/1,294 = 0.39%, arithmetic, not stated by source). Recall denominator not stated.
- GPT-4o with policy definitions and seven examples: 0.0% Bank/UPI FPR, 1.5% global FPR, 0.949 recall. Claude (model unnamed): 1.8% FPR at 0.960 recall. GPT-4o-mini: 77.3% and 56.9% FPR (Bank/UPI mapping not explicit). Six LLM APIs exceed a "1% FPR gate" on Bank and UPI under a minimal prompt. [ACL](https://aclanthology.org/2026.groundlm-1.79/)
- Label protocol: source labels are separated from fraud/non-fraud/unclear adjudication and from allow/review/block actions. GPT-4o and Gemini helped adjudicate source-positive messages, so positive labels depend partly on LLMs. Authors state positive metrics (recall, F1) are secondary. [ACL](https://aclanthology.org/2026.groundlm-1.79/)
- Zenodo 20480151 (read via summariser): "Anonymous Supplementary Artifact, EMNLP 2026 Industry Track Submission (Multilingual Indian SMS Scam Detection)". v4, published 31 May 2026; supersedes v3 (record text gives DOI 10.5281/zenodo.20478597 for v3; v3 "should not be used for review"). Apache-2.0. Authors "Anonymous". [Zenodo](https://zenodo.org/records/20480151)
- Zenodo paper title: "Default LLM APIs Over Flag Indian Financial SMS: A Cleaned Label Benchmark and Template Level Deployment Audit" (under double-blind review; not accepted per record). Contents: cleaned 1,393-row test set; two XLM-R base checkpoints; per-sample predictions from eight systems; label cleanup pipeline; 9-bucket template taxonomy; inter-annotator agreement on a 276-row subset (Fleiss κ 0.82, Krippendorff α 0.82); threshold sweep τ ∈ {0.30..0.90}; audit CSV 7,234 rows; 76 files, 2.26 GB uncompressed. [Zenodo](https://zenodo.org/records/20480151)
- Zenodo states it reports no headline detection results (no F1, accuracy, precision, or FPR), though its description says a Makefile reproduces "all reported paper numbers exactly". The numbers are not on the record page. [Zenodo](https://zenodo.org/records/20480151)
- Other Indian/Hinglish candidates (all UNVERIFIED unless stated):
  - HinSpam (HF Keshav0av/HinSpam): self-reported F1 0.949, acc 0.953, MCC 0.906; provenance undescribed. [HF](https://huggingface.co/Keshav0av/HinSpam) (from SOTA_TARGETS)
  - Hinglish & English Financial Scam Dataset (HF bolewara/hinglish-scam-text-dataset): about 3,787 examples, only 35 Hinglish scam and 28 Hinglish benign; English part from UCI. No numbers. [HF](https://huggingface.co/bolewara/hinglish-scam-text-dataset) (search summary)
  - OTP/phishing SMS (HF gandharvbakshi/SMS-dataset-OTP-OTP_INTENT_Phishing): about 73,470 synthetic Indian SMS rows; gated. Search summary only.
  - SMS-Shield (Ganmoor AI Labs, LinkedIn post): ~17k synthetic-teacher-labelled Indian SMS; 4B model claimed to beat 27B. Self-reported, LinkedIn only. Not a benchmark.
  - RupeeBee technical report: BERT with attention, 94% acc, not peer-reviewed. [PDF](https://rupeebee.vercel.app/doc/rupeebee-publication.pdf)
  - Hindi Spam SMS dataset (NIT Silchar, IEEE DataPort, DOI 10.21227/5y8x-n678): 3,894 messages; no accepted model numbers; subscription required (SOTA_TARGETS).
  - ICFD-31k (IJCAI-ECAI 2026, [PDF](https://www.ijcai.org/proceedings/2026/0786.pdf)): Hinglish/English synthetic phone-call transcripts, not SMS. Annotator κ 0.534 (search summary).
  - Bangla-English financial fraud (arXiv 2603.11358, CC BY 4.0): Bangladesh, not India. [HTML](https://arxiv.org/html/2603.11358)

### Inferences
- The GroundLM paper is the only Indian financial-SMS benchmark that combines a named denominator, a stated protocol, and a released artifact. Its headline metric is FPR, not F1, so "beat it" means matching the FPR at equal or better recall, on the same 1,294 negatives.
- The "270M XLM-R" is most likely XLM-R base (about 270M total parameters, most in the 250K-vocab embedding; inference, not stated by source). Zenodo confirms "XLM RoBERTa base". Size is therefore not a small-model result; the on-device relevance is limited.
- Two records (ACL 2026 workshop, Zenodo EMNLP 2026 Industry) are versions of the same work with different titles and framing ("policy-aligned benchmark" vs "cleaned label benchmark"). Their label sets may differ. Do not mix numbers across them without checking.

### Gaps
- Recall denominator (positives count) for 0.980 not stated on the ACL page. Needs the PDF.
- Threshold selection for the 0.4% FPR not stated in the summary read.
- Whether the ACL attachment zip (https://aclanthology.org/attachments/2026.groundlm-1.79.attachment.zip) contains the same labels as the Zenodo "cleaned" version: not verified.
- SOTA_TARGETS said IIITD-sourced rows are hash-only; this was not seen on the Zenodo page read this session. Treat as UNVERIFIED.
- Zenodo v4 numbers (cleaned XLM-R FPR, recall): absent from the record page; need the paper PDF or the Makefile output.
- No peer-reviewed Hinglish SMS spam benchmark with leakage-controlled protocol exists in our searches.

---

## Q3. Small and on-device models (published size + accuracy)

### Takeaway
Only COPS (3.46 MB, abstract verified) gives a size and a headline accuracy from a verified abstract. EI-BERT's 1.91 MB is verified but is an NLU compression paper, not SMS. PhishLang's numbers are not confirmed by the abstract. The Samsung ~127 kB figure remains a search-summary claim.

### Cited Findings
- COPS (arXiv 2402.04173; IEEE CCNC 2024, DOI 10.1109/ccnc51664.2024.10454635): accuracy 98.15% (smishing) and 99.5% (URL phishing), "respectively"; FNR 0.037 and FPR 0.015 (attribution to task unclear in the abstract); model size 3.46 MB (DVAE detection module). Datasets: "open datasets", unnamed on the abstract page. Split not stated. [arXiv](https://arxiv.org/abs/2402.04173)
- EI-BERT / cross-distillation (arXiv 2507.04636): "only 1.91 MB"; accepted at KDD 2025 (per the arXiv page; not matched on kdd.org). Alipay edge recommendation, 8.4M daily devices. Abstract reports NLU tasks, no SMS result, no accuracy in abstract. [arXiv](https://arxiv.org/abs/2507.04636). Not an SMS benchmark; drop from the SMS target table.
- PhishLang (arXiv 2408.05667, Roy & Nilizadeh; v3 16 Apr 2025; no peer-reviewed venue on arXiv page): abstract gives no accuracy, size, or dataset numbers; claims it outperforms anti-phishing tools (no figures). It is a website/URL phishing detector, not SMS. [arXiv](https://arxiv.org/abs/2408.05667). SOTA_TARGETS figures (acc 0.96, F1 0.96, 74 MB memory; 94.7% on evasive attacks; NDSS 2026 DOI 10.14722/ndss.2026.231037) are NOT confirmed by the abstract read: UNVERIFIED.
- Samsung on-device (see Q1): 0.99 acc, 127 kB, ASR 0.41. UNVERIFIED numbers.

### Inferences
- The defensible small-model comparators for SMS are COPS (size and accuracy, unnamed datasets) and the Samsung paper (size, accuracy, Korean dataset; split unknown). Both lack reproducible protocols in our reads.
- Size reporting differs: COPS 3.46 MB is a module; Samsung 127 kB may be a quantised classifier; PhishLang 74 MB is memory. Compare on-disk artifact size only, consistent with SOTA_TARGETS guidance.

### Gaps
- COPS datasets, sizes, and split: need full paper (ar5iv or PDF).
- Samsung: full paper for split, dedup, and model architecture.
- PhishLang: full HTML/PDF for tables and dataset; not read.
- Model size of PhishLang MobileBERT not stated in abstract.

---

## Q4. Robustness and adversarial SMS benchmarks (Johnny et al., arXiv 2609.01171)

### Takeaway
Johnny (1 Sep 2026, arXiv preprint, no venue on arXiv page) gives the most complete smishing-robustness protocol, but it is accuracy only, uses random splits, and has internal table inconsistencies. The strongest defended number is BERT with full adversarial training: Homoglyph 99.5%, Invisible 94.6%. Minimal adversarial training leaves Homoglyph at 79.7%. The EMNLP 2026 acceptance claim in earlier notes is not confirmed.

### Cited Findings
- Abstract (summariser): title "Johnny Still Receives Spam SMS: Assessing the Robustness of SMS Spam Detection"; Salman, Islam, Ikram, Kaafar; submitted 1 Sep 2026; arXiv preprint. Abstract says adversarial training alone is insufficient; multi-model ensemble with minority voting "substantially improves robustness" (no figures in abstract). [arXiv abs](https://arxiv.org/abs/2609.01171)
- Dataset (HTML, summariser): "Super SMS", stated 67,010 messages (2012–2024; sources include UCI, NUS SMS, SpamHunter, X, ScamWatch, UK Action Fraud). CONFLICT: 40,837 benign + 26,181 spam = 67,018, not 67,010. Test: 5,000 benign + 5,000 spam, random split. Holdout: 300 test spam messages used to generate adversarial examples. [HTML](https://arxiv.org/html/2609.01171v1)
- Attack families (Table 2): perceptible = Spacing, Delete Chars, Swap Chars, Insert Chars, Sub-Chars, Sub-Word; imperceptible = Invisible, Homoglyphs, Reordering (RTLO). Reordering excluded from threat model after failing SMS/RCS delivery (Table 3). [HTML](https://arxiv.org/html/2609.01171v1)
- Full adversarial training, BERT (Table 14): Original 97.9; Spacing 98.7; Insert 99.7; Delete 98.7; Sub-Chars 99.2; Sub-Word 98.9; Swap 99.5; Invisible 94.6; Homoglyphs 99.5; Reordering 98.2. [HTML](https://arxiv.org/html/2609.01171v1)
- Minimal adversarial training (Table 15, BERTadverse_mini; held-in = Insert and Invisible): Original 99.7; Spacing 96.4; Insert 99.5; Delete 98.5; Sub-Chars 99.5; Sub-Word 99.2; Swap 99.2; Invisible 98.7; Homoglyphs 79.7; Reordering 94.2. [HTML](https://arxiv.org/html/2609.01171v1)
- Clean BERT baselines differ across tables: 99.33% acc / 0.30% FPR (Table 11); 99.0% (Table 12); 98.93% after AT (Table 13); 97.9% (Table 14). [HTML](https://arxiv.org/html/2609.01171v1)
- Homoglyph 41.3% without adversarial training (from SOTA_TARGETS, Table 12): NOT re-read this session. UNVERIFIED for this run.
- Open-weight HF spam models under Homoglyph: 17.6% to 70.9% (SOTA_TARGETS; UNVERIFIED).
- Real-world delivery: invisible-character and homoglyph perturbations survive SMS and RCS delivery (Table 3, carried). Third-party anti-spam services and commercial apps are evaluated.

### Inconsistencies (CONFLICT list, from HTML read)
1. Dataset total 67,010 vs 67,018 (sum of labels).
2. Adversarial training sample: Section 3.6 says "from the test set"; Section 6 says "from the training set"; Section 6.1 says none from holdout.
3. BERT clean accuracy: 99.33 / 99.0 / 98.93 / 97.9 across Tables 11 to 14 (see above).
4. Invisible robustness: text says AT improved imperceptible-attack resistance, but Invisible falls from 95.9 (Table 12) to 94.6 (Table 14).
5. Mobile apps and third-party services: FPR/FNR in Section 4.2 differ from Tables 5 and 6 (e.g., SMS Shield FPR 4.05% text vs 7.86% table; OOPspam FPR 0.25% text vs 0.53% table; SpamHunter FPR 3.80% text vs 7.94% table).
6. App clean accuracy: Table 5 gives 93.99 / 82.65 / 81.82; Table 10 gives 89.6 / 78.5 / 79.3 as "Original".
7. Rounding in Table 16 (minor).
- The paper reports accuracy under attack only, not attack success rate. Section 7 (proposed ensemble) was truncated in the read; ensemble numbers UNVERIFIED.

### Other smishing adversarial items
- Samsung EVA (IEEE Access 2024): ASR 0.41 against variant smishing; adversarial training with EVA improves robustness (search summary; UNVERIFIED). See Q1.
- No other 2025 to 2026 smishing adversarial benchmark with numbers was found in our search (2026 searches returned synthetic conversational and DGA sets only).

### Inferences
- Robust target to beat for our canonicaliser: Homoglyph and Invisible under AT, i.e. 99.5 and 94.6 (BERT, full AT). Johnny's own minimal-AT result (Homoglyph 79.7) shows that partial defence does not transfer.
- Because Johnny reports accuracy only, any ASR comparison needs our own re-run on the same attack generator.

### Gaps
- Johnny Section 7 (ensemble) and the Table 12 attack numbers: not re-read; need HTML pages beyond character 100,000 (summariser read only the first pages).
- Resolve Table 11 vs 12 vs 13 baseline discrepancy from the PDF.
- Johnny split: "random" in HTML; no dedup or temporal split stated.
- EMNLP 2026 acceptance: still NOT FOUND (arXiv only).

---

## Changes vs SOTA_TARGETS.md (2026-10-30)

| Item | Earlier status | Now |
|---|---|---|
| GroundLM 2026 (Gupta) | "UNVERIFIED" for 0.4% FPR, 0.980 recall | Confirmed on ACL page (summariser). DOI 10.18653/2026.groundlm-1.79 added. Recall denominator still missing |
| Zenodo 20480151 | "v4, Apache-2.0, IIITD-sourced rows hash-only" | Confirmed v4 and Apache-2.0. Title is "Multilingual Indian SMS Scam Detection" (anonymous supplement). Headline numbers absent on record. Hash-only claim not seen |
| Johnny (2609.01171) | Table 12 values; EMNLP 2026 not found | Table 14 (full AT) and Table 15 (mini AT) confirmed. Dataset-total conflict (67,010 vs 67,018) and split-source conflict added. EMNLP still not found |
| COPS | 3.46 MB; abstract numbers | Confirmed: 98.15 / 99.5, FNR 0.037, FPR 0.015, 3.46 MB (DVAE module), CCNC 2024 |
| EI-BERT | 1.91 MB | Confirmed size; reclassified as non-SMS NLU compression |
| PhishLang | acc 0.96 / 74 MB; 94.7% on evasive | NOT confirmed by abstract; still UNVERIFIED |
| Nahapetyan | no classifier metric | Unchanged; metadata numbers from search |
| Samsung | 0.99, 127 kB | Unchanged; still search summary |
| Hinglish | HinSpam only | Added HF bolewara (35 Hinglish scam examples), HF OTP/phishing (gated), SMS-Shield (LinkedIn, unverified) |
| Synthetic smishing | none | Added 2604.11752 (3,201 conv.), COVA-X 2606.06879 (10,985 conv.), Gravity Falls 2603.03270 (C2 domains) |

Not re-checked this session (carry UNVERIFIED): UCI and RoBERTa figures, Mishra & Soni primary numbers, PhreshPhish v2 table (note: a search result suggests v1.0.1 Feb 2026 added ~200k samples for Mar to Dec 2025, which conflicts with the v2 text in SOTA_TARGETS; the HF mirrors zhanfwei-0226/phreshphish and KevinRoshan8/phreshphish are not the official release), URLTran, GramBeddings, PhiUSIIL, SmishX.

## Reproducibility verdict (protocol-reproducible?)

- Protocol-reproducible in principle: PhreshPhish (temporal split, dedup, leakage filter; HF release) for URL-only reruns; GroundLM/Zenodo (named denominators, released labels/prompts per ACL claim, Apache-2.0 artifact) for the Indian FPR endpoint, pending a check that the labels match.
- Not reproducible from our reads: Johnny (random split, internal inconsistencies, no code noted), COPS (datasets unnamed), Samsung (dataset private, split unknown), PhishLang (abstract only), all UCI/Mishra numbers (random, no dedup).
- Not a benchmark: Nahapetyan (measurement), EI-BERT (NLU).

## Sources (key)
- ACL GroundLM 2026: https://aclanthology.org/2026.groundlm-1.79/ · https://doi.org/10.18653/2026.groundlm-1.79
- Zenodo 20480151: https://zenodo.org/records/20480151
- Johnny: https://arxiv.org/abs/2609.01171 · https://arxiv.org/html/2609.01171v1
- COPS: https://arxiv.org/abs/2402.04173
- EI-BERT: https://arxiv.org/abs/2507.04636
- PhishLang: https://arxiv.org/abs/2408.05667
- Samsung: https://research.samsung.com/research-papers/On-device-Smishing-Classifier-Resistant-to-Text-Evasion-Attack · https://doi.org/10.1109/ACCESS.2024.3349577
- Nahapetyan: https://doi.org/10.1109/SP54263.2024.00169 · https://par.nsf.gov/servlets/purl/10581077
- Review: https://arxiv.org/html/2604.11429
- Synthetic smishing: https://arxiv.org/abs/2604.11752 · https://papers.cool/arxiv/2606.06879 · https://arxiv.org/html/2603.03270v1
- Hinglish: https://huggingface.co/Keshav0av/HinSpam · https://huggingface.co/bolewara/hinglish-scam-text-dataset · https://www.ijcai.org/proceedings/2026/0786.pdf
- Bangla-English: https://arxiv.org/html/2603.11358
