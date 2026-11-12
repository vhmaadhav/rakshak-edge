# NIDS state of the art for cross-dataset generalisation and edge deployment (as of Nov 2026)

Status key: VERIFIED = read from the primary page in this pass. SEARCH = seen only in a search-result summary; re-check before citing. UNVERIFIED = no source read. Nothing was downloaded or run. The search budget was about 15 tool calls, so this is a partial pass and the gaps are large.

Corrections to `docs/NETWORK_IDS_PLAN.md` section 3 are called out inline.

## Cross-dataset numbers on NetFlow-v2 and similar datasets

### Takeaway
The only primary-source cross-dataset numbers found are Cantone et al. 2024 (CICIDS2017 and CSE-CIC-IDS2018, plus LycoS variants, not NF-v2) and Atif and Ali 2026 (NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2, binary only). Both show a strong collapse in the CIC-direction-to-UNSW and reverse directions. No leave-one-dataset-out (LODO) result on NF-v2 was found in this pass.

### Cited Findings
- Cantone, Marrocco, Bria, "On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection". Within-dataset performance is "nearly perfect"; cross-dataset accuracy is "largely commensurate with random chance", with exceptions for "a few combinations". The paper attributes this to "anomalies in the data" and "data heterogeneity". VERIFIED: https://arxiv.org/abs/2402.10974. Venue per arXiv record: IEEE Access 12:144489–144508 (2024), DOI 10.1109/ACCESS.2024.3472907. Only v1 (15 Feb 2024) is listed. Venue discrepancy with the plan's "Journal of Information Security and Applications" is unresolved.
- Cantone Table 3, best F1 by train→test (from plan §3.1, read in an earlier pass; not re-fetched this session): CIC-IDS2017→CSE-CIC-IDS2018 39.12% (LDA); CSE-CIC-IDS2018→CIC-IDS2017 51.21% (RF); LycoS-IDS2017→LycoS-Unicas-IDS2018 14.42% (XGB); LycoS-Unicas-IDS2018→LycoS-IDS2017 70.46% (LDA). Metric definition (binary or macro) still unverified.
- Atif and Ali, "A Three-Axis Stress Test of LLM vs Classical ML for Network Intrusion Detection under Distribution Shift and Adversarial Evasion", arXiv 2609.13511 (v1, PDF metadata 15 Sep 2026). VERIFIED via https://arxiv.org/html/2609.13511v1. Datasets: NF-UNSW-NB15-v2 (about 2.39M flows, 3.98% attack) and NF-CSE-CIC-IDS2018-v2 (about 18.9M flows, 11.95% attack), re-extracted into a shared 43-feature schema. Final model uses a locked 29-feature subset ("Set C"). Both datasets balanced to 50,000 flows per class. Attack labels are disjoint across the two datasets, so cross-dataset evaluation is binary only. Protocol: train on full source, test on full target, both directions.
  - Cross-dataset F1, Set C: UNSW→CIC: XGBoost 0.8116; RandomForest 0.5662–0.7587 (seed range); RoBERTa-LoRA 0.6582. CIC→UNSW: XGBoost 0.0730; RandomForest 0.0201–0.0378; RoBERTa-LoRA 0.0599. VERIFIED.
  - UNSW→CIC detail: XGBoost precision 0.7601, recall 0.8707, FPR 0.2748, balanced accuracy 0.7979. RoBERTa-LoRA precision 0.5290, recall 0.8709, FPR 0.7756, balanced accuracy 0.5477. VERIFIED.
  - CIC→UNSW balanced accuracy near chance: XGBoost 0.5050 (recall 0.0390); RoBERTa-LoRA 0.5042 (recall 0.0316). VERIFIED.
  - Not reported in the paper: MCC anywhere; plain accuracy cross-dataset; multi-seed results for RoBERTa-LoRA; per-attack-subtype results; cross-dataset adversarial evaluation. VERIFIED from the paper's own gaps list via the fetch.
  - Correction to plan §7 item 15: the "Atif and Ali zero-shot preprint" is now identified as arXiv 2609.13511. The plan's concern that it used target labels for feature selection was NOT checked. Set C's selection procedure must be read before using these numbers. Treat it as UNVERIFIED on protocol.
- The "UNSW→CIC XGBoost F1 0.81, reverses to 0.02–0.07; does not hold after removing TTL, packet-length and TCP-window features; does not reproduce on NetFlow v3" statement (from a search summary) is SEARCH-level only. The 0.81 and 0.02–0.07 values match Atif and Ali's table above; the ablation and v3 claims could not be traced to a primary source in this pass.
- Layeghy, Baktashmotlagh, Portmann, "DI-NIDS: Domain Invariant Network Intrusion Detection System", arXiv 2210.08252 (submitted 15 Oct 2022). VERIFIED abstract. Datasets: NFv2-CIC-2018 and NFv2-UNSW-NB15. Method: domain-adversarial network on labelled source domains, then One-Class SVM on extracted features. Target labels not used. The page reports no F1 or accuracy numbers and does not define transfer pairs. It says that swapping train and test "makes this even more severe" in some cases. The DOI 10.1016/j.knosys.2023.110626 is linked but the venue is not named on the page. The abstract claims "superior cross-domain performance" without numbers on the page. Full paper not read.
- Raskovalov, Gabdullin, Androsov, "NIDS Neural Networks Using Sliding Time Window Data Processing with Trainable Activations and its Generalization Capability", arXiv 2410.18658 (submitted 24 Oct 2024, v2 25 Oct 2024). VERIFIED abstract. Datasets: CICIDS2017 and a custom dataset (not NF-v2, not UNSW-NB15). Eleven features without deep packet inspection. Training accuracy above 99%. Abstract: "performance metrics decline significantly when changing datasets". No cross-dataset numbers on the abstract page. Full paper not read.
- Layeghy, Portmann and others (2026 survey) El Mahdaouy et al., "Deep Learning for Contextualized NetFlow-Based Network Intrusion Detection: Methods, Data, Evaluation and Deployment", arXiv 2602.05594 (v3 March 2026). VERIFIED fetch of v2 HTML. Survey with no cross-dataset numbers. Names CIC-IDS-2017, CSE-CIC-IDS-2018, CTU-13, UNSW-NB15, ToN-IoT, and others; does NOT name NF-v2. Says "CIC-IDS-2017 and CSE-CIC-IDS-2018 contain labeling errors" (no rate). Gives no LODO results. Useful as a critique source, not as a number source.
- TAN-IDS (PLoS ONE, DOI 10.1371/journal.pone.0346801 per plan and an aggregator). Aggregator-only, UNVERIFIED: the fetch of the PLOS page returned HTTP 429. Abstract per synapsesocial.com (SEARCH): maps several datasets to an 8-dimensional NetFlow feature interface; scenarios are in-dataset, direct cross-dataset, mixed-domain training and lightweight target fine-tuning; "strong in-dataset results do not carry over"; mixed-domain training helps; fine-tuning on 5% labelled target data "recovers attack-class performance in several scenarios". Numbers not read. Do not cite numbers until the PLOS page is read.
- Houssel et al. 2024 (search summary, SEARCH): zero-shot GPT-4 and Llama3-8B-Instruct on NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 give macro precision and recall of about 40–54%. The authors judged them not precise enough for primary detection. Not fetched. Not a cross-dataset protocol per the summary.
- Rehman et al., arXiv 2510.17883 "From Flows to Words: Can Zero-/Few-Shot LLMs Detect Network Intrusions?" (SEARCH). Macro F1 0.78 (7B) and 0.68 (3B) on original UNSW-NB15, not NetFlow, not cross-dataset.
- LLM comparison result: Atif and Ali above (VERIFIED). XGBoost beats RoBERTa-LoRA by 15.3 F1 points on UNSW→CIC. Both near chance on CIC→UNSW. LLMs did not outperform classical ML under shift in that paper.

### Inferences
- Under a shared NF-v2 43-feature schema, the UNSW→CIC direction can reach F1 around 0.8 with gradient boosting, but the reverse direction collapses to near-zero recall (about 0.03–0.04). A single "cross-dataset F1" number hides this asymmetry. Report both directions and the FPR.
- The 0.81 figure rests on a 29-feature subset chosen by a process not yet verified. Any headline needs a frozen-before-target feature set (plan §2.2 rule 2).
- The attack-label disjointness in NF-UNSW-NB15-v2 vs NF-CSE-CIC-IDS2018-v2 means binary cross-dataset is the only fair cross-dataset metric. Multi-class LODO on these pairs is not defined.

### Gaps
- No LODO (leave-one-dataset-out) result on NF-v2 or the other NF-*-v2 sets was found in this pass. The "2026 TAN-IDS" paper is the nearest, but its numbers were not read.
- Cantone's metric definition (binary vs macro) was not verified this session.
- No primary numbers found for NF-ToN-IoT-v2, NF-BoT-IoT-v2 or NF-UQ-NIDS-v2 cross-dataset transfer.
- No 2025–2026 domain-adaptation paper with NF-v2 numbers was read. A 2025 SHAP-based study evaluating five unsupervised DA methods is reported only by a search summary (SEARCH, no URL or numbers captured). Conclusion per summary: "conventional DA methods fail to deliver robust cross-domain NIDS".

## Methods that help transfer with few or no target labels

### Takeaway
The evidence seen this pass is thin. Mixed-domain training and small labelled target fine-tuning (about 5%) are reported to help (TAN-IDS, SEARCH). Adversarial domain-invariant features (DI-NIDS) are claimed to help with no numbers read. Target-only training with 5% labels matches source-based adaptation in one 2026 NetFlow v3 preprint (SEARCH). No primary-source benign-only calibration numbers were found.

### Cited Findings
- DI-NIDS (VERIFIED abstract, no numbers): domain-adversarial feature extraction, target labels not used, claims superior cross-domain performance. See above.
- TAN-IDS (SEARCH via aggregator): mixed-domain training helps generalisation; 5% target labels recover attack-class performance in several scenarios.
- A 2026 NetFlow v3 preprint (SEARCH, quoted as summarised by a search aggregator; primary not captured): at a 5% label budget, target-only training matches source-based adaptation. This is v3, not v2, so transfer to our schema is not established.
- Arp et al., "Dos and Don'ts of Machine Learning in Computer Security", USENIX Security 2022 (plan §2.1, primary page and slides read in an earlier pass). Pitfalls include label inaccuracy, data snooping and inappropriate baselines. Used to set the protocol, not for numbers.
- Apruzzese, Pajola, Conti, "The Cross-evaluation of Machine Learning-based NIDS", arXiv 2203.04686, IEEE TNSM DOI 10.1109/TNSM.2022.3157344 (plan §2.1, abstract read earlier). Cross-evaluation exposes both potential and risk.

### Inferences
- Benign-only anomaly calibration on the deployment network (home baseline) is the approach most consistent with the plan's false-positive target (≤1% FPR on held-out benign traffic). No published number for it on NF-v2 was found in this pass. This is a gap, not a finding.
- Feature subsets that drop TTL, TCP window and sequence fields are repeatedly proposed (plan §4.2, DI-NIDS-style ablations). This matches the plan. The effect size is still unverified.

### Gaps
- No primary-source numbers for benign-only anomaly detection calibrated on a home baseline across NF-v2 datasets.
- No primary numbers for rules + ML hybrids on NF-v2.
- No primary numbers for few-shot target calibration with a specific label budget beyond the TAN-IDS aggregator summary and the 2026 v3 preprint summary.

## Edge and Raspberry Pi deployments (throughput, latency, memory, size)

### Takeaway
No published Raspberry Pi 5 result with flows/s, latency, memory and model size together was found. The best-documented Pi study in this pass is on Pi 4, not Pi 5, and reports only accuracy and a 50 ms average response time with no method for measuring it. The plan's Pi 5 gap stands.

### Cited Findings
- Wijethilaka, Yapa, Siriwardena, "A cost effective machine learning based network intrusion detection system using Raspberry Pi for real time analysis", PLoS ONE 20(12):e0331123, published 29 Dec 2025, DOI 10.1371/journal.pone.0331123. VERIFIED via https://pmc.ncbi.nlm.nih.gov/articles/PMC12747338/. Hardware: Raspberry Pi 4 (RAM variant not stated). Dataset: NF-UQ-NIDS. Random Forest selected: accuracy 96.5%, F1 94.8%, precision 94.1%, recall 95.5%. Other models evaluated: Naive Bayes, XGBoost, SVM, LSTM, and ANN. Latency: "average response time of 50 milliseconds"; the measurement method is not described. Not reported: model size, throughput (flows/s or packets/s), memory or CPU. Capture: pyshark and tcpdump. Caveats: XGBoost F1 82.7% in Table 3 is inconsistent with its listed precision 90.0% and recall 91.2% (which imply about 90.6%). The plan's search summary listed this as Pi 5; the fetched article says Pi 4. CORRECTION to the plan's implied hardware.
- Pi 4 TinyML feedforward network (SEARCH, search summary only, not fetched): about 120,000 packets/s inference throughput and about 31 KB model. Metric is packets, not flows. Dataset not confirmed. This may be the NTNU ToN_IoT study in plan §3.3 (SEARCH) and is not independently checked.
- Optimised ensemble deep learning on Raspberry Pi, IEEE Access, June 2025, DOI 10.1109/ACCESS.2025.3584373 (SEARCH): reported 97.3% accuracy across scenarios, pruned and quantised, Kafka pipeline, peak CPU 111.8%. Throughput not in the summary. Not fetched.
- Hobby project (search result, not peer-reviewed): Random Forest on CIC-IDS-2017 with about 30 ms per flow on Pi 5 and 0.94 macro F1 (https://liamdoocey.github.io/ml-nids/, SEARCH). Not citable as a paper.
- OpenPiMap (https://github.com/evilbotnet/OpenPiMap, SEARCH): NetFlow v5 analyser on a Pi 1 Model B. Uptime only, no throughput.
- Plan §3.3 "2026 quantum-assisted TinyML IDS, Raspberry Pi 4, about 0.65 MB, about 16.8 ms inference" (arXiv 2609.06814, SEARCH). Not fetched in this pass; still UNVERIFIED.
- Plan §3.3 "Scapy vs Pyshark vs Pypcap on Raspberry Pi 3 Model B" (SEARCH, from plan). Not re-checked. Pi 3B only.

### Inferences
- The Pi 4 study's feature path (pyshark + tcpdump) is a per-packet Python route. The plan already rejects this for the hot path. This is a reasonable argument for nfstream on the Pi 5, but nfstream on Pi was not found in any paper.
- The 50 ms "response time" is not a per-flow latency. It cannot be compared with the plan's p95 flow-to-verdict target without the measurement method.

### Gaps
- No Pi 5 flows/s, latency, RSS or model-size measurement was found in any source read this pass.
- No nfstream-on-Raspberry-Pi paper or benchmark found. Plan §4.1 "nfstream aarch64 wheel exists" remains as the plan states it (from PyPI, read in an earlier pass).
- No nfstream vs CICFlowMeter feature-parity study was found in this pass. The LycoS motivation (plan §4.4) is from a search summary only.

## How credible papers report, and label quality

### Takeaway
Credible NIDS papers in this pass were the ones that stated the feature set, split protocol, and both directions. Most of the strongest-looking results are within-dataset. Label-error work on CIC-IDS2017 and CSE-CIC-IDS2018 is established (Engelen et al. 2021; Liu et al. 2022), but the corrected release was not located in this pass.

### Cited Findings
- Liu, Engelen, Lynar, Essam, Joosen, "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018", IEEE CNS 2022, pp. 254–262, DOI 10.1109/CNS56114.2022.9947235. Abstract (SEARCH, search summary): reports many previously undocumented errors in attack orchestration, feature generation, documentation and labelling; provides a recreated dataset with reverse-engineered, corrected labelling logic "made publicly available". Code: https://www.github.com/GintsEngelen/CNS2022_Code (README says the paper link would be added on publication). No download link for the corrected dataset was confirmed. Corrected dataset location: UNVERIFIED.
- Engelen, Rimmer, Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE SPW 2021, DOI 10.1109/SPW53761.2021.00009 (plan §1.3, SEARCH for numbers; PDF not fetched). "More than 20% of traces reconstructed or relabelled" is a search-summary number, not read.
- Survey El Mahdaouy et al. 2026, arXiv 2602.05594 (VERIFIED): says CIC-IDS-2017 and CSE-CIC-IDS-2018 "contain labeling errors", and lists temporal leakage, splitting, dataset design flaws and weak cross-dataset generalisation as failure modes.
- Arp et al. 2022 (plan §2.1, read earlier): pitfalls list.
- Dube 2023, "Faulty use of the CIC-IDS 2017 dataset in information security research", J. Computer Virology and Hacking Techniques, DOI 10.1007/s11416-023-00509-7 (plan §1.3, VERIFIED via OpenAlex in an earlier pass).
- Atif and Ali 2026 (VERIFIED above): reports a binary-only cross-dataset protocol because attack labels are disjoint. This is an explicit label-compatibility caveat that many papers omit.

### Inferences
- Any headline on CIC-IDS2017 or CSE-CIC-IDS2018 must say which label version was used (uncorrected, or Engelen/Liu corrected). Otherwise it is not comparable.
- Papers that do not report the protocol (direction, feature set, threshold rule) cannot be used as targets. Several search-summary "SOTA" claims in this pass fall in that category.

### Gaps
- The Engelen/Liu corrected release location and licence not confirmed (plan §7 item 1 remains open).
- Cantone's own metric definition not confirmed.
- No quantified label-error rate for NF-v2 datasets found. The NF-v2 labels inherit the raw captures' label issues (plan inference, UNVERIFIED).

## Corrections to the plan (section 3 and section 7)
- Plan §3.1: Cantone Table 3 numbers were read in an earlier pass and not re-fetched in this session. Metric definition still open.
- Plan §3.3 and §7 item 15: the "Atif and Ali" preprint is now identified as arXiv 2609.13511 (VERIFIED). Set C selection procedure not checked, so the protocol claim in the plan stays open.
- Plan §3.3 and search summary: the "Pi 5, PLOS ONE Dec 2025" study is a Pi 4 study with no flows/s figure (VERIFIED).
- Plan §3.3 "Pi 3B packet-capture study" and "NTNU ToN_IoT Pi 4" entries: still SEARCH only.

## Sources read in this pass (URLs)
- https://arxiv.org/abs/2402.10974 (VERIFIED metadata and abstract)
- https://arxiv.org/html/2609.13511v1 (VERIFIED numbers)
- https://arxiv.org/abs/2210.08252 (VERIFIED abstract)
- https://arxiv.org/abs/2410.18658 (VERIFIED abstract)
- https://arxiv.org/html/2602.05594v2 (VERIFIED survey text)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12747338/ (VERIFIED, Pi 4 study)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12747338/ (redirected to the link above)
- https://synapsesocial.com/papers/69d8958f6c1944d70ce06a5e (aggregator; PLOS page returned HTTP 429)
- https://github.com/GintsEngelen/CNS2022_Code (repo name from search; not fetched)
- Search-only: https://arxiv.org/html/2510.17883v2; https://arxiv.org/pdf/2408.04342; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12747338/ (fetched)

## Not found in this pass (explicit gaps)
- LODO numbers on NF-v2 datasets.
- 2026 foundation-model or LLM NIDS with primary cross-dataset numbers on NF-v2.
- Pi 5 flows/s, latency, memory, size on any published paper.
- nfstream on Raspberry Pi performance or feature-parity study.
- Corrected CIC-IDS2017 / CSE-CIC-IDS2018 download location.
