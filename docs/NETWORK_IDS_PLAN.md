# Network-IDS module: research plan and targets

Scope: the flow-based home/SME/IoT network intrusion detector described in `docs/CONTEXT.md` ("Scope addition: network-intrusion module"). SMS and URL remain the headline research claim. This module runs on a Raspberry Pi 5, uses the Evidence-Locked Advisory pattern, and is evaluated with the same rigour as the SMS work.

Status key used below:
- **VERIFIED**: confirmed by fetching the primary page or the Kaggle CLI output in this pass.
- **SEARCH**: seen only in a search-result summary. Re-check the primary source before citing a number.
- **UNVERIFIED**: no source confirmed it. Do not use it in a report until checked.

Nothing here has been downloaded in bulk. Kaggle file lists and metadata were read with `kaggle datasets files|metadata`. Two small files were fetched to the scratchpad: the NetFlow-v2 feature list and the UNSW-NB15 feature list.

---

## 1. Datasets

### 1.1 Summary table

| Dataset | Flow features | Labels | Licence (primary or Kaggle copy) | Known problems | Use in our plan |
|---|---|---|---|---|---|
| **UNSW-NB15** | Argus-style, 49 columns, includes IPs, ports, `Stime`/`Ltime`, and host-window `ct_*` counts (VERIFIED: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 feature file) | `attack_cat` (9 categories, VERIFIED same file) + binary `Label` | Kaggle copies: CC-BY-NC-SA-4.0 (`dhoogla/unswnb15`), unknown (`mrwellsdavid/unsw-nb15`). Official terms: https://research.unsw.edu.au/projects/unsw-nb15-dataset (SEARCH) | Identifier columns (`srcip`, `sport`, `dstip`, `dsport`) and timestamps invite shortcut learning. Label-error audit: UNVERIFIED (none found) | Cross-dataset source/target, feature-ablation study |
| **NF-UNSW-NB15-v2** (NetFlow v2 derivative of UNSW-NB15) | 43 NetFlow-style features, shared with the other NF-*-v2 sets (VERIFIED: `dhoogla/nfunswnb15v2`, file `NetFlow v2 Features.csv` has 43 rows) | Multi-class + binary | CC-BY-NC-SA-4.0 (VERIFIED from Kaggle metadata) | Keeps IP and port identifiers in the feature list (VERIFIED in the file). Inherits any label errors from the raw capture (inference, UNVERIFIED) | Primary cross-dataset feature space |
| **CIC-IDS2017** (original CSVs) | CICFlowMeter, about 80 features (UNB says "more than 80", VERIFIED: https://www.unb.ca/cic/datasets/ids-2017.html). Counts reported as 78, 79 and 84 depending on source (SEARCH) | 15 labels: BENIGN + 14 attacks (SEARCH; the 14 attacks are listed on the UNB page) | Kaggle copy `bertvankeulen/cicids-2017` is Apache-2.0 (VERIFIED metadata). Official terms: UNB page (UNVERIFIED for licence text) | Labelling, attack-orchestration and feature-extraction errors (Engelen et al. 2021; Liu et al. 2022; Rosay et al. 2022, sources in §1.3). Identifier columns (Flow ID, IPs, ports, Timestamp) | Baseline only, and only with a corrected labelling source |
| **CIC-IDS2017 corrected** (Engelen/Liu relabelled; LycoS-IDS2017) | Engelen/Liu release: CICFlowMeter-style, regenerated. LycoS-IDS2017: LycoSTand features from PCAPs (Rosay et al. 2022, SEARCH) | Relabelled | Not confirmed. LycoS tools and data: https://lycos-ids.univ-lemans.fr (SEARCH) | Corrected labels are the point of the dataset | **Must-have** for any CIC-IDS2017 result |
| **CSE-CIC-IDS2018** | CICFlowMeter-V3, 80 features (UNB page https://www.unb.ca/cic/datasets/ids-2018.html exists; the "80" figure is SEARCH-level, not read on that page). Flow count not confirmed | 7 attack scenarios (Brute-force, Heartbleed, Botnet, DoS, DDoS, Web, Infiltration; VERIFIED UNB page) | Official terms: UNB page. AWS Open Data: https://registry.opendata.aws/cse-cic-ids2018 (SEARCH). Kaggle copies: see §1.2 | Engelen et al. 2021 report that the labelling problems persist in CSE-CIC-IDS2018 (SEARCH: https://lirias.kuleuven.be/retrieve/713070) | Cross-dataset target |
| **LycoS-Unicas-IDS2018** (corrected CSE-CIC-IDS2018) | Same as LycoS-IDS2017 method (Cantone et al. 2024) | Corrected labels | Availability not confirmed. Cantone et al. 2024 is the only source found (https://arxiv.org/abs/2402.10974, VERIFIED) | Not checked | Preferred corrected 2018 variant if obtainable; ask the authors |
| **CIC-IoT-2023** | 46/47 features per record, sources disagree (SEARCH). 105 devices, 33 attacks in 7 classes (VERIFIED: https://www.unb.ca/cic/datasets/iotdataset-2023.html). Records about 46.7–47 M (SEARCH) | 7 classes, 33 attacks | Kaggle `himadri07/ciciot2023` is MIT (VERIFIED metadata). Original terms: UNB page | Duplicate rows: a Kaggle derivative reports 61,326 exact duplicates (5.25%) in its 1,168,147-row subset and calls the original "widely reported >99%" accuracy inflated (SEARCH: Kaggle `sohamkatlariwala/ciciot2023-preprocessed-leakage-fixed`, description.md, VERIFIED text, figures self-reported) | Secondary IoT check. Use the deduplicated, fixed split only as a cross-check |
| **ToN_IoT** (network part) | 45 flow features in the standardised version (SEARCH, arXiv 2212.13994 describes ToN-IoT versions). Full network part about 21.98 M records (SEARCH, UNVERIFIED) | 9 attack types (SEARCH) | Terms: https://research.unsw.edu.au/projects/toniot-datasets (VERIFIED page; licence text not captured) | Raskovalov et al. (arXiv 2212.13994, VERIFIED) show inconsistencies between versions and publish a filtered "ToN-IoT-R" | Modern IoT cross-check; use only the ToN-IoT-R-style filtered version |
| **Bot-IoT** | Pcap, Argus and CSV. CSV about 16.7 GB; pcap about 69.3 GB; over 72 M records; 5% subset about 3 M records, 1.07 GB (VERIFIED: https://research.unsw.edu.au/projects/bot-iot-dataset) | DDoS, DoS, OS/service scan, keylogging, data exfiltration (VERIFIED) | Free for academic research; commercial use needs agreement with the authors (VERIFIED, same page) | Feature-set limits: unverified | Botnet/IoT cross-check; use the NF-BoT-IoT-v2 copy |
| **NSL-KDD** (legacy only) | 41 features (original KDD, SEARCH) | Normal + attack classes | Not verified. The UNB NSL page reportedly says the dataset is not currently available (SEARCH: https://unb.ca/cic/datasets/nsl.html) | Tavallaee et al. 2009 report 78%/75% redundancy in KDD train/test (SEARCH); McHugh 2000 realism critique (SEARCH) | **Legacy comparison only.** Do not headline |
| **CTU-13** (botnet) | Argus `binetflow` flows, labels manual per scenario (SEARCH: https://stratosphereips.org/datasets-ctu13; Garcia et al. 2014, Computers & Security 45:100–123 from the Stratosphere FAQ, SEARCH) | 13 botnet scenarios (SEARCH) | Kaggle `dhoogla/ctu13` is CC-BY-NC-SA-4.0 (VERIFIED metadata). Original terms: UNVERIFIED | Argus features differ from CICFlowMeter. Scenarios are few and per-botnet | Botnet-beaconing evidence check (BOTNET_BEACONING) |
| **Modern: NF-ToN-IoT-v2, NF-UQ-NIDS-v2, NF-BoT-IoT-v2** (Sarhan et al.) | 43 NetFlow features (VERIFIED for NF-UNSW-NB15-v2; the others are the same family, SEARCH) | Multi-class | CC-BY-NC-SA-4.0 for `dhoogla` copies (VERIFIED for `nfunswnb15v2`, `nftoniotv2`, `nfbotiotv2`, `nfcsecicids2018v2` via Kaggle metadata; `nfuqnidsv2` not checked) | As above | The shared-schema backbone (§4) |
| **CIC-IoMT-2024** | Not inspected | Not inspected | `limamateus/cic-iomt-2024` (licence not checked) | Not inspected | Only if time allows (2024-2026 medical IoT) |

### 1.2 Kaggle copies (exact slugs, sizes from `kaggle datasets list` and `files`)

| Slug | Content | Size (CLI) | Licence (metadata) | Notes |
|---|---|---|---|---|
| `dhoogla/unswnb15` | UNSW-NB15 train and test as parquet | 12.3 MB | CC-BY-NC-SA-4.0 | Parquet, not the official CSVs. Good for a smoke test |
| `mrwellsdavid/unsw-nb15` | UNSW-NB15 CSVs incl. `UNSW_NB15_training-set.csv` (15.4 MB) and `UNSW_NB15_testing-set.csv` (32.3 MB), `NUSW-NB15_features.csv` | 156 MB | unknown | Closest to the official CSVs. Identity with the UNSW release not verified (check the row count and checksum) |
| `dhoogla/nfunswnb15v2` | NF-UNSW-NB15-v2 parquet + `NetFlow v2 Features.csv` (43 features) | 49.5 MB | CC-BY-NC-SA-4.0 | Primary cross-dataset source |
| `dhoogla/nfcsecicids2018v2` | NF-CSE-CIC-IDS2018-V2 parquet + feature list | 641.6 MB | CC-BY-NC-SA-4.0 | Primary cross-dataset target |
| `dhoogla/nfbotiotv2` | NF-BoT-IoT-V2 parquet + feature list | 508.6 MB (parquet) | CC-BY-NC-SA-4.0 | |
| `dhoogla/nftoniotv2` | NF-ToN-IoT-V2 | 182.7 MB | CC-BY-NC-SA-4.0 | |
| `dhoogla/nfuqnidsv2` | NF-UQ-NIDS-V2 | 1.85 GB | not checked | Large; use only if needed |
| `dhoogla/csecicids2018` | CSE-CIC-IDS2018 per-day CICFlowMeter parquet (10 files shown, one per day or attack) | 634 MB | CC-BY-NC-SA-4.0 | CICFlowMeter features, comparable with CIC-IDS2017 CSVs |
| `bertvankeulen/cicids-2017` | CIC-IDS2017 CSVs: monday, tuesday, wednesday, thursday, friday, plus `*_plus` variants | 533 MB | apache-2.0 | `_plus` files not inspected. Check whether they are corrected labels before use |
| `shadman1028/cicids2017-official-flow-feature-csv-files` | CIC-IDS2017 "official flow feature" CSVs | 241 MB | not checked | Likely the original labels |
| `ernie55ernie/improved-cicids2017-and-csecicids2018` | `CICIDS2017_improved/*.csv` and `CSECICIDS2018_improved/*.csv` | 10.99 GB | "other" | Provenance not confirmed. Could be a corrected release. Check the Kaggle description before use (UNVERIFIED) |
| `ericanacletoribeiro/cicids2017-cleaned-and-preprocessed` | CIC-IDS2017 cleaned and preprocessed | 210 MB | CC0-1.0 | Cleaned, not relabelled. Do not treat as corrected labels |
| `himadri07/ciciot2023` | CICIoT2023 train/test/validation CSVs | 500.6 MB | MIT | Official-style split. Check duplicates before use |
| `sohamkatlariwala/ciciot2023-preprocessed-leakage-fixed` | 19-feature deduplicated CICIoT2023 train/val/test, 1,106,821 rows | 50.3 MB | "other" | Description says redistribution terms are unconfirmed and it is "published privately pending that confirmation". Use internally only |
| `arnobbhowmik/ton-iot-network-dataset` | ToN_IoT network subset | 1.8 MB | not checked | Looks like a subset. Confirm the row count |
| `vigneshvenkateswaran/bot-iot` | Bot-IoT CSVs `data_1.csv`…, 1.26 GB total | 1.26 GB | unknown | |
| `dhoogla/cicbotiot` | CIC-BoT-IoT-V2 parquet + feature list | 1.1 GB | CC-BY-NC-SA-4.0 (metadata not individually verified, same publisher family) | |
| `dhoogla/nslkdd` | NSL-KDD | 2.0 MB | not checked | Legacy only |
| `dhoogla/ctu13` | CTU-13 13 scenarios as `.binetflow.parquet` | 106.9 MB | CC-BY-NC-SA-4.0 | |
| `limamateus/cic-iomt-2024` | CIC-IoMT-2024 | 127.6 MB | not checked | |
| `yasiralifarrukh/unsw-and-cicids2017-labelled-pcap-data` | Labelled PCAPs, UNSW-NB15 and CIC-IDS2017 (name from CLI) | 345.7 MB | not checked | Contents not inspected. Candidate raw source for re-extraction (§4.4) |
| `sonalimishra3/bot-iot-pcap` | Bot-IoT PCAPs | 5.2 GB | not checked | Contents not inspected |

**Searched and not found:** corrected CIC-IDS2017 (Engelen/Liu) and LycoS-IDS2017 on Kaggle (`kaggle datasets list -s lycos`, `corrected cicids2017`, `relabeled cicids2017` returned no matching dataset); corrected CSE-CIC-IDS2018 (LycoS-Unicas or Liu et al.) on Kaggle.

Kaggle CLI commands (for later use inside kernels, not on the laptop):
`kaggle datasets files <slug>`, `kaggle datasets metadata <slug>`, `kaggle datasets download <slug> -f <file>`.

### 1.3 Known flaws and sources

- **CIC-IDS2017.** Engelen, Rimmer and Joosen (2021, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE SPW, DOI 10.1109/SPW53761.2021.00009) report attack-simulation, feature-extraction and labelling errors. Their search-summary account: more than 20% of traces reconstructed or relabelled, artefacts removed (about 20% of data), labels corrected (about 10% of samples), and the feature extractor fixed. Source: https://lirias.kuleuven.be/retrieve/713070 (SEARCH for the numbers, the PDF itself was not fetched). Liu, Engelen, Lynar, Essam and Joosen (2022, "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018", IEEE CNS, DOI 10.1109/CNS56114.2022.9947235, VERIFIED via OpenAlex) publish reverse-engineered labelling logic and a recreated dataset, and recommend that labelling logic be released with any dataset (SEARCH). Rosay et al. (2022, ICISSP, DOI 10.5220/0010774000003120, SEARCH) found problems in both the CSVs and CICFlowMeter and released LycoSTand and LYCOS-IDS2017, with "significant performance improvements" on the corrected data. Dube (2023, "Faulty use of the CIC-IDS 2017 dataset in information security research", J. Computer Virology and Hacking Techniques, DOI 10.1007/s11416-023-00509-7, VERIFIED via OpenAlex) is a critique of how the data is used. Lanvin et al. (2022, CRiSIS) is cited in the search summary as an independent check of the labels (UNVERIFIED, no URL retrieved).
- **Identifier and leakage columns (CIC-IDS2017 and CSE-CIC-IDS2018).** Flow ID, Source IP, Destination IP, Source/Destination Port and Timestamp are in the CSVs. Cantone et al. (2024) drop "7 identifier and redundant fields" from CIC-IDS2017 and drop Timestamp and Protocol from CSE-CIC-IDS2018 (VERIFIED: https://arxiv.org/html/2402.10974v1, Section 3, via WebFetch).
- **CSE-CIC-IDS2018.** Engelen et al. (2021) report that the errors persist in this version (SEARCH, see above).
- **UNSW-NB15.** Identifier columns (`srcip`, `sport`, `dstip`, `dsport`), timestamps (`Stime`, `Ltime`), and host-window counts (`ct_srv_src`, `ct_dst_ltm`, `ct_src_dport_ltm`, …) are in the feature file (VERIFIED: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15 `NUSW-NB15_features.csv`). Their values depend on the window and the capture, so they are a shortcut risk. A published label-error audit was not found: UNVERIFIED.
- **NF-*-v2 (NetFlow).** Feature list includes `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, and DNS/FTP fields (VERIFIED, 43-row file). Sarhan, Layeghy and Portmann (arXiv 2101.11315, VERIFIED abstract, DOI 10.1007/s11036-021-01843-0 from arXiv page) argue for a standard NetFlow feature set. Inference, UNVERIFIED: these datasets inherit the labels of the raw captures, so they inherit CIC/UNSW label errors.
- **CICIoT2023.** The leakage-fixed derivative reports 61,326 exact duplicates (5.25%) and a dedup-before-split issue (Kaggle description, SEARCH-level, self-reported; see §1.1 row).
- **ToN-IoT.** Raskovalov et al. (arXiv 2212.13994, VERIFIED abstract) "inspect different versions of ToN-IoT dataset and point out inconsistencies in some versions". They publish a filtered ToN-IoT-R and a compact 'standardised' NetFlowv5-compatible feature set. Use ToN-IoT-R, not the raw version.
- **NSL-KDD.** Redundancy and realism critiques (Tavallaee et al. 2009; McHugh 2000) are SEARCH-level only. Legacy use only.

---

## 2. Evaluation protocols

### 2.1 Pitfalls to pre-register

- Arp, Quiring, Pendlebury, Warnecke, Pierazzi, Wressnegger, Cavallaro and Rieck, "Dos and Don'ts of Machine Learning in Computer Security", USENIX Security 2022, pp. 3971–3988, Distinguished Paper Award (SEARCH: https://www.bifold.berlin/news-events/news/view/news-detail/cybersecurity-under-scrutiny). Paper and pages VERIFIED via the USENIX presentation page https://usenix.org/conference/usenixsecurity22/presentation/arp; the list of pitfalls is from the USENIX slides https://www.usenix.net/system/files/sec22_slides-arp.pdf (SEARCH). Pitfalls named on the slides include sampling bias, label inaccuracy, data snooping, spurious correlations, biased parameter selection and inappropriate baselines. The condensed CACM version is "Pitfalls in Machine Learning for Computer Security" (SEARCH).
- Apruzzese, Pajola and Conti, "The Cross-evaluation of Machine Learning-based NIDS", arXiv 2203.04686, IEEE TNSM DOI 10.1109/TNSM.2022.3157344 (VERIFIED abstract). Proposes XeNIDS for cross-evaluation on six datasets and says cross-evaluation exposes both potential and risk.
- Apruzzese, Laskov and Schneider, "SoK: Pragmatic Assessment of Machine Learning for Network Intrusion Detection", arXiv 2305.00550 (SEARCH for the title and submission date; body not read). Argues for changing NIDS evaluation methodology.
- Cantone, Marrocco and Bria, "On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection", arXiv 2402.10974 (VERIFIED abstract and Table 3 via WebFetch). Within-dataset performance is "nearly perfect"; cross-dataset accuracy is "largely commensurate with random chance". Note: arXiv lists IEEE Access 12:144489–144508 (2024); the HTML header says Journal of Information Security and Applications. Resolve the venue before citing.
- Liu et al. 2022 (CNS, DOI above): publish the labelling logic with the dataset.
- Sudyana et al. 2024, "Quality Analysis in IDS Dataset: Impact on Model Generalization", IEEE CNS, DOI 10.1109/cns62487.2024.10735570 (VERIFIED via OpenAlex title only; body not read). Related: "Improving Generalization of ML-Based IDS With Lifecycle-Based Dataset, Auto-Learning Features, and Deep Learning", DOI 10.1109/tmlcn.2024.3402158 (VERIFIED via OpenAlex title only).
- Alani and Miri, "Towards an Explainable Universal Feature Set for IoT Intrusion Detection", Sensors 2022, DOI 10.3390/s22155690 (VERIFIED via OpenAlex title only; body not read).

### 2.2 Splits

| ID | Protocol | Definition | Why |
|---|---|---|---|
| **T0** | Random split, same dataset | Stratified random 70/15/15 after deduplication | Reference only. Never the headline. This is where ~99% numbers come from (Cantone et al. 2024: within-dataset average MCC 94.63%, VERIFIED text) |
| **T1** | Cross-dataset | Train on dataset A, test on B, both directions. Feature set = frozen shared schema (§4.2). Thresholds and hyper-parameters set on A's validation split only | Tests whether the detector learns attacks or the capture. Cantone et al.: cross-dataset MCC average 29.35% (VERIFIED Table 3 text) |
| **T2** | Leave-one-attack-family-out | Hold out one attack family (e.g. DoS) from training, test on it. Repeat per family; also do it cross-dataset | Tests unseen-family generalisation (UNSW-NB15 has nine categories, VERIFIED feature file) |
| **T3** | Temporal | CIC-IDS2017 by day: Monday is benign only; Tuesday–Friday contain attacks (VERIFIED UNB page). Train on earlier days, test on the later day. For UNSW-NB15, sort by `Stime` and split by time | Tests drift. Strong in CIC-IDS2017 because the day structure is known |
| **T4** | Host or campaign disjoint | Group by source IP/subnet and attack campaign or time window; no group in both train and test | Stops the model memorising a host. Use clustered bootstrap CIs over these groups |

Rules for every split:
1. Deduplicate exact-duplicate feature vectors before splitting (Kaggle CICIoT2023 derivative: 5.25% exact duplicates, SEARCH-level).
2. Freeze the feature set and the decision threshold before touching the target test set. Do not select features by target-set F1.
3. Report the random-split number (T0) next to every cross-dataset number so the collapse is visible.
4. Use the corrected labels when they exist (Engelen/Liu/LycoS). Report the uncorrected numbers separately and say so.
5. Report: macro-F1, per-family recall at fixed benign FPR (e.g. 0.1% and 1%), PR-AUC, benign FPR per dataset, ECE/Brier, and risk–coverage for the UNCERTAIN state.

### 2.3 How much random-split numbers collapse

- Cantone et al. (2024): within-dataset MCC averages 94.63%; cross-dataset averages 29.35% (VERIFIED text). Per-direction F1 values in §3.
- Their stated conclusion: "training a ML model on a single publicly available dataset is not sufficient to create a robust NIDS". They attribute the collapse to dataset composition, label errors and dataset-specific feature correlations (VERIFIED text).
- The collapse is also driven by feature choice: CICFlowMeter-style features with IP, port and timestamp columns (CIC-IDS2017 and CSE-CIC-IDS2018) carry dataset-specific shortcuts. Cantone et al. remove identifiers before training (VERIFIED).

---

## 3. Published results and targets

### 3.1 Cross-dataset numbers (primary source: Cantone et al. 2024, Table 3, VERIFIED via https://arxiv.org/html/2402.10974v1)

Each row: train → test, best classifier by F1 in the paper's table. Feature set: their 77 common features (VERIFIED text). Metric definition (binary or macro): UNVERIFIED.

| Train → Test | Best F1 (classifier) | Best MCC (classifier) | AUROC best | Accuracy |
|---|---|---|---|---|
| CIC-IDS2017 → CSE-CIC-IDS2018 | 39.12% (LDA) | 35.68% (XGB) | 81.70% (LDA) | not reported (authors chose not to) |
| CSE-CIC-IDS2018 → CIC-IDS2017 | 51.21% (RF) | 44.44% (RF) | 85.06% (LDA) | not reported |
| LycoS-IDS2017 → LycoS-Unicas-IDS2018 | 14.42% (XGB) | 10.17% (XGB) | 65.46% (RF) | not reported |
| LycoS-Unicas-IDS2018 → LycoS-IDS2017 | 70.46% (LDA) | 60.41% (LDA) | 91.66% (LDA) | not reported |

Notes:
- The paper's text gives the average for LycoS17→LycoS18 MCC as 10.83%, but the four table values average about 3.17%. Use the table values and flag the discrepancy.
- Their "accuracy is at random chance" claim in the abstract has no accuracy number in the paper (VERIFIED text).

### 3.2 Within-dataset (reference only)

- Cantone et al. 2024: average within-dataset MCC 94.63% (VERIFIED text).

### 3.3 Edge and Pi results found

| Item | Hardware | Reported | Source | Gap |
|---|---|---|---|---|
| 2026 quantum-assisted TinyML IDS, classical INT8 variant | Raspberry Pi 4 | About 0.65 MB memory, about 16.8 ms inference, about 105.9k parameters (full precision) | https://arxiv.org/pdf/2609.06814 (SEARCH) | Dataset and accuracy not confirmed. Use only as a size and latency reference |
| NTNU ToN_IoT lightweight models (CART, XGBoost) | Raspberry Pi 4 | About 120,000 pps packet throughput under constrained settings | https://www.ntnu.no/ojs/index.php/nikt/article/view/6513/5844 (SEARCH) | Accuracy and latency not confirmed |
| Scapy vs Pyshark vs Pypcap on Raspberry Pi 3 Model B | Pi 3B | Pypcap+dpkt outperformed Pyshark and Scapy in the paper's experiments (SEARCH) | https://sol.sbc.org.br/index.php/wperformance/article/view/36283 | Numbers not confirmed. Pi 3B, not Pi 5 |
| INT8 MLP on UNSW-NB15/CIC-IDS2017 on STM32 NPU, Raspberry Pi 3B+ inference 27 ms | Pi 3B+ | 27 ms (SEARCH, no URL retrieved) | UNVERIFIED | Excluded from targets |

**Gap:** no published result was found that reports accuracy, latency, model size and INT8/ONNX together on a Pi 4 or 5 with CICIDS2017 or UNSW-NB15 (SEARCH, one search only). This limited search does not establish a first; Pi measurements remain OPEN.

### 3.4 "Target to beat" (proposed; the literature values above are floors, not ceilings)

| Target | Metric | Floor from literature | Our proposed bar | How measured |
|---|---|---|---|---|
| T1 CIC17→CSE18 | macro-F1 (and binary F1, both reported) | 39.12% (LDA, Cantone) | Beat the re-run baseline of Cantone's LDA and XGB under our frozen shared schema, and report the Cantone numbers re-run under our metric definition | Kaggle re-run, not the paper's number |
| T1 CSE18→CIC17 | same | 51.21% (RF, Cantone) | same | same |
| T1 LycoS17→LycoS18 | same | 14.42% (XGB, Cantone) | same | same |
| T1 LycoS18→LycoS17 | same | 70.46% (LDA, Cantone) | same | same |
| T0 random split (reference) | macro-F1, PR-AUC | Cantone within-dataset MCC 94.63% | Report only | Required for the collapse figure |
| Benign FPR (campaign-disjoint) | FPR at fixed TPR | Not in literature | Proposed ≤ 1% FPR on held-out benign-day traffic (CIC-IDS2017 Monday, plus home captures) | Our own captures (§6.4) |
| Edge model size | INT8 ONNX file size | 0.65 MB (Pi 4 reference, SEARCH) | ≤ 1 MB for the student | Measured on Kaggle and on the Pi |
| Live pipeline | p95 flow-to-verdict latency on Pi 5 at the test load | No published Pi 5 number | Set after the first measurement | `tcpreplay` load, §6.3 |
| Pi 5 capture | packets/s at zero drop, CPU % | NTNU ~120k pps on Pi 4 (SEARCH) | Measured on Pi 5 with nfstream and Scapy | `tcpreplay` at 10, 100, 500 Mbps |

Predeclare go/kill thresholds before any target run (rule in `docs/CONTEXT.md`: freeze the benchmark before chasing models).

---

## 4. On-device capture and features

### 4.1 Live-capture tools

| Tool | Licence | Linux aarch64 wheel | Facts | Verdict |
|---|---|---|---|---|
| **nfstream** (v6.6.0) | LGPL-3.0-or-later (VERIFIED: https://pypi.org/project/nfstream/ and https://github.com/nfstream/nfstream) | Yes: `manylinux2014_aarch64` and `musllinux_1_2_aarch64` wheels for cp310–cp314 (VERIFIED from the PyPI JSON listing for 6.6.0) | Uses AF_PACKET_V3/FANOUT on Linux, CFFI computation engine, nDPI for layer-7 (VERIFIED: https://raw.githubusercontent.com/nfstream/nfstream/master/README.md). Early flow features: first n packet sizes, inter-arrival times and directions. Post-mortem statistics: min/mean/std/max of packet size and inter-arrival time. Parameters `idle_timeout=120`, `active_timeout=1800`, `statistical_analysis`, `n_dissections`, `splt_analysis` (VERIFIED from README). Dependencies: cffi, psutil, dpkt, numpy, pandas < 3 (VERIFIED PyPI) | **Recommended** capture and feature extractor. Pi 5 throughput not measured |
| **Scapy** | GPL-2.0 (UNVERIFIED, no source pulled) | Pure Python, wheels not needed | Per-packet Python. On a Pi 3B, Pypcap+dpkt beat Scapy and Pyshark in the cited study (SEARCH) | Use for protocol parsing of DNS fields and for tests. Not for the hot path |
| **pyshark** | MIT (UNVERIFIED) | Pure Python wrapper | Wraps `tshark` (UNVERIFIED) | Slower. Use only for offline debugging |
| **dpkt + pypcap** | BSD-3 (UNVERIFIED) | Wheels likely needed | Pi 3B study favoured this route (SEARCH) | Fallback if nfstream does not hold line rate on Pi 5 |
| **Zeek** | BSD (UNVERIFIED) | Packaged on Debian (UNVERIFIED) | Used by ToN_IoT network logs (VERIFIED: UNSW ToN-IoT page) | Option for richer logs; heavier than needed |

**Licence note for us:** nfstream is LGPL-3.0. Using it unmodified as a pip dependency is the normal case; if we modify nfstream itself, the LGPL terms apply to those changes. Take the final call to Maadhav or a licence reviewer before distribution (not legal advice).

Hardware baseline (VERIFIED: https://www.raspberrypi.com/products/raspberry-pi-5/): BCM2712 quad-core Cortex-A76 at 2.4 GHz, 2 MB shared L3, up to 16 GB LPDDR4X, Gigabit Ethernet, 2 × USB 3.0, PCIe 2.0 x1. Target the 4 GB or 8 GB model. Measure with the same board for every run.

### 4.2 Shared feature schema v1 (proposal)

Goal: (a) computable live from nfstream (or nfstream plus Scapy for DNS fields) and (b) derivable from the NF-*-v2 feature files, and approximately from CICFlowMeter CSVs for the CIC datasets.

**Keep (candidate set, about 25 features).** Names are from the NF-v2 feature file (VERIFIED).
- Protocol and service: `PROTOCOL`, `L7_PROTO` (ablate: nDPI numbering differs between tools, UNVERIFIED), `ICMP_TYPE`, `ICMP_IPV4_TYPE`
- Volume: `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`, `FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `DURATION_OUT`
- Rate: `SRC_TO_DST_SECOND_BYTES`, `DST_TO_SRC_SECOND_BYTES`, `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`
- Packet size: `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`, `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`, `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES`
- TCP flags: `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`
- Retransmission: `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_IN_PKTS`, `RETRANSMITTED_OUT_BYTES`, `RETRANSMITTED_OUT_PKTS` (live computability UNVERIFIED for nfstream; if not available, drop)
- DNS (for DNS_TUNNEL): `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER` (parse with Scapy/dpkt; live computability UNVERIFIED)

**Drop (identifiers, leakage, or not live-computable):**
- Identifiers: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `L4_SRC_PORT`, `L4_DST_PORT`, MAC and OUI fields, flow ID, Timestamp and `Stime`/`Ltime`, `DNS_QUERY_ID`. Ablate `L4_DST_PORT` as a bucketed service-port feature only.
- Dataset-specific Argus/Bro fields not produced live: `trans_depth`, `res_bdy_len`, `is_ftp_login`, `ct_flw_http_mthd`, `ct_ftp_cmd` (UNSW-NB15 fields, VERIFIED names).
- Host-window counters (`ct_*`) and the aggregation of "100 connections" (VERIFIED name in UNSW feature file): compute only in the live pipeline with the same window definition, otherwise drop.
- Handshake timing (`tcprtt`, `synack`, `ackdat`, `Sjit`, `Djit`): not verified as live-computable with nfstream. Drop unless shown to be reproducible.
- TCP sequence numbers (`stcpb`, `dtcpb`) and raw window sizes (`swin`, `dwin`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`): ablate. They are OS fingerprints and can shortcut across datasets.
- TTL (`MIN_TTL`, `MAX_TTL`, `sttl`, `dttl`): ablate for the same reason.

Feature-set decision is made once on the T1 validation split, then frozen (§2.2 rule 2).

### 4.3 Mapping to CICFlowMeter CSVs (for CIC-IDS2017 and CSE-CIC-IDS2018)

- A mapping table will be written in code, not by hand, and checked for each dataset. Approximate mapping (UNVERIFIED, to confirm against the CSV headers): flow duration, forward/backward packet counts and byte totals, packet-length min/max/mean/std, flag counts, and IAT statistics map to the schema above.
- CICFlowMeter fields with no NF-v2 counterpart (for example "Init_Win_bytes", "Active"/"Idle" statistics) are dropped from the shared set and kept only in a CICFlowMeter-only ablation.

### 4.4 Preferred route for train/live parity

Recompute features from the public PCAPs using the same nfstream pipeline that runs on the Pi, on Kaggle. This avoids the mismatch between CICFlowMeter and nfstream (the LycoS authors' motivation, SEARCH). Candidate PCAP sources are in §1.2 (`yasiralifarrukh/unsw-and-cicids2017-labelled-pcap-data`, `sonalimishra3/bot-iot-pcap`). Check their contents before relying on them. The CSE-CIC-IDS2018 PCAPs are on the AWS bucket (SEARCH: `s3://cse-cic-ids2018`, no-sign-request sync) and are large: run that on Kaggle as well, never on the laptop.

### 4.5 Throughput limits

- No measured nfstream or Scapy numbers for the Pi 5 exist (SEARCH, no result found). Measure:
  - `tcpreplay` a known pcap at 10, 100 and 500 Mbps into the Pi's interface.
  - Record flow-export rate, dropped packets (interface counters), CPU per core, RSS and p95 flow-to-verdict latency.
- Pi 5 Gigabit Ethernet is the ceiling for capture. Home and SME traffic is far below that, so the realistic test is a sustained load of a few hundred Mbps, not line rate.

---

## 5. Evidence mapping (Evidence-Locked Advisory)

### 5.1 Design

- Same pattern as `src/rakshak/evidence.py`: the detector emits evidence IDs with types and weights; the package has an `allowed_actions` list; the verifier rejects anything outside the package.
- Evidence types are lower snake case, like the SMS types (`credential_request`, `apk_install`, …). Action codes are upper snake case.
- **Add a separate `NetAction` enum** rather than extending the SMS `Action` enum, so that SMS behaviour does not change. The verifier must accept actions from the union of the two enums only for the matching evidence type.
- The network `ACTION_MAP` is the same shape as the SMS one. The existing SMS codes `IGNORE` and `VERIFY_INDEPENDENTLY` are reused.
- Nothing here blocks traffic automatically in Phase 1. Every action is a suggestion the user applies, and the device-side agent never changes router or firewall state without user approval.

### 5.2 Evidence types → allowed action codes (proposal)

| Evidence type | Example trigger on flows | Allowed action codes | Plain-language advice |
|---|---|---|---|
| `port_scan` | Many distinct destination ports or hosts from one source in a short window, mostly SYN-only flows | `BLOCK_SOURCE_IP`, `CHECK_ROUTER_FIREWALL`, `VERIFY_INDEPENDENTLY` | Something on your network or outside it is probing your devices. Block the source or check your router firewall |
| `brute_force_ssh` | Repeated short SSH flows to one host on port 22 with failed-login-like sizes | `CHANGE_PASSWORD`, `DISABLE_PASSWORD_LOGIN`, `BLOCK_SOURCE_IP` | A device is being guessed at over SSH. Change the password and turn off password login |
| `brute_force_other` | Repeated short flows to FTP, Telnet or RDP ports | `CHANGE_PASSWORD`, `BLOCK_SOURCE_IP` | A login service is being guessed at. Change the password and block the source |
| `dos_flood` | High packet or byte rate to one target, SYN-flood shape, UDP flood shape | `RATE_LIMIT`, `CONTACT_ISP`, `CHECK_ROUTER_FIREWALL` | Your device or connection is being flooded. Limit the rate and contact your ISP if it continues |
| `botnet_beaconing` | Periodic outbound flows with near-constant size and interval to one external endpoint | `ISOLATE_DEVICE`, `FACTORY_RESET_IOT`, `CHANGE_PASSWORD` | A device is calling home on a regular schedule. Isolate it and consider a factory reset |
| `dns_tunnel` | Many DNS queries with long names or unusual query types, high `DNS_TTL_ANSWER` variance, from one host | `ISOLATE_DEVICE`, `VERIFY_INDEPENDENTLY` | A device may be tunnelling data through DNS. Isolate it and check it |
| `data_exfil` | Large sustained outbound byte totals to an external endpoint, with asymmetric `OUT_BYTES` ≫ `IN_BYTES` | `ISOLATE_DEVICE`, `CONTACT_ISP`, `VERIFY_INDEPENDENTLY` | Much more data is leaving a device than arriving. Isolate it and check what it is |
| `new_device_rare_port` | Device not seen before on the network, using an unusual port or protocol | `VERIFY_INDEPENDENTLY`, `ISOLATE_DEVICE` | A new device is talking on an unusual port. Confirm it is yours |
| (low risk) | No evidence above, benign-state | `IGNORE` | No action |
| (uncertain) | Scores in the UNCERTAIN band | `VERIFY_INDEPENDENTLY` | Not sure. Check the device list |

### 5.3 Action codes

| Code | Meaning (what the user does) |
|---|---|
| `BLOCK_SOURCE_IP` | Add a block for the source address in the router or firewall (user applies it) |
| `CHECK_ROUTER_FIREWALL` | Review the router's firewall and port-forward rules |
| `CHANGE_PASSWORD` | Change the password of the affected account or device |
| `DISABLE_PASSWORD_LOGIN` | Turn off password logins and use keys (SSH) |
| `RATE_LIMIT` | Apply a rate limit to the affected service or connection |
| `CONTACT_ISP` | Contact the internet provider |
| `ISOLATE_DEVICE` | Move the device to a guest network or unplug it |
| `FACTORY_RESET_IOT` | Factory-reset the IoT device, then update its firmware and change its password |
| `USE_TRUSTED_DNS` | Switch the router to a trusted DNS resolver (optional, added if the DNS-tunnel evidence is used) |
| `VERIFY_INDEPENDENTLY` | Check with the device owner or the vendor (existing SMS code) |
| `IGNORE` | No action (existing SMS code) |

Note on codes: `DO_NOT_CLICK`, `DO_NOT_SHARE_OTP`, `REPORT_1930_OR_CHAKSHU` and `BLOCK_SENDER` remain SMS-only and must not appear in network packages (verifier test required).

---

## 6. Build plan

### 6.1 Kaggle jobs (all training, HPO and evaluation on Kaggle, per `docs/CONTEXT.md`)

| Job | Inputs | Output | Notes |
|---|---|---|---|
| `net_ingest` | `dhoogla/nfunswnb15v2`, `dhoogla/nfcsecicids2018v2`, `dhoogla/nfbotiotv2`, `dhoogla/nftoniotv2`, `bertvankeulen/cicids-2017` (after checking `_plus`), `dhoogla/csecicids2018`, `dhoogla/ctu13` | Shared-schema parquet, one per dataset, plus the mapping table | Check licence in each kernel log. Dedupe here |
| `net_baselines` | Output above | LightGBM and XGBoost results for T0–T4, with clustered bootstrap CIs | Random split reported, not headlined |
| `net_student` | Teacher outputs from baselines | Small MLP and 1D-CNN student, distilled, then INT8 static quantisation and ONNX export | Report size and accuracy before and after INT8 |
| `net_reextract` (optional) | PCAPs from §1.2 | nfstream-extracted features for train/live parity | Only after the Pi pipeline is fixed |

Model choices and their rationale:
- GBDT (LightGBM or XGBoost) is the teacher and the strong baseline on tabular flow features.
- MLP or 1D-CNN is the Pi student. 1D-CNN over packet-size sequences only if the nfstream early features are available live.
- Quantisation: INT8, export to ONNX, measure size and latency on the Pi with ONNX Runtime (aarch64 wheel availability UNVERIFIED; check before committing).

### 6.2 Evaluation splits

Run T0 (reference), T1 (both directions), T2 (leave-one-family-out, within and across), T3 (CIC-IDS2017 by day; UNSW by `Stime`), T4 (group by source host or campaign). Use the corrected labels wherever they exist and say when they are not used.

Pre-register: the feature schema (§4.2), the threshold rule (set on source validation, FPR target), and go/kill thresholds (§3.4).

### 6.3 Pi pipeline

1. nfstream live on the Pi's Ethernet interface (`NFStreamer(source=<iface>, statistical_analysis=True, …)`). Use AF_PACKET (default on Linux).
2. Feature assembler maps flows to the shared schema v1.
3. ONNX Runtime INT8 student scores each flow; host-window aggregation for `port_scan`, `brute_force_*`, `dos_flood`, `botnet_beaconing`.
4. Evidence builder emits the package (§5). Llama via ExecuTorch (quantisation name fixed per `docs/CONTEXT.md`, exact recipe per `docs/EXECUTORCH_PLAN.md`) only verbalises the package. The verifier rejects anything else.
5. Benchmarks on the Pi 5: `tcpreplay` at 10, 100, 500 Mbps; record packets/s, drops, CPU, RSS, p50/p95/p99 flow-to-verdict latency. Airplane-mode and thermal logs as for SMS.

### 6.4 Demo scenario (safe, legal, self-generated only, on an isolated home LAN)

Run only against devices and accounts Maadhav owns, on a separate network with no internet exposure. Nothing targets a third party.
1. Port scan: `nmap` against a test IoT device or a Raspberry Pi on the isolated LAN.
2. SSH brute force: `hydra` against a test SSH container on the same LAN, with a test wordlist.
3. Flood: `hping3` against a VM the user owns, at a low rate with a hard cap.
4. Beaconing: a script on a test device that calls a self-run HTTPS endpoint on a fixed schedule.
5. DNS tunnelling: a synthetic generator sending long-label queries to a self-run resolver.
6. Benign baseline: a home capture of normal traffic (browsing, streaming, IoT chatter), captured by the user, not downloaded from anywhere.

Every attack command runs against the user's own hosts only. Do not run any scan or flood against an address the user does not own.

### 6.5 Timeline

Phase 1 is due 23 Nov 2026 (`docs/CONTEXT.md`). The network module is a coverage add-on. Minimum viable for Phase 1: T1 cross-dataset numbers on the shared schema, the Pi 5 latency and throughput table, and the demo in §6.4 items 1–2 plus 4. Anything beyond that is Phase 2.

---

## 7. Open items (UNVERIFIED or needing a primary source)

1. Corrected CIC-IDS2017 release location and licence (Liu et al. 2022 / Engelen et al. 2021 data-availability sections). Needed before any headline CIC-IDS2017 result.
2. Corrected CSE-CIC-IDS2018 (LycoS-Unicas-IDS2018) availability. Cantone et al. 2024 do not say where to download it.
3. Engelen et al. 2021 PDF numbers (20%, 20%, 10%) are from a search summary. Read the PDF before quoting.
4. Cantone et al. 2024 venue discrepancy (IEEE Access 12 vs. Journal of Information Security and Applications).
5. Metric definition in Cantone et al. 2024 (binary or macro F1). Needed before our T1 comparison.
6. Whether `bertvankeulen/cicids-2017` `_plus` files are corrected or just enriched.
7. Whether `ernie55ernie/improved-cicids2017-and-csecicids2018` is a corrected release. Read its Kaggle description before use.
8. Licence text for UNSW-NB15, CIC-IDS2017/2018/IoT-2023 (UNB), CTU-13, ToN-IoT (official pages not fully read).
9. NSL-KDD counts (125,973 / 22,544) and availability. Not confirmed.
10. ToN-IoT network dataset counts and feature count (45 per arXiv 2212.13994 summary, UNVERIFIED on the primary).
11. nfstream aarch64 install on Pi OS (wheel exists per PyPI; run-time on Pi 5 not tested). Also its exact per-flow feature names: read https://www.nfstream.org/docs/api before writing the mapping.
12. Whether nfstream computes retransmission and DNS-field features live. Needed for `dns_tunnel` and the retransmission features.
13. ONNX Runtime aarch64 wheel and ExecuTorch Pi 5 build (UNVERIFIED here; check in `docs/EXECUTORCH_PLAN.md`).
14. Raw PCAP contents of the Kaggle candidates in §1.2 (not inspected).
15. Search-only leads not used for numbers: Atif and Ali zero-shot preprint (UNVERIFIED, no URL retrieved; reportedly used target labels for feature selection, so it would not meet our protocol anyway); an IEEE Access 2026 paper reporting macro-F1 around 0.12 for CICIDS2017 to CSE-CIC-IDS2018 with 80 features (UNVERIFIED, no URL retrieved); Sarhan et al. 2021 per-direction F1 tables (not read).

---

## 8. Sources (primary first)

- Cantone, Marrocco, Bria. arXiv 2402.10974: https://arxiv.org/abs/2402.10974 (HTML: https://arxiv.org/html/2402.10974v1). IEEE Access 12:144489–144508, 2024 (arXiv record), DOI 10.1109/ACCESS.2024.3472907
- Sarhan, Layeghy, Portmann. "Towards a Standard Feature Set for NIDS Datasets", arXiv 2101.11315: https://arxiv.org/abs/2101.11315
- Raskovalov, Gabdullin, Dolmatov. arXiv 2212.13994: https://arxiv.org/abs/2212.13994
- Arp et al. "Dos and Don'ts of Machine Learning in Computer Security", USENIX Security 2022: https://usenix.org/conference/usenixsecurity22/presentation/arp ; slides: https://www.usenix.net/system/files/sec22_slides-arp.pdf
- Apruzzese, Pajola, Conti. arXiv 2203.04686: https://arxiv.org/abs/2203.04686 ; DOI 10.1109/TNSM.2022.3157344
- Apruzzese, Laskov, Schneider. SoK, arXiv 2305.00550: https://arxiv.org/abs/2305.00550 (title and date via search only)
- Engelen, Rimmer, Joosen. "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE SPW 2021, DOI 10.1109/SPW53761.2021.00009: https://lirias.kuleuven.be/retrieve/713070 (not read in full)
- Liu, Engelen, Lynar, Essam, Joosen. "Error Prevalence in NIDS datasets", IEEE CNS 2022, DOI 10.1109/CNS56114.2022.9947235
- Rosay, Cheval, Carlier, Leroux. "Network Intrusion Detection: A Comprehensive Analysis of CIC-IDS2017", ICISSP 2022, DOI 10.5220/0010774000003120 (search only)
- Dube. "Faulty use of the CIC-IDS 2017 dataset in information security research", DOI 10.1007/s11416-023-00509-7
- Sudyana et al. 2024, DOI 10.1109/cns62487.2024.10735570 (title only)
- UNB CIC datasets: https://www.unb.ca/cic/datasets/ids-2017.html ; https://www.unb.ca/cic/datasets/ids-2018.html ; https://www.unb.ca/cic/datasets/iotdataset-2023.html
- UNSW Bot-IoT: https://research.unsw.edu.au/projects/bot-iot-dataset ; UNSW ToN-IoT: https://research.unsw.edu.au/projects/toniot-datasets ; UNSW-NB15 project: https://research.unsw.edu.au/projects/unsw-nb15-dataset (search)
- Stratosphere CTU-13: https://stratosphereips.org/datasets-ctu13 ; FAQ: https://www.stratosphereips.org/datasets-faq (search)
- Neto et al. 2023, CICIoT2023, Sensors 23(13):5941 (PMC record https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10346235/, search)
- nfstream: https://pypi.org/project/nfstream/ ; https://github.com/nfstream/nfstream ; README https://raw.githubusercontent.com/nfstream/nfstream/master/README.md ; docs https://www.nfstream.org/docs/api
- Raspberry Pi 5 specification: https://www.raspberrypi.com/products/raspberry-pi-5/
- Pi 3B packet-capture study: https://sol.sbc.org.br/index.php/wperformance/article/view/36283 (search)
- Quantum-assisted TinyML IDS (Pi 4 INT8 numbers): https://arxiv.org/pdf/2609.06814 (search)
- NTNU ToN_IoT Pi 4 study: https://www.ntnu.no/ojs/index.php/nikt/article/view/6513/5844 (search)
- Kaggle CLI results (all slugs in §1.2): run in this session with `kaggle datasets list|files|metadata`

# Ported Codex notes: NIDS historical-run audit

## 2026-11-02 — audit of the actual historical run

The generic planned host-cluster protocol is not evidence that host columns were
present. Saved nids_baselines results (tree dea7270125e726d5e70c4ddf139673062be2870309a81b8cec9b15a175a337ec)
show allfour input copies lacked host columns; T5K5000 intervals used
`iid_stratified`, NOThost/campaign clusters. T4 was unavailable. Keep this scope
when citing those intervals. K5000 target attack fractions were .2374/.1886/.982/
.7798, so this was not generally a mostly-benign-home calibration demonstration.
Original target labels were only diagnostic, never used to clean the slices.
No new clustering, split, labels or data download is introduced by this audit.
Next provenance/research checks and primary links: NIDS_NEXT_RESEARCH.md.
