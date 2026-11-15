# Real labelled pcaps for the W3 NIDS run (research notes, 2026-11-12)

Scope: find real labelled packet captures (not flow CSVs) for the rules-first pipeline in `docs/NIDS_EDGE.md`
(port_scan, brute_force_ssh, brute_force_other, dns_tunnel, dos_flood). Read-only research. Nothing was
downloaded beyond the small Kaggle `.md5` files and metadata JSON. No accounts, no posts.

Sources cited inline. "Not verified" marks anything I could not confirm from a primary source.

## 1. Ranked table

| Rank | Dataset | Families covered (rule mapping) | Size | Licence / terms | Kaggle-attachable | Label quality |
|---|---|---|---|---|---|---|
| 1 | CIC-IDS2017 raw pcaps, Kaggle mirror `congyuanxu/cicids2017` | brute_force_ssh (Tue SSH-Patator), brute_force_other (Tue FTP-Patator), port_scan (Fri PortScan), dos_flood (Wed DoS, Fri DDoS LOIT); benign days | Mon 10.82 GB, Tue 11.05, Wed 13.42, Thu 8.30, Fri 8.84 (decimal GB, from Kaggle listing); total 52.4 GB | Wrapper MIT (Kaggle metadata). Underlying data is UNB's; UNB page says labelled flows are "publicly available for researchers" and asks for citation, no licence named | **Yes**: `congyuanxu/cicids2017`, files `Tuesday-WorkingHours/Tuesday-WorkingHours.pcap` etc. Per-day `.md5` files included | Known errors (Engelen 2021; Liu 2022). Attack windows published (see section 2) |
| 2 | CSE-CIC-IDS2018 raw pcap.zip per day (AWS open data) | brute_force_ssh (Wed 14-02 SSH-Bruteforce), brute_force_other (Wed 14-02 FTP-BruteForce), dos_flood (Wed 21-02, Thu 15-02, Fri 16-02), dns none | pcap.zip per day 38.7 to 59.3 GB incl. logs; whole bucket about 470 GB (summed from HTTPS listing) | Redistribution allowed "in any form" with citation and link to AWS page (UNB 2018 page) | **No** raw-pcap mirror found on Kaggle. Only flow CSVs (`masjohncook/cse-cic-ids2018-cicflowmeter`, 1.7 GB) | Errors also reported in Liu 2022 and Engelen 2021 abstract-level summaries |
| 3 | DNS exfiltration pcaps `daumel/dns-tunneling-dataset` | dns_tunnel (plain UDP/53 tunnels: ozymandns, dnscat2, cobaltstrike, iodine, dnspot, dnsexfiltrator, dns2tcp, tcp_over_dns); benign DNS (university, top-1M) | Malicious pcaps about 0.53 GB total; benign about 31 GB (university) plus 0.17 GB (top-1M). Kaggle reports 15.9 GB dataset size | CC BY 4.0 (Kaggle metadata) | **Yes**: `daumel/dns-tunneling-dataset`, paths `dns-exfiltration-dataset/01_original_pcap_files/{malicious,benign}/<tool>/*.pcap`. Kaggle listing only partly paged (see caveats) | Label = folder (tool). Tools and pcaps collected from Singh et al., Gao et al., Chen et al. and the author's own lab, per the dataset description |
| 4 | IoT-23 (Stratosphere) pcaps | port_scan only (11 of 23 scenarios list `PartOfAHorizontalPortScan`); no SSH brute force (page mentions Telnet only); DNS column present but unreadable | Full set about 20 to 21 GB (UNB-hosted page, Stratosphere IoT-23 page) | Stratosphere page names no licence, only citation request. Kaggle full copy `surajsooraj26/iot-23` 9.2 GB is **CSV, not pcap** | Full pcap copy: not verified. Kaggle `iot-23` is CSV (CC0-1.0 wrapper, 23 CSVs) | Flow labels from Zeek, `detailed-label` column |
| 5 | UNSW-NB15 raw pcaps | port_scan-like (Reconnaissance class); no explicit SSH brute force (page does not mention SSH) | 100 GB raw pcaps per UNSW page; per-file names not given | Academic use free "in perpetuity"; commercial use needs agreement; cite the listed papers | Kaggle has CSV only (`dhoogla/unswnb15`, 12 MB); no raw pcap mirror found | Ground truth in `UNSW-NB15_GT.csv` |
| 6 | CTU-13 (Stratosphere botnet captures) | Not verified for SSH or port scan or DNS (per-scenario table is an image on the page) | Scenario pcaps from 58 MB (scenario 1) to 70.7 GB (scenario 10, Kaggle listing); total about 2.5 GB CSV plus pcaps | CC-BY (Stratosphere, via search summary) | Yes, `berty2/ctu-13-dataset` (listing shows `botnet-capture-*.pcap` files) | Manually labelled flows (Stratosphere) |
| 7 | CIRA-CIC-DoHBrw-2020 (UNB) | Only DoH tunnels (dns2tcp, DNSCat2, Iodine over HTTPS). Does **not** match the plain UDP/53 rule | Raw pcap present; no total size given | Redistribution "in any form", cite DoHMeter and Montazeri Shatoori 2020 (UNB page) | Not found on Kaggle | Labelled by server IP (UNB page) |
| 8 | MAWI WIDE archive | Benign only; no attack labels for our rules | Daily or 15-min trace sizes not stated on the page | "Research purposes only"; privacy prohibitions (MAWI page) | Not on Kaggle (not searched further) | Anomaly labels from MAWILab, which the page says was terminated Dec 2024 |

Families with no good real-pcap source found: **brute_force_other on a non-CIC source** (only CIC-IDS2017 FTP-Patator); **dos_flood** outside CIC-IDS2017 (not searched further); **SSH brute force** outside CIC-IDS2017/2018 (search returned nothing relevant).

## 2. Published attack schedules (for time-window ground truth)

**CIC-IDS2017** (UNB page, `https://www.unb.ca/cic/datasets/ids-2017.html`; times as printed there, no timezone given):

| Day | Attack | Window (page) |
|---|---|---|
| Tue 4 Jul | FTP-Patator | 9:20 to 10:20 a.m. |
| Tue 4 Jul | SSH-Patator | 2:00 to 3:00 p.m. |
| Wed 5 Jul | DoS slowloris / Slowhttptest / Hulk / GoldenEye | 9:47 to 10:10 / 10:14 to 10:35 / 10:43 to 11:00 / 11:10 to 11:23 a.m. |
| Wed 5 Jul | Heartbleed | 3:12 to 3:32 p.m. |
| Thu 6 Jul | Web attacks (brute force, XSS, SQLi) | 9:20 to 10:00 / 10:15 to 10:35 / 10:40 to 10:42 a.m. |
| Thu 6 Jul | Infiltration (several) | 2:19 p.m. onward; Dropbox 3:04 to 3:45 p.m. |
| Fri 7 Jul | Botnet ARES | 10:02 to 11:02 a.m. |
| Fri 7 Jul | Port scan (sub-scans listed) | 1:55 to 3:29 p.m. |
| Fri 7 Jul | DDoS LOIT | 3:56 to 4:16 p.m. |

Daily sizes printed on the same page (GB): Mon 11.0, Tue 11, Wed 13, Thu 7.8, Fri 8.3. **These do not match the Kaggle file sizes** (Thu 8.30 GB and Fri 8.84 GB in decimal). Not resolved; the Kaggle md5 values could be compared with a UNB checksum if one is published (none seen).

**CSE-CIC-IDS2018** (UNB page, `https://www.unb.ca/cic/datasets/ids-2018.html`), Table 2 (start to finish, no timezone given):
- Wed 14-02-2018: FTP-BruteForce 10:32 to 12:09; SSH-Bruteforce 14:01 to 15:31.
- Thu 15-02-2018: DoS GoldenEye 9:26 to 10:09; Slowloris 10:59 to 11:40.
- Fri 16-02-2018: DoS SlowHTTPTest 10:12 to 11:08; Hulk 13:45 to 14:19.
- Tue 20-02 and Wed 21-02: DDoS LOIC/HOIC variants.
- Others: web brute force, XSS, SQLi, infiltration, bot.

## 3. Labelling errors (CIC-IDS2017 and 2018)

- Engelen, Rimmer, Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study", IEEE S&P Workshops 2021. Abstract (via search summary, full text not readable): problems in traffic generation, flow construction, feature extraction and labelling; more than 20% of traces reconstructed or relabelled; the issue persists in CSE-CIC-IDS2018. Source: `https://lirias.kuleuven.be/retrieve/713070` (PDF binary, not readable here).
- Liu, Engelen, Lynar, Essam, Joosen, "Error Prevalence in NIDS datasets", IEEE CNS 2022, DOI 10.1109/CNS56114.2022.9947235. Reports undocumented errors in attack orchestration, feature generation, documentation, labelling. Provides a recreated dataset with labelling logic "reverse-engineered, corrected" (`https://intrusion-detection.distrinet-research.be/CNS2022/`, download page not readable). Code: `https://github.com/GintsEngelen/CNS2022_Code`, contains `CICIDS2017_labelling_fixed_CICFlowMeter.ipynb` and an `original_version` script. README says **no** corrected dataset is shipped there; PCAPs must be downloaded from the dataset sites. The README also tells users to delete a day-boundary capture file (`... 172.31.69 - Copy.24`) that duplicates flows.
- Lanvin et al., CRiSIS 2022, DOI 10.1007/978-3-031-31108-6_2: packet misorder, packet duplication, attacks performed but not labelled (search summary). Per-day percentages **not verified**.

Implication for W3: the shipped pipeline should be scored against the corrected labelling (Liu 2022 scripts) as well as the original CICFlowMeter labels, and the pcap-to-window mapping above should be used as the primary ground truth because it is independent of the flow labeller.

## 4. Published rule-based baselines

Searched for Zeek or Snort per-family results on CIC-IDS2017. **None found** with per-family detection and false-positive numbers. Closest items: a profile-based port-scan detector reported better than Snort (2017 paper, dataset not confirmed) and a Lisbon thesis on port-scan rules on CIC-IDS-2017 (numbers not readable). No published comparison to use as a reference point. The W3 result will therefore be a first-of-its-kind baseline; report it as such.

## 5. Recommended minimal set (Kaggle CPU kernel, about 20 GB disk, 12 h)

Disk is the binding limit. A single CIC-IDS2017 day pcap is 8.3 to 13.4 GB, so only one day should be on disk at a time.

1. **Friday, CIC-IDS2017** (`congyuanxu/cicids2017`, 8.84 GB): covers port_scan (1:55 to 3:29 p.m.) and dos_flood (DDoS LOIT, 3:56 to 4:16 p.m.). Benign morning traffic in the same file. Run first.
2. **Tuesday, CIC-IDS2017** (11.05 GB): covers brute_force_ssh (2:00 to 3:00 p.m.) and brute_force_other (FTP, 9:20 to 10:20 a.m.). Run second, after deleting Friday. Attach as a kernel input, not a download.
3. **Wednesday, CIC-IDS2017** (13.42 GB, optional): DoS families. Likely does not fit with another day; run alone.
4. **DNS tunnel set**: `daumel/dns-tunneling-dataset` malicious folders only (about 0.53 GB) plus a benign subset (for example 2 GB of university traffic) for the false-positive side. Tests the `dns_tunnel` rule (plain UDP/53), which CIRA DoH cannot test.
5. Optional cross-check: IoT-23 pcaps for port scan, only if a pcap copy is located (not verified).

Use the Liu 2022 labelling scripts for the corrected ground truth. Score per host window as `docs/NIDS_EDGE.md` specifies, with no rule retuning.

Compute caveat: nfstream throughput on 11 GB pcaps on the Kaggle CPU was not measured; check the 12 h budget with a small run first.

## Caveats and unverified items

- Kaggle file sizes are from the listing (decimal GB); UNB's printed sizes disagree for Thu and Fri.
- The DNS dataset's Kaggle listing was paged only in part; the total pcap size (about 31 GB) comes from partial listing sums.
- The CIC-IDS2017 Kaggle mirror is a third-party copy. Its md5 values (Friday 7366c552..., Monday 8525733..., Thursday 043d37cd..., Tuesday 2d976fec..., Wednesday a8c9cd0e...) were not compared with an official checksum.
- The 2018 bucket is public; the per-day `pcap.zip` must be unzipped (partial extraction via HTTP range requests is possible but untested).
- Engelen and Lanvin full texts were not readable; error details above are abstract-level.
- CTU-13 SSH or DNS scenarios and IoT-23 full-pcap licence were not confirmed.

## Sources

- UNB CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNB CIC-IDS2018: https://www.unb.ca/cic/datasets/ids-2018.html
- AWS registry (cse-cic-ids2018): https://registry.opendata.aws/cse-cic-ids2018/
- Kaggle `congyuanxu/cicids2017` (listing and metadata, MIT)
- Kaggle `daumel/dns-tunneling-dataset` (listing, CC BY 4.0)
- Kaggle `surajsooraj26/iot-23` (CC0-1.0 wrapper, CSV)
- Stratosphere IoT-23: https://www.stratosphereips.org/datasets-iot23
- Stratosphere CTU-13: https://www.stratosphereips.org/datasets-ctu13
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- MAWI: https://mawi.wide.ad.jp/mawi/
- UNB CIC-Bell-DNS2021: https://www.unb.ca/cic/datasets/dns-2021.html
- UNB CIRA-CIC-DoHBrw-2020: https://www.unb.ca/cic/datasets/dohbrw-2020.html
- Engelen et al. 2021: https://lirias.kuleuven.be/retrieve/713070
- Liu et al. 2022 (CNS): https://intrusion-detection.distrinet-research.be/CNS2022/ and https://github.com/GintsEngelen/CNS2022_Code
- Lanvin et al. 2022 (CRiSIS): https://doi.org/10.1007/978-3-031-31108-6_2
