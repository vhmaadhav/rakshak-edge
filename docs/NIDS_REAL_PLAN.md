# Real labelled pcaps through the rules-first NIDS path: pre-registration

Registered 2026-11-12, before any scoring run. The job is `kaggle/jobs/nids_real_pcaps.py`. Nothing in this
plan is retuned after results are seen. A miss is reported as a miss.

## 1. What is frozen

- Host rules and thresholds: `src/rakshak/nids/host_rules.py` (brute force 10 flows / 60 s, dns_tunnel 5 queries / 60 s,
  dos_flood 200 SYN-only flows / 10 s). Port scan: `HostWindow`, 10 s, 20 ports (`evidence.py`).
- Alert and uncertain thresholds, quiet mode, host alert logic: `deploy/pi/service/nids_pipeline.py` and
  `src/rakshak/nids/host_alerts.py`. Quiet mode: a model-only flag is an UNCERTAIN advisory, never an alert.
- Model: `student_cse_cic_ids2018_int8.onnx` from the `nids_baselines` kernel output. The job refuses to run unless the
  file's sha256 starts with `d3dfdcb16dcaa278` (the sha256 the Pi holdout used).
- Flow reader: nfstream 6.6.0 (PyPI release 2026-02-16, at least two weeks old on 2026-11-12). The job installs it
  only if it is missing and the kernel has internet. The stdlib reader is not used on these captures: it reads the
  whole file into memory, and the CIC days are 7.8 to 13 GB.
- Each capture is one request, as in `POST /analyze_flows` and in the holdout runner: host windows are per capture.

## 2. Data (read-only from `/kaggle/input`, never copied to `/kaggle/working`)

| Key | Kaggle source | File | Size (Kaggle listing) | Role |
|---|---|---|---|---|
| `friday` | `congyuanxu/cicids2017` | `Friday*.pcap` (one match, else the capture is MISSING) | 8.84 GB | port scan, DDoS LOIT; benign traffic |
| `tuesday` | `congyuanxu/cicids2017` | `Tuesday*.pcap` (one match) | 11.05 GB | FTP-Patator, SSH-Patator; benign traffic |
| `dns_mal/*` | `daumel/dns-tunneling-dataset` | `dns-exfiltration-dataset/01_original_pcap_files/malicious/<tool>/*.pcap`, sorted | about 0.53 GB | dns_tunnel units, one per file |
| `dns_ben/*` | same | `.../benign/<set>/*.pcap`, sorted | about 31 GB (partly listed) | benign host-hours only |

Wednesday (DoS) and CSE-CIC-IDS2018 are not in this run. Order: friday, tuesday, dns_mal (sorted), dns_ben (sorted).
The Kaggle size of Thursday and Friday differs from the UNB page (research notes); the job does not depend on sizes.

## 3. Ground truth

### 3a. CIC-IDS2017 (friday, tuesday): published schedule, not the CSV labels

Source: UNB page `https://www.unb.ca/cic/datasets/ids-2017.html`, fetched 2026-11-12. The CSV labels are not used,
because the Engelen et al. 2021 and Liu et al. 2022 work reports labelling errors in them. The schedule is the
ground truth, and the CSV labels are not read by the job.

| Unit id | Day | Family (rule name) | Activity (page) | Local window (page) | Attacker IPs (page) | Victim IPs (page) |
|---|---|---|---|---|---|---|
| F1 | Fri 2017-07-07 | port_scan | PortScan, firewall on | 13:55 to 14:43 | Kali 205.174.165.73, and NAT 172.16.0.1 | 192.168.10.50 (= 205.174.165.68) |
| F2 | Fri 2017-07-07 | port_scan | PortScan, firewall off | 14:51 to 15:29 | same | same |
| F3 | Fri 2017-07-07 | dos_flood | DDoS LOIT | 15:56 to 16:16 | Windows 205.174.165.69, .70, .71 | 192.168.10.50 (= 205.174.165.68) |
| T1 | Tue 2017-07-04 | brute_force_other | FTP-Patator | 09:20 to 10:20 | same as F1 | same |
| T2 | Tue 2017-07-04 | brute_force_ssh | SSH-Patator | 14:00 to 15:00 | same as F1 | same |

- Port scan: F1 and F2 are two units, one per sub-scan. Multi-slot windows are not merged.
- A unit's sources are its listed attacker IPs. A flow matches a unit if its source is in the unit's set. Victim IPs
  are not used for matching; they are logged.
- Wednesday DoS (page 9:47 to 11:23) is excluded from this run.

**Timezone (unresolved).** The page gives local times and states no timezone. Its only date reference is "9 a.m.,
Monday, July 3, 2017". Primary assumption, fixed now: **UTC-3** (Atlantic Daylight Time, UNB's location). Sensitivity
offsets **UTC-4, UTC-5 and UTC+0** are computed and reported, but the primary is never switched after results.
If a sensitivity offset gives a clearly better match, that is a finding for a later plan, not a change now.

**NAT and IP uncertainty.** The page says the Kali attacker's NAT address is `172.16.0.1` on other lines and mentions
`172.16.0.11` once on the Heartbleed line, which may be a typo. The job matches both `205.174.165.73` and `172.16.0.1`
and logs every attacker-candidate IP that appears in each window (top talkers), so a wrong IP set is visible in the
report, not hidden.

**Known schedule-level caveats** (Engelen 2021; Liu 2022 abstract-level, full text not read): traffic generation and
flow construction problems, attacks performed but not labelled (Lanvin 2022). The schedule may not match every packet
in a window. A miss inside a window is reported with its top talkers.

### 3b. DNS tunnels (dns_mal): one unit per malicious capture

- Family `dns_tunnel`. The tunnel host is the source with the most flow packets to UDP port 53 in that capture
  (`src2dst_packets`, from the flow reader), chosen before scoring. Ties go to the lexically smaller IP.
- Unit window: the span from the first to the last port-53 flow of that source.
- Tool name (folder) is logged; it is not used for matching.
- Benign DNS captures (dns_ben) have no units. Every host-hour in them counts as benign.

## 4. Unit of evaluation and metrics

Unit = (source IP set, family, attack window). For each unit:

- **Detected (TP)**: some rule alert record has a source in the unit's set and overlaps the window. Overlap: record
  first <= window end and record last >= window start. Otherwise **FN**.
- **Family correct**: a detected record has the rule type equal to the unit family.
- **Time to first alert**: earliest overlapping record first timestamp minus window start, in seconds, floored at 0.
  A record that starts before the window is counted, and flagged as "before window".
- **Model-only** (reported separately, never counted as a detection): the unit has a model-only advisory on a source
  in its set (`alert.is_model_only`), and/or model-flagged flows (family != benign) from its sources inside the window.

Rule alert records are per (source, rule type): the first and last timestamp of the flows that the rule hit or the
window flagged. They come from `host_rules.evaluate` and the `HostWindow` evidence, the same calls the service makes.
The job checks that the set of hosts with records equals the set of hosts with non-model-only alerts in
`host_alerts`, and reports any mismatch.

Metrics reported per family: units, detected, family-correct, misses (unit ids and top talkers), median time to first
alert. Overall: all units, all detections.

## 5. Benign false alerts

- **Benign host-hour**: a pair (source IP, UTC hour of the flow's first timestamp) that has at least one flow in the
  capture and is not a labelled host-window. A labelled host-window is an (attacker source, hour) pair that overlaps a
  unit window for that source.
- **False alert**: a rule record whose (source, UTC hour of its first timestamp) is not a labelled host-window.
  Overall rate = false alerts / benign host-hours. Also reported per rule type.
- Model-only advisories on hosts with no unit are counted per capture (host count). No rate is claimed for them.

## 6. Budget and partial results

- `--max-hours` default **10**, measured from the start of `run`. Checked before each capture and every 5,000 flows
  while reading and scoring.
- A capture that runs out of budget is **not scored** (no partial scoring). It is recorded as `budget_exhausted`, and
  the run stops.
- After each capture, and at the end, the job writes `results.json`, `metrics.json`, `alerts.parquet` and `report.md`
  atomically (write a `.tmp` file, then replace). A killed run leaves the last complete write.
- A capture that cannot be read (for example pcapng with no nfstream, or a missing file) is recorded as `unmeasured`
  with the reason, and the run goes on.

## 7. Throughput

Per capture: flows, read seconds, scoring seconds, flows/s. Labelled "Kaggle CPU, not Pi". Not compared with the Pi
numbers in NIDS_EDGE.md.

## 8. Reporting rules

- Every miss, false alert and budget stop is in `report.md`.
- The sensitivity table for the offsets is in `metrics.json` and `report.md`, labelled as sensitivity.
- No result is used to change a rule, threshold, schedule, offset, the model or the flow reader.
- Nothing here is a published-SOTA claim. Results are first-of-their-kind for this rule set on these captures.

## 9. Not done in this run

- Wednesday (DoS), CSE-CIC-IDS2018, IoT-23, UNSW-NB15, CTU-13: not in the set.
- Schedule timezone and NAT IPs are not verified against a primary source (see 3a).
- No outcome is predicted for any unit. The DDoS LOIT unit is an HTTP-level flood, so a SYN-only rule may miss it.
  That miss is expected to be reported as a miss, not reclassified.
