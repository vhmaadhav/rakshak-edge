# Network path on the Raspberry Pi 5 (POST /analyze_flows)

Date: 2026-11-06 (laptop tests and Pi run). Scope: Problem Statement 3, network intrusion part.
Everything runs on the device. Nothing captures live traffic, and no test sends a packet.

## What runs

- **Endpoint** `POST /analyze_flows` in the same resident service as `/analyze` (SMS). Body is either a
  classic pcap (raw bytes, magic `a1b2c3d4` family) or JSON `{"flows": [...]}` of nfstream-style flow
  dicts (`rakshak.nids.live.NFSTREAM_KEYS`). Body cap 20 MB (413 above it, 411 without Content-Length,
  400 for a body that is neither). Same 503 `cooling` refusal as `/analyze` at 75 C.
- **Flow source.** A pcap goes to nfstream 6.6.0 when it imports (`--nids-flow-source auto`, the default).
  Otherwise the stdlib reader `src/rakshak/nids/pcap_flows.py` is used (the fallback, see below). JSON
  flows skip this step (`flow_source: json`).
- **Scoring.** Each flow goes through `pi_monitor.flow_decision`: `live.featurize_flow` (19 live
  features), the INT8 ONNX student, and host-window evidence (`HostWindow`, 10 s, distinct destination
  ports per source, the same code as `pi_monitor`). The window is per request.
- **Per-host alerts** (`src/rakshak/nids/host_alerts.py`). One advisory per source host, one evidence
  item per type (largest weight kept). A host is an alert when one of its flows is in the ALERT state
  (score >= the calibrated threshold) or the host window flags it (a port scan). A host whose flows are
  only UNCERTAIN from the model is not an alert: it goes to `watchlist` with its counts. The package,
  template and verifier are the existing network code (`build_network_package`, `net_template`,
  `verify`, with its network checks).
- **Advisory explainer:** template only (`net_template`). The Llama explainer is not wired for network
  evidence. The `--net-explainer llm` option from the plan was not built. This is a gap, not a choice
  that was measured.
- **Remediation** (`src/rakshak/remediation.py`): deterministic `{code, title, steps[]}` list for the
  chosen action first, then the other allowed actions (IGNORE dropped when another code is present).
  Same field on `/analyze` (SMS) and on each network alert. See "Remediation" below.
- **Page:** a "Network" section in `deploy/pi/service/index.html` uploads a .pcap and shows per-host
  alerts, evidence, advisory, remediation, the watchlist count and timings. No external assets.

## Model choice

Four INT8 students were trained, one per dataset (`kaggle/out/nids_baselines/output/student_*_int8.onnx`,
about 6.7 KB each, all within the 1 MB budget). The shipped one is **cse_cic_ids2018 INT8**
(registry id `nids_cse_cic_ids2018_int8`, sha256 pinned, synced to `models/nids/` on the Pi).

Reasons, from `report.md` "Student (INT8 ONNX, live features)":

| Student (trained on) | Within-dataset F1 | MCC | PR-AUC | Recall @ FPR 0.1% | Recall @ FPR 1% | Benign / attack used |
|---|---|---|---|---|---|---|
| unsw_nb15 | 0.830 | 0.785 | 0.815 | 0.068 | 0.238 | 13097 / 4130 |
| **cse_cic_ids2018** | 0.796 | 0.777 | 0.794 | **0.622** | **0.679** | 43511 / 9783 |
| bot_iot | 0.997 | 0.873 | 1.000 | 0.870 | 0.870 | 391 / 18834 |
| ton_iot | 0.903 | 0.458 | 0.911 | 0.008 | 0.068 | 9850 / 36108 |

- unsw_nb15 has the best F1 but a 13% false-positive rate at its max-F1 point (FPR 0.130), against 0.5%
  for cse. For a home device the low-FPR operating point matters more, and cse keeps about 9 times the
  recall of unsw at FPR 0.1%.
- bot_iot looks best, but 95% of its rows are attacks (391 benign). The benign-dominated deployment
  assumption fails there (the report says so), so its numbers say little about home traffic. It also
  flags every synthetic flow below (see the synthetic results).
- ton_iot has low recall at low FPR.
- **Thresholds** are the calibration thresholds from `results.json` (`protocols/student/results/student
  cse_cic_ids2018`): alert = max-F1 point **0.0015007** (the F1 in the table), uncertain = 1%-FPR point
  **0.0005502**. Scores are 1 - P(benign), as in the Kaggle evaluation. The `pi_monitor` CLI still has its
  own defaults (0.5 / 0.2); those were not changed, and they do not match this student's scale.

### Cross-dataset (honest, near chance for the student)

Within-dataset numbers are high. Cross-dataset numbers are not useful for deployment on an unseen network:

| Student (trained on -> tested on) | F1 | MCC | ROC-AUC | PR-AUC | Recall @ FPR 1% |
|---|---|---|---|---|---|
| unsw -> cse_cic_ids2018 | 0.129 | -0.291 | 0.340 | 0.167 | 0.210 |
| unsw -> bot_iot | 0.958 | -0.002 | 0.133 | 0.936 | 0.042 |
| unsw -> ton_iot | 0.555 | -0.155 | 0.398 | 0.768 | 0.240 |
| cse -> unsw_nb15 | 0.297 | -0.361 | 0.423 | 0.285 | 0.805 |
| cse -> bot_iot | 0.332 | 0.020 | 0.662 | 0.981 | 0.282 |
| cse -> ton_iot | 0.718 | 0.183 | 0.608 | 0.844 | 0.659 |
| bot -> unsw_nb15 | 0.351 | -0.022 | 0.574 | 0.364 | 0.049 |
| bot -> cse_cic_ids2018 | 0.353 | 0.150 | 0.589 | 0.224 | 0.336 |
| bot -> ton_iot | 0.817 | 0.266 | 0.655 | 0.822 | 0.124 |
| ton -> unsw_nb15 | 0.351 | -0.212 | 0.176 | 0.153 | 0.003 |
| ton -> cse_cic_ids2018 | 0.268 | 0.002 | 0.500 | 0.353 | 0.072 |
| ton -> bot_iot | 0.783 | -0.021 | 0.503 | 0.980 | 0.001 |

Source: `report.md`, "Student (INT8 ONNX, live features)" (the within rows are in the same table).
Caveats from the report: F1 and PR-AUC are dominated by the attack base rate of the target (bot_iot is
mostly attacks, so high PR-AUC there is not a detection result). Several cross rows have ROC-AUC below
0.5, so the ranking itself reverses across sites.

The leave-one-dataset-out baselines (`report.md` T5, headline K=5000, LightGBM raw) say the same thing.
Target cse_cic_ids2018: F1 0.343, MCC 0.145, ROC-AUC 0.622, FPR @ R=0.90 0.738. Target unsw_nb15: F1
0.128, MCC 0.005, ROC-AUC 0.450, FPR @ R=0.90 0.996. Target bot_iot: F1 0.203, ROC-AUC 0.710, FPR @ R=0.90
0.924 (attack-dominated). Target ton_iot: F1 0.603, MCC 0.177, ROC-AUC 0.650, FPR @ R=0.90 0.701. The
best T5 baseline per target does not change this picture (hybrid IF on cse: MCC 0.337, ROC-AUC 0.745).

**Conclusion for the device:** the detector is a within-dataset model. It is not shown to generalise to
an unseen home network. A shipped alert is a signal to check, not a verdict.

## Rules-first policy (pre-registered 2026-11-07, before any run)

Decision by the owner: deterministic host-window rules decide network alerts. A model-only flag never
raises an alert by itself. The thresholds below were written before any pcap of the new rules was run,
and they are not to be changed after seeing results. A bad result is reported as a fail, not retuned.

### Rule thresholds (fixed)

| Rule | Shape that fires | Window | Weight | Rationale |
|---|---|---|---|---|
| `port_scan` (existing, unchanged) | >= 20 distinct destination ports from one source | 10 s | 0.9 | Kept from the 2026-11-06 build so the old result stays comparable. |
| `brute_force_ssh` | >= 10 **short** flows (TCP, duration <= 5 s, total bytes <= 10 KB, read as 10,240 bytes; clarified before any run) from one source to one destination on port 22 | 60 s | 0.9 | Ten failed logins is a clear count and a short session is what a failed login looks like. 60 s, not 10 s: a slow attack at 2 s spacing puts ten attempts in 18 s, which a 10 s window cannot hold. |
| `brute_force_other` | same shape, ports 21, 23, 3389, 5900 | 60 s | 0.9 | Same reasoning as SSH, one rule for the service ports that take passwords. |
| `dns_tunnel` | >= 5 DNS queries (UDP, dst port 53) from one source whose name has a label >= 40 chars, or a name >= 100 chars, or the longest label has Shannon entropy >= 3.8 bits/char | 60 s | 0.9 | Long or random-looking labels are the usual carrier of DNS tunnelling. Five queries stops a single odd name from firing. The name comes from the query question (nfstream `requested_server_name`, or the stdlib DNS parser). |
| `dos_flood` | >= 200 SYN-only flows (TCP, at least one SYN, no packet back from the destination) from one source to one destination | 10 s | 0.85 | A SYN flood shows as many unanswered connection attempts. 200 in 10 s is far above a normal browser load. The packet-rate option (>= 1,000 pps for >= 5 s) is **not built**: flows only give first/last times, so a per-second rate would be a guess. |

Window is a span test: a set of events fires when the last minus the first is <= the window length.
All rule counts are per request, the same as `HostWindow`. Window evidence gets source `window` (port scan,
brute force, dos) or `rule` (DNS). Rule evidence drives risk, state, family and action exactly as the
existing port-scan rule does: the family is the rule name (`port_scan`, `brute_force_ssh`,
`brute_force_other`, `dns_tunnel`, `dos_flood`).

### Model roles (`--net-model-mode {quiet,alert}`, env `RAKSHAK_NET_MODEL_MODE`, default `quiet`)

- **quiet:** a host with a rule hit uses only the rule evidence for family, action and allowed actions. Its
  model score (p_attack, flagged flows) is reported next to it and cannot change the family.
  A host with only model evidence (no rule) is an advisory with state UNCERTAIN, risk capped at UNCERTAIN,
  action VERIFY_INDEPENDENTLY, and advisory path `model_only_quiet`. Its remediation is VERIFY_INDEPENDENTLY
  only. The summary is a fixed neutral sentence: "The traffic model marked some flows as unusual, but no
  known attack pattern was found. Check the device if you don't recognise it."
  It does not count in `decision.alerts`; it is counted in `decision.model_only_hosts`.
- **alert:** the 2026-11-06 behaviour (model ALERT flows raise an alert with the model family), plus the new
  rules. Kept for comparison only.

### Pre-registered expected outcomes (quiet mode unless stated)

| Profile, pcap | Expected rule outcome | Expected state / risk | Expected action |
|---|---|---|---|
| design, benign | no rule alert | no alert (model-only advisory allowed) | - |
| design, port_scan | port_scan | UNCERTAIN / HIGH | BLOCK_SOURCE_IP |
| design, ssh_brute | brute_force_ssh | UNCERTAIN / HIGH | CHANGE_PASSWORD |
| design, dns_long | dns_tunnel | UNCERTAIN / HIGH | ISOLATE_DEVICE |
| holdout, benign | no rule alert | no alert | - |
| holdout, ssh_slow (12 at 2 s) | brute_force_ssh | UNCERTAIN / HIGH | CHANGE_PASSWORD |
| holdout, port_scan (25 ports) | port_scan | UNCERTAIN / HIGH | BLOCK_SOURCE_IP |
| holdout, dns_tunnel (base32-like labels) | dns_tunnel | UNCERTAIN / HIGH | ISOLATE_DEVICE |
| holdout, syn_flood (300 SYN) | dos_flood | UNCERTAIN / HIGH | RATE_LIMIT |
| holdout, near_miss (6 SSH logins, 3 long CDN lookups) | no rule alert | no alert | - |

"Action" is the template's chosen action (first code of the primary rule's map that the package allows).
Alert mode on the design profile is expected to give the same rule family and action, with the model
evidence listed as well. The model is not expected to decide any family.

### Holdout profile (`make_test_pcaps.py --profile holdout`)

Different seed and different parameters from the design profile (no parameter of the rules was tuned on it).
Addresses are RFC 1918 only: attacker 10.0.1.23, scan source 10.0.2.50, flood source 10.0.3.77, clients
192.168.2.x, servers 192.168.2.10 to .13. mDNS is sent unicast to 192.168.2.1 (port 5353), not to the
multicast group, so the file stays inside RFC 1918. Files are written with `wrpcap` only.
Nothing is sent, and no scanner or socket is involved.

### Holdout set (generated 2026-11-12, before scoring)

Written before any rule was run on these files. Generator: `deploy/pi/service/make_holdout_pcaps.py`
(seed label `holdout-2026-11-12:<file name>`, scapy `wrpcap`, RFC 1918 only, nothing sent). Output goes to
`results/nids_holdout/pcaps/` (gitignored). This set is not the design profile and not the earlier
`--profile holdout` of `make_test_pcaps.py`: new seed, new addresses (172.16.5.x clients and servers, attacker
10.20.30.40, scanner 10.20.31.41, flooder 10.20.32.42) and new parameters. Each file has one expected outcome,
fixed here. "Near" rows are threshold cases, where the expected outcome is the rule as written. Two benign
files are look-alikes on purpose; the CDN look-alike of six lookups meets the length rule by design and is
expected to be a false alert. A miss is a miss: it is reported, not retuned.

Predictions written in advance: a slow SSH rate of 7 s (span 63 s) and a slow SYN scan of 1 s per port are
expected to fall outside the window rules. The 24-char hex DNS names have no prediction (the entropy is near
the 3.8 bit line).

| File | Group | Settings | Expected | Expected source |
|---|---|---|---|---|
| `ho_benign_web_dns.pcap` | benign | 200-packet HTTPS download, 10 HTTPS sessions to 4 servers, 12 DNS names, NTP, 2 mDNS | none | - |
| `ho_benign_ssh_interactive.pcap` | benign | 8 interactive SSH sessions, 40 keystroke/echo exchanges each, 25 s, 30 s apart | none | - |
| `ho_benign_ssh_bulk.pcap` | benign | one SSH bulk download, 300 x 1400 B, 0.05 s gap (about 15 s) | none | - |
| `ho_benign_near_miss.pcap` | benign | 6 short SSH logins 4 s apart (below 10), 3 lookups of the 42-char CDN label 5 s apart | none | - |
| `ho_benign_cdn_lookalike_4.pcap` | benign | 4 lookups of the 42-char CDN label in 30 s (below 5), then 2 normal names | none | - |
| `ho_benign_cdn_lookalike_6.pcap` | benign | 6 lookups of the 42-char CDN label 8 s apart (rule shape met on purpose; expected FP) | none | - |
| `ho_ssh_brute_fast.pcap` | main | 40 short SSH attempts 0.5 s apart from 10.20.30.40 to :22 | brute_force_ssh | 10.20.30.40 |
| `ho_ssh_brute_slow.pcap` | main | 12 short SSH attempts 6 s apart (span 66 s, 11 fit in 60 s) | brute_force_ssh | 10.20.30.40 |
| `ho_telnet_brute.pcap` | main | 12 short sessions to :23 1 s apart | brute_force_other | 10.20.30.40 |
| `ho_port_scan_syn_fast.pcap` | main | 40 ports, SYN then RST-ACK, 0.05 s apart | port_scan | 10.20.31.41 |
| `ho_port_scan_connect_fast.pcap` | main | 30 ports, TCP connect (SYN, SYN-ACK, ACK, RST), 0.1 s apart | port_scan | 10.20.31.41 |
| `ho_dns_tunnel_b32.pcap` | main | 20 TXT queries, 32-char base32-like labels, about 1 s apart, no answers | dns_tunnel | 172.16.5.60 |
| `ho_dns_tunnel_long.pcap` | main | 8 TXT queries, 45 to 60 char lowercase+digit labels, 3 s apart | dns_tunnel | 172.16.5.60 |
| `ho_dos_syn_300_3s.pcap` | main | 300 SYN-only connections to :80 in 3 s | dos_flood | 10.20.32.42 |
| `ho_dos_syn_1000_20s.pcap` | main | 1000 SYN-only connections to :80 in 20 s (50/s) | dos_flood | 10.20.32.42 |
| `ho_mixed_attacks.pcap` | mixed | 10 SSH attempts 1 s apart from 10.20.30.40, 30-port SYN scan from 10.20.31.41, 4 HTTPS | brute_force_ssh + port_scan | 10.20.30.40 + 10.20.31.41 |
| `ho_ssh_brute_9.pcap` | near | 9 short SSH attempts 3 s apart (one below the line) | none | - |
| `ho_ssh_brute_10_5s.pcap` | near | 10 short SSH attempts 5 s apart (span 45 s, exactly the line) | brute_force_ssh | 10.20.30.40 |
| `ho_ssh_brute_10_7s.pcap` | near | 10 short SSH attempts 7 s apart (span 63 s, at most 9 fit in 60 s) | none | - |
| `ho_port_scan_20.pcap` | near | 20 ports, SYN scan, 0.5 s apart (exactly the 10 s line) | port_scan | 10.20.31.41 |
| `ho_port_scan_slow.pcap` | near | 40 ports, SYN scan, 1.0 s apart (10 s window holds about 11 ports) | none | - |
| `ho_dns_tunnel_5.pcap` | near | 5 base32-like queries 4 s apart (exactly the count line) | dns_tunnel | 172.16.5.60 |
| `ho_dns_tunnel_4.pcap` | near | 4 base32-like queries 4 s apart (one below the line) | none | - |
| `ho_dns_hex24.pcap` | near | 10 hex labels of 24 chars, 2 s apart (entropy near the 3.8 bit line; no prediction made) | no prediction | - |
| `ho_dos_syn_150.pcap` | near | 150 SYN-only connections in 10 s (below 200) | none | - |
| `ho_dos_syn_200.pcap` | near | 200 SYN-only connections, 0.05 s apart (span 9.95 s, exactly the line) | dos_flood | 10.20.32.42 |

Scoring rules for this set (fixed before the run): a family case is a TP when the expected source raises an
alert, and "family right" when the alert family equals the expected one. A benign file with any alert is an
FP. Model-only mode alerts on any host with a flow in the model's ALERT state (score >= 0.0015007), with the
host window and the rules left out. Combined mode is `--net-model-mode alert` (model ALERT flows and the rules
together; the strict 2026-11-06 behaviour with no rules is no longer selectable).

### Holdout results (2026-11-12)

Scored on the laptop (`deploy/pi/service/run_holdout_nids.py`, output `results/nids_holdout/results_laptop.md`
and `.json`). Model: `student_cse_cic_ids2018_int8.onnx`, sha256 prefix `d3dfdcb16dcaa278` (the copy under
`kaggle/out/`). nfstream is not installed on the laptop, so all 26 files went through the stdlib reader
(`pcap_stdlib`). 2017 flows in total. Scored after the set above was fixed; no rule or threshold was changed.

Summary (TP/FN for attack instances: a family instance is a TP when the expected source raised an alert;
"family right" means the alert carried the expected family, with the model's `brute_force` and `dos_ddos`
names counted as the matching rule family):

| Family (instances) | (a) rules-first TP/FN, family right | (b) model-only TP/FN, family right | (c) combined TP/FN, family right |
|---|---|---|---|
| brute_force_ssh (3) | 3/0, 3/3 | 3/0, 0/3 | 3/0, 3/3 |
| brute_force_other (1) | 1/0, 1/1 | 1/0, 0/1 | 1/0, 1/1 |
| port_scan (3) | 3/0, 3/3 | 0/3, 0/3 | 3/0, 3/3 |
| dns_tunnel (2) | 2/0, 2/2 | 0/2, 0/2 | 2/0, 2/2 |
| dos_flood (2) | 2/0, 2/2 | 0/2, 0/2 | 2/0, 2/2 |
| Benign files with any alert (FP/n) | 1/6 | 5/6 | 6/6 |

The `ho_mixed_attacks` file counts once in brute_force_ssh and once in port_scan (two sources, both found in
rules mode). The one rules-mode benign alert is `ho_benign_cdn_lookalike_6`, which meets the length rule on
purpose (six lookups of a 42-character label in 40 s). It is a false alert of the rule as written, not a
surprise.

Near-threshold files (the expected outcome is the rule as written):

| Near-threshold file | Expected | (a) rules-first alerts (as specified) | (b) model-only alerts (as specified) | (c) combined alerts (as specified) |
|---|---|---|---|---|
| `ho_ssh_brute_9` | none | none (yes) | 10.20.30.40 dos_ddos (no, alert raised) | 10.20.30.40 dos_ddos (no, alert raised) |
| `ho_ssh_brute_10_5s` | brute_force_ssh | 10.20.30.40 brute_force_ssh (yes) | 10.20.30.40 dos_ddos (no) | 10.20.30.40 brute_force_ssh (yes) |
| `ho_ssh_brute_10_7s` | none | none (yes) | 10.20.30.40 dos_ddos (no, alert raised) | 10.20.30.40 dos_ddos (no, alert raised) |
| `ho_port_scan_20` | port_scan | 10.20.31.41 port_scan (yes) | none (no) | 10.20.31.41 port_scan (yes) |
| `ho_port_scan_slow` | none | none (yes) | none (yes) | none (yes) |
| `ho_dns_tunnel_5` | dns_tunnel | 172.16.5.60 dns_tunnel (yes) | none (no) | 172.16.5.60 dns_tunnel (yes) |
| `ho_dns_tunnel_4` | none | none (yes) | none (yes) | none (yes) |
| `ho_dns_hex24` | no prediction | none | none | none |
| `ho_dos_syn_150` | none | none (yes) | none (yes) | none (yes) |
| `ho_dos_syn_200` | dos_flood | 10.20.32.42 dos_flood (yes) | none (no) | 10.20.32.42 dos_flood (yes) |

Per file (rules alerts with state, risk, action; model-only and combined alerts; model flags in rules mode):

| File | Group | Expected | (a) rules alerts | (b) model-only alerts | (c) combined alerts | Model flags (a) |
|---|---|---|---|---|---|---|
| `ho_benign_web_dns` | benign | none | none | 172.16.5.20 dos_ddos | 172.16.5.20 dos_ddos | 11 flows, 1 model-only host, p max 0.9986 |
| `ho_benign_ssh_interactive` | benign | none | none | 172.16.5.44 dos_ddos | 172.16.5.44 dos_ddos | 8 flows, 1 model-only host, p max 0.9887 |
| `ho_benign_ssh_bulk` | benign | none | none | 172.16.5.44 brute_force | 172.16.5.44 brute_force | 1 flow, 1 model-only host, p max 0.977 |
| `ho_benign_near_miss` | benign | none | none | 172.16.5.44 dos_ddos | 172.16.5.44 dos_ddos | 6 flows, 1 model-only host, p max 0.9606 |
| `ho_benign_cdn_lookalike_4` | benign | none | none | 172.16.5.20 brute_force | 172.16.5.20 brute_force | 2 flows, 1 model-only host, p max 0.0024 |
| `ho_benign_cdn_lookalike_6` | benign | none | 172.16.5.20 dns_tunnel (UNCERTAIN, HIGH, ISOLATE_DEVICE) | none | 172.16.5.20 dns_tunnel | 0 flows, p max 0.0 |
| `ho_ssh_brute_fast` | main | brute_force_ssh | 10.20.30.40 brute_force_ssh (UNCERTAIN, HIGH, CHANGE_PASSWORD) | 10.20.30.40 dos_ddos | 10.20.30.40 brute_force_ssh | 40 flows, p max 0.055 |
| `ho_ssh_brute_slow` | main | brute_force_ssh | 10.20.30.40 brute_force_ssh (UNCERTAIN, HIGH, CHANGE_PASSWORD) | 10.20.30.40 dos_ddos | 10.20.30.40 brute_force_ssh | 12 flows, p max 0.0545 |
| `ho_telnet_brute` | main | brute_force_other | 10.20.30.40 brute_force_other (UNCERTAIN, HIGH, CHANGE_PASSWORD) | 10.20.30.40 dos_ddos | 10.20.30.40 brute_force_other | 12 flows, p max 0.055 |
| `ho_port_scan_syn_fast` | main | port_scan | 10.20.31.41 port_scan (UNCERTAIN, HIGH, BLOCK_SOURCE_IP) | none | 10.20.31.41 port_scan | 0 flows |
| `ho_port_scan_connect_fast` | main | port_scan | 10.20.31.41 port_scan (UNCERTAIN, HIGH, BLOCK_SOURCE_IP) | none | 10.20.31.41 port_scan | 0 flows |
| `ho_dns_tunnel_b32` | main | dns_tunnel | 172.16.5.60 dns_tunnel (UNCERTAIN, HIGH, ISOLATE_DEVICE) | none | 172.16.5.60 dns_tunnel | 0 flows |
| `ho_dns_tunnel_long` | main | dns_tunnel | 172.16.5.60 dns_tunnel (UNCERTAIN, HIGH, ISOLATE_DEVICE) | none | 172.16.5.60 dns_tunnel | 0 flows |
| `ho_dos_syn_300_3s` | main | dos_flood | 10.20.32.42 dos_flood (UNCERTAIN, HIGH, RATE_LIMIT) | none | 10.20.32.42 dos_flood | 0 flows |
| `ho_dos_syn_1000_20s` | main | dos_flood | 10.20.32.42 dos_flood (UNCERTAIN, HIGH, RATE_LIMIT) | none | 10.20.32.42 dos_flood | 0 flows |
| `ho_mixed_attacks` | mixed | brute_force_ssh + port_scan | 10.20.30.40 brute_force_ssh, 10.20.31.41 port_scan (both UNCERTAIN, HIGH) | 10.20.30.40 dos_ddos, 172.16.5.20 dos_ddos | 172.16.5.20 dos_ddos, 10.20.30.40 brute_force_ssh, 10.20.31.41 port_scan | 14 flows, p max 0.9982 |
| `ho_ssh_brute_9` | near | none | none | 10.20.30.40 dos_ddos | 10.20.30.40 dos_ddos | 9 flows, p max 0.0545 |
| `ho_ssh_brute_10_5s` | near | brute_force_ssh | 10.20.30.40 brute_force_ssh (UNCERTAIN, HIGH, CHANGE_PASSWORD) | 10.20.30.40 dos_ddos | 10.20.30.40 brute_force_ssh | 10 flows, p max 0.0498 |
| `ho_ssh_brute_10_7s` | near | none | none | 10.20.30.40 dos_ddos | 10.20.30.40 dos_ddos | 10 flows, p max 0.0545 |
| `ho_port_scan_20` | near | port_scan | 10.20.31.41 port_scan (UNCERTAIN, HIGH, BLOCK_SOURCE_IP) | none | 10.20.31.41 port_scan | 0 flows |
| `ho_port_scan_slow` | near | none | none | none | none | 0 flows |
| `ho_dns_tunnel_5` | near | dns_tunnel | 172.16.5.60 dns_tunnel (UNCERTAIN, HIGH, ISOLATE_DEVICE) | none | 172.16.5.60 dns_tunnel | 0 flows |
| `ho_dns_tunnel_4` | near | none | none | none | none | 0 flows |
| `ho_dns_hex24` | near | no prediction | none | none | none | 0 flows |
| `ho_dos_syn_150` | near | none | none | none | none | 0 flows |
| `ho_dos_syn_200` | near | dos_flood | 10.20.32.42 dos_flood (UNCERTAIN, HIGH, RATE_LIMIT) | none | 10.20.32.42 dos_flood | 0 flows |

Throughput, laptop, not Pi: 27,816 flows/s (scoring only: featurize, ONNX, window; median of 3 runs per file).
Per request p50 1.2 ms, p95 15.8 ms (whole request, parse to verifier). The Pi numbers come from
`run_holdout_nids.py --label "Pi 5"` and are not in this table yet.

What this says, plainly:

- **Rules-first detected every attack instance in the main and mixed sets (11 of 11), with the right family in all
  11.** That is the claim this set can support. It is a holdout in the sense that the files, addresses and
  counts were fixed before the run; it is not blind to the thresholds, because the parameters were chosen by
  reading the rule constants so that each case sits at a stated edge. Real traffic was not tested.
- **Misses, all predicted, all by the rule as written:** a slow SSH run at 7 s spacing (span 63 s, 9 attempts
  fit in the 60 s window) and a slow SYN scan at 1 s per port (about 11 ports in the 10 s window). Both are the
  price of the pre-registered windows, and they are reported as misses, not retuned.
- **Benign:** 5 of 6 benign files raised no alert. That includes 8 interactive SSH sessions, a bulk SSH download,
  six short SSH logins 4 s apart, and four lookups of the long CDN name. The sixth (six lookups in 40 s) alerts
  because the length rule cannot tell it from a tunnel.
- **Model-only (b) is not an alerting mode on this set.** It detected the SSH and telnet brute-force files
  (4 of 11 attack instances, all labelled dos_ddos), missed every port scan, DNS tunnel and SYN flood, and
  alerted on 5 of 6 benign files. This is consistent with the 2026-11-07 design finding that the model does not
  separate these shapes.
- **Combined (c) detects and names the attacks but also alerts on all 6 benign files.** The benign alerts come
  from the model's ALERT flows (p up to 0.999 on ordinary web and SSH traffic). Combined mode is not safe for
  a home device on this evidence.
- **Near-threshold, rules-first:** every case behaved as written. Fired: 10 SSH attempts at 5 s, 20 scan ports,
  5 DNS queries, 200 SYN. Did not fire: 9 SSH attempts, 10 SSH attempts at 7 s, 40 slow scan ports, 4 DNS
  queries, 150 SYN. The 24-char hex DNS set had no prediction (see below).
- **`ho_dns_hex24`:** no prediction was made, and the rules did not fire. The label entropy was not measured,
  so the reason is not known.

Not tested or not done: real traffic; the nfstream reader (the laptop used the stdlib reader only; the Pi
run uses `--flow-source auto`, which takes nfstream); the older `make_test_pcaps.py --profile holdout` set
(not run here); the Pi itself (no Pi access in this run).

### Running the holdout on the Pi (later)

The pcaps and `manifest.json` are made on the laptop with `make_holdout_pcaps.py` (scapy is not needed on the
Pi). Copy `results/nids_holdout/pcaps/` to the Pi, then from the repo root:

```
.venv/bin/python deploy/pi/service/run_holdout_nids.py --pcaps <copied pcaps dir> \
    --model models/nids/student_cse_cic_ids2018_int8.onnx --label "Pi 5" --repeat 3 \
    --json <out dir>/results_pi.json
```

The runner has no thermal guard of its own. Check the Pi's temperature before the run, as the service does
(it refuses at 75 C), and do not run it while the service is busy.

## Synthetic pcaps (offline, not real traffic)

`deploy/pi/service/make_test_pcaps.py` writes four files with scapy `wrpcap` (no socket, nothing sent,
RFC 1918 addresses only). They are synthetic test shapes, so they say nothing about real attacks or
real hosts:

- `benign.pcap`: 6 TCP sessions to 192.168.1.10:443 (5 to 6 packets each, payload) and 8 DNS queries.
- `port_scan.pcap`: 10.0.0.66 sends one SYN to each of 100 ports of 192.168.1.10, each answered by RST.
- `ssh_brute.pcap`: 60 short TCP sessions from 10.0.0.66 to 192.168.1.10:22.
- `dns_long.pcap`: 40 DNS TXT queries with 60-character labels from 192.168.1.20.

Results, same shipped student and thresholds. Laptop uses the stdlib reader, the Pi uses nfstream
(the service's auto path):

| pcap | Pi (nfstream), 5 runs | Laptop (stdlib) | Reading |
|---|---|---|---|
| benign | ALERT, HIGH, dos_ddos, 6 of 14 flows flagged (p 0.998), evidence `dos_flood`, remediation RATE_LIMIT, CHECK_ROUTER_FIREWALL, CONTACT_ISP | same | **False alert.** The synthetic HTTPS sessions have short flows the model reads as flood-like. |
| port_scan | UNCERTAIN, HIGH, port_scan, 81 of 100 flows flagged, evidence `port_scan` (0.9, window), remediation BLOCK_SOURCE_IP, CHECK_ROUTER_FIREWALL, VERIFY_INDEPENDENTLY | same | Detected by the host window. Matches the plan. |
| ssh_brute | ALERT, MEDIUM, **dos_ddos**, 60 of 60 flagged (p 0.055), evidence `dos_flood`, remediation RATE_LIMIT, CHECK_ROUTER_FIREWALL, CONTACT_ISP, VERIFY_INDEPENDENTLY | same | Alert raised, but the **family is wrong** (the plan expects `brute_force_ssh`, CHANGE_PASSWORD, DISABLE_PASSWORD_LOGIN). The model does not separate these short SSH flows. Not fixed. |
| dns_long | SAFE, no alert | same | **Missed.** DNS query types and TTL are not live features (no DNS parser), and no evidence type fires on the name. Expected from the feature set. |

Scores are not a measure of detection quality on real traffic: one synthetic capture per shape, and the
benign capture is a fabricated shape. These are functional checks of the path, not accuracy numbers.

## Pi timings (Pi 5, 4 GB, no cooler, 3 threads)

Driver: `run/synthetic/` pcaps posted 5 times each to `127.0.0.1:8770/analyze_flows` on the Pi
(stdlib client). Summary: `~/rakshak-bench/run/nids_pi_eval/summary_20261106T183349Z.json`.

| pcap | flows | client p50 / p95 (ms) | service wall p50 / p95 (ms) | per-flow p50 / p95 (ms) | flows/s (median) | energy per request (J) |
|---|---|---|---|---|---|---|
| benign | 14 | 122.5 / 183.4 | 68.6 / 127.6 | 0.149 / 0.283 | 5173 | 0.23 |
| port_scan | 100 | 142.6 / 142.7 | 88.8 / 89.2 | 0.146 / 0.164 | 6557 | 0.30 |
| ssh_brute | 60 | 136.3 / 140.7 | 82.4 / 82.8 | 0.148 / 0.175 | 6325 | 0.28 |
| dns_long | 40 | 124.6 / 125.6 | 70.8 / 71.8 | 0.138 / 0.187 | 6535 | 0.24 |

- Client time is 40 to 55 ms above the service wall time: the HTTP and Unix-socket relay path (not measured
  separately). The service wall time includes parsing, scoring, host alerts and the verifier.
  The per-flow number is the flow scoring only (featurize, ONNX, window).
- RSS: the service stayed at about 1660 MB across the 40 requests of the endpoint and the 40 SMS rows
  (first and last request logged 1660.7 and 1660.8 MB; peak 1840 MB at `/health`). No growth per request.
- Temperature: 51.6 C at the start of the final endpoint run, 52.7 C at the end, 54.3 C at most. Throttle
  flags `0x0` before and after. No batch was started at 65 C or above.
- The SMS eval run (below) peaked at 70.8 C under the explainer.
- Energy: PMIC estimate, not a calibrated meter (no 27 W PSU).

## Flow extraction: nfstream on the Pi, and the fallback

- **nfstream installed on the Pi** from PyPI with `pip install nfstream` into `~/rakshak-bench/.venv`
  (6.6.0, with cffi, psutil, dpkt and pycparser wheels). No sudo, no system package. It imports and reads
  pcap files. Live capture was not started, and the service never uses it.
- **Fallback (stdlib):** `src/rakshak/nids/pcap_flows.py`, a classic-pcap reader with `struct` only (no
  dpkt, no scapy). It handles Ethernet (with one VLAN tag), raw IP, Linux SLL, IPv4 TCP, UDP and other
  protocols, and gives nfstream key names. It rejects pcapng and other link types with a clear error.
- **Parity:** packet and byte sizes are the captured frame length (Ethernet included), as nfstream reports
  them. The first version used the IP length, which moved every byte feature by 14 bytes per packet. That
  was fixed before the Pi run. On the synthetic captures the byte, packet and flag features match nfstream,
  and the detection results are the same on laptop and Pi. Duration and throughput features differ by up to
  about 1 ms on some flows (the cause was not chased).

## Remediation

`remediation` is a list of `{code, title, steps[]}` in both `/analyze` and per alert in `/analyze_flows`.
Codes come from the package's allowed actions only. A check (`remediation_problems`) rejects a code from
the other domain (SMS-only codes in a network package and the reverse). The verifier output has
`remediation_ok`.

- SMS: 8 codes (`DO_NOT_CLICK`, `DO_NOT_SHARE_OTP`, `OPEN_OFFICIAL_APP`, `CONTACT_BANK_OFFICIAL`,
  `REPORT_1930_OR_CHAKSHU`, `BLOCK_SENDER`, `VERIFY_INDEPENDENTLY`, `IGNORE`). English and Hinglish.
  `REPORT_1930_OR_CHAKSHU` links https://sancharsaathi.gov.in/sfc/ (text only, no API).
- Network: 11 codes (`BLOCK_SOURCE_IP`, `CHECK_ROUTER_FIREWALL`, `CHANGE_PASSWORD`,
  `DISABLE_PASSWORD_LOGIN`, `RATE_LIMIT`, `CONTACT_ISP`, `ISOLATE_DEVICE`, `FACTORY_RESET_IOT`,
  `USE_TRUSTED_DNS`, `VERIFY_INDEPENDENTLY`, `IGNORE`). English only. Steps say what to do with the code
  and make no new claim.
- **Hinglish SMS text is a first draft and needs native-speaker review** before any user sees it.

## SMS regression (40-row eval, Pi)

`eval_service.py run` and `report` on the same 40 rows, at 3 threads, after the change (`responses.jsonl`
under `~/rakshak-bench/results/edge_eval/`). The previous outputs are in `results/edge_eval_pre_nids_backup/`.

- The only added field is `remediation` (and `verifier.remediation_ok`). Otherwise the responses match,
  except for timings and one new decision field `lang_detected` that was already in the code at HEAD.
- Rows 22 and 29 differ: they now route to Hinglish (`template_hinglish`), while the backup run used the
  English explainer. That routing is HEAD code (`route_lang` was not changed), and the backup predates it
  (it has no `lang_detected` field). So the backup is not a clean A/B against HEAD. A clean A/B means
  running HEAD on the Pi and was not done.
- All 40 rows: HTTP 200, `remediation_ok` true, no cooling refusal.

## Tests (laptop)

- `tests/test_edge_nids_service.py` (13): JSON and pcap bodies, one alert per host with distinct evidence,
  the watchlist (uncertain only), remediation separation and Hinglish, the 503 cooling path, missing model,
  body limits (400, 411, 413), and the model-contract mismatch (an INT8 copy with a changed schema
  is refused).
- `tests/test_nids_pcap_flows.py` (9): grouping, flags, idle timeout, UDP and ICMP and IPv6, big-endian
  nanosecond headers, bad input.
- Required globs: `tests/test_edge_*.py tests/test_nids*.py tests/test_verifier*.py` plus the two new files:
  276 passed. `ruff check .` clean. `ruff format --check` clean on the new files.

## Not done or untested

- The Llama explainer for network evidence (`--net-explainer llm`) is not built. The template is the only
  explainer for network alerts.
- A real 503 cooling refusal on the Pi for `/analyze_flows` (the path is tested with a mock).
- Real traffic. No capture from a home network was taken. The synthetic captures are fabricated shapes.
- Accuracy: no labelled pcap of real attacks was run through the service, so no precision or recall
  is claimed here.
- SSH brute force family (dos_ddos instead of brute_force_ssh) and the DNS tunnel shape (missed).
- The benign false alert on the synthetic HTTPS sessions. The cause is not analysed.
- Throughput at Gigabit line rate, and a load test beyond 100 flows per request. The 50,000-flow JSON cap
  is not tested at size.
- Restart after a crash (same as the SMS service).
- `pi_monitor` CLI defaults (0.5 / 0.2) are not the calibrated student thresholds. Left unchanged.

### Holdout on the Pi 5 (2026-11-12)

`deploy/pi/service/run_holdout_nids.py --repeat 3` on the Pi 5 (3-thread service stack, INT8 model sha `d3dfdcb1…`),
run under `thermal_run.py --pause 63 --resume 58 --kill 65`: start 54 °C, peak 59.0 °C, 0 pauses, throttle `0x0`,
11.3 s wall. The flow source on the Pi is nfstream (`pcap_nfstream`); the laptop run used the stdlib reader.
**Every alert and verdict matches the laptop table above, file for file**: rules-first 11/11 attacks with the family
right and 1/6 benign files alerted; model-only 4/11 (family right 0/11) and 5/6 benign; combined 11/11 and 6/6 benign.
Pi throughput: **6 188 flows/s** (scoring only, rules mode, median of 3 runs per file); per request p50 **56.4 ms**,
p95 **125.8 ms** (whole request, parse to verifier). Raw output: `results/nids_holdout/results_pi.json` (gitignored).
