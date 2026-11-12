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
