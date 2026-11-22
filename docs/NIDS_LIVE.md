# NIDS live monitor: alert-only mode on the Raspberry Pi

This describes `python -m rakshak.nids.pi_monitor` as a live, alert-only monitor on a network
interface you own. It reads flow records, scores them, groups them into time windows, keeps
bounded per-host state across windows, and prints one JSON line per host alert. It never blocks,
drops or rewrites traffic.

## Scope and status

- The evaluation so far uses supplied and synthetic captures. No field measurement has been made.
- The false-alarm rate on real benign traffic is **not yet measured**. Nothing in this document is
  a measured result.
- Suggested target to measure against in the field: **at most 0.1 high-severity alerts per device-hour**.
  Report the observation duration and confidence intervals with any rate you quote. A rate from a
  short run is not evidence for the target.

## Running it on the Pi (alert-only, owned interface)

Install the optional capture package on the device (`nfstream`). The live path needs it. Offline
input (`--pcap`, `--flows-json`) works without it.

```bash
python -m rakshak.nids.pi_monitor \
  --iface eth0 \
  --model /opt/rakshak/nids/student_int8.onnx \
  --mode alert-only \
  --window-s 60 \
  --max-hosts 512 \
  --max-history 256 \
  --alert-cooldown-s 300 \
  --alias-salt "$(cat /etc/rakshak/alias.salt)" \
  --summary-out /var/log/rakshak/nids_summary.json \
  --summary-every-s 300
```

Notes:
- Capture needs raw-socket rights. Run it under a service account you control, not an
  interactive login. Only point it at an interface you own and are allowed to monitor.
- `--alias-salt` takes the salt as text. Use the same salt on every run if you want the same
  alias for the same device across runs. Without it, aliases are random per run.
  Keep the salt file readable only by the service account.
- Alert lines go to stdout (JSON lines). The summary goes to `--summary-out`, at exit and every
  `--summary-every-s` seconds. Without `--summary-out` the summary is printed to stderr at exit.
- Stop it with Ctrl-C or SIGTERM. The open window is processed and the final summary is written.

Offline runs use the same code path:

```bash
python -m rakshak.nids.pi_monitor --pcap capture.pcap --model student_int8.onnx --summary-out s.json
python -m rakshak.nids.pi_monitor --flows-json flows.json --model student_int8.onnx
```

Without nfstream, `--pcap` uses the built-in pure-Python reader, and `--iface` exits with a clear
error (code 2) that names the missing package.

## How windows and host state work

- Each flow is placed in a window by its start time (`--window-s`). A window is closed in one of
  three ways: when a flow for a later window arrives, at exit, or on a wall-clock timer. The timer
  closes a window `--window-s` seconds after its first flow arrived, plus `--grace-s` (default
  10 s). This is what makes a quiet link produce alerts and periodic summaries on time. The timer
  does not wait for the next flow.
- nfstream reports a flow when it expires, so a flow's start time can be well before its arrival.
  A flow that arrives after its window was closed is counted as `late` and scored in a new batch.
  Raise `--grace-s` if `late` is large relative to `flows.seen`.
- A flow that arrives after its window has been processed is counted as `late` and scored in the
  window that is still open.
- Each flow is scored by the student model and the port-scan window check, as in the per-request
  path. The rules in `host_rules` run over the retained history of each host plus the current
  window. A rule can therefore span windows (for example, SSH attempts split over two windows).
- Host alerts are rule-based only. A host the model flags without any rule hit gives a
  **model-only advisory**. It is counted in the summary and is not an alert. Print these with
  `--include-model-only` if you want them.
- The same alert is not repeated for the same host within `--alert-cooldown-s`. Repeats are counted
  as `suppressed_repeats`.

## What is and is not retained

Alert-only mode is enforced at ingest, not only in the output:

- **Kept per flow:** the numeric flow statistics the model and rules read (byte, packet, duration,
  TCP flag counts, protocol, destination port, start time), and the destination alias.
- **Kept per host:** a bounded list of the rule-relevant flows from the last 60 s, as those numbers.
  Tracked hosts and history length are capped (see the bounds below).
- **Replaced before any state is kept:** source and destination addresses become salted hash
  aliases (`h-` plus 12 hex characters). A DNS query name is replaced by a marker that records only
  "suspicious" or "not suspicious". The name itself is never kept.
- **Not kept:** raw addresses, DNS names, the source port, and any payload bytes. The pcap reader
  reads headers and lengths only. A test checks that no raw address or name appears in state,
  alerts or the summary.
- **Never done:** blocking, dropping, rewriting or resetting traffic. The only outputs are alert
  lines and the summary. A test checks that the live module imports no socket, process or firewall API.

Alert records contain the alias, severity, family, the rule evidence (type and weight), the advised
action, and the verifier result. They do not contain addresses.

## Bounds

| Setting | Default | Meaning |
|---|---|---|
| `--max-hosts` | 512 | Tracked source hosts. The least recently seen is evicted first. |
| `--max-history` | 256 | Rule-relevant flows kept per host. Oldest evicted first. Must be at least 200, or the rule thresholds cannot be reached. |
| `--max-flows-window` | 50000 | Flows kept per window. The rest are dropped and counted. |
| `--idle-evict-s` | 900 | A host unseen this long is forgotten. |

Every drop is counted in the summary under `flows.dropped_bounds`. Nothing is dropped silently.
The state is a few tens of megabytes at the default limits. Check `process.peak_rss_kb` on the
device rather than relying on this estimate.

## Summary fields

The summary is written as JSON (`schema` `rakshak.nids.live_summary/1`). It has counts and rates
only, with no addresses.

| Field | Meaning |
|---|---|
| `observed.duration_s` | Span of flow start times seen (last minus first), in seconds. This is capture time, not wall-clock time. |
| `distinct_hosts` | Number of distinct source aliases seen during the run. |
| `device_hours` | Host-active exposure. For each processed window, every distinct host with at least one flow counts for `window_s`. Summed over windows and divided by 3600. This is the denominator for the rates below. It is the benign exposure only if the traffic is benign. |
| `device_hours_upper_bound` | `distinct_hosts` times the observed span. This assumes every host was present for the whole run, so it overstates the exposure. It is kept for comparison only and is not used for any rate. |
| `alerts.total` | Rule-based host alerts emitted (after cooldown). |
| `alerts.by_severity` | `high`, `medium`, `low`. High means the rule's risk is HIGH (top rule weight 0.85 or more, the same thresholds as the SMS path). Low includes UNCERTAIN. |
| `alerts.per_device_hour` | `alerts.total` / `device_hours`. Null if no hours were observed. |
| `alerts.high_per_device_hour` | High-severity alerts / `device_hours`. This is the figure to compare with the target. |
| `alerts.suppressed_repeats` | Alerts withheld because the same host and rule were already reported within the cooldown. |
| `alerts.model_only_advisories` | Model-only flags. These are not alerts. |
| `time_to_first_alert_s` | Seconds from the first flow seen to the window that produced the first alert. Null if none. |
| `windows_processed` | Number of time windows processed. |
| `flows.seen / invalid / scored / late` | Flows received, flows without an address or start time, flows scored, flows that arrived for an earlier window. |
| `flows.dropped_window_cap` | Flows dropped because a window was full. |
| `flows.history_evicted` | Rule-relevant flows discarded from host history by the per-host cap, or because their host was evicted. |
| `flows.dropped_bounds` | The two lines above added together. |
| `bounds.*` | The limits in effect, the tracked host count at exit, and eviction counts. |
| `process.cpu_time_s` | CPU time of the process (user and system). |
| `process.peak_rss_kb` | Peak resident memory of the process, in kilobytes (Linux `ru_maxrss`). Null on systems without `resource`. |
| `process.wall_s` | Wall-clock time from start to the summary. |
| `retained_fields` | The keys the session keeps per flow, so a reader can check what is stored. |

`distinct_hosts` is the one counter that grows with the number of distinct hosts seen. It holds one
short string per host. Everything else is bounded.

## Scope statement

- The design follows the rules-first policy in `docs/NIDS_EDGE.md`. Rule thresholds are unchanged.
- Results so far come from supplied and synthetic captures. They do not show behaviour on a live
  home or office network.
- The false-alarm rate on real benign traffic is not yet measured. Before quoting a rate, run the
  monitor for a stated duration on traffic you own, and report the rate with its confidence
  interval and the duration.
- Window boundaries follow flow start times. A window closes on the wall-clock timer described
  above, so a quiet link still reports, but alert timing is only as precise as `--window-s` plus
  `--grace-s`.
- Each window is processed as a batch. A flood or scan is detected at the end of its window, not
  in the middle of it.
