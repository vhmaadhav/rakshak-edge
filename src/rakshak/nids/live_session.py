"""Bounded live NIDS session: flows in time windows, per-host state across windows, alert-only.

``LiveSession.feed`` takes nfstream-style flow dicts (from an interface, a pcap or a flow JSON
file) in any order. Flows are grouped into windows by ``bidirectional_first_seen_ms``. When a flow
arrives for a later window, the earlier window is processed: each flow is scored by
``pi_monitor.flow_decision``, the rules in ``host_rules`` run over the retained history plus the
window, and ``host_alerts`` gives one advisory per host (quiet mode). Hosts keep a bounded history
of rule-relevant flows across windows, so a rule can span windows.

Alert-only, explicitly:
- Nothing here blocks, drops or rewrites traffic. The output is alert records and a summary.
- Addresses are replaced by per-run aliases (``HostAliaser``) before any state is kept. The DNS
  query name is replaced by a sentinel that keeps only the rule's yes/no answer. Every field not
  in ``RETAINED_KEYS`` is discarded at ingest. Nothing here reads payload bytes; the pcap reader
  keeps header statistics only.
- The module imports no socket, process or firewall API (a test checks the source).

Bounds (``LiveConfig``): tracked hosts (least recently seen evicted first), history per host (oldest
evicted), flows per window (excess dropped), idle host eviction. Every drop is counted in the
summary. Nothing is silently skipped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rakshak.nids import host_rules
from rakshak.nids.evidence import STATE_LABEL
from rakshak.nids.host_alerts import HostAlert, host_alerts
from rakshak.nids.live import NFSTREAM_KEYS, HostWindow
from rakshak.nids.pi_monitor import flow_decision

try:  # resource is Unix-only; the summary reports None elsewhere
    import resource
except ImportError:  # pragma: no cover - not on the Pi
    resource = None

SCHEMA = "rakshak.nids.live_summary/1"
MODE_ALERT_ONLY = "alert-only"
MODES = (MODE_ALERT_ONLY,)
SEVERITY = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low", "UNCERTAIN": "low"}
# Stands for "the DNS name was suspicious". The name itself is never kept. The length is chosen so
# that host_rules.dns_name_is_suspicious is true for it.
DNS_SUSPICIOUS_SENTINEL = "x" * host_rules.DNS_LABEL_LEN

# Keys the session keeps for one flow. Addresses are aliases. The source port and the DNS name are
# not kept (the name becomes the sentinel above, or None).
_NUMERIC_KEYS = tuple(
    k for k in NFSTREAM_KEYS if k not in ("src_ip", "dst_ip", "src_port", "requested_server_name")
)
RETAINED_KEYS = frozenset(_NUMERIC_KEYS) | {"src_ip", "dst_ip", "requested_server_name"}
# The per-host history keeps only what the rules read, as a tuple (compact).
_EVENT_KEYS = (
    "bidirectional_first_seen_ms",
    "protocol",
    "dst_port",
    "dst_ip",
    "bidirectional_duration_ms",
    "bidirectional_bytes",
    "bidirectional_syn_packets",
    "dst2src_packets",
    "requested_server_name",
)
_HORIZON_MS = max(host_rules.AUTH_WINDOW_MS, host_rules.DNS_WINDOW_MS, host_rules.DOS_WINDOW_MS)
_MIN_HISTORY = max(host_rules.AUTH_MIN_FLOWS, host_rules.DNS_MIN_QUERIES, host_rules.DOS_MIN_FLOWS)


@dataclass(frozen=True)
class LiveConfig:
    """Window and bound settings. Defaults keep the state well under 100 MB on a Raspberry Pi 5."""

    window_s: float = 60.0
    grace_s: float = 10.0  # wall-clock grace after a window's window_s has elapsed
    max_hosts: int = 512
    max_history_per_host: int = 256
    max_flows_per_window: int = 50_000
    idle_evict_s: float = 900.0
    alert_cooldown_s: float = 300.0
    include_model_only: bool = False
    summary_every_s: float = 0.0
    mode: str = MODE_ALERT_ONLY

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not self.window_s > 0:
            raise ValueError("window_s must be positive")
        if self.grace_s < 0:
            raise ValueError("grace_s must not be negative")
        if self.max_hosts < 1:
            raise ValueError("max_hosts must be at least 1")
        if self.max_history_per_host < _MIN_HISTORY:
            # Below this the rule thresholds (up to 200 flows in 10 s) can never be reached.
            raise ValueError(f"max_history_per_host must be at least {_MIN_HISTORY}")
        if self.max_flows_per_window < 1:
            raise ValueError("max_flows_per_window must be at least 1")
        if not self.idle_evict_s > 0:
            raise ValueError("idle_evict_s must be positive")
        if self.alert_cooldown_s < 0 or self.summary_every_s < 0:
            raise ValueError("alert_cooldown_s and summary_every_s must not be negative")


class HostAliaser:
    """Per-run aliases for addresses: a SHA-256 prefix over a salt. Same salt, same aliases."""

    def __init__(self, salt: str | None = None) -> None:
        self._salt = (salt if salt is not None else os.urandom(16).hex()).encode("utf-8")

    def alias(self, address: Any) -> str | None:
        if address is None:
            return None
        digest = hashlib.sha256(self._salt + b"\x00" + str(address).encode("utf-8")).hexdigest()
        return "h-" + digest[:12]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def sanitize_flow(flow: Mapping[str, Any], aliaser: HostAliaser) -> dict[str, Any]:
    """The only copy of a flow the session keeps: retained keys, aliased addresses, no DNS name."""
    out: dict[str, Any] = {k: flow.get(k) for k in _NUMERIC_KEYS}
    out["src_ip"] = aliaser.alias(flow.get("src_ip"))
    out["dst_ip"] = aliaser.alias(flow.get("dst_ip"))
    name = flow.get("requested_server_name")
    suspicious = isinstance(name, str) and host_rules.dns_name_is_suspicious(name)
    out["requested_server_name"] = DNS_SUSPICIOUS_SENTINEL if suspicious else None
    return out


@dataclass
class _Host:
    last_ms: float
    events: deque  # tuples of _EVENT_KEYS, rule-relevant flows only


def _usage_peak_rss_kb() -> int | None:
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # kilobytes on Linux


def _alert_record(a: HostAlert, window_end_ms: float, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "time_ms": window_end_ms,
        "host": a.src,
        "severity": SEVERITY.get(a.package.risk.value, "low"),
        "state": STATE_LABEL[a.state],
        "family": a.family,
        "path": a.path,
        "p_attack": round(a.p_attack, 4),
        "evidence": [
            {"id": e.id, "type": e.type, "weight": round(e.weight, 3)} for e in a.evidence
        ],
        "action": a.template["action"],
        "summary": a.template["summary"],
        "verified": a.verified,
        "mode": MODE_ALERT_ONLY,
    }


class LiveSession:
    """Windowed, bounded, alert-only scoring of a flow stream. Feed flows, then call ``close``."""

    def __init__(
        self,
        probs_fn: Callable[..., Any],
        *,
        alert: float,
        uncertain: float,
        window_ms: int = 10_000,
        config: LiveConfig | None = None,
        aliaser: HostAliaser | None = None,
        on_alert: Callable[[dict[str, Any]], None] | None = None,
        summary_sink: Callable[[dict[str, Any]], None] | None = None,
        source: str = "unknown",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= uncertain <= alert <= 1.0:
            raise ValueError("need 0 <= uncertain <= alert <= 1")
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        self.cfg = config or LiveConfig()
        self._probs = probs_fn
        self._alert = float(alert)
        self._uncertain = float(uncertain)
        self._window_ms = self.cfg.window_s * 1000.0
        self._portscan = HostWindow(window_ms=window_ms, max_sources=self.cfg.max_hosts)
        self._aliaser = aliaser or HostAliaser()
        self._on_alert = on_alert
        self._sink = summary_sink
        self._source = source
        self._clock = clock
        self._hosts: OrderedDict[str, _Host] = OrderedDict()
        self._buffer: list[dict[str, Any]] = []
        self._window_idx: int | None = None
        self._deadline: float | None = None  # clock time at which the open window must close
        self._last_closed: int | None = None
        self._max_ts: float | None = None
        self._min_ts: float | None = None
        self._first_alert_ms: float | None = None
        self._seen_hosts: set[str] = set()
        self._last_emit: dict[tuple, float] = {}
        self._by_sev = {"high": 0, "medium": 0, "low": 0}
        self._n = {
            "flows_seen": 0,
            "flows_invalid": 0,
            "flows_scored": 0,
            "late_flows": 0,
            "flows_dropped_window_cap": 0,
            "history_evicted": 0,
            "hosts_evicted_cap": 0,
            "hosts_evicted_idle": 0,
            "windows_processed": 0,
            "active_host_windows": 0,
            "alerts_suppressed": 0,
            "model_only_advisories": 0,
        }
        self._started = clock()
        self._last_sink = self._started
        self._closed = False

    @property
    def tracked_hosts(self) -> int:
        return len(self._hosts)

    def feed(self, flow: Mapping[str, Any]) -> None:
        """Add one flow. A flow that starts a later window first processes the window before it."""
        if self._closed:
            raise RuntimeError("session is closed")
        self._n["flows_seen"] += 1
        ts = _as_float(flow.get("bidirectional_first_seen_ms"))
        if ts is None or flow.get("src_ip") is None:
            self._n["flows_invalid"] += 1
            return
        widx = int(ts // self._window_ms)
        if self._window_idx is not None and widx > self._window_idx:
            self._flush()
        if self._window_idx is None:
            if self._last_closed is not None and widx <= self._last_closed:
                self._n["late_flows"] += 1  # its window already closed; scored in a new batch
            self._window_idx = widx
            self._deadline = self._clock() + self.cfg.window_s + self.cfg.grace_s
        elif widx < self._window_idx:
            self._n["late_flows"] += 1  # counted in the window still open
        self._max_ts = ts if self._max_ts is None else max(self._max_ts, ts)
        self._min_ts = ts if self._min_ts is None else min(self._min_ts, ts)
        if len(self._buffer) >= self.cfg.max_flows_per_window:
            self._n["flows_dropped_window_cap"] += 1
            return
        kept = sanitize_flow(flow, self._aliaser)
        kept["bidirectional_first_seen_ms"] = ts
        self._buffer.append(kept)

    def poll(self) -> None:
        """Call on a timer, also when no flow arrives. Closes the open window once its wall-clock
        deadline (window_s after it opened, plus grace_s) has passed, and writes any periodic
        summary, so a quiet link still reports on time."""
        if self._closed:
            return
        if (self._window_idx is not None and self._deadline is not None
                and self._clock() >= self._deadline):
            self._flush()
        else:
            self._maybe_sink()

    def close(self) -> dict[str, Any]:
        """Process the open window, write the final summary to the sink, and return it."""
        if not self._closed:
            self._flush()
            self._closed = True
            if self._sink is not None:
                self._sink(self.summary())
        return self.summary()

    # --- processing -------------------------------------------------------------------------

    def _flush(self) -> None:
        """Close the open window: process its buffered flows, then housekeeping and summary."""
        if self._window_idx is not None:
            self._last_closed = self._window_idx
        self._window_idx = None
        self._deadline = None
        if self._buffer:
            window = sorted(self._buffer, key=lambda f: f["bidirectional_first_seen_ms"])
            self._buffer = []
            self._process(window)
        self._housekeeping()
        self._maybe_sink()

    def _process(self, window: list[dict[str, Any]]) -> None:
        self._n["windows_processed"] += 1
        self._n["active_host_windows"] += len({f["src_ip"] for f in window})
        window_end = window[-1]["bidirectional_first_seen_ms"]
        for f in window:
            self._touch(f["src_ip"], f["bidirectional_first_seen_ms"])

        horizon = self._max_ts - _HORIZON_MS  # type: ignore[operator]
        history: list[dict[str, Any]] = []
        for src, host in self._hosts.items():
            self._prune(host, horizon)
            history.extend(_event_flow(src, ev) for ev in host.events)
        offset = len(history)

        decisions = [
            flow_decision(f, self._probs, self._portscan, alert=self._alert,
                          uncertain=self._uncertain)
            for f in window
        ]
        self._n["flows_scored"] += len(decisions)
        hits = [
            replace(h, flow_index=frozenset(i - offset for i in h.flow_index if i >= offset))
            for h in host_rules.evaluate(history + window)
        ]
        for a in host_alerts(decisions, hits, mode="quiet"):
            self._emit(a, window_end)

        for f in window:
            host = self._hosts.get(f["src_ip"])
            if not host_rules.rule_relevant(f):
                continue
            if host is None:  # evicted earlier in this window, so it has no slot
                self._n["history_evicted"] += 1
                continue
            if len(host.events) == host.events.maxlen:
                self._n["history_evicted"] += 1
            host.events.append(tuple(f[k] for k in _EVENT_KEYS))

    def _touch(self, alias: str, ts: float) -> None:
        host = self._hosts.get(alias)
        if host is None:
            if len(self._hosts) >= self.cfg.max_hosts:
                _, gone = self._hosts.popitem(last=False)
                self._n["hosts_evicted_cap"] += 1
                self._n["history_evicted"] += len(gone.events)
            host = self._hosts[alias] = _Host(
                last_ms=ts, events=deque(maxlen=self.cfg.max_history_per_host)
            )
        else:
            self._hosts.move_to_end(alias)
        host.last_ms = max(host.last_ms, ts)
        self._seen_hosts.add(alias)

    def _prune(self, host: _Host, horizon: float) -> None:
        if any(ev[0] < horizon for ev in host.events):
            host.events = deque(
                (ev for ev in host.events if ev[0] >= horizon),
                maxlen=self.cfg.max_history_per_host,
            )

    def _housekeeping(self) -> None:
        if self._max_ts is None:
            return
        idle_ms = self.cfg.idle_evict_s * 1000.0
        for alias in [a for a, h in self._hosts.items() if self._max_ts - h.last_ms > idle_ms]:
            del self._hosts[alias]
            self._n["hosts_evicted_idle"] += 1
        cooldown_ms = self.cfg.alert_cooldown_s * 1000.0
        now = self._max_ts
        self._last_emit = {k: v for k, v in self._last_emit.items() if now - v < cooldown_ms}

    def _emit(self, a: HostAlert, window_end: float) -> None:
        if a.is_model_only and not self.cfg.include_model_only:
            self._n["model_only_advisories"] += 1
            return
        key = (a.src, a.path, a.family, tuple(sorted({e.type for e in a.evidence})))
        last = self._last_emit.get(key)
        if last is not None and window_end - last < self.cfg.alert_cooldown_s * 1000.0:
            self._n["alerts_suppressed"] += 1
            return
        self._last_emit[key] = window_end
        if a.is_model_only:
            self._n["model_only_advisories"] += 1
            record = _alert_record(a, window_end, "model_only_advisory")
        else:
            record = _alert_record(a, window_end, "alert")
            sev = record["severity"]
            self._by_sev[sev] += 1
            if self._first_alert_ms is None:
                self._first_alert_ms = window_end
        if self._on_alert is not None:
            self._on_alert(record)

    def _maybe_sink(self) -> None:
        if self._sink is None or self.cfg.summary_every_s <= 0:
            return
        now = self._clock()
        if now - self._last_sink >= self.cfg.summary_every_s:
            self._last_sink = now
            self._sink(self.summary())

    # --- metrics ----------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Field metrics so far. Contains aliases and counts only, never a raw address."""
        start, end = self._min_ts, self._max_ts
        span_s = (end - start) / 1000.0 if start is not None and end is not None else 0.0
        hours = span_s / 3600.0
        hosts = len(self._seen_hosts)
        # Host-active exposure: for each window, every distinct host with >= 1 flow counts for
        # window_s. This is the rate denominator.
        device_hours = self._n["active_host_windows"] * self.cfg.window_s / 3600.0
        upper_bound = hosts * hours  # every host counted for the whole span: an overstatement
        total = sum(self._by_sev.values())

        def per_hour(n: int) -> float | None:
            return round(n / device_hours, 6) if device_hours > 0 else None

        first = self._first_alert_ms
        n = self._n
        return {
            "schema": SCHEMA,
            "mode": self.cfg.mode,
            "source": self._source,
            "observed": {
                "start_ms": start,
                "end_ms": end,
                "duration_s": round(span_s, 3),
            },
            "distinct_hosts": hosts,
            "device_hours": round(device_hours, 4),
            "device_hours_basis": "host-active exposure: sum over processed windows of the "
                                  "distinct hosts with at least one flow, times window_s",
            "device_hours_upper_bound": round(upper_bound, 4),
            "device_hours_upper_bound_basis": "distinct hosts times the observed span; "
                                              "overstates exposure, kept for comparison",
            "alerts": {
                "total": total,
                "by_severity": dict(self._by_sev),
                "per_device_hour": per_hour(total),
                "high_per_device_hour": per_hour(self._by_sev["high"]),
                "suppressed_repeats": n["alerts_suppressed"],
                "model_only_advisories": n["model_only_advisories"],
            },
            "time_to_first_alert_s": (
                None if first is None or start is None else round((first - start) / 1000.0, 3)
            ),
            "windows_processed": n["windows_processed"],
            "flows": {
                "seen": n["flows_seen"],
                "invalid": n["flows_invalid"],
                "scored": n["flows_scored"],
                "late": n["late_flows"],
                "dropped_window_cap": n["flows_dropped_window_cap"],
                "history_evicted": n["history_evicted"],
                "dropped_bounds": n["flows_dropped_window_cap"] + n["history_evicted"],
            },
            "bounds": {
                "window_s": self.cfg.window_s,
                "max_hosts": self.cfg.max_hosts,
                "max_history_per_host": self.cfg.max_history_per_host,
                "max_flows_per_window": self.cfg.max_flows_per_window,
                "idle_evict_s": self.cfg.idle_evict_s,
                "tracked_hosts": len(self._hosts),
                "hosts_evicted_cap": n["hosts_evicted_cap"],
                "hosts_evicted_idle": n["hosts_evicted_idle"],
            },
            "process": {
                "cpu_time_s": round(time.process_time(), 3),
                "peak_rss_kb": _usage_peak_rss_kb(),
                "wall_s": round(self._clock() - self._started, 3),
            },
            "retained_fields": sorted(RETAINED_KEYS),
        }


def _event_flow(src: str, ev: tuple) -> dict[str, Any]:
    out = dict(zip(_EVENT_KEYS, ev, strict=True))
    out["src_ip"] = src
    return out


def run_session(
    flows: Iterable[Mapping[str, Any]],
    probs_fn: Callable[..., Any],
    *,
    alert: float,
    uncertain: float,
    window_ms: int = 10_000,
    config: LiveConfig | None = None,
    aliaser: HostAliaser | None = None,
    on_alert: Callable[[dict[str, Any]], None] | None = None,
    summary_sink: Callable[[dict[str, Any]], None] | None = None,
    source: str = "unknown",
    max_flows: int | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Feed a flow iterable through a ``LiveSession`` and return the final summary."""
    session = LiveSession(
        probs_fn,
        alert=alert,
        uncertain=uncertain,
        window_ms=window_ms,
        config=config,
        aliaser=aliaser,
        on_alert=on_alert,
        summary_sink=summary_sink,
        source=source,
        clock=clock,
    )
    for n, flow in enumerate(flows):
        if max_flows is not None and n >= max_flows:
            break
        session.feed(flow)
    return session.close()


def drive(session: LiveSession, flows: Iterable[Mapping[str, Any]], *,
          poll_every_s: float = 0.5) -> None:
    """Feed ``flows`` to ``session`` and call ``poll`` between flows, so windows close on the timer
    even when a blocking source (an interface) is silent. A reader thread pulls from ``flows``.
    Only the calling thread touches the session. The session is not closed here."""
    pending: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=10_000)

    def pump() -> None:
        try:
            for flow in flows:
                pending.put(("flow", flow))
        except BaseException as exc:  # re-raised in the calling thread
            pending.put(("error", exc))
            return
        pending.put(("done", None))

    threading.Thread(target=pump, daemon=True).start()
    while True:
        try:
            kind, value = pending.get(timeout=poll_every_s)
        except queue.Empty:
            session.poll()
            continue
        if kind == "flow":
            session.feed(value)
            session.poll()
        elif kind == "error":
            raise value
        else:
            return


def load_flows_json(path: str | Path) -> list[dict[str, Any]]:
    """Flows from a JSON file: either a list, or an object with a ``flows`` list."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    flows = doc.get("flows") if isinstance(doc, dict) else doc
    if not isinstance(flows, list):
        raise ValueError("flow file must hold a list, or an object with a 'flows' list")
    if any(not isinstance(f, dict) for f in flows):
        raise ValueError("every flow must be a JSON object")
    return flows


def write_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    """Write the summary JSON atomically (temporary file, then rename)."""
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
