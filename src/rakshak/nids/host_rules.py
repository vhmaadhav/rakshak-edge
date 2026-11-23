"""Deterministic host-window rules for network flows (rules-first policy, docs/NIDS_EDGE.md).

Pre-registered on 2026-11-07, before any run. The thresholds are the constants below and must not be
changed after seeing results. ``evaluate`` takes the flows of one request (nfstream-style dicts, see
``rakshak.nids.live.NFSTREAM_KEYS``) and returns the rule hits. The model does not take part.

Rules (all counts per request; a window fires when its last event minus its first is <= the length):

- ``brute_force_ssh`` / ``brute_force_other``: >= 10 short TCP flows (duration <= 5 s and total
  bytes <= 10 KB) from one source to one destination on the port, in 60 s. SSH is port 22; other is
  21, 23, 3389 and 5900.
- ``dns_tunnel``: >= 5 DNS queries (UDP to port 53) from one source in 60 s whose name has a label
  of >= 40 chars, or a name of >= 100 chars, or the longest label has Shannon entropy >= 3.8
  bits/char.
- ``dos_flood``: >= 200 SYN-only TCP flows (at least one SYN, no packet back from the
  destination) from one source to one destination in 10 s. The packet-rate option is not built.
- ``port_scan`` is not here. It stays in ``evidence.evidence_for_flow`` (``HostWindow``, 10 s,
  >= 20).

Flows missing the statistics a rule reads are not counted (no guessed value).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# --- pre-registered thresholds (docs/NIDS_EDGE.md, "Rules-first policy") ---
SSH_PORT = 22
OTHER_AUTH_PORTS = (21, 23, 3389, 5900)
AUTH_WINDOW_MS = 60_000
AUTH_MIN_FLOWS = 10
AUTH_MAX_DURATION_MS = 5_000
AUTH_MAX_BYTES = 10_240
AUTH_WEIGHT = 0.9
DNS_PORT = 53
DNS_WINDOW_MS = 60_000
DNS_MIN_QUERIES = 5
DNS_LABEL_LEN = 40
DNS_NAME_LEN = 100
DNS_ENTROPY_BITS = 3.8
DNS_WEIGHT = 0.9
DOS_WINDOW_MS = 10_000
DOS_MIN_FLOWS = 200
DOS_WEIGHT = 0.85

RULE_SOURCE_WINDOW = "window"
RULE_SOURCE_RULE = "rule"


@dataclass(frozen=True)
class RuleHit:
    """One rule that fired for a source (and destination, where the rule has one).

    ``flow_index`` is the set of request-order flow indices that qualified inside a firing window.
    """

    src: str
    type: str
    weight: float
    source: str
    dst: str | None
    dst_port: int | None
    flow_index: frozenset[int]


def shannon_bits(text: str) -> float:
    """Shannon entropy of the characters of ``text``, in bits per character (0.0 for empty)."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def dns_name_is_suspicious(name: str | None) -> bool:
    """True when a DNS query name meets one of the three pre-registered tests."""
    if not name:
        return False
    clean = name.strip().rstrip(".").lower()
    if not clean:
        return False
    labels = clean.split(".")
    longest = max(labels, key=len)
    return (
        len(longest) >= DNS_LABEL_LEN
        or len(clean) >= DNS_NAME_LEN
        or shannon_bits(longest) >= DNS_ENTROPY_BITS
    )


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_short_tcp(flow: Mapping[str, Any]) -> bool:
    duration = _num(flow.get("bidirectional_duration_ms"))
    size = _num(flow.get("bidirectional_bytes"))
    return (
        duration is not None
        and size is not None
        and duration <= AUTH_MAX_DURATION_MS
        and size <= AUTH_MAX_BYTES
    )


def _is_syn_only(flow: Mapping[str, Any]) -> bool:
    syn = _num(flow.get("bidirectional_syn_packets"))
    back = _num(flow.get("dst2src_packets"))
    return syn is not None and syn >= 1 and back == 0


def rule_relevant(flow: Mapping[str, Any]) -> bool:
    """True when ``flow`` is one that ``evaluate`` can count toward a rule (same tests, no window).

    The live session keeps only these flows per host, so its state stays small. A flow that is not
    relevant can never change a rule hit.
    """
    if flow.get("src_ip") is None or _num(flow.get("bidirectional_first_seen_ms")) is None:
        return False
    proto = _num(flow.get("protocol"))
    dport = _num(flow.get("dst_port"))
    if proto is None or dport is None:
        return False
    proto_i, dport_i = int(proto), int(dport)
    if proto_i == 6 and (dport_i == SSH_PORT or dport_i in OTHER_AUTH_PORTS):
        if _is_short_tcp(flow):
            return True
    if proto_i == 17 and dport_i == DNS_PORT:
        name = flow.get("requested_server_name")
        return dns_name_is_suspicious(name if isinstance(name, str) else None)
    return proto_i == 6 and _is_syn_only(flow)


def _qualifying_in_windows(
    events: list[tuple[float, int]], window_ms: float, min_count: int
) -> set[int]:
    """Indices (the second item of each event) that lie in some window of >= ``min_count`` events.

    ``events`` are ``(ts_ms, flow_index)``. A window starts at an event and holds the events whose
    time is within ``window_ms`` of it (inclusive). Returns the union of those windows' events.
    """
    events = sorted(events)
    n = len(events)
    diff = [0] * (n + 1)
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j + 1 < n and events[j + 1][0] - events[i][0] <= window_ms:
            j += 1
        if j - i + 1 >= min_count:
            diff[i] += 1
            diff[j + 1] -= 1
    out: set[int] = set()
    run = 0
    for k in range(n):
        run += diff[k]
        if run > 0:
            out.add(events[k][1])
    return out


def _qualifying_in_windows_distinct(
    events: list[tuple[float, int, str]],
    window_ms: float,
    min_count: int,
    min_distinct: int,
) -> set[int]:
    """Like ``_qualifying_in_windows``, additionally requiring ``>= min_distinct`` distinct keys."""
    if min_distinct <= 1:
        return _qualifying_in_windows([(ts, idx) for ts, idx, _ in events], window_ms, min_count)
    events = sorted(events)
    n = len(events)
    diff = [0] * (n + 1)
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j + 1 < n and events[j + 1][0] - events[i][0] <= window_ms:
            j += 1
        if j - i + 1 >= min_count:
            distinct = len({events[k][2] for k in range(i, j + 1)})
            if distinct >= min_distinct:
                diff[i] += 1
                diff[j + 1] -= 1
    out: set[int] = set()
    run = 0
    for k in range(n):
        run += diff[k]
        if run > 0:
            out.add(events[k][1])
    return out


def evaluate(flows: Iterable[Mapping[str, Any]], *, dns_min_distinct: int = 1) -> list[RuleHit]:
    """Rule hits for one request. ``flows`` must be in request order (the index is the position).

    ``dns_min_distinct`` defaults to 1 (exact pre-registered 2026-11-07 rule). Setting it > 1
    (e.g. 3) additionally requires at least that many distinct normalized DNS query names inside
    the 60 s window, suppressing repeated lookups of a single long CDN/asset hostname while
    preserving DNS tunneling detection (which varies encoded subdomain labels across queries).
    """
    auth: dict[tuple[str, str | None, int], list[tuple[float, int]]] = {}
    dns: dict[str, list[tuple[float, int, str]]] = {}
    dos: dict[tuple[str, str | None], list[tuple[float, int]]] = {}
    for idx, flow in enumerate(flows):
        src = flow.get("src_ip")
        if src is None:
            continue
        src = str(src)
        ts = _num(flow.get("bidirectional_first_seen_ms"))
        proto = _num(flow.get("protocol"))
        dport_f = _num(flow.get("dst_port"))
        if ts is None or proto is None or dport_f is None:
            continue
        dport = int(dport_f)
        dst = flow.get("dst_ip")
        dst = None if dst is None else str(dst)
        if int(proto) == 6 and (dport == SSH_PORT or dport in OTHER_AUTH_PORTS):
            if _is_short_tcp(flow):
                auth.setdefault((src, dst, dport), []).append((ts, idx))
        if int(proto) == 17 and dport == DNS_PORT:
            name = flow.get("requested_server_name")
            if dns_name_is_suspicious(name if isinstance(name, str) else None):
                norm_name = str(name).strip().rstrip(".").lower()
                dns.setdefault(src, []).append((ts, idx, norm_name))
        if int(proto) == 6 and _is_syn_only(flow):
            dos.setdefault((src, dst), []).append((ts, idx))

    hits: list[RuleHit] = []
    for (src, dst, dport), events in auth.items():
        matched = _qualifying_in_windows(events, AUTH_WINDOW_MS, AUTH_MIN_FLOWS)
        if matched:
            kind = "brute_force_ssh" if dport == SSH_PORT else "brute_force_other"
            hits.append(
                RuleHit(src, kind, AUTH_WEIGHT, RULE_SOURCE_WINDOW, dst, dport, frozenset(matched))
            )
    for src, events in dns.items():
        matched = _qualifying_in_windows_distinct(
            events, DNS_WINDOW_MS, DNS_MIN_QUERIES, max(1, int(dns_min_distinct))
        )
        if matched:
            hits.append(
                RuleHit(
                    src,
                    "dns_tunnel",
                    DNS_WEIGHT,
                    RULE_SOURCE_RULE,
                    None,
                    DNS_PORT,
                    frozenset(matched),
                )
            )
    for (src, dst), events in dos.items():
        matched = _qualifying_in_windows(events, DOS_WINDOW_MS, DOS_MIN_FLOWS)
        if matched:
            hits.append(
                RuleHit(
                    src, "dos_flood", DOS_WEIGHT, RULE_SOURCE_WINDOW, dst, None, frozenset(matched)
                )
            )
    hits.sort(key=lambda h: (h.src, -h.weight, h.type, h.dst or ""))
    return hits
