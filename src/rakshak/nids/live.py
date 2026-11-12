"""nfstream flows to the shared NetFlow-v2 schema, for the Raspberry Pi and for tests.

``featurize_flow`` takes a plain dict with nfstream attribute names (``NFSTREAM_KEYS``) and
returns the 33 shared features in ``SHARED_FEATURES`` order. A feature that nfstream cannot give
is NaN there, and it is listed in ``UNMAPPABLE_LIVE`` with the reason. The model is trained and
run on ``LIVE_FEATURES`` only (the mappable subset), so no feature is filled with a made-up value.

nfstream is imported lazily in ``iter_nfstream``. Importing this module never starts a capture.
Mapping conventions (not checked against the NF-*-v2 CSVs, so verify them on Kaggle first):
- ``IN_*`` is the src-to-dst direction (the flow initiator), ``OUT_*`` is dst-to-src.
- Rates use the bidirectional duration. A zero-duration flow has rate 0.
- TCP flag masks are the OR of the flags seen, which is the same meaning as NetFlow TCP_FLAGS.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from rakshak.nids.evidence import WindowStats
from rakshak.nids.schema import SHARED_FEATURES

NFSTREAM_KEYS: tuple[str, ...] = (
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "bidirectional_first_seen_ms",
    "bidirectional_duration_ms",
    "bidirectional_bytes",
    "bidirectional_min_ps",
    "bidirectional_max_ps",
    "bidirectional_syn_packets",
    "bidirectional_cwr_packets",
    "bidirectional_ece_packets",
    "bidirectional_urg_packets",
    "bidirectional_ack_packets",
    "bidirectional_psh_packets",
    "bidirectional_rst_packets",
    "bidirectional_fin_packets",
    "src2dst_duration_ms",
    "src2dst_packets",
    "src2dst_bytes",
    "src2dst_syn_packets",
    "src2dst_cwr_packets",
    "src2dst_ece_packets",
    "src2dst_urg_packets",
    "src2dst_ack_packets",
    "src2dst_psh_packets",
    "src2dst_rst_packets",
    "src2dst_fin_packets",
    "dst2src_duration_ms",
    "dst2src_packets",
    "dst2src_bytes",
    "dst2src_syn_packets",
    "dst2src_cwr_packets",
    "dst2src_ece_packets",
    "dst2src_urg_packets",
    "dst2src_ack_packets",
    "dst2src_psh_packets",
    "dst2src_rst_packets",
    "dst2src_fin_packets",
    "requested_server_name",  # DNS query name for DNS flows (rule input only, not a model feature)
)

_HIST = "nfstream gives packet-size min/mean/stddev/max, not a histogram"

UNMAPPABLE_LIVE: dict[str, str] = {
    "L7_PROTO": "nfstream exposes application_name (a string), no numeric nDPI id",
    "ICMP_TYPE": "nfstream exposes no ICMP type",
    "ICMP_IPV4_TYPE": "nfstream exposes no ICMP type",
    "NUM_PKTS_UP_TO_128_BYTES": _HIST,
    "NUM_PKTS_128_TO_256_BYTES": _HIST,
    "NUM_PKTS_256_TO_512_BYTES": _HIST,
    "NUM_PKTS_512_TO_1024_BYTES": _HIST,
    "NUM_PKTS_1024_TO_1514_BYTES": _HIST,
    "RETRANSMITTED_IN_BYTES": "nfstream computes no retransmission counts",
    "RETRANSMITTED_IN_PKTS": "nfstream computes no retransmission counts",
    "RETRANSMITTED_OUT_BYTES": "nfstream computes no retransmission counts",
    "RETRANSMITTED_OUT_PKTS": "nfstream computes no retransmission counts",
    "DNS_QUERY_TYPE": "nfstream has no DNS fields (only requested_server_name); needs a DNS parser",
    "DNS_TTL_ANSWER": "nfstream has no DNS fields (only requested_server_name); needs a DNS parser",
}

LIVE_FEATURES: tuple[str, ...] = tuple(f for f in SHARED_FEATURES if f not in UNMAPPABLE_LIVE)
FEATURE_INDEX: dict[str, int] = {f: i for i, f in enumerate(SHARED_FEATURES)}
LIVE_COLUMNS: np.ndarray = np.asarray([FEATURE_INDEX[f] for f in LIVE_FEATURES], dtype=np.int64)

# nfstream TCP flag counters and their NetFlow bit values (RFC 793 order).
FLAG_BITS: dict[str, int] = {
    "fin": 0x01,
    "syn": 0x02,
    "rst": 0x04,
    "psh": 0x08,
    "ack": 0x10,
    "urg": 0x20,
    "ece": 0x40,
    "cwr": 0x80,
}

_NAN = float("nan")


def _num(flow: Mapping[str, Any], key: str) -> float:
    value = flow.get(key)
    if value is None:
        return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def _flag_mask(flow: Mapping[str, Any], prefix: str) -> float:
    """OR of the flag bits whose counter is above zero. NaN when no counter is present."""
    keys = [f"{prefix}_{name}_packets" for name in FLAG_BITS]
    if all(flow.get(k) is None for k in keys):
        return _NAN
    mask = 0
    for name, bit in FLAG_BITS.items():
        count = _num(flow, f"{prefix}_{name}_packets")
        if math.isfinite(count) and count > 0:
            mask |= bit
    return float(mask)


def _duration_s(flow: Mapping[str, Any]) -> float:
    return _num(flow, "bidirectional_duration_ms") / 1000.0


def _rate(bytes_value: float, dur_s: float) -> float:
    if not math.isfinite(bytes_value) or not math.isfinite(dur_s):
        return _NAN
    return bytes_value / dur_s if dur_s > 0 else 0.0


def _throughput_bps(bytes_value: float, dur_s: float) -> float:
    return _rate(bytes_value * 8.0, dur_s)


_FEATURE_FNS: dict[str, Any] = {
    "PROTOCOL": lambda f: _num(f, "protocol"),
    "IN_BYTES": lambda f: _num(f, "src2dst_bytes"),
    "OUT_BYTES": lambda f: _num(f, "dst2src_bytes"),
    "IN_PKTS": lambda f: _num(f, "src2dst_packets"),
    "OUT_PKTS": lambda f: _num(f, "dst2src_packets"),
    "FLOW_DURATION_MILLISECONDS": lambda f: _num(f, "bidirectional_duration_ms"),
    "DURATION_IN": lambda f: _num(f, "src2dst_duration_ms"),
    "DURATION_OUT": lambda f: _num(f, "dst2src_duration_ms"),
    "SRC_TO_DST_SECOND_BYTES": lambda f: _rate(_num(f, "src2dst_bytes"), _duration_s(f)),
    "DST_TO_SRC_SECOND_BYTES": lambda f: _rate(_num(f, "dst2src_bytes"), _duration_s(f)),
    "SRC_TO_DST_AVG_THROUGHPUT": lambda f: _throughput_bps(_num(f, "src2dst_bytes"),
                                                          _duration_s(f)),
    "DST_TO_SRC_AVG_THROUGHPUT": lambda f: _throughput_bps(_num(f, "dst2src_bytes"),
                                                          _duration_s(f)),
    "MIN_IP_PKT_LEN": lambda f: _num(f, "bidirectional_min_ps"),
    "MAX_IP_PKT_LEN": lambda f: _num(f, "bidirectional_max_ps"),
    "LONGEST_FLOW_PKT": lambda f: _num(f, "bidirectional_max_ps"),
    "SHORTEST_FLOW_PKT": lambda f: _num(f, "bidirectional_min_ps"),
    "TCP_FLAGS": lambda f: _flag_mask(f, "bidirectional"),
    "CLIENT_TCP_FLAGS": lambda f: _flag_mask(f, "src2dst"),
    "SERVER_TCP_FLAGS": lambda f: _flag_mask(f, "dst2src"),
}
assert set(_FEATURE_FNS) | set(UNMAPPABLE_LIVE) == set(SHARED_FEATURES)
assert len(LIVE_FEATURES) == len(_FEATURE_FNS)


def featurize_flow(flow: Mapping[str, Any]) -> np.ndarray:
    """One flow dict to the 33 shared features (float64, ``SHARED_FEATURES`` order).

    Unmappable features are NaN. Missing keys give NaN too, never an error, so a flow that
    lacks statistics is still scored on the live subset. Pass the result to
    ``schema.coerce_matrix`` after selecting ``LIVE_COLUMNS``.
    """
    out = np.full(len(SHARED_FEATURES), _NAN, dtype=np.float64)
    for name, fn in _FEATURE_FNS.items():
        out[FEATURE_INDEX[name]] = fn(flow)
    return out


def flow_from_nfstream(obj: Any) -> dict[str, Any]:
    """Copy the nfstream attributes this module reads from an NFlow object into a dict."""
    return {key: getattr(obj, key, None) for key in NFSTREAM_KEYS}


def iter_nfstream(
    source: str,
    *,
    idle_timeout: int = 120,
    active_timeout: int = 1800,
) -> Iterator[dict[str, Any]]:
    """Flows from an interface name or a pcap path, as dicts. nfstream is imported here only."""
    from nfstream import NFStreamer  # lazy: not a project dependency

    streamer = NFStreamer(
        source=source,
        statistical_analysis=True,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
    )
    for flow in streamer:
        yield flow_from_nfstream(flow)


class HostWindow:
    """Distinct destination ports and flow count per source address over a sliding window.

    Used for host-window evidence (port scans). Old entries are dropped on each ``observe``.
    At most ``max_sources`` sources are kept; the oldest-seen one is evicted first.
    """

    def __init__(self, window_ms: int = 10_000, max_sources: int = 4096) -> None:
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        self.window_ms = window_ms
        self.max_sources = max_sources
        self._events: dict[str, deque[tuple[float, int]]] = {}

    def observe(self, src_ip: str, dst_port: int | None, ts_ms: float) -> WindowStats:
        events = self._events.get(src_ip)
        if events is None:
            if len(self._events) >= self.max_sources:
                self._events.pop(next(iter(self._events)))
            events = self._events[src_ip] = deque()
        events.append((float(ts_ms), int(dst_port) if dst_port is not None else -1))
        cutoff = ts_ms - self.window_ms
        while events and events[0][0] < cutoff:
            events.popleft()
        ports = {p for _, p in events if p >= 0}
        return WindowStats(src_ip=src_ip, distinct_dst_ports=len(ports), flows=len(events))
