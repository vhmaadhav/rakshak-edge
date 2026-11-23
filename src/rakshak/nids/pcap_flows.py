"""Pure-Python pcap to nfstream-style flow dicts (the fallback when nfstream is not installed).

``flows_from_pcap_bytes`` reads a classic libpcap file (not pcapng) with ``struct`` only and
groups IPv4 TCP, UDP and other packets into bidirectional flows. It emits dicts with the keys in
``live.NFSTREAM_KEYS``, so ``live.featurize_flow`` and the model see the same shape as with
nfstream. Only the statistics ``featurize_flow`` reads are filled. IPv6, fragments and tunnels are
skipped or counted as ports 0. Packet and byte sizes are captured frame lengths (Ethernet included),
as nfstream reports them. The timeouts follow nfstream's defaults (idle 120 s, active 1800 s).

This module never opens a socket and never captures live traffic. It reads bytes it is given.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass, field
from typing import Any

from rakshak.nids.live import FLAG_BITS, NFSTREAM_KEYS

_MAGICS: dict[bytes, tuple[str, int]] = {
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000),  # big-endian, microseconds
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000),  # little-endian, microseconds
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),  # big-endian, nanoseconds
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000),  # little-endian, nanoseconds
}
_LINKTYPES = {1: "ether", 101: "raw", 113: "sll"}
_VLAN = (0x8100, 0x88A8)
MAX_PACKETS = 2_000_000


@dataclass
class _Dir:
    packets: int = 0
    bytes: int = 0
    first_ms: float | None = None
    last_ms: float | None = None
    flags: dict[str, int] = field(default_factory=lambda: {name: 0 for name in FLAG_BITS})


@dataclass
class _Flow:
    init: tuple[str, int]  # (ip, port) of the first packet: the flow initiator
    peer: tuple[str, int]
    proto: int
    first_ms: float
    last_ms: float
    min_ps: int | None = None
    max_ps: int | None = None
    fwd: _Dir = field(default_factory=_Dir)  # initiator to peer
    back: _Dir = field(default_factory=_Dir)  # peer to initiator
    qname: str | None = None  # DNS question name of the first query (client to port 53)

    def add(self, forward: bool, ts_ms: float, size: int, tcp_flags: int | None) -> None:
        d = self.fwd if forward else self.back
        d.packets += 1
        d.bytes += size
        d.first_ms = ts_ms if d.first_ms is None else min(d.first_ms, ts_ms)
        d.last_ms = ts_ms if d.last_ms is None else max(d.last_ms, ts_ms)
        self.last_ms = max(self.last_ms, ts_ms)
        self.min_ps = size if self.min_ps is None else min(self.min_ps, size)
        self.max_ps = size if self.max_ps is None else max(self.max_ps, size)
        if tcp_flags is not None:
            for name, bit in FLAG_BITS.items():
                if tcp_flags & bit:
                    d.flags[name] += 1

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {key: None for key in NFSTREAM_KEYS}
        ip_a, port_a = self.init
        ip_b, port_b = self.peer
        out.update(
            src_ip=ip_a,
            dst_ip=ip_b,
            src_port=port_a,
            dst_port=port_b,
            protocol=self.proto,
            bidirectional_first_seen_ms=self.first_ms,
            bidirectional_duration_ms=self.last_ms - self.first_ms,
            bidirectional_bytes=self.fwd.bytes + self.back.bytes,
            bidirectional_min_ps=self.min_ps,
            bidirectional_max_ps=self.max_ps,
            src2dst_duration_ms=_span(self.fwd),
            src2dst_packets=self.fwd.packets,
            src2dst_bytes=self.fwd.bytes,
            dst2src_duration_ms=_span(self.back),
            dst2src_packets=self.back.packets,
            dst2src_bytes=self.back.bytes,
        )
        out["requested_server_name"] = self.qname
        for name in FLAG_BITS:
            total = self.fwd.flags[name] + self.back.flags[name]
            out[f"bidirectional_{name}_packets"] = total
            out[f"src2dst_{name}_packets"] = self.fwd.flags[name]
            out[f"dst2src_{name}_packets"] = self.back.flags[name]
        return out


def _span(d: _Dir) -> float:
    if d.first_ms is None or d.last_ms is None:
        return 0.0
    return d.last_ms - d.first_ms


def dns_question_name(payload: bytes) -> str | None:
    """The first question name of a DNS message (lower case, no trailing dot), or None.

    Only plain labels are read. A compression pointer in the question is not expected in a query
    and gives None, as does any malformed message. Matches nfstream's ``requested_server_name``
    for a DNS query (the name is the same string, so the DNS rules see the same field).
    """
    if len(payload) < 13 or struct.unpack(">H", payload[4:6])[0] < 1:
        return None
    labels: list[str] = []
    pos = 12
    total = 0
    while True:
        if pos >= len(payload):
            return None
        n = payload[pos]
        if n == 0:
            break
        if n & 0xC0 or n > 63:
            return None
        label = payload[pos + 1 : pos + 1 + n]
        if len(label) != n:
            return None
        labels.append(label.decode("ascii", "replace"))
        total += n + 1
        pos += 1 + n
        if total > 255:
            return None
    return ".".join(labels).lower() if labels else None


def _parse_packet(
    data: bytes, linktype: str
) -> tuple[str, str, int, int, int, int, int | None, str | None] | None:
    """(src, dst, proto, sport, dport, frame_len, tcp_flags, dns_qname) for IPv4 packets, else None.

    The size is the captured frame length (Ethernet header included), as nfstream reports
    packet and byte counts. Using the IP total length instead would move every byte feature.
    """
    if linktype == "ether":
        if len(data) < 14:
            return None
        ethertype = struct.unpack(">H", data[12:14])[0]
        offset = 14
        if ethertype in _VLAN:
            if len(data) < 18:
                return None
            ethertype = struct.unpack(">H", data[16:18])[0]
            offset = 18
    elif linktype == "sll":
        if len(data) < 16:
            return None
        ethertype = struct.unpack(">H", data[14:16])[0]
        offset = 16
    else:
        ethertype, offset = 0x0800, 0
    if ethertype != 0x0800 or len(data) < offset + 20:
        return None
    ip = data[offset:]
    if ip[0] >> 4 != 4:
        return None
    ihl = (ip[0] & 0x0F) * 4
    if ihl < 20 or len(ip) < ihl:
        return None
    proto = ip[9]
    src = socket.inet_ntoa(ip[12:16])
    dst = socket.inet_ntoa(ip[16:20])
    l4 = ip[ihl:]
    sport = dport = 0
    flags: int | None = None
    qname: str | None = None
    if proto in (6, 17) and len(l4) >= 4:
        sport, dport = struct.unpack(">HH", l4[:4])
    if proto == 6 and len(l4) >= 14:
        flags = l4[13]
    if proto == 17 and dport == 53 and len(l4) > 8:
        qname = dns_question_name(l4[8:])
    return src, dst, proto, sport, dport, len(data), flags, qname


def flows_from_pcap_bytes(
    data: bytes,
    *,
    idle_timeout_s: int = 120,
    active_timeout_s: int = 1800,
) -> list[dict[str, Any]]:
    """Flow dicts (``live.NFSTREAM_KEYS``) from a classic pcap, sorted by first-seen time.

    Raises ValueError for a file that is not a classic pcap, a link type other than Ethernet,
    raw IP or Linux SLL, or more than ``MAX_PACKETS`` packets.
    """
    if len(data) < 24:
        raise ValueError("pcap is shorter than its global header")
    magic = data[:4]
    if magic == b"\x0a\x0d\x0d\x0a":
        raise ValueError("pcapng is not supported; save the capture as classic pcap")
    if magic not in _MAGICS:
        raise ValueError("not a pcap file (bad magic)")
    endian, divisor = _MAGICS[magic]
    network = struct.unpack(endian + "I", data[20:24])[0]
    if network not in _LINKTYPES:
        raise ValueError(f"unsupported pcap link type {network}")
    linktype = _LINKTYPES[network]
    idle_ms = idle_timeout_s * 1000.0
    active_ms = active_timeout_s * 1000.0

    flows: list[dict[str, Any]] = []
    table: dict[tuple, _Flow] = {}
    pos = 24
    count = 0
    while pos + 16 <= len(data):
        ts_sec, ts_frac, incl_len, _orig = struct.unpack(endian + "IIII", data[pos : pos + 16])
        pos += 16
        packet = data[pos : pos + incl_len]
        pos += incl_len
        count += 1
        if count > MAX_PACKETS:
            raise ValueError(f"pcap has more than {MAX_PACKETS} packets")
        ts_ms = ts_sec * 1000.0 + ts_frac * 1000.0 / divisor
        parsed = _parse_packet(packet, linktype)
        if parsed is None:
            continue
        src, dst, proto, sport, dport, size, flags, qname = parsed
        a, b = (src, sport), (dst, dport)
        key = (proto,) + tuple(sorted([a, b]))
        flow = table.get(key)
        if flow is not None and (
            ts_ms - flow.last_ms > idle_ms or ts_ms - flow.first_ms > active_ms
        ):
            flows.append(flow.to_dict())
            flow = None
        if flow is None:
            flow = _Flow(init=a, peer=b, proto=proto, first_ms=ts_ms, last_ms=ts_ms)
            table[key] = flow
        flow.add(forward=(a == flow.init), ts_ms=ts_ms, size=size, tcp_flags=flags)
        if qname is not None and dport == 53 and flow.qname is None:
            flow.qname = qname
    flows.extend(f.to_dict() for f in table.values())
    flows.sort(key=lambda f: f["bidirectional_first_seen_ms"])
    return flows
