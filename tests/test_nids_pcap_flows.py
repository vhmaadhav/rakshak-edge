"""Pure-Python pcap reader (rakshak.nids.pcap_flows): flow grouping, idle split, flags, rejects."""

from __future__ import annotations

import struct

import pytest

from rakshak.nids.live import NFSTREAM_KEYS
from rakshak.nids.pcap_flows import flows_from_pcap_bytes

ETH = b"\x02" * 6 + b"\x02" * 6 + b"\x08\x00"


def _ip(src, dst, proto, l4, total=None):
    a = bytes(map(int, src.split(".")))
    b = bytes(map(int, dst.split(".")))
    total = 20 + len(l4) if total is None else total
    return struct.pack(">BBHHHBBH4s4s", 0x45, 0, total, 0, 0, 64, proto, 0, a, b) + l4


def _tcp(sport, dport, flags):
    return struct.pack(">HHIIBBHHH", sport, dport, 0, 0, 0x50, flags, 1024, 0, 0)


def _pcap(records, magic=b"\xd4\xc3\xb2\xa1", endian="<", network=1):
    out = bytearray(
        struct.pack(endian + "IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, network)
        if magic == b"\xd4\xc3\xb2\xa1"
        else b""
    )
    for ts_ms, frame in records:
        sec, ms = divmod(ts_ms, 1000)
        out += struct.pack(endian + "IIII", int(sec), int(ms * 1000), len(frame), len(frame))
        out += frame
    return bytes(out)


def test_groups_both_directions_into_one_flow_with_flags():
    recs = [
        (1000, ETH + _ip("10.0.0.1", "10.0.0.2", 6, _tcp(40000, 80, 0x02))),  # SYN
        (1005, ETH + _ip("10.0.0.2", "10.0.0.1", 6, _tcp(80, 40000, 0x12))),  # SYN-ACK
        (1010, ETH + _ip("10.0.0.1", "10.0.0.2", 6, _tcp(40000, 80, 0x10))),  # ACK
    ]
    flows = flows_from_pcap_bytes(_pcap(recs))
    assert len(flows) == 1
    f = flows[0]
    assert set(NFSTREAM_KEYS) <= set(f)
    assert (f["src_ip"], f["dst_ip"], f["dst_port"]) == ("10.0.0.1", "10.0.0.2", 80)
    assert f["src2dst_packets"] == 2 and f["dst2src_packets"] == 1
    assert f["bidirectional_syn_packets"] == 2
    assert f["src2dst_syn_packets"] == 1 and f["dst2src_syn_packets"] == 1
    assert f["bidirectional_duration_ms"] == 10.0
    assert f["bidirectional_bytes"] == 3 * 54  # captured frame: 14-byte Ethernet + 40-byte IP+TCP


def test_idle_timeout_starts_a_new_flow():
    frame = ETH + _ip("10.0.0.1", "10.0.0.2", 6, _tcp(40000, 22, 0x02))
    recs = [(1000, frame), (1000 + 121_000, frame)]  # 121 s apart, idle timeout is 120 s
    assert len(flows_from_pcap_bytes(_pcap(recs))) == 2


def test_udp_and_icmp_and_non_ip_frames():
    udp = struct.pack(">HHHH", 50000, 53, 8, 0)
    recs = [
        (1000, ETH + _ip("192.168.1.2", "192.168.1.1", 17, udp)),
        (1001, ETH + _ip("192.168.1.2", "192.168.1.1", 1, b"\x08\x00" + b"\x00" * 6)),
        (1002, b"\x00" * 6 + b"\x00" * 6 + b"\x86\xdd" + b"\x00" * 40),  # IPv6: skipped
    ]
    flows = flows_from_pcap_bytes(_pcap(recs))
    protos = sorted((f["protocol"], f["dst_port"]) for f in flows)
    assert protos == [(1, 0), (17, 53)]


def test_big_endian_nanosecond_header_is_read():
    frame = ETH + _ip("10.1.1.1", "10.1.1.2", 6, _tcp(1, 2, 0x02))
    out = bytearray(struct.pack(">IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 65535, 1))
    out += struct.pack(">IIII", 1_700_000_000, 5_000_000, len(frame), len(frame)) + frame
    flows = flows_from_pcap_bytes(bytes(out))
    assert flows[0]["bidirectional_first_seen_ms"] == 1_700_000_000_000 + 5.0


@pytest.mark.parametrize(
    "data, message",
    [
        (b"short", "shorter"),
        (b"\x00\x01\x02\x03" + b"\x00" * 20, "bad magic"),
        (b"\x0a\x0d\x0d\x0a" + b"\x00" * 40, "pcapng"),
        (_pcap([], network=147), "link type"),
    ],
)
def test_rejects_bad_input(data, message):
    with pytest.raises(ValueError, match=message):
        flows_from_pcap_bytes(data)


def test_truncated_record_is_ignored_not_fatal():
    frame = ETH + _ip("10.0.0.1", "10.0.0.2", 6, _tcp(1, 2, 0x02))
    good = _pcap([(1000, frame)])
    flows = flows_from_pcap_bytes(good + struct.pack("<IIII", 1, 0, 100, 100) + b"\x00" * 10)
    assert len(flows) == 1


def test_ipv4_packet_with_invalid_ihl_is_ignored():
    # Version 4, IHL 0 (< 5, invalid) should be skipped rather than parsing IP header bytes as L4.
    bad_ip = struct.pack(
        ">BBHHHBBH4s4s", 0x40, 0, 20, 0, 0, 64, 6, 0, b"\x0a\x00\x00\x01", b"\x0a\x00\x00\x02"
    )
    assert flows_from_pcap_bytes(_pcap([(1000, ETH + bad_ip)])) == []
