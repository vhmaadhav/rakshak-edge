#!/usr/bin/env python3
"""Holdout pcaps for the NIDS rules, fixed before any rule is scored (docs/NIDS_EDGE.md,
"Holdout set (generated 2026-11-12, before scoring)").

    .venv/bin/python deploy/pi/service/make_holdout_pcaps.py --out results/nids_holdout/pcaps
    .venv/bin/python deploy/pi/service/make_holdout_pcaps.py --manifest-md   # settings table

Offline: packets are built and written with scapy's wrpcap. No socket is opened and nothing is
sent. Every address is RFC 1918 (172.16/12 and 10/8 here). The rng of each file is seeded from
HOLDOUT_TAG and the file name, so the same code gives the same flows. The design files of
make_test_pcaps.py were used while the rules were written and are not used here.

Each case is either benign (``group="benign"``, expected no alert, including two benign
look-alikes), a main attack (``group="main"``), a threshold case (``group="near"``, the expected
outcome is the rule as written, so a miss or a fire there is a result, not a surprise), or a mixed
file (``group="mixed"``, two attackers and benign traffic in one capture).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scapy.all import DNS, DNSQR, IP, TCP, Raw, wrpcap  # type: ignore

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from make_test_pcaps import _ether, _tcp_session, _udp_pair  # noqa: E402

HOLDOUT_TAG = "holdout-2026-11-12"
T0 = 1_700_200_000.0
CLIENT_A = "172.16.5.20"  # web, benign mix, CDN look-alikes
CLIENT_B = "172.16.5.44"  # interactive SSH, SSH near miss
CLIENT_T = "172.16.5.60"  # DNS tunnel client
SERVERS = ("172.16.5.10", "172.16.5.11", "172.16.5.12", "172.16.5.13")
SSH_HOST = SERVERS[0]
RESOLVER = "172.16.5.1"
ATTACKER = "10.20.30.40"  # SSH and telnet brute force
SCANNER = "10.20.31.41"  # port scans
FLOODER = "10.20.32.42"  # SYN floods
CDN_NAME = "assets-prod-eu-west-1-7f3a9c2e1b0d4a6f8e5c.cdn.example.net"  # 42-char label
B32 = "abcdefghijklmnopqrstuvwxyz234567"
LOWER_DIGIT = "abcdefghijklmnopqrstuvwxyz0123456789"
HEX = "0123456789abcdef"
SAFE_NAMES = (
    "news.example.org",
    "mail.example.net",
    "docs.example.org",
    "static.example.org",
    "time.example.net",
    "api.example.net",
    "img.example.org",
    "cdn-small.example.net",
    "login.example.org",
    "updates.example.net",
    "weather.example.org",
    "shop.example.net",
)


def _rng(name: str) -> random.Random:
    return random.Random(f"{HOLDOUT_TAG}:{name}")


def _label(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _emit(steps, t_start):
    """Steps are (src, sport, dst, dport, flags, seq, ack, gap_s, payload)."""
    pkts = []
    t = t_start
    for s_ip, s_p, d_ip, d_p, flags, seq, ack, gap, load in steps:
        t += gap
        pkt = _ether(
            IP(src=s_ip, dst=d_ip)
            / TCP(sport=s_p, dport=d_p, flags=flags, seq=seq, ack=ack, window=64240)
            / Raw(load)
        )
        pkt.time = t
        pkts.append(pkt)
    return pkts, t


def _handshake(client, cport, server, sport, rng):
    seq, srv = rng.randrange(1, 2**31), rng.randrange(1, 2**31)
    steps = [
        (client, cport, server, sport, "S", seq, 0, 0.0, b""),
        (server, sport, client, cport, "SA", srv, seq + 1, 0.02, b""),
        (client, cport, server, sport, "A", seq + 1, srv + 1, 0.02, b""),
    ]
    return steps, seq + 1, srv + 1


def _close(steps, client, cport, server, sport, c, s):
    steps.append((client, cport, server, sport, "FA", c, s, 0.1, b""))
    steps.append((server, sport, client, cport, "FA", s, c + 1, 0.02, b""))
    steps.append((client, cport, server, sport, "A", c + 1, s + 1, 0.02, b""))


def _bulk(t, client, cport, server, sport, rng, packets, gap):
    """One long server-to-client transfer (HTTPS or SSH data), 1400-byte segments."""
    steps, c, s = _handshake(client, cport, server, sport, rng)
    for i in range(packets):
        steps.append((server, sport, client, cport, "PA", s, c, gap, b"d" * 1400))
        s += 1400
        if i % 4 == 3:
            steps.append((client, cport, server, sport, "A", c, s, 0.01, b""))
    _close(steps, client, cport, server, sport, c, s)
    return _emit(steps, t)


def _interactive(t, client, cport, server, sport, rng, exchanges, span):
    """An interactive SSH session: small keystrokes and echoes spread over ``span`` seconds."""
    steps, c, s = _handshake(client, cport, server, sport, rng)
    mean_gap = span / exchanges
    for _ in range(exchanges):
        n = rng.randrange(1, 9)
        steps.append(
            (client, cport, server, sport, "PA", c, s, rng.uniform(0.5, 1.5) * mean_gap, b"k" * n)
        )
        c += n
        m = rng.randrange(20, 121)
        steps.append((server, sport, client, cport, "PA", s, c, 0.01, b"e" * m))
        s += m
    _close(steps, client, cport, server, sport, c, s)
    return _emit(steps, t)


def _brute(port, attacker, count, spacing, rng, server=SSH_HOST):
    """``count`` short TCP sessions (a failed-login shape) at a fixed spacing."""
    pkts = []
    for k in range(count):
        sess, _ = _tcp_session(
            T0 + 1.0 + k * spacing,
            attacker,
            52000 + k,
            server,
            port,
            rng,
            payload_len=rng.randrange(40, 90),
        )
        pkts += sess
    return pkts


def _scan(t0, scanner, target, ports, spacing, rng, connect):
    """One SYN per port. ``connect=False``: the target answers RST-ACK (SYN scan). ``True``: the
    target answers SYN-ACK, the scanner ACKs and resets (TCP connect scan)."""
    pkts = []
    t = t0
    for i, port in enumerate(ports):
        t += spacing
        sport = 46000 + i
        seq = rng.randrange(2**31)
        syn = _ether(
            IP(src=scanner, dst=target)
            / TCP(sport=sport, dport=port, flags="S", seq=seq, window=1024)
        )
        syn.time = t
        pkts.append(syn)
        if connect:
            sa = _ether(
                IP(src=target, dst=scanner)
                / TCP(
                    sport=port,
                    dport=sport,
                    flags="SA",
                    seq=rng.randrange(2**31),
                    ack=seq + 1,
                    window=64240,
                )
            )
            sa.time = t + 0.001
            ack = _ether(
                IP(src=scanner, dst=target)
                / TCP(sport=sport, dport=port, flags="A", seq=seq + 1, ack=1, window=1024)
            )
            ack.time = t + 0.002
            rst = _ether(
                IP(src=scanner, dst=target)
                / TCP(sport=sport, dport=port, flags="R", seq=seq + 1, window=0)
            )
            rst.time = t + 0.003
            pkts += [sa, ack, rst]
        else:
            rst = _ether(
                IP(src=target, dst=scanner)
                / TCP(sport=port, dport=sport, flags="RA", seq=0, ack=1, window=0)
            )
            rst.time = t + 0.001
            pkts.append(rst)
    return pkts


def _dns(t0, client, names, spacing, rng, answer=True):
    """DNS queries (UDP, port 53) from ``client`` to the resolver, one name per query."""
    pkts = []
    t = t0
    for i, name in enumerate(names):
        t += spacing(rng) if callable(spacing) else spacing
        req = DNS(
            rd=1, qd=DNSQR(qname=name, qtype="TXT" if name.count(".") and len(name) > 60 else "A")
        )
        rep = DNS(qr=1, rd=1, ra=1, qd=DNSQR(qname=name), an=None) if answer else None
        pkts += _udp_pair(t, client, 50000 + i, RESOLVER, 53, req, rep)
    return pkts


def _syns(t0, src, dst, dport, count, spacing, rng):
    """``count`` SYN-only TCP connections (no reply) from ``src`` to ``dst``:``dport``."""
    pkts = []
    t = t0
    ports = rng.sample(range(1024, 65535), count)
    for port in ports:
        t += spacing
        syn = _ether(
            IP(src=src, dst=dst)
            / TCP(sport=port, dport=dport, flags="S", seq=rng.randrange(2**31), window=2048)
        )
        syn.time = t
        pkts.append(syn)
    return pkts


def _web_sessions(t, client, rng, count, base_port=42000):
    pkts = []
    for i in range(count):
        t += rng.uniform(1.0, 4.0)
        sess, t = _tcp_session(
            t,
            client,
            base_port + i,
            SERVERS[i % len(SERVERS)],
            443,
            rng,
            payload_len=rng.randrange(400, 1400),
        )
        pkts += sess
    return pkts, t


# ---------------------------------------------------------------- benign ---------------------
def b_benign_web_dns(rng):
    pkts, t = _bulk(T0 + 1.0, CLIENT_A, 44000, SERVERS[0], 443, rng, packets=200, gap=0.15)
    sess, t = _web_sessions(t, CLIENT_A, rng, count=10)
    pkts += sess
    pkts += _udp_pair(
        t + 2.0,
        CLIENT_A,
        123,
        RESOLVER,
        123,
        Raw(b"\x1b" + b"\x00" * 47),
        Raw(b"\x1c" + b"\x00" * 47),
    )
    for k in range(2):  # mDNS sent unicast to the resolver (stays inside RFC 1918)
        pkts += _udp_pair(
            t + 6.0 + 4.0 * k,
            CLIENT_A,
            5353,
            RESOLVER,
            5353,
            DNS(rd=0, qd=DNSQR(qname="printer.local")),
        )
    pkts += _dns(t + 1.0, CLIENT_A, list(SAFE_NAMES[:12]), 1.5, rng)
    return pkts


def b_benign_ssh_interactive(rng):
    pkts = []
    for k in range(8):
        sess, _ = _interactive(
            T0 + 1.0 + 30.0 * k, CLIENT_B, 56000 + k, SSH_HOST, 22, rng, exchanges=40, span=25.0
        )
        pkts += sess
    return pkts


def b_benign_ssh_bulk(rng):
    pkts, _ = _bulk(T0 + 1.0, CLIENT_B, 56100, SSH_HOST, 22, rng, packets=300, gap=0.05)
    return pkts


def b_benign_near_miss(rng):
    """Six short SSH logins four seconds apart, and three lookups of the long CDN name."""
    pkts = []
    for k in range(6):
        sess, _ = _tcp_session(
            T0 + 1.0 + 4.0 * k,
            CLIENT_B,
            54000 + k,
            SSH_HOST,
            22,
            rng,
            payload_len=rng.randrange(200, 600),
        )
        pkts += sess
    pkts += _dns(T0 + 2.0, CLIENT_B, [CDN_NAME] * 3, 5.0, rng)
    return pkts


def b_benign_cdn_lookalike_4(rng):
    """Four lookups of a long CDN name in 60 s: below the five-query line, so no alert."""
    return _dns(T0 + 1.0, CLIENT_A, [CDN_NAME] * 4, 10.0, rng) + _dns(
        T0 + 60.0, CLIENT_A, list(SAFE_NAMES[:2]), 2.0, rng
    )


def b_benign_cdn_lookalike_6(rng):
    """Six lookups of the long CDN name in 40 s. The rule shape is met on purpose (a known weak
    point of the length test); expected to be a benign false alert."""
    return _dns(T0 + 1.0, CLIENT_A, [CDN_NAME] * 6, 8.0, rng)


# ----------------------------------------------------------------- main attacks --------------
def b_ssh_brute_fast(rng):
    return _brute(22, ATTACKER, 40, 0.5, rng)


def b_ssh_brute_slow(rng):
    return _brute(22, ATTACKER, 12, 6.0, rng)


def b_telnet_brute(rng):
    return _brute(23, ATTACKER, 12, 1.0, rng)


def b_port_scan_syn_fast(rng):
    ports = sorted(rng.sample(range(1, 1025), 40))
    return _scan(T0 + 1.0, SCANNER, SERVERS[1], ports, 0.05, rng, connect=False)


def b_port_scan_connect_fast(rng):
    ports = sorted(rng.sample(range(1, 1025), 30))
    return _scan(T0 + 1.0, SCANNER, SERVERS[2], ports, 0.1, rng, connect=True)


def b_dns_tunnel_b32(rng):
    names = [_label(rng, B32, 32) + ".t.holdout-lab.test" for _ in range(20)]
    return _dns(T0 + 1.0, CLIENT_T, names, lambda r: r.uniform(0.8, 1.2), rng, answer=False)


def b_dns_tunnel_long(rng):
    names = [
        _label(rng, LOWER_DIGIT, rng.randrange(45, 61)) + ".t.holdout-lab.test" for _ in range(8)
    ]
    return _dns(T0 + 1.0, CLIENT_T, names, 3.0, rng, answer=False)


def b_dos_syn_300_3s(rng):
    return _syns(T0 + 1.0, FLOODER, SERVERS[0], 80, 300, 0.01, rng)


def b_dos_syn_1000_20s(rng):
    return _syns(T0 + 1.0, FLOODER, SERVERS[0], 80, 1000, 0.02, rng)


def b_mixed_attacks(rng):
    pkts = _brute(22, ATTACKER, 10, 1.0, rng)
    ports = sorted(rng.sample(range(1, 1025), 30))
    pkts += _scan(T0 + 1.0, SCANNER, SERVERS[2], ports, 0.05, rng, connect=False)
    sess, _ = _web_sessions(T0 + 1.0, CLIENT_A, rng, count=4)
    return pkts + sess


# ------------------------------------------------------------ threshold cases ---------------
def b_ssh_brute_9(rng):
    return _brute(22, ATTACKER, 9, 3.0, rng)


def b_ssh_brute_10_5s(rng):
    return _brute(22, ATTACKER, 10, 5.0, rng)


def b_ssh_brute_10_7s(rng):
    return _brute(22, ATTACKER, 10, 7.0, rng)


def b_port_scan_20(rng):
    ports = sorted(rng.sample(range(1, 1025), 20))
    return _scan(T0 + 1.0, SCANNER, SERVERS[3], ports, 0.5, rng, connect=False)


def b_port_scan_slow(rng):
    ports = sorted(rng.sample(range(1, 1025), 40))
    return _scan(T0 + 1.0, SCANNER, SERVERS[3], ports, 1.0, rng, connect=False)


def b_dns_tunnel_5(rng):
    names = [_label(rng, B32, 32) + ".t.holdout-lab.test" for _ in range(5)]
    return _dns(T0 + 1.0, CLIENT_T, names, 4.0, rng, answer=False)


def b_dns_tunnel_4(rng):
    names = [_label(rng, B32, 32) + ".t.holdout-lab.test" for _ in range(4)]
    return _dns(T0 + 1.0, CLIENT_T, names, 4.0, rng, answer=False)


def b_dns_hex24(rng):
    names = [_label(rng, HEX, 24) + ".t.holdout-lab.test" for _ in range(10)]
    return _dns(T0 + 1.0, CLIENT_T, names, 2.0, rng, answer=False)


def b_dos_syn_150(rng):
    return _syns(T0 + 1.0, FLOODER, SERVERS[0], 80, 150, 10.0 / 150.0, rng)


def b_dos_syn_200(rng):
    return _syns(T0 + 1.0, FLOODER, SERVERS[0], 80, 200, 0.05, rng)


@dataclass(frozen=True)
class Case:
    name: str
    group: str  # benign | main | near | mixed
    expect: tuple[str, ...]  # () no alert; ("open",) no prediction; else the expected families
    srcs: tuple[str, ...]  # the expected alerting source for each family in ``expect``
    builder: Callable[[random.Random], list]
    settings: str


CASES: tuple[Case, ...] = (
    Case(
        "ho_benign_web_dns",
        "benign",
        (),
        (),
        b_benign_web_dns,
        "200-packet HTTPS download, 10 HTTPS sessions to 4 servers, 12 DNS names, NTP, 2 mDNS",
    ),
    Case(
        "ho_benign_ssh_interactive",
        "benign",
        (),
        (),
        b_benign_ssh_interactive,
        "8 interactive SSH sessions, 40 keystroke/echo exchanges each, 25 s, 30 s apart",
    ),
    Case(
        "ho_benign_ssh_bulk",
        "benign",
        (),
        (),
        b_benign_ssh_bulk,
        "one SSH bulk download, 300 x 1400 B, 0.05 s gap (about 15 s)",
    ),
    Case(
        "ho_benign_near_miss",
        "benign",
        (),
        (),
        b_benign_near_miss,
        "6 short SSH logins 4 s apart (below 10), 3 lookups of the 42-char CDN label 5 s apart",
    ),
    Case(
        "ho_benign_cdn_lookalike_4",
        "benign",
        (),
        (),
        b_benign_cdn_lookalike_4,
        "4 lookups of the 42-char CDN label in 30 s (below 5), then 2 normal names",
    ),
    Case(
        "ho_benign_cdn_lookalike_6",
        "benign",
        (),
        (),
        b_benign_cdn_lookalike_6,
        "6 lookups of the 42-char CDN label 8 s apart (rule shape met on purpose; expected FP)",
    ),
    Case(
        "ho_ssh_brute_fast",
        "main",
        ("brute_force_ssh",),
        (ATTACKER,),
        b_ssh_brute_fast,
        "40 short SSH attempts 0.5 s apart from 10.20.30.40 to :22",
    ),
    Case(
        "ho_ssh_brute_slow",
        "main",
        ("brute_force_ssh",),
        (ATTACKER,),
        b_ssh_brute_slow,
        "12 short SSH attempts 6 s apart (span 66 s, 11 fit in 60 s)",
    ),
    Case(
        "ho_telnet_brute",
        "main",
        ("brute_force_other",),
        (ATTACKER,),
        b_telnet_brute,
        "12 short sessions to :23 1 s apart",
    ),
    Case(
        "ho_port_scan_syn_fast",
        "main",
        ("port_scan",),
        (SCANNER,),
        b_port_scan_syn_fast,
        "40 ports, SYN then RST-ACK, 0.05 s apart",
    ),
    Case(
        "ho_port_scan_connect_fast",
        "main",
        ("port_scan",),
        (SCANNER,),
        b_port_scan_connect_fast,
        "30 ports, TCP connect (SYN, SYN-ACK, ACK, RST), 0.1 s apart",
    ),
    Case(
        "ho_dns_tunnel_b32",
        "main",
        ("dns_tunnel",),
        (CLIENT_T,),
        b_dns_tunnel_b32,
        "20 TXT queries, 32-char base32-like labels, about 1 s apart, no answers",
    ),
    Case(
        "ho_dns_tunnel_long",
        "main",
        ("dns_tunnel",),
        (CLIENT_T,),
        b_dns_tunnel_long,
        "8 TXT queries, 45 to 60 char lowercase+digit labels, 3 s apart",
    ),
    Case(
        "ho_dos_syn_300_3s",
        "main",
        ("dos_flood",),
        (FLOODER,),
        b_dos_syn_300_3s,
        "300 SYN-only connections to :80 in 3 s",
    ),
    Case(
        "ho_dos_syn_1000_20s",
        "main",
        ("dos_flood",),
        (FLOODER,),
        b_dos_syn_1000_20s,
        "1000 SYN-only connections to :80 in 20 s (50/s)",
    ),
    Case(
        "ho_mixed_attacks",
        "mixed",
        ("brute_force_ssh", "port_scan"),
        (ATTACKER, SCANNER),
        b_mixed_attacks,
        "10 SSH attempts 1 s apart from 10.20.30.40, 30-port SYN scan from 10.20.31.41, 4 HTTPS",
    ),
    Case(
        "ho_ssh_brute_9",
        "near",
        (),
        (),
        b_ssh_brute_9,
        "9 short SSH attempts 3 s apart (one below the line)",
    ),
    Case(
        "ho_ssh_brute_10_5s",
        "near",
        ("brute_force_ssh",),
        (ATTACKER,),
        b_ssh_brute_10_5s,
        "10 short SSH attempts 5 s apart (span 45 s, exactly the line)",
    ),
    Case(
        "ho_ssh_brute_10_7s",
        "near",
        (),
        (),
        b_ssh_brute_10_7s,
        "10 short SSH attempts 7 s apart (span 63 s, at most 9 fit in 60 s)",
    ),
    Case(
        "ho_port_scan_20",
        "near",
        ("port_scan",),
        (SCANNER,),
        b_port_scan_20,
        "20 ports, SYN scan, 0.5 s apart (exactly the 10 s line)",
    ),
    Case(
        "ho_port_scan_slow",
        "near",
        (),
        (),
        b_port_scan_slow,
        "40 ports, SYN scan, 1.0 s apart (10 s window holds about 11 ports)",
    ),
    Case(
        "ho_dns_tunnel_5",
        "near",
        ("dns_tunnel",),
        (CLIENT_T,),
        b_dns_tunnel_5,
        "5 base32-like queries 4 s apart (exactly the count line)",
    ),
    Case(
        "ho_dns_tunnel_4",
        "near",
        (),
        (),
        b_dns_tunnel_4,
        "4 base32-like queries 4 s apart (one below the line)",
    ),
    Case(
        "ho_dns_hex24",
        "near",
        ("open",),
        (),
        b_dns_hex24,
        "10 hex labels of 24 chars, 2 s apart (entropy near the 3.8 bit line; no prediction made)",
    ),
    Case(
        "ho_dos_syn_150",
        "near",
        (),
        (),
        b_dos_syn_150,
        "150 SYN-only connections in 10 s (below 200)",
    ),
    Case(
        "ho_dos_syn_200",
        "near",
        ("dos_flood",),
        (FLOODER,),
        b_dos_syn_200,
        "200 SYN-only connections, 0.05 s apart (span 9.95 s, exactly the line)",
    ),
)
CASE_BY_NAME = {c.name: c for c in CASES}


def manifest_markdown() -> str:
    head = "| File | Group | Settings | Expected | Expected source |\n|---|---|---|---|---|\n"
    rows = []
    for c in CASES:
        exp = (
            "none"
            if not c.expect
            else ("no prediction" if c.expect == ("open",) else " + ".join(c.expect))
        )
        src = " + ".join(c.srcs) if c.srcs else "-"
        rows.append(f"| `{c.name}.pcap` | {c.group} | {c.settings} | {exp} | {src} |")
    return head + "\n".join(rows) + "\n"


def write_holdout(out: Path) -> list[tuple[str, int]]:
    """Write the pcaps and ``manifest.json`` (the expectations, read by run_holdout_nids.py)."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    rows = []
    for case in CASES:
        pkts = case.builder(_rng(case.name))
        pkts.sort(key=lambda p: float(p.time))
        path = out / f"{case.name}.pcap"
        wrpcap(str(path), pkts)
        written.append((str(path), len(pkts)))
        rows.append(
            {
                "file": path.name,
                "group": case.group,
                "expect": list(case.expect),
                "srcs": list(case.srcs),
                "settings": case.settings,
                "packets": len(pkts),
            }
        )
    manifest = {"tag": HOLDOUT_TAG, "cases": rows}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", help="directory for the .pcap files")
    p.add_argument("--manifest-md", action="store_true", help="print the settings table and exit")
    args = p.parse_args(argv)
    if args.manifest_md:
        print(manifest_markdown(), end="")
        return 0
    if not args.out:
        p.error("--out is required unless --manifest-md is given")
    for path, n in write_holdout(Path(args.out)):
        print(f"{path} packets={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
