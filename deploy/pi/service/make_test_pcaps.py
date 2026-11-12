#!/usr/bin/env python3
"""Write synthetic test pcaps with scapy. Offline: writes files only, never sends a packet.

    .venv/bin/python deploy/pi/service/make_test_pcaps.py --out DIR [--seed 0]

Writes four files into DIR:
  benign.pcap       normal-looking TCP HTTPS sessions and DNS queries between private addresses
  port_scan.pcap    one private host sends SYNs to ~100 ports of another, which answers with RST
  ssh_brute.pcap    many short TCP sessions to port 22 of one private host (failed-login shape)
  dns_long.pcap     DNS queries with long labels from one private host (optional tunnel shape)

Every address is RFC 1918 (10/8, 172.16/12 or 192.168/16). The packets are built and written with
scapy's wrpcap. No socket is opened and nothing is sent, so these files are synthetic and offline:
they say nothing about real attacks or real hosts. Use them only for the service's functional test.

    .venv/bin/python deploy/pi/service/make_test_pcaps.py --out DIR --profile holdout

The holdout profile writes six other files (holdout_*.pcap, with a different seed, addresses and
parameters; docs/NIDS_EDGE.md, "Holdout profile"): a benign mix with a long download, NTP, mDNS and
several HTTPS sessions; a slow SSH brute force (12 attempts 2 s apart); a short port scan (25
ports); DNS tunnelling with base32-looking labels; a SYN flood of 300 SYNs; and a near-miss benign
case (6 SSH logins and 3 lookups of a long CDN name).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from scapy.all import DNS, DNSQR, IP, TCP, UDP, Ether, Raw, wrpcap  # type: ignore

CLIENT = "192.168.1.20"
SERVER = "192.168.1.10"
ATTACKER = "10.0.0.66"
DNS_RESOLVER = "192.168.1.1"
SSH_HOST = "192.168.1.10"
T0 = 1_700_000_000.0  # fixed epoch, so the files are reproducible


def _ether(pkt):
    return Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / pkt


def _tcp_session(t, src, sport, dst, dport, rng, payload_len=0):
    """A short TCP session: SYN, SYN-ACK, ACK, optional data both ways, FIN exchange."""
    pkts = []
    seq = rng.randrange(1, 2**31)
    srv = rng.randrange(1, 2**31)
    steps = [
        (src, sport, dst, dport, "S", seq, 0, 0.0),
        (dst, dport, src, sport, "SA", srv, seq + 1, 0.02),
        (src, sport, dst, dport, "A", seq + 1, srv + 1, 0.02),
    ]
    if payload_len:
        steps.append((src, sport, dst, dport, "PA", seq + 1, srv + 1, 0.03))
        steps.append((dst, dport, src, sport, "PA", srv + 1, seq + 1 + payload_len, 0.05))
    steps.append((src, sport, dst, dport, "FA", seq + 1 + payload_len, srv + 1 + payload_len, 0.1))
    steps.append((dst, dport, src, sport, "FA", srv + 1 + payload_len, seq + 2 + payload_len, 0.02))
    steps.append((src, sport, dst, dport, "A", seq + 2 + payload_len, srv + 2 + payload_len, 0.02))
    t_cur = t
    for s_ip, s_p, d_ip, d_p, flags, s_seq, s_ack, gap in steps:
        t_cur += gap
        load = b""
        if "P" in flags and payload_len:
            load = (b"x" if s_ip == src else b"y") * payload_len
        pkt = _ether(
            IP(src=s_ip, dst=d_ip)
            / TCP(sport=s_p, dport=d_p, flags=flags, seq=s_seq, ack=s_ack, window=64240)
            / Raw(load)
        )
        pkt.time = t_cur
        pkts.append(pkt)
    return pkts, t_cur


def benign(rng) -> list:
    pkts = []
    t = T0
    for i in range(6):
        t += rng.uniform(0.5, 3.0)
        sess, t = _tcp_session(
            t, CLIENT, 40000 + i, SERVER, 443, rng, payload_len=rng.randrange(300, 1200)
        )
        pkts += sess
    for i in range(8):
        t += rng.uniform(0.2, 2.0)
        q = _ether(
            IP(src=CLIENT, dst=DNS_RESOLVER)
            / UDP(sport=50000 + i, dport=53)
            / DNS(rd=1, qd=DNSQR(qname=f"host{i}.example.internal"))
        )
        q.time = t
        r = _ether(
            IP(src=DNS_RESOLVER, dst=CLIENT)
            / UDP(sport=53, dport=50000 + i)
            / DNS(qr=1, rd=1, ra=1, qd=DNSQR(qname=f"host{i}.example.internal"), an=None)
        )
        r.time = t + 0.01
        pkts += [q, r]
    return pkts


def port_scan(rng) -> list:
    pkts = []
    t = T0
    ports = rng.sample(range(1, 1025), 100)
    for i, port in enumerate(sorted(ports)):
        t += 0.01
        syn = _ether(
            IP(src=ATTACKER, dst=SERVER)
            / TCP(sport=45000 + i, dport=port, flags="S", seq=rng.randrange(2**31), window=1024)
        )
        syn.time = t
        rst = _ether(
            IP(src=SERVER, dst=ATTACKER)
            / TCP(sport=port, dport=45000 + i, flags="RA", seq=0, ack=1, window=0)
        )
        rst.time = t + 0.001
        pkts += [syn, rst]
    return pkts


def ssh_brute(rng) -> list:
    pkts = []
    t = T0
    for i in range(60):
        t += rng.uniform(0.8, 2.5)
        sess, t = _tcp_session(
            t, ATTACKER, 52000 + i, SSH_HOST, 22, rng, payload_len=rng.randrange(40, 90)
        )
        pkts += sess
    return pkts


def dns_long(rng) -> list:
    pkts = []
    t = T0
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    for i in range(40):
        t += rng.uniform(0.3, 1.2)
        label = "".join(rng.choice(alphabet) for _ in range(60))
        q = _ether(
            IP(src=CLIENT, dst=DNS_RESOLVER)
            / UDP(sport=51000 + i, dport=53)
            / DNS(rd=1, qd=DNSQR(qname=f"{label}.tunnel.example.internal", qtype="TXT"))
        )
        q.time = t
        pkts.append(q)
    return pkts


# --- holdout profile: different seed, addresses and parameters (not used to tune any rule) ---
T0_HOLDOUT = 1_700_100_000.0
HO_CLIENT = "192.168.2.20"
HO_CLIENT_B = "192.168.2.40"
HO_SERVERS = ("192.168.2.10", "192.168.2.11", "192.168.2.12", "192.168.2.13")
HO_RESOLVER = "192.168.2.1"
HO_ATTACKER = "10.0.1.23"
HO_SCANNER = "10.0.2.50"
HO_FLOODER = "10.0.3.77"
HO_TUNNEL_CLIENT = "192.168.2.30"
BASE32 = "abcdefghijklmnopqrstuvwxyz234567"


def _download(t, server, sport, client, rng, packets=150, gap=0.2):
    """A long HTTPS download: handshake, server data every ``gap`` s, client ACK every 4 packets."""
    pkts = []
    seq, srv = rng.randrange(1, 2**31), rng.randrange(1, 2**31)
    hs = [
        (client, sport, server, 443, "S", seq, 0, 0.0, b""),
        (server, 443, client, sport, "SA", srv, seq + 1, 0.02, b""),
        (client, sport, server, 443, "A", seq + 1, srv + 1, 0.02, b""),
    ]
    cur_c, cur_s = seq + 1, srv + 1
    for i in range(packets):
        hs.append((server, 443, client, sport, "PA", cur_s, cur_c, gap, b"d" * 1400))
        cur_s += 1400
        if i % 4 == 3:
            hs.append((client, sport, server, 443, "A", cur_c, cur_s, 0.01, b""))
    hs.append((client, sport, server, 443, "FA", cur_c, cur_s, 0.1, b""))
    hs.append((server, 443, client, sport, "FA", cur_s, cur_c + 1, 0.02, b""))
    hs.append((client, sport, server, 443, "A", cur_c + 1, cur_s + 1, 0.02, b""))
    t_cur = t
    for s_ip, s_p, d_ip, d_p, flags, s_seq, s_ack, gap_s, load in hs:
        t_cur += gap_s
        pkt = _ether(
            IP(src=s_ip, dst=d_ip)
            / TCP(sport=s_p, dport=d_p, flags=flags, seq=s_seq, ack=s_ack, window=64240)
            / Raw(load)
        )
        pkt.time = t_cur
        pkts.append(pkt)
    return pkts, t_cur


def _udp_pair(t, src, sport, dst, dport, req, rep=None):
    pkts = []
    q = _ether(IP(src=src, dst=dst) / UDP(sport=sport, dport=dport) / req)
    q.time = t
    pkts.append(q)
    if rep is not None:
        r = _ether(IP(src=dst, dst=src) / UDP(sport=dport, dport=sport) / rep)
        r.time = t + 0.01
        pkts.append(r)
    return pkts


def holdout_benign(rng) -> list:
    pkts = []
    t = T0_HOLDOUT
    dl, t = _download(t + 1.0, HO_SERVERS[0], 44000, HO_CLIENT, rng)
    pkts += dl
    for i, server in enumerate(HO_SERVERS):
        t += rng.uniform(1.0, 4.0)
        sess, t = _tcp_session(
            t, HO_CLIENT, 42000 + i, server, 443, rng, payload_len=rng.randrange(400, 1400)
        )
        pkts += sess
    ntp_req = Raw(b"\x1b" + b"\x00" * 47)
    t += 2.0
    pkts += _udp_pair(t, HO_CLIENT, 123, HO_RESOLVER, 123, ntp_req, ntp_req)
    for _ in range(2):
        t += rng.uniform(3.0, 6.0)
        pkts += _udp_pair(
            t,
            HO_CLIENT,
            5353,
            HO_RESOLVER,
            5353,
            DNS(rd=0, qd=DNSQR(qname="printer.local")),
        )
    names = [
        "api.example.net",
        "news.example.org",
        "mail.example.net",
        "static.example.org",
        "time.example.net",
        "docs.example.org",
    ]
    for i, name in enumerate(names):
        t += rng.uniform(0.5, 2.0)
        pkts += _udp_pair(
            t,
            HO_CLIENT,
            50100 + i,
            HO_RESOLVER,
            53,
            DNS(rd=1, qd=DNSQR(qname=name)),
            DNS(qr=1, rd=1, ra=1, qd=DNSQR(qname=name), an=None),
        )
    return pkts


def holdout_ssh_slow(rng) -> list:
    pkts = []
    t = T0_HOLDOUT
    for i in range(12):
        sess, _ = _tcp_session(
            t, HO_ATTACKER, 53000 + i, HO_SERVERS[0], 22, rng, payload_len=rng.randrange(40, 90)
        )
        pkts += sess
        t += 2.0
    return pkts


def holdout_port_scan(rng) -> list:
    pkts = []
    t = T0_HOLDOUT
    for i, port in enumerate(sorted(rng.sample(range(1, 1025), 25))):
        t += 0.1
        syn = _ether(
            IP(src=HO_SCANNER, dst=HO_SERVERS[1])
            / TCP(sport=46000 + i, dport=port, flags="S", seq=rng.randrange(2**31), window=1024)
        )
        syn.time = t
        rst = _ether(
            IP(src=HO_SERVERS[1], dst=HO_SCANNER)
            / TCP(sport=port, dport=46000 + i, flags="RA", seq=0, ack=1, window=0)
        )
        rst.time = t + 0.001
        pkts += [syn, rst]
    return pkts


def holdout_dns_tunnel(rng) -> list:
    pkts = []
    t = T0_HOLDOUT
    for i in range(20):
        t += rng.uniform(0.8, 1.2)
        label = "".join(rng.choice(BASE32) for _ in range(32))
        pkts += _udp_pair(
            t,
            HO_TUNNEL_CLIENT,
            52000 + i,
            HO_RESOLVER,
            53,
            DNS(rd=1, qd=DNSQR(qname=f"{label}.tun.example.test", qtype="TXT")),
        )
    return pkts


def holdout_syn_flood(rng) -> list:
    pkts = []
    t = T0_HOLDOUT
    ports = rng.sample(range(1024, 65535), 300)
    for port in ports:
        t += 0.01
        syn = _ether(
            IP(src=HO_FLOODER, dst=HO_SERVERS[0])
            / TCP(sport=port, dport=80, flags="S", seq=rng.randrange(2**31), window=2048)
        )
        syn.time = t
        pkts.append(syn)
    return pkts


def holdout_near_miss(rng) -> list:
    pkts = []
    t = T0_HOLDOUT
    for i in range(6):
        sess, _ = _tcp_session(
            t, HO_CLIENT_B, 54000 + i, HO_SERVERS[0], 22, rng, payload_len=rng.randrange(200, 600)
        )
        pkts += sess
        t += 20.0
    name = "assets-prod-eu-west-1-7f3a9c2e1b0d4a6f8e5c.cdn.example.net"
    for i in range(3):
        t += 5.0
        pkts += _udp_pair(
            t,
            HO_CLIENT_B,
            55000 + i,
            HO_RESOLVER,
            53,
            DNS(rd=1, qd=DNSQR(qname=name)),
            DNS(qr=1, rd=1, ra=1, qd=DNSQR(qname=name), an=None),
        )
    return pkts


DESIGN_BUILDERS = {
    "benign": benign,
    "port_scan": port_scan,
    "ssh_brute": ssh_brute,
    "dns_long": dns_long,
}
HOLDOUT_BUILDERS = {
    "holdout_benign": holdout_benign,
    "holdout_ssh_slow": holdout_ssh_slow,
    "holdout_port_scan": holdout_port_scan,
    "holdout_dns_tunnel": holdout_dns_tunnel,
    "holdout_syn_flood": holdout_syn_flood,
    "holdout_near_miss": holdout_near_miss,
}
PROFILES = {"design": DESIGN_BUILDERS, "holdout": HOLDOUT_BUILDERS}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="directory for the .pcap files")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="design",
        help="design (the 2026-11-06 files) or holdout (six new files)",
    )
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, build in PROFILES[args.profile].items():
        # Design seed label as in 2026-11-06, so the design files stay byte-identical.
        label = f"{args.seed}:{name}" if args.profile == "design" else f"holdout:{args.seed}:{name}"
        rng = random.Random(label)
        pkts = build(rng)
        path = out / f"{name}.pcap"
        wrpcap(str(path), pkts)
        print(f"{path} packets={len(pkts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
