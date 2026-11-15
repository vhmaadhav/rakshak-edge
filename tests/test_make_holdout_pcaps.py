"""Holdout generator (deploy/pi/service/make_holdout_pcaps.py): files, manifest, RFC 1918, shape.

Checks the construction only (file set, packet and flow counts that the manifest states, private
addresses, determinism). It does not score anything with the rules."""

from __future__ import annotations

import ipaddress
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (SERVICE, REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("scapy")

from make_holdout_pcaps import CASES, write_holdout  # noqa: E402

from rakshak.nids.pcap_flows import flows_from_pcap_bytes  # noqa: E402


@pytest.fixture(scope="module")
def holdout(tmp_path_factory):
    out = tmp_path_factory.mktemp("holdout")
    write_holdout(out)
    return out


def _flows(path: Path):
    return flows_from_pcap_bytes(path.read_bytes())


def test_manifest_matches_files(holdout):
    doc = json.loads((holdout / "manifest.json").read_text(encoding="utf-8"))
    names = {row["file"] for row in doc["cases"]}
    assert names == {f"{c.name}.pcap" for c in CASES}
    assert len(names) == len(CASES) == 26
    for row in doc["cases"]:
        assert (holdout / row["file"]).stat().st_size > 24
        assert row["packets"] > 0


def test_every_address_is_rfc1918(holdout):
    for path in holdout.glob("*.pcap"):
        for flow in _flows(path):
            for ip in (flow["src_ip"], flow["dst_ip"]):
                assert ipaddress.ip_address(ip).is_private, (path.name, ip)


def test_burst_counts_match_the_design_of_the_files(holdout):
    """The number of short SSH flows to port 22 from the attacker is the attempt count the
    manifest states (9, 10 and 40): a check on the generator, not on the rules."""
    for name, attempts in (
        ("ho_ssh_brute_9", 9),
        ("ho_ssh_brute_10_5s", 10),
        ("ho_ssh_brute_fast", 40),
    ):
        flows = [f for f in _flows(holdout / f"{name}.pcap") if f["dst_port"] == 22]
        assert len(flows) == attempts, name
        assert {f["src_ip"] for f in flows} == {"10.20.30.40"}


def test_generation_is_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    write_holdout(a)
    write_holdout(b)
    for path in sorted(a.glob("*.pcap")):
        fa = [
            (f["src_ip"], f["dst_ip"], f["dst_port"], f["bidirectional_first_seen_ms"])
            for f in _flows(path)
        ]
        fb = [
            (f["src_ip"], f["dst_ip"], f["dst_port"], f["bidirectional_first_seen_ms"])
            for f in _flows(b / path.name)
        ]
        assert fa == fb, path.name
