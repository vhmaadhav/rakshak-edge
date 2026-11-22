"""Review fixes: the relay finds a single-worker socket beside stale per-worker sockets, and the
verdict-first worker configuration carries the explainer candidate and cap. No model is loaded."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import socket_bridge  # noqa: E402


def test_upstream_paths_falls_back_to_the_plain_socket(tmp_path):
    base = str(tmp_path / "edge.sock")
    assert socket_bridge.upstream_paths(base) == [base]


def test_upstream_paths_lists_worker_sockets_in_numeric_order(tmp_path):
    base = tmp_path / "edge.sock"
    for n in (0, 1, 10, 2):
        (tmp_path / f"edge.sock.{n}").write_text("")
    got = socket_bridge.upstream_paths(str(base))
    assert [Path(x).name for x in got] == [
        "edge.sock.0",
        "edge.sock.1",
        "edge.sock.2",
        "edge.sock.10",
    ]


def test_stale_worker_socket_does_not_hide_the_single_worker_socket(tmp_path):
    (tmp_path / "edge.sock.0").write_text("")  # left by a killed multi-worker service
    (tmp_path / "edge.sock").write_text("")  # the current single-worker service
    got = [Path(x).name for x in socket_bridge.upstream_paths(str(tmp_path / "edge.sock"))]
    assert "edge.sock" in got


def test_cfg_from_args_carries_the_explainer_candidate_and_cap():
    import edge_workers

    class Args:
        explainer_candidate = "describe_kw"
        explainer_cap = 48
        llm_cpus = None

        def __getattr__(self, name):  # every other option is irrelevant here
            return None

    cfg = edge_workers.cfg_from_args(Args(), Path("/tmp/root"))
    assert cfg["explainer_candidate"] == "describe_kw"
    assert cfg["explainer_cap"] == 48
