"""Raspberry Pi flow monitor: nfstream flows, shared features, ONNX score, evidence, JSON lines.

    python -m rakshak.nids.pi_monitor --iface eth0 --model model.onnx
    python -m rakshak.nids.pi_monitor --pcap capture.pcap --model model.onnx --max-flows 500

Per flow it prints one JSON line with the state (SAFE, UNCERTAIN or ALERT), the attack
probability, the family, the evidence items and the advised action. The explanation is the
deterministic network template, checked by ``rakshak.verifier.verify``. The line says whether the
check passed. There is no automatic blocking: every action is advice.

Model contract (``model.onnx`` metadata, written by kaggle/jobs/nids_baselines.py): the schema
version, the live feature list and the family order. A mismatch stops the monitor before the
first flow is scored. onnxruntime and nfstream are imported only when they are needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rakshak.nids.evidence import (
    STATE_LABEL,
    NetEvidence,
    build_network_package,
    evidence_for_flow,
    net_template,
)
from rakshak.nids.live import LIVE_COLUMNS, LIVE_FEATURES, HostWindow, featurize_flow
from rakshak.nids.schema import FAMILIES, SCHEMA_VERSION, coerce_matrix
from rakshak.verifier import verify

ProbsFn = Callable[[np.ndarray], np.ndarray]  # (1, n_live) float32 -> (1, n_families) probs


@dataclass(frozen=True)
class FlowDecision:
    """The scored state of one flow, before any advisory is built."""

    state: str  # "scam", "uncertain" or "benign" (the detector and window decision)
    p_attack: float
    family: str
    src: str
    dst: Any
    dst_port: int | None
    evidence: list[NetEvidence]


def flow_decision(
    flow: Mapping[str, Any],
    probs_fn: ProbsFn,
    window: HostWindow,
    *,
    alert: float,
    uncertain: float,
) -> FlowDecision:
    """Score one flow and its window evidence. Pure apart from ``window``."""
    if not 0.0 <= uncertain <= alert <= 1.0:
        raise ValueError("need 0 <= uncertain <= alert <= 1")
    x = coerce_matrix(featurize_flow(flow)[LIVE_COLUMNS][None, :], LIVE_FEATURES)
    probs = np.asarray(probs_fn(x), dtype=np.float64).reshape(-1)
    if probs.shape[0] != len(FAMILIES):
        raise ValueError(f"expected {len(FAMILIES)} family probabilities, got {probs.shape[0]}")
    p_attack = float(np.clip(1.0 - probs[0], 0.0, 1.0))
    if p_attack >= alert:
        state = "scam"
    elif p_attack >= uncertain:
        state = "uncertain"
    else:
        state = "benign"
    family = FAMILIES[1 + int(np.argmax(probs[1:]))] if state != "benign" else "benign"

    src = flow.get("src_ip")
    dst_port = flow.get("dst_port")
    ts = flow.get("bidirectional_first_seen_ms")
    stats = window.observe(str(src), None if dst_port is None else int(dst_port),
                           float(ts) if ts is not None else 0.0)
    evidence = evidence_for_flow(family, p_attack,
                                 dst_port=None if dst_port is None else int(dst_port),
                                 window=stats)
    if state == "benign" and evidence:
        # Window evidence can raise the state, never lower it (as the SMS rules do).
        state = "uncertain"
    return FlowDecision(state=state, p_attack=p_attack, family=family, src=str(src),
                        dst=flow.get("dst_ip"),
                        dst_port=None if dst_port is None else int(dst_port),
                        evidence=evidence)


def assess_flow(
    flow: Mapping[str, Any],
    probs_fn: ProbsFn,
    window: HostWindow,
    *,
    alert: float,
    uncertain: float,
) -> dict[str, Any]:
    """Score one flow and return the JSON-ready record. Pure apart from ``window``."""
    dec = flow_decision(flow, probs_fn, window, alert=alert, uncertain=uncertain)
    package = build_network_package(dec.evidence, dec.state)
    template = net_template(package)
    verified = verify(json.dumps(template), package).ok
    return {
        "state": STATE_LABEL[dec.state],
        "p_attack": round(dec.p_attack, 4),
        "family": dec.family,
        "src": dec.src,
        "dst": dec.dst,
        "dst_port": dec.dst_port,
        "risk": package.risk.value,
        "evidence": [{"id": e.id, "type": e.type, "weight": round(e.weight, 3)}
                     for e in dec.evidence],
        "action": template["action"],
        "summary": template["summary"],
        "verified": verified,
    }


def run_monitor(
    flows: Iterable[Mapping[str, Any]],
    probs_fn: ProbsFn,
    *,
    alert: float,
    uncertain: float,
    window_ms: int = 10_000,
    max_flows: int | None = None,
) -> Iterator[dict[str, Any]]:
    window = HostWindow(window_ms=window_ms)
    for n, flow in enumerate(flows):
        if max_flows is not None and n >= max_flows:
            return
        yield assess_flow(flow, probs_fn, window, alert=alert, uncertain=uncertain)


class OnnxScorer:
    """Single-thread onnxruntime session over the student model. Checks the model contract."""

    def __init__(self, path: str) -> None:
        import onnxruntime as ort  # lazy: only on the device

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        if meta.get("rakshak.schema") != SCHEMA_VERSION:
            raise ValueError(f"model schema {meta.get('rakshak.schema')!r} != {SCHEMA_VERSION!r}")
        if json.loads(meta.get("rakshak.features", "[]")) != list(LIVE_FEATURES):
            raise ValueError("model features differ from the live feature list")
        if json.loads(meta.get("rakshak.families", "[]")) != list(FAMILIES):
            raise ValueError("model family order differs from schema.FAMILIES")
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rakshak.nids.pi_monitor", description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--iface", help="live capture interface, for example eth0")
    src.add_argument("--pcap", help="offline pcap file (testing)")
    p.add_argument("--model", required=True, help="student ONNX file")
    p.add_argument("--alert-threshold", type=float, default=0.5,
                   help="attack probability at or above which the state is ALERT")
    p.add_argument("--uncertain-threshold", type=float, default=0.2,
                   help="attack probability at or above which the state is UNCERTAIN")
    p.add_argument("--window-ms", type=int, default=10_000, help="host-window length")
    p.add_argument("--idle-timeout", type=int, default=120, help="nfstream idle timeout (s)")
    p.add_argument("--active-timeout", type=int, default=1800, help="nfstream active timeout (s)")
    p.add_argument("--max-flows", type=int, default=None, help="stop after this many flows")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from rakshak.nids.live import iter_nfstream  # lazy: needs nfstream

    scorer = OnnxScorer(args.model)
    flows = iter_nfstream(args.pcap or args.iface, idle_timeout=args.idle_timeout,
                          active_timeout=args.active_timeout)
    try:
        for record in run_monitor(flows, scorer, alert=args.alert_threshold,
                                  uncertain=args.uncertain_threshold,
                                  window_ms=args.window_ms, max_flows=args.max_flows):
            print(json.dumps(record, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
