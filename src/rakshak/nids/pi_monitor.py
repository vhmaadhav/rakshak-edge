"""Raspberry Pi flow monitor: nfstream flows, shared features, ONNX score, evidence, JSON lines.

Alert-only, live (docs/NIDS_LIVE.md):

    python -m rakshak.nids.pi_monitor --iface eth0 --model model.onnx --summary-out summary.json
    python -m rakshak.nids.pi_monitor --pcap capture.pcap --model model.onnx --summary-out s.json
    python -m rakshak.nids.pi_monitor --flows-json flows.json --model model.onnx

Flows are grouped into time windows (``--window-s``). Per-host state is kept across windows and
bounded (``--max-hosts``, ``--max-history``). Each window emits one JSON line per host alert (rule
evidence only). Nothing is blocked, and no payload bytes or raw addresses are kept. The summary
(``--summary-out``) holds the field metrics. nfstream is needed only for ``--iface`` (and for
``--pcap`` when it is installed); ``--pcap`` falls back to the pure-Python reader otherwise.

``--per-flow`` keeps the older output: one JSON line per flow with its state, evidence and
verifier result, without host state across windows.

Model contract (``model.onnx`` metadata, written by kaggle/jobs/nids_baselines.py): the schema
version, the live feature list and the family order. A mismatch stops the monitor before the
first flow is scored. onnxruntime is imported only when it is needed. nfstream is never imported
unless an interface or a pcap (with nfstream installed) is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rakshak.nids.evidence import (
    STATE_LABEL,
    NetEvidence,
    build_network_package,
    evidence_for_flow,
    net_template,
)
from rakshak.nids.live import (
    LIVE_COLUMNS,
    LIVE_FEATURES,
    HostWindow,
    featurize_flow,
    iter_nfstream,
    nfstream_available,
)
from rakshak.nids.pcap_flows import flows_from_pcap_bytes
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

        from rakshak.ort_offline import offline_session_options

        opts = offline_session_options(ort)
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
    src.add_argument("--iface", help="live capture interface, for example eth0 (needs nfstream)")
    src.add_argument("--pcap", help="offline pcap file")
    src.add_argument("--flows-json",
                     help='flow JSON file: {"flows": [...]} or a list of nfstream-style flows')
    p.add_argument("--model", required=True, help="student ONNX file")
    p.add_argument("--alert-threshold", type=float, default=0.5,
                   help="attack probability at or above which the state is ALERT")
    p.add_argument("--uncertain-threshold", type=float, default=0.2,
                   help="attack probability at or above which the state is UNCERTAIN")
    p.add_argument("--window-ms", type=int, default=10_000,
                   help="sliding window for the per-flow port-scan evidence (ms)")
    p.add_argument("--idle-timeout", type=int, default=120, help="nfstream idle timeout (s)")
    p.add_argument("--active-timeout", type=int, default=1800, help="nfstream active timeout (s)")
    p.add_argument("--max-flows", type=int, default=None, help="stop after this many flows")

    live = p.add_argument_group("live session (bounded, alert-only)")
    live.add_argument("--mode", default="alert-only", choices=("alert-only",),
                      help="alert-only is the only mode: nothing is blocked or stored raw")
    live.add_argument("--window-s", type=float, default=60.0,
                      help="processing window by flow start time (s)")
    live.add_argument("--grace-s", type=float, default=10.0,
                      help="wall-clock grace after a window's --window-s before it is closed (s)")
    live.add_argument("--max-hosts", type=int, default=512,
                      help="tracked source hosts; the least recently seen is evicted first")
    live.add_argument("--max-history", type=int, default=256,
                      help="rule-relevant flows kept per host (oldest evicted first, min 200)")
    live.add_argument("--max-flows-window", type=int, default=50_000,
                      help="flows kept per window; the rest are dropped and counted")
    live.add_argument("--idle-evict-s", type=float, default=900.0,
                      help="forget a host unseen for this long (s)")
    live.add_argument("--alert-cooldown-s", type=float, default=300.0,
                      help="do not repeat the same host alert within this time (s)")
    live.add_argument("--include-model-only", action="store_true",
                      help="also print model-only advisories (they are not alerts)")
    live.add_argument("--alias-salt", default=None,
                      help="salt for host aliases; the same salt gives the same aliases "
                           "(default: random for each run)")
    live.add_argument("--summary-out", default=None, metavar="PATH.json",
                      help="write the field-metrics summary here (at exit and periodically)")
    live.add_argument("--summary-every-s", type=float, default=0.0,
                      help="also rewrite --summary-out every N seconds (0: only at exit)")
    live.add_argument("--per-flow", action="store_true",
                      help="older output: one line per flow, no state across windows")
    return p


def _open_flows(args: argparse.Namespace) -> tuple[Iterable[Mapping[str, Any]], str]:
    """(flows, source label). Raises OSError or ValueError with a message for the user."""
    from rakshak.nids.live_session import load_flows_json

    if args.flows_json:
        return load_flows_json(args.flows_json), "flows_json"
    if args.pcap:
        if nfstream_available():
            return iter_nfstream(args.pcap, idle_timeout=args.idle_timeout,
                                 active_timeout=args.active_timeout), "pcap_nfstream"
        return flows_from_pcap_bytes(Path(args.pcap).read_bytes()), "pcap_stdlib"
    return iter_nfstream(args.iface, idle_timeout=args.idle_timeout,
                         active_timeout=args.active_timeout), "interface"


def main(argv: list[str] | None = None) -> int:
    from rakshak.nids.live_session import (
        HostAliaser,
        LiveConfig,
        LiveSession,
        drive,
        write_summary,
    )

    args = build_parser().parse_args(argv)
    if args.iface is not None and not nfstream_available():
        print("error: --iface needs the optional nfstream package, which is not installed. "
              "Install nfstream on the device, or use --pcap or --flows-json for offline input.",
              file=sys.stderr)
        return 2
    try:
        flows, label = _open_flows(args)
        config = LiveConfig(
            window_s=args.window_s,
            grace_s=args.grace_s,
            max_hosts=args.max_hosts,
            max_history_per_host=args.max_history,
            max_flows_per_window=args.max_flows_window,
            idle_evict_s=args.idle_evict_s,
            alert_cooldown_s=args.alert_cooldown_s,
            include_model_only=args.include_model_only,
            summary_every_s=args.summary_every_s,
            mode=args.mode,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    scorer = OnnxScorer(args.model)
    if args.per_flow:
        try:
            for record in run_monitor(flows, scorer, alert=args.alert_threshold,
                                      uncertain=args.uncertain_threshold,
                                      window_ms=args.window_ms, max_flows=args.max_flows):
                print(json.dumps(record, sort_keys=True), flush=True)
        except KeyboardInterrupt:
            return 0
        return 0

    def on_alert(record: dict[str, Any]) -> None:
        print(json.dumps(record, sort_keys=True), flush=True)

    sink = (lambda s: write_summary(args.summary_out, s)) if args.summary_out else None
    session = LiveSession(
        scorer,
        alert=args.alert_threshold,
        uncertain=args.uncertain_threshold,
        window_ms=args.window_ms,
        config=config,
        aliaser=HostAliaser(args.alias_salt),
        on_alert=on_alert,
        summary_sink=sink,
        source=label,
    )
    try:
        drive(session, itertools.islice(flows, args.max_flows))
    except KeyboardInterrupt:
        pass
    finally:
        summary = session.close()
    if sink is None:
        print(json.dumps(summary, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
