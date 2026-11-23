"""The edge network path: flows in, per-host alerts out (``POST /analyze_flows``).

Input is either a classic pcap (raw body, starts with a pcap magic) or JSON ``{"flows": [...]}``
of nfstream-style flow dicts (``rakshak.nids.live.NFSTREAM_KEYS``). A pcap is turned into flows by
nfstream when it imports on this machine (``flow_source`` "auto" or "nfstream"), else by the
pure-Python reader ``rakshak.nids.pcap_flows`` ("stdlib", the fallback). Each flow is scored by the
INT8 student through ``pi_monitor.flow_decision`` (with a per-request ``HostWindow``), then
``host_alerts.host_alerts`` gives one advisory per source host. Remediation comes from
``rakshak.remediation``. No flow is sent anywhere; nothing here captures traffic.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rakshak.nids import host_rules
from rakshak.nids.host_alerts import NET_MODEL_MODES, host_alerts, watchlist
from rakshak.nids.live import NFSTREAM_KEYS, HostWindow
from rakshak.nids.pcap_flows import flows_from_pcap_bytes
from rakshak.nids.pi_monitor import flow_decision
from rakshak.nids.schema import SCHEMA_VERSION
from rakshak.remediation import remediation_for, remediation_problems

MAX_BODY_BYTES = 20 * 1024 * 1024  # request body cap (pcap or JSON)
MAX_FLOWS = 50_000
FLOW_SOURCES = ("auto", "nfstream", "stdlib")
PCAP_MAGICS = (b"\xa1\xb2\xc3\xd4", b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\x3c\x4d", b"\x4d\x3c\xb2\xa1")
NET_MODEL_ID = "nids_cse_cic_ids2018_int8"
# Calibrated operating points of the shipped student, from kaggle/out/nids_baselines/output/
# results.json (protocols "student" / "results" / "student cse_cic_ids2018"): the max-F1 threshold
# on the calibration split is the alert threshold, and the threshold for 1% FPR on calibration is
# the uncertain threshold. Scores are 1 - P(benign), as in the Kaggle evaluation.
NET_ALERT_THRESHOLD = 0.0015007257461547852
NET_UNCERTAIN_THRESHOLD = 0.0005502104759216309
NET_MODEL_MODE = "quiet"  # rules first: a model-only flag is advice to verify, never an alert


class FlowRequestError(ValueError):
    """The request body is not usable (maps to HTTP 400)."""


def is_pcap(raw: bytes) -> bool:
    return raw[:4] in PCAP_MAGICS


def _nfstream_flows(raw: bytes, scratch: Path) -> list[dict[str, Any]]:
    """Flows from nfstream on a temporary copy of the pcap. The copy is removed afterwards."""
    from rakshak.nids.live import iter_nfstream

    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="upload-", suffix=".pcap", dir=str(scratch))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        return [{k: f.get(k) for k in NFSTREAM_KEYS} for f in iter_nfstream(name)]
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def flows_from_body(
    raw: bytes, flow_source: str, scratch: Path
) -> tuple[list[dict[str, Any]], str]:
    """(flows, source label). Raises FlowRequestError on a body that is neither pcap nor JSON."""
    if flow_source not in FLOW_SOURCES:
        raise FlowRequestError(f"flow_source must be one of {FLOW_SOURCES}")
    if is_pcap(raw):
        if flow_source in ("auto", "nfstream"):
            try:
                return _nfstream_flows(raw, scratch), "pcap_nfstream"
            except ImportError:
                if flow_source == "nfstream":
                    raise FlowRequestError("nfstream is not installed on this device") from None
        try:
            return flows_from_pcap_bytes(raw), "pcap_stdlib"
        except ValueError as exc:
            raise FlowRequestError(str(exc)) from exc
    try:
        doc = json.loads(raw.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        raise FlowRequestError('body must be a pcap file or JSON {"flows": [...]}') from None
    flows = doc.get("flows") if isinstance(doc, dict) else None
    if not isinstance(flows, list) or not flows:
        raise FlowRequestError("field 'flows' must be a non-empty list of flow objects")
    if len(flows) > MAX_FLOWS:
        raise FlowRequestError(f"more than {MAX_FLOWS} flows in one request")
    for i, f in enumerate(flows):
        if not isinstance(f, dict) or not isinstance(f.get("src_ip"), str):
            raise FlowRequestError(f"flow {i} must be an object with a string 'src_ip'")
        ts = f.get("bidirectional_first_seen_ms")
        if ts is not None:
            if isinstance(ts, bool):
                raise FlowRequestError(
                    f"flow {i} field 'bidirectional_first_seen_ms' must be numeric"
                )
            try:
                float(ts)
            except (TypeError, ValueError):
                raise FlowRequestError(
                    f"flow {i} field 'bidirectional_first_seen_ms' must be numeric"
                ) from None
        dport = f.get("dst_port")
        if dport is not None:
            if isinstance(dport, bool):
                raise FlowRequestError(f"flow {i} field 'dst_port' must be an integer")
            try:
                int(dport)
            except (TypeError, ValueError):
                raise FlowRequestError(f"flow {i} field 'dst_port' must be an integer") from None
    return flows, "json"


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[idx], 4)


class NidsAnalyzer:
    """Scores flows and builds the per-host response. The scorer is injected (tests use a fake)."""

    def __init__(
        self,
        probs_fn: Callable[..., Any],
        *,
        alert: float = NET_ALERT_THRESHOLD,
        uncertain: float = NET_UNCERTAIN_THRESHOLD,
        window_ms: int = 10_000,
        model_id: str = NET_MODEL_ID,
        scratch: Path,
        flow_source: str = "auto",
        model_mode: str = NET_MODEL_MODE,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if flow_source not in FLOW_SOURCES:
            raise ValueError(f"flow_source must be one of {FLOW_SOURCES}")
        if model_mode not in NET_MODEL_MODES:
            raise ValueError(f"model_mode must be one of {NET_MODEL_MODES}")
        self.model_mode = model_mode
        self.probs_fn = probs_fn
        self.alert = float(alert)
        self.uncertain = float(uncertain)
        self.window_ms = window_ms
        self.model_id = model_id
        self.scratch = scratch
        self.flow_source = flow_source
        self.clock = clock

    def analyse_body(self, raw: bytes) -> dict[str, Any]:
        t0 = self.clock()
        flows, source = flows_from_body(raw, self.flow_source, self.scratch)
        t_parse = self.clock()
        return self.analyse_flows(flows, source=source, parse_ms=(t_parse - t0) * 1000.0)

    def analyse_flows(
        self, flows: list[dict[str, Any]], *, source: str = "json", parse_ms: float = 0.0
    ) -> dict[str, Any]:
        clock = self.clock
        window = HostWindow(window_ms=self.window_ms)
        ordered = sorted(flows, key=lambda f: float(f.get("bidirectional_first_seen_ms") or 0.0))
        decisions = []
        per_flow: list[float] = []
        t_score0 = clock()
        for flow in ordered:
            t = clock()
            decisions.append(
                flow_decision(
                    flow, self.probs_fn, window, alert=self.alert, uncertain=self.uncertain
                )
            )
            per_flow.append((clock() - t) * 1000.0)
        t_score1 = clock()
        hits = host_rules.evaluate(ordered)
        alerts = host_alerts(decisions, hits, mode=self.model_mode)
        # The watchlist is the alert-mode list of uncertain-only hosts. Quiet mode reports those
        # hosts as model_only advisories instead, so the list is empty there.
        watch = watchlist(decisions) if self.model_mode == "alert" else []
        t_alerts = clock()

        out_alerts = []
        for a in alerts:
            d = a.to_dict()
            rem = remediation_for(a.package, a.template["action"], "en")
            d["remediation"] = rem
            d["remediation_ok"] = not remediation_problems(rem, a.package)
            out_alerts.append(d)
        t_end = clock()
        n = len(decisions)
        score_s = max(t_score1 - t_score0, 1e-9)
        # Rule and alert hosts sort first (model_only advisories last), so alerts[0] is the worst.
        worst = out_alerts[0] if out_alerts else None
        rule_alerts = [x for x in out_alerts if x["advisory"]["path"] != "model_only_quiet"]
        model_flagged = [d for d in decisions if d.family != "benign"]
        return {
            "flows_processed": n,
            "flow_source": source,
            "flows_per_s": round(n / score_s, 2) if n else 0.0,
            "model": {
                "id": self.model_id,
                "schema": SCHEMA_VERSION,
                "mode": self.model_mode,
                "alert_threshold": round(self.alert, 4),
                "uncertain_threshold": round(self.uncertain, 4),
                "features": 19,
            },
            "decision": {
                "risk": worst["decision"]["risk"] if worst else "LOW",
                "state": worst["decision"]["state"] if worst else "SAFE",
                "family": worst["decision"]["family"] if worst else "benign",
                "p_attack": worst["decision"]["p_attack"] if worst else 0.0,
                "alerts": len(rule_alerts),
                "model_only_hosts": len(out_alerts) - len(rule_alerts),
                "rule_hits": len(hits),
                "hosts_seen": len({f.get("src_ip") for f in ordered}),
                "flagged_flows": sum(x["decision"]["flagged_flows"] for x in out_alerts),
                "model_flagged_flows": len(model_flagged),
                "model_p_attack_max": round(max((d.p_attack for d in decisions), default=0.0), 4),
            },
            "alerts": out_alerts,
            "watchlist": watch,
            "timings_ms": {
                "parse": round(parse_ms, 3),
                "score": round((t_score1 - t_score0) * 1000.0, 3),
                "per_flow_p50": _pct(per_flow, 0.5),
                "per_flow_p95": _pct(per_flow, 0.95),
                "per_flow_mean": round(sum(per_flow) / n, 4) if n else None,
                "host_alerts": round((t_alerts - t_score1) * 1000.0, 3),
                "remediation_and_verify": round((clock() - t_alerts) * 1000.0, 3),
                "total": round((t_end - t_score0) * 1000.0 + parse_ms, 3),
            },
        }
