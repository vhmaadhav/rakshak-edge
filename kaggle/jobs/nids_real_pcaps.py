"""Real labelled pcaps through the shipped rules-first NIDS path (docs/NIDS_REAL_PLAN.md).

Pre-registered 2026-11-12. Rules, thresholds, quiet mode and the INT8 student are the service's
(deploy/pi/service/nids_pipeline.py, src/rakshak/nids/host_rules.py, host_alerts.py). Nothing here
retunes them. Each capture is one request, read as a stream from /kaggle/input (read-only) with
nfstream, scored in order friday -> tuesday -> dns malicious -> dns benign under a
wall-clock budget.

Outputs in the kernel working directory: results.json, metrics.json, alerts.parquet, report.md.
They are rewritten atomically after every capture, so a stopped run keeps its last complete write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (REPO / "src", REPO / "deploy" / "pi" / "service"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import nids_pipeline as P  # noqa: E402  (the service's thresholds and flow reader)

from rakshak.nids import host_rules  # noqa: E402
from rakshak.nids.host_alerts import host_alerts  # noqa: E402
from rakshak.nids.live import NFSTREAM_KEYS, HostWindow  # noqa: E402
from rakshak.nids.pi_monitor import OnnxScorer, flow_decision  # noqa: E402

# --- pre-registered constants (docs/NIDS_REAL_PLAN.md). Change only by a new plan. ---
MODEL_NAME = "student_cse_cic_ids2018_int8.onnx"
MODEL_SHA256_PREFIX = "d3dfdcb16dcaa278"
NFSTREAM_PIN = "nfstream==6.6.0"
PRIMARY_OFFSET_H = -3.0  # UTC-3 (ADT): assumption, the page gives no timezone
SENSITIVITY_OFFSETS_H = (-4.0, -5.0, 0.0)
HOUR_MS = 3_600_000
PROGRESS_EVERY = 5_000
DEFAULT_MAX_HOURS = 10.0
KALI_IPS = ("205.174.165.73", "172.16.0.1")
DDOS_LOIT_IPS = ("205.174.165.69", "205.174.165.70", "205.174.165.71")
VICTIM_IPS = ("192.168.10.50", "205.174.165.68")

# (unit id, capture key, day, family, label, start, end, attacker IPs)
SCHEDULE = (
    (
        "F1",
        "friday",
        "2017-07-07",
        "port_scan",
        "PortScan, firewall on",
        "13:55",
        "14:43",
        KALI_IPS,
    ),
    (
        "F2",
        "friday",
        "2017-07-07",
        "port_scan",
        "PortScan, firewall off",
        "14:51",
        "15:29",
        KALI_IPS,
    ),
    ("F3", "friday", "2017-07-07", "dos_flood", "DDoS LOIT", "15:56", "16:16", DDOS_LOIT_IPS),
    ("T1", "tuesday", "2017-07-04", "brute_force_other", "FTP-Patator", "09:20", "10:20", KALI_IPS),
    ("T2", "tuesday", "2017-07-04", "brute_force_ssh", "SSH-Patator", "14:00", "15:00", KALI_IPS),
)
CIC_PATTERNS = {"friday": "Friday*.pcap", "tuesday": "Tuesday*.pcap"}
DNS_ROOT_PART = "01_original_pcap_files"
PARQUET_COLUMNS = (
    "capture",
    "kind",
    "host",
    "family",
    "first_ts_ms",
    "last_ts_ms",
    "first_utc",
    "last_utc",
    "state",
    "risk",
    "path",
)


class BudgetExceeded(RuntimeError):
    """The wall-clock budget ran out. The capture in progress is not scored."""


@dataclass
class Unit:
    id: str
    capture: str
    family: str
    label: str
    srcs: frozenset
    start_ms: float
    end_ms: float


@dataclass
class Capture:
    key: str
    kind: str  # "cic" or "dns_mal" or "dns_ben"
    path: Path | None
    note: str = ""


@dataclass
class CaptureScore:
    """Everything the metrics need from one scored capture. Schedule-free, so offsets are cheap."""

    key: str
    kind: str
    path: str
    flows: int
    observed_hours: set
    records: list  # rule records: dict(src, type, first_ms, last_ms, weight)
    flagged: list  # (src, ts_ms) for model-flagged flows (family != benign)
    model_only_srcs: set
    alert_srcs: set  # hosts with a non-model-only alert
    alerts: list  # alert dicts for the parquet and report
    dns_src: str | None = None
    dns_span: tuple | None = None
    read_s: float = 0.0
    score_s: float = 0.0
    top_talkers: dict = field(default_factory=dict)


# ---------------------------------------------------------------- time and units


def epoch_ms(day: str, hhmm: str, offset_h: float) -> float:
    """A local wall-clock time at a fixed UTC offset, as epoch milliseconds."""
    tz = timezone(timedelta(hours=offset_h))
    when = datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    return when.timestamp() * 1000.0


def hour_of(ts_ms: float) -> int:
    return int(ts_ms // HOUR_MS)


def cic_units(offset_h: float) -> list[Unit]:
    out = []
    for uid, cap, day, fam, label, start, end, ips in SCHEDULE:
        out.append(
            Unit(
                uid,
                cap,
                fam,
                label,
                frozenset(ips),
                epoch_ms(day, start, offset_h),
                epoch_ms(day, end, offset_h),
            )
        )
    return out


def dns_source(flows: list[dict]) -> tuple[str | None, tuple | None]:
    """The tunnel host: most flow packets to UDP/53 (ties to the smaller IP), before scoring."""
    pk: Counter = Counter()
    span: dict[str, list[float]] = {}
    for f in flows:
        if int(_num(f.get("protocol"), -1)) != 17 or int(_num(f.get("dst_port"), -1)) != 53:
            continue
        src = str(f.get("src_ip"))
        pk[src] += int(_num(f.get("src2dst_packets"), 0))
        first = _num(f.get("bidirectional_first_seen_ms"), None)
        last = first if first is None else first + _num(f.get("bidirectional_duration_ms"), 0)
        if first is not None:
            lo_hi = span.setdefault(src, [first, last])
            lo_hi[0] = min(lo_hi[0], first)
            lo_hi[1] = max(lo_hi[1], last)
    if not pk:
        return None, None
    best = sorted(pk.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    lo, hi = span[best]
    return best, (lo, hi)


def dns_units(captures: list[Capture], scored: dict[str, CaptureScore]) -> list[Unit]:
    """One unit per malicious capture, in discovery order. Source and window come from the
    scored capture (the tunnel host is fixed before scoring). Unscored captures get no source."""
    out = []
    for i, cap in enumerate(c for c in captures if c.kind == "dns_mal"):
        cs = scored.get(cap.key)
        src = cs.dns_src if cs else None
        span = cs.dns_span if cs else None
        if src is None or span is None:
            out.append(
                Unit(
                    f"D{i + 1}",
                    cap.key,
                    "dns_tunnel",
                    f"{cap.key} (no DNS source)",
                    frozenset(),
                    0.0,
                    0.0,
                )
            )
            continue
        out.append(
            Unit(
                f"D{i + 1}",
                cap.key,
                "dns_tunnel",
                f"{cap.key} DNS tunnel",
                frozenset({src}),
                span[0],
                span[1],
            )
        )
    return out


def _num(value: Any, default: Any = 0.0):
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and abs(out) != float("inf") else default


# ---------------------------------------------------------------- discovery


def discover(root: Path) -> list[Capture]:
    """Ordered captures: friday, tuesday, DNS malicious (sorted), DNS benign (sorted).

    A CIC day with no single match is recorded as missing. Nothing is guessed."""
    caps: list[Capture] = []
    for key in ("friday", "tuesday"):
        hits = sorted(root.rglob(CIC_PATTERNS[key])) if root.is_dir() else []
        if len(hits) == 1:
            caps.append(Capture(key, "cic", hits[0]))
        else:
            caps.append(Capture(key, "cic", None, f"missing: {len(hits)} matches"))
    dns = sorted(root.rglob("*.pcap")) if root.is_dir() else []
    mal = [p for p in dns if DNS_ROOT_PART in p.parts and "malicious" in p.parts]
    ben = [p for p in dns if DNS_ROOT_PART in p.parts and "benign" in p.parts]
    for p in mal:
        caps.append(Capture(f"dns-mal/{p.parent.name}/{p.name}", "dns_mal", p))
    for p in ben:
        caps.append(Capture(f"dns-ben/{p.parent.name}/{p.name}", "dns_ben", p))
    return caps


# ---------------------------------------------------------------- model and flows


def find_model(root: Path) -> Path:
    hits = sorted(root.rglob(MODEL_NAME)) if root.is_dir() else []
    if len(hits) != 1:
        raise RuntimeError(f"expected one {MODEL_NAME} under {root}, found {len(hits)}")
    digest = hashlib.sha256(hits[0].read_bytes()).hexdigest()
    if not digest.startswith(MODEL_SHA256_PREFIX):
        raise RuntimeError(f"model sha256 {digest[:16]} is not the pinned {MODEL_SHA256_PREFIX}")
    return hits[0]


def ensure_nfstream(install: bool, log) -> str | None:
    """None when nfstream imports; else a reason. Installs the pin only when allowed."""
    try:
        import nfstream  # noqa: F401

        return None
    except ImportError:
        pass
    if not install:
        return "nfstream not importable and install disabled"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            NFSTREAM_PIN,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "pip install of " + NFSTREAM_PIN + " failed: " + proc.stderr.strip()[-300:]
    log(f"installed {NFSTREAM_PIN}")
    return None


def read_flows(path: Path, flow_source: str, check) -> list[dict]:
    """Flows for one capture. nfstream streams the file from /kaggle/input; stdlib reads it whole
    (only for small test pcaps, enforced below)."""
    flows: list[dict] = []
    if flow_source == "nfstream":
        from rakshak.nids.live import iter_nfstream

        for i, f in enumerate(iter_nfstream(str(path))):
            flows.append({k: f.get(k) for k in NFSTREAM_KEYS})
            if i % PROGRESS_EVERY == 0:
                check()
    elif flow_source == "stdlib":
        if path.stat().st_size > 50 * 1024 * 1024:
            raise RuntimeError("stdlib reader is for small test pcaps only")
        from rakshak.nids.pcap_flows import flows_from_pcap_bytes

        flows = flows_from_pcap_bytes(path.read_bytes())
        check()
    else:
        raise ValueError(f"flow_source {flow_source!r}")
    return flows


def score_flows(ordered: list[dict], scorer, check) -> tuple[list, list, list]:
    """Per-flow decisions, rule hits and host alerts: the calls NidsAnalyzer.analyse_flows makes,
    kept here so the decisions and the window flags are visible to the metrics."""
    window = HostWindow(window_ms=10_000)
    decisions = []
    for i, flow in enumerate(ordered):
        decisions.append(
            flow_decision(
                flow,
                scorer,
                window,
                alert=P.NET_ALERT_THRESHOLD,
                uncertain=P.NET_UNCERTAIN_THRESHOLD,
            )
        )
        if i % PROGRESS_EVERY == 0:
            check()
    hits = host_rules.evaluate(ordered)
    alerts = host_alerts(decisions, hits, mode=P.NET_MODEL_MODE)
    return decisions, hits, alerts


def rule_records(ordered: list[dict], decisions: list, hits: list) -> list[dict]:
    """Per (source, rule type): first and last timestamp of the flows the rule hit or the window
    flagged. Same evidence the service uses; no new rule."""
    idx_by_key: dict[tuple, set] = {}
    weight: dict[tuple, float] = {}
    for h in hits:
        key = (h.src, h.type)
        idx_by_key.setdefault(key, set()).update(h.flow_index)
        weight[key] = max(weight.get(key, 0.0), h.weight)
    for i, d in enumerate(decisions):
        for e in d.evidence:
            if e.source == "window":
                key = (d.src, e.type)
                idx_by_key.setdefault(key, set()).add(i)
                weight[key] = max(weight.get(key, 0.0), e.weight)
    out = []
    for (src, typ), idx in sorted(idx_by_key.items()):
        ts = [_num(ordered[i].get("bidirectional_first_seen_ms"), None) for i in idx]
        ts = [t for t in ts if t is not None]
        if not ts:
            continue
        out.append(
            {
                "src": src,
                "type": typ,
                "first_ms": min(ts),
                "last_ms": max(ts),
                "weight": weight.get((src, typ), 0.0),
            }
        )
    return out


def score_capture(cap: Capture, scorer, flow_source: str, check, log) -> CaptureScore:
    t0 = time.monotonic()
    flows = read_flows(cap.path, flow_source, check)
    t1 = time.monotonic()
    ordered = sorted(flows, key=lambda f: float(_num(f.get("bidirectional_first_seen_ms"), 0.0)))
    decisions, hits, alerts = score_flows(ordered, scorer, check)
    t2 = time.monotonic()
    recs = rule_records(ordered, decisions, hits)
    flagged = [
        (d.src, _num(f.get("bidirectional_first_seen_ms"), 0.0))
        for f, d in zip(ordered, decisions, strict=False)
        if d.family != "benign"
    ]
    observed = {
        (str(f.get("src_ip")), hour_of(_num(f.get("bidirectional_first_seen_ms"), 0.0)))
        for f in ordered
    }
    model_only = {a.src for a in alerts if a.is_model_only}
    alert_srcs = {a.src for a in alerts if not a.is_model_only}
    dns_src, dns_span = dns_source(ordered) if cap.kind == "dns_mal" else (None, None)
    rec_srcs = {r["src"] for r in recs}
    if rec_srcs != alert_srcs:
        log(f"consistency: record hosts {sorted(rec_srcs)} != alert hosts {sorted(alert_srcs)}")
    talkers = Counter(str(f.get("src_ip")) for f in ordered)
    alert_rows = []
    for a in alerts:
        d = a.to_dict()
        alert_rows.append(
            {
                "src": a.src,
                "state": d["decision"]["state"],
                "risk": d["decision"]["risk"],
                "path": d["advisory"]["path"],
                "family": d["decision"]["family"],
            }
        )
    return CaptureScore(
        key=cap.key,
        kind=cap.kind,
        path=str(cap.path),
        flows=len(ordered),
        observed_hours=observed,
        records=recs,
        flagged=flagged,
        model_only_srcs=model_only,
        alert_srcs=alert_srcs,
        alerts=alert_rows,
        dns_src=dns_src,
        dns_span=dns_span,
        read_s=round(t1 - t0, 3),
        score_s=round(t2 - t1, 3),
        top_talkers=dict(talkers.most_common(10)),
    )


# ---------------------------------------------------------------- evaluation


def evaluate(cs: CaptureScore, units: list[Unit]) -> dict:
    """Unit rows, false alerts and benign host-hours for one capture at one schedule."""
    labelled = set()
    for u in units:
        for s in u.srcs:
            for h in range(hour_of(u.start_ms), hour_of(u.end_ms) + 1):
                labelled.add((s, h))
    benign_hours = cs.observed_hours - labelled
    rows = []
    for u in units:
        if u.capture != cs.key:
            continue
        matches = [
            r
            for r in cs.records
            if r["src"] in u.srcs and r["first_ms"] <= u.end_ms and r["last_ms"] >= u.start_ms
        ]
        first = min((max(0.0, r["first_ms"] - u.start_ms) / 1000.0 for r in matches), default=None)
        before = any(r["first_ms"] < u.start_ms for r in matches)
        flagged = sum(1 for s, ts in cs.flagged if s in u.srcs and u.start_ms <= ts <= u.end_ms)
        rows.append(
            {
                "unit": u.id,
                "capture": u.capture,
                "family": u.family,
                "label": u.label,
                "srcs": sorted(u.srcs),
                "detected": bool(matches),
                "family_correct": any(r["type"] == u.family for r in matches),
                "ttfa_s": None if first is None else round(first, 3),
                "alert_before_window": before,
                "model_only_advisory": any(s in cs.model_only_srcs for s in u.srcs),
                "model_flagged_flows_in_window": flagged,
                # top talkers of the capture, kept for misses so a wrong IP set is visible
                "top_talkers": None if matches else cs.top_talkers,
            }
        )
    false_alerts = [r for r in cs.records if (r["src"], hour_of(r["first_ms"])) not in labelled]
    model_only_benign = [
        s for s in cs.model_only_srcs if s not in {x for u in units for x in u.srcs}
    ]
    return {
        "capture": cs.key,
        "kind": cs.kind,
        "units": rows,
        "benign_host_hours": len(benign_hours),
        "false_alerts": false_alerts,
        "model_only_benign_hosts": len(model_only_benign),
    }


def summarise(scores: list[CaptureScore], captures: list[Capture], offset_h: float) -> dict:
    """Metrics at one schedule offset. Units whose capture was not scored are listed as not
    measured and left out of every denominator."""
    scored = {cs.key: cs for cs in scores}
    units_all = cic_units(offset_h) + dns_units(captures, scored)
    units = [u for u in units_all if u.capture in scored]
    not_measured = [u.id for u in units_all if u.capture not in scored]
    per_cap = [evaluate(cs, units) for cs in scores]
    rows = [r for pc in per_cap for r in pc["units"]]
    fams: dict[str, dict] = {}
    for r in rows:
        f = fams.setdefault(
            r["family"],
            {"units": 0, "detected": 0, "family_correct": 0, "misses": [], "ttfa_s": []},
        )
        f["units"] += 1
        f["detected"] += int(r["detected"])
        f["family_correct"] += int(r["family_correct"])
        if not r["detected"]:
            f["misses"].append(r["unit"])
        if r["ttfa_s"] is not None:
            f["ttfa_s"].append(r["ttfa_s"])
    for f in fams.values():
        f["median_ttfa_s"] = round(statistics.median(f["ttfa_s"]), 3) if f["ttfa_s"] else None
        del f["ttfa_s"]
    benign_hours = sum(pc["benign_host_hours"] for pc in per_cap)
    fa = [r for pc in per_cap for r in pc["false_alerts"]]
    by_rule: dict[str, int] = Counter(r["type"] for r in fa)
    scored_flows = sum(c.flows for c in scores)
    return {
        "offset_h": offset_h,
        "not_measured_units": not_measured,
        "units": len(rows),
        "detected": sum(int(r["detected"]) for r in rows),
        "family_correct": sum(int(r["family_correct"]) for r in rows),
        "families": fams,
        "benign": {
            "host_hours": benign_hours,
            "false_alerts": len(fa),
            "false_alerts_per_host_hour": round(len(fa) / benign_hours, 6)
            if benign_hours
            else None,
            "per_rule": {
                k: {
                    "false_alerts": v,
                    "per_host_hour": round(v / benign_hours, 6) if benign_hours else None,
                }
                for k, v in sorted(by_rule.items())
            },
        },
        "model_only": {
            "unit_advisories": sum(int(r["model_only_advisory"]) for r in rows),
            "unit_windows_with_model_flags": sum(
                int(r["model_flagged_flows_in_window"] > 0) for r in rows
            ),
            "benign_hosts_with_advisory": sum(pc["model_only_benign_hosts"] for pc in per_cap),
        },
        "units_detail": rows,
        "flows_scored": scored_flows,
    }


# ---------------------------------------------------------------- outputs


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _dash(value):
    return "-" if value is None else value


def _utc(ms):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def alert_table(scores: list[CaptureScore]):
    """One row per (host, rule type) with its window, plus one row per model-only advisory.
    Host state and risk come from the host's rule alert. No payloads are stored."""
    import pandas as pd

    rows = []
    for cs in scores:
        for r in cs.records:
            meta = next(
                (a for a in cs.alerts if a["src"] == r["src"] and a["path"] != "model_only_quiet"),
                {},
            )
            rows.append(
                {
                    "capture": cs.key,
                    "kind": cs.kind,
                    "host": r["src"],
                    "family": r["type"],
                    "first_ts_ms": r["first_ms"],
                    "last_ts_ms": r["last_ms"],
                    "first_utc": _utc(r["first_ms"]),
                    "last_utc": _utc(r["last_ms"]),
                    "state": meta.get("state", ""),
                    "risk": meta.get("risk", ""),
                    "path": meta.get("path", "rule"),
                }
            )
        for a in cs.alerts:
            if a["path"] == "model_only_quiet":
                rows.append(
                    {
                        "capture": cs.key,
                        "kind": cs.kind,
                        "host": a["src"],
                        "family": "model_only",
                        "first_ts_ms": None,
                        "last_ts_ms": None,
                        "first_utc": None,
                        "last_utc": None,
                        "state": a["state"],
                        "risk": a["risk"],
                        "path": a["path"],
                    }
                )
    return pd.DataFrame(rows, columns=list(PARQUET_COLUMNS))


def report_markdown(state: dict, primary: dict, sens: list[dict]) -> str:
    L = [
        f"# Real pcaps through the rules-first NIDS path (status: {state['status']})",
        "",
        "Pre-registered in docs/NIDS_REAL_PLAN.md. Rules, thresholds and model are the service's.",
        f"Model sha256 prefix {MODEL_SHA256_PREFIX}; flow reader: {state.get('flow_source')}; "
        f"budget {state['max_hours']} h; elapsed {state['elapsed_s']} s.",
        "",
        "## Captures",
        "",
        "| Capture | Status | Flows | Read s | Score s | Flows/s (Kaggle CPU, not Pi) |",
        "|---|---|---|---|---|---|",
    ]
    for c in state["captures"]:
        fps = (
            round(c["flows"] / (c["read_s"] + c["score_s"]), 1)
            if c.get("flows") and (c["read_s"] + c["score_s"]) > 0
            else ""
        )
        L.append(
            f"| `{c['key']}` | {c['status']} {c.get('note', '')} | {c.get('flows', '')} | "
            f"{c.get('read_s', '')} | {c.get('score_s', '')} | {fps} |"
        )
    L += [
        "",
        f"## Detection, primary offset UTC{primary['offset_h']:+.0f} (assumed, see plan 3a)",
        "",
        "| Unit | Family | Window (label) | Detected | Family right | Time to first alert s | "
        "Model-only advisory | Model-flag flows in window |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in primary["units_detail"]:
        L.append(
            f"| {r['unit']} | {r['family']} | {r['label']} | {'yes' if r['detected'] else 'NO'} | "
            f"{'yes' if r['family_correct'] else 'no'} | {_dash(r['ttfa_s'])} | "
            f"{'yes' if r['model_only_advisory'] else 'no'} | "
            f"{r['model_flagged_flows_in_window']} |"
        )
    L += [
        "",
        "| Family | Units | Detected | Family right | Misses | Median time to first alert s |",
        "|---|---|---|---|---|---|",
    ]
    for fam, f in sorted(primary["families"].items()):
        L.append(
            f"| {fam} | {f['units']} | {f['detected']} | {f['family_correct']} | "
            f"{', '.join(f['misses']) or 'none'} | {_dash(f['median_ttfa_s'])} |"
        )
    b = primary["benign"]
    L += [
        "",
        "## Benign false alerts (quiet mode, rule alerts only)",
        "",
        f"{b['false_alerts']} false alerts over {b['host_hours']} benign host-hours "
        f"= {b['false_alerts_per_host_hour']} per host-hour.",
        "",
    ]
    for rule, v in b["per_rule"].items():
        L.append(f"- {rule}: {v['false_alerts']} ({v['per_host_hour']} per host-hour)")
    m = primary["model_only"]
    L += [
        "",
        "## Model-only (reported separately, not alerts)",
        "",
        "Units with a model-only advisory on a unit source: "
        f"{m['unit_advisories']} of {primary['units']}. "
        f"Units with model-flagged flows in the window: {m['unit_windows_with_model_flags']}. "
        f"Model-only advisories on hosts with no unit: {m['benign_hosts_with_advisory']}.",
        "",
    ]
    L += [
        "## Sensitivity (not used to choose anything)",
        "",
        "| Offset | Detected | Family right | Benign false alerts per host-hour |",
        "|---|---|---|---|",
    ]
    for s in sens:
        L.append(
            f"| UTC{s['offset_h']:+.0f} | {s['detected']}/{s['units']} | "
            f"{s['family_correct']}/{s['units']} | "
            f"{s['benign']['false_alerts_per_host_hour']} |"
        )
    return "\n".join(L) + "\n"


def write_outputs(
    out: Path, scores: list[CaptureScore], captures: list[Capture], state: dict
) -> dict:
    """Recompute metrics from the scored captures and write all four files atomically."""
    out.mkdir(parents=True, exist_ok=True)
    primary = summarise(scores, captures, PRIMARY_OFFSET_H)
    sens = [summarise(scores, captures, o) for o in SENSITIVITY_OFFSETS_H]
    metrics = {
        "status": state["status"],
        "primary": primary,
        "sensitivity": [{k: v for k, v in s.items() if k != "units_detail"} for s in sens],
    }
    _atomic_text(out / "metrics.json", json.dumps(metrics, indent=2, default=str) + "\n")
    public = {k: v for k, v in state.items() if not k.startswith("_")}
    _atomic_text(out / "results.json", json.dumps(public, indent=2, default=str) + "\n")
    _atomic_text(out / "report.md", report_markdown(public, primary, sens))
    tmp = out / "alerts.parquet.tmp"
    alert_table(scores).to_parquet(tmp, index=False)
    os.replace(tmp, out / "alerts.parquet")
    return metrics


# ---------------------------------------------------------------- run


def run_captures(
    captures: list[Capture],
    scorer,
    flow_source: str,
    out: Path,
    state: dict,
    deadline: float,
    clock,
    log,
) -> list[CaptureScore]:
    scores: list[CaptureScore] = []

    def check():
        if clock() >= deadline:
            raise BudgetExceeded("wall-clock budget")

    for cap in captures:
        entry = {
            "key": cap.key,
            "kind": cap.kind,
            "path": str(cap.path) if cap.path else None,
            "status": "pending",
        }
        state["captures"].append(entry)
        if cap.path is None:
            entry.update(status="missing", note=cap.note)
            write_outputs(out, scores, captures, state)
            continue
        if clock() >= deadline:
            entry.update(status="budget_not_started", note="not scored")
            state["status"] = "budget_exhausted"
            write_outputs(out, scores, captures, state)
            break
        try:
            cs = score_capture(cap, scorer, flow_source, check, log)
        except BudgetExceeded:
            entry.update(status="budget_exhausted", note="not scored")
            state["status"] = "budget_exhausted"
            write_outputs(out, scores, captures, state)
            break
        except Exception as exc:  # one bad capture does not end the run
            entry.update(status="unmeasured", note=f"{type(exc).__name__}: {str(exc)[:200]}")
            log(f"{cap.key}: unmeasured: {entry['note']}")
            write_outputs(out, scores, captures, state)
            continue
        scores.append(cs)
        entry.update(
            status="scored",
            flows=cs.flows,
            read_s=cs.read_s,
            score_s=cs.score_s,
            rule_records=len(cs.records),
            alert_hosts=sorted(cs.alert_srcs),
            top_talkers=cs.top_talkers,
        )
        log(f"{cap.key}: {cs.flows} flows, read {cs.read_s}s, score {cs.score_s}s")
        state["elapsed_s"] = round(clock() - state["_t0"], 1)
        write_outputs(out, scores, captures, state)
    return scores


def run(ctx) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--flow-source", default="nfstream", choices=("nfstream", "stdlib"))
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args(ctx.argv)
    out = Path(ctx.work)
    t0 = time.monotonic()
    state = {
        "status": "complete",
        "max_hours": args.max_hours,
        "flow_source": args.flow_source,
        "captures": [],
        "_t0": t0,
        "elapsed_s": 0.0,
        "pins": {"nfstream": NFSTREAM_PIN},
    }
    deadline = t0 + args.max_hours * 3600.0
    root = Path(args.input)
    model = find_model(root)
    reason = (
        ensure_nfstream(not args.no_install, ctx.log) if args.flow_source == "nfstream" else None
    )
    captures = discover(root)
    if reason:
        state["status"] = "unmeasured_no_flow_reader"
        state["note"] = reason
        write_outputs(out, [], captures, state)
        ctx.results["summary"] = reason
        return
    scorer = OnnxScorer(str(model))
    scores = run_captures(
        captures, scorer, args.flow_source, out, state, deadline, time.monotonic, ctx.log
    )
    state["elapsed_s"] = round(time.monotonic() - t0, 1)
    metrics = write_outputs(out, scores, captures, state)
    p = metrics["primary"]
    family_right = sum(f["family_correct"] for f in p["families"].values())
    ctx.results["summary"] = (
        f"{state['status']}: {p['detected']}/{p['units']} units detected, "
        f"{family_right} family right, {p['benign']['false_alerts']} benign "
        f"false alerts, {len(p['not_measured_units'])} units not measured"
    )


if __name__ == "__main__":
    raise SystemExit("run by the kernel runner (build_kernel.py), not directly")
