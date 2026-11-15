#!/usr/bin/env python3
"""Score the holdout pcaps (make_holdout_pcaps.py) in three alert modes and print the tables.

    python3 deploy/pi/service/run_holdout_nids.py --pcaps results/nids_holdout/pcaps \\
        --model models/nids/student_cse_cic_ids2018_int8.onnx --label "Pi 5" --repeat 3 \\
        --json results/nids_holdout/results_pi.json

Modes (docs/NIDS_EDGE.md, "Holdout results"):
  rules       quiet mode, the shipped default: host rules decide, the model cannot raise an alert.
  model-only  the INT8 student alone: a host with a flow in the model's ALERT state (score >=
              0.0015007). No host window and no rules.
  combined    ``--net-model-mode alert``: model ALERT flows and the rules together.

The same NidsAnalyzer and flow reader as POST /analyze_flows score each file. ``--flow-source
auto`` (the service default) uses nfstream when it imports, else the stdlib reader; the output
says which. Needs onnxruntime and numpy; does not need scapy (the expectations come from
manifest.json). Nothing is sent or captured. Throughput lines carry ``--label``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (HERE, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nids_pipeline import (  # noqa: E402
    NET_ALERT_THRESHOLD,
    NET_UNCERTAIN_THRESHOLD,
    NidsAnalyzer,
    flows_from_body,
)

from rakshak.nids.live import HostWindow  # noqa: E402
from rakshak.nids.pi_monitor import OnnxScorer, flow_decision  # noqa: E402

FAMILIES_ORDER = ("brute_force_ssh", "brute_force_other", "port_scan", "dns_tunnel", "dos_flood")
MODE_NAMES = {
    "rules": "(a) rules-first",
    "model_only": "(b) model-only",
    "combined": "(c) combined",
}


def model_only(raw: bytes, scorer, flow_source: str, scratch: Path) -> dict:
    """Model-only mode: hosts with a flow in the model's ALERT state. Host window and rules off."""
    flows, source = flows_from_body(raw, flow_source, scratch)
    ordered = sorted(flows, key=lambda f: float(f.get("bidirectional_first_seen_ms") or 0.0))
    window = HostWindow(window_ms=10_000)
    decs = [
        flow_decision(
            f, scorer, window, alert=NET_ALERT_THRESHOLD, uncertain=NET_UNCERTAIN_THRESHOLD
        )
        for f in ordered
    ]
    by_src: dict[str, list] = {}
    for d in decs:
        if d.state == "scam":
            by_src.setdefault(d.src, []).append(d)
    alerts = []
    for src in sorted(by_src):
        fam = Counter(d.family for d in by_src[src]).most_common(1)[0][0]
        alerts.append({"src": src, "family": fam, "flows": len(by_src[src])})
    return {
        "alerts": alerts,
        "flows": len(decs),
        "flow_source": source,
        "model_flagged_flows": sum(d.family != "benign" for d in decs),
        "p_max": round(max((d.p_attack for d in decs), default=0.0), 4),
    }


def _rule_alerts(out: dict) -> list[dict]:
    return [a for a in out["alerts"] if a["advisory"]["path"] != "model_only_quiet"]


def _slim(alerts: list[dict]) -> list[dict]:
    return [
        {
            "src": a["src"],
            "family": a["decision"]["family"],
            "state": a["decision"]["state"],
            "risk": a["decision"]["risk"],
            "action": a["advisory"]["action"],
            "path": a["advisory"]["path"],
            "verifier_ok": a["verifier"]["ok"],
        }
        for a in alerts
    ]


def score_file(path: Path, row: dict, scorer, flow_source: str, scratch: Path, repeat: int):
    raw = path.read_bytes()
    quiet = NidsAnalyzer(scorer, scratch=scratch, flow_source=flow_source, model_mode="quiet")
    combined = NidsAnalyzer(scorer, scratch=scratch, flow_source=flow_source, model_mode="alert")
    runs = [quiet.analyse_body(raw) for _ in range(max(1, repeat))]
    q = runs[0]
    comb = combined.analyse_body(raw)
    mo = model_only(raw, scorer, flow_source, scratch)
    return {
        "file": row["file"],
        "group": row["group"],
        "expect": row["expect"],
        "srcs": row["srcs"],
        "flows": q["flows_processed"],
        "flow_source": q["flow_source"],
        "rules": {
            "alerts": _slim(_rule_alerts(q)),
            "model_only_hosts": q["decision"]["model_only_hosts"],
            "model_flagged_flows": q["decision"]["model_flagged_flows"],
            "model_p_max": q["decision"]["model_p_attack_max"],
        },
        "model_only": {
            "alerts": mo["alerts"],
            "model_flagged_flows": mo["model_flagged_flows"],
            "p_max": mo["p_max"],
        },
        "combined": {"alerts": _slim(comb["alerts"])},
        "timing": {
            "total_ms": [r["timings_ms"]["total"] for r in runs],
            "score_ms": [r["timings_ms"]["score"] for r in runs],
        },
    }


def fam_ok(expected: str, got: str) -> bool:
    """The model names families coarsely: brute_force and dos_ddos."""
    if got == expected:
        return True
    if got == "brute_force" and expected.startswith("brute_force"):
        return True
    return got == "dos_ddos" and expected == "dos_flood"


def judge(alerts: list[dict], expect: list[str], srcs: list[str]) -> list[dict]:
    """One entry per expected (family, source): detected (source alerted), family right."""
    out = []
    for fam, src in zip(expect, srcs, strict=False):
        mine = [a for a in alerts if a["src"] == src]
        out.append(
            {
                "family": fam,
                "src": src,
                "detected": bool(mine),
                "family_right": any(fam_ok(fam, a["family"]) for a in mine),
            }
        )
    return out


def _mode_alerts(rec: dict, mode: str) -> list[dict]:
    return rec[{"rules": "rules", "model_only": "model_only", "combined": "combined"}[mode]][
        "alerts"
    ]


def _p(values: list[float], q: float):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 3)


def _near_row(rec: dict, alerts: list[dict]) -> dict:
    if rec["expect"] == ["open"]:
        as_spec = "no prediction"
    elif not rec["expect"]:
        as_spec = "yes" if not alerts else "no (alert raised)"
    else:
        js = judge(alerts, rec["expect"], rec["srcs"])
        as_spec = "yes" if all(j["detected"] and j["family_right"] for j in js) else "no"
    return {
        "file": rec["file"],
        "expect": rec["expect"] or ["none"],
        "alerts": [f"{a['src']}:{a['family']}" for a in alerts],
        "as_specified": as_spec,
    }


def summarise(records: list[dict], label: str, flow_source: str, model: Path, repeat: int) -> dict:
    per_mode = {}
    for mode in MODE_NAMES:
        fam_rows: dict[str, dict] = {}
        fp = {"fp": 0, "n": 0, "files": []}
        near = []
        for rec in records:
            alerts = _mode_alerts(rec, mode)
            if rec["group"] == "benign":
                fp["n"] += 1
                if alerts:
                    fp["fp"] += 1
                    fp["files"].append(rec["file"])
            if rec["group"] in ("main", "mixed"):
                for j in judge(alerts, rec["expect"], rec["srcs"]):
                    row = fam_rows.setdefault(
                        j["family"], {"n": 0, "tp": 0, "right": 0, "misses": []}
                    )
                    row["n"] += 1
                    row["tp"] += int(j["detected"])
                    row["right"] += int(j["family_right"])
                    if not j["detected"]:
                        row["misses"].append(rec["file"])
            if rec["group"] == "near":
                near.append(_near_row(rec, alerts))
        per_mode[mode] = {"families": fam_rows, "benign_fp": fp, "near": near}
    flows = sum(r["flows"] for r in records)
    per_req = [t for r in records for t in r["timing"]["total_ms"]]
    score_s = sum(statistics.median(r["timing"]["score_ms"]) for r in records) / 1000.0
    return {
        "label": label,
        "machine": platform.machine(),
        "model": {
            "path": str(model),
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest()[:16],
        },
        "flow_source_requested": flow_source,
        "flow_sources_used": sorted({r["flow_source"] for r in records}),
        "repeat": repeat,
        "files": len(records),
        "flows": flows,
        "throughput": {
            "flows_per_s": round(flows / score_s, 1) if score_s > 0 else None,
            "request_ms_p50": _p(per_req, 0.5),
            "request_ms_p95": _p(per_req, 0.95),
        },
        "modes": per_mode,
    }


def _file_row(r: dict) -> str:
    exp = "none"
    if r["expect"] == ["open"]:
        exp = "no prediction"
    elif r["expect"]:
        exp = " + ".join(r["expect"])
    ra = ", ".join(
        f"{a['src']} {a['family']} ({a['state']}, {a['risk']}, {a['action']})"
        for a in r["rules"]["alerts"]
    )
    mo = ", ".join(f"{a['src']} {a['family']}" for a in r["model_only"]["alerts"])
    co = ", ".join(f"{a['src']} {a['family']}" for a in r["combined"]["alerts"])
    rr = r["rules"]
    flags = (
        f"{rr['model_flagged_flows']} flows, {rr['model_only_hosts']} model-only hosts, "
        f"p max {rr['model_p_max']}"
    )
    return (
        f"| `{r['file']}` | {r['group']} | {exp} | {ra or 'none'} | {mo or 'none'} "
        f"| {co or 'none'} | {flags} |"
    )


def markdown(s: dict, records: list[dict]) -> str:
    lines = [
        f"Machine: {s['label']} ({s['machine']}). Model sha256 {s['model']['sha256']}. "
        f"Flow source requested `{s['flow_source_requested']}`, "
        f"used {', '.join(s['flow_sources_used'])}. "
        f"{s['files']} files, {s['flows']} flows, repeat {s['repeat']}.",
        "",
        "| File | Group | Expected | (a) rules alerts | (b) model-only alerts "
        "| (c) combined alerts | Model flags (a) |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_file_row(r) for r in records]
    lines.append("")
    head = " | ".join(f"{MODE_NAMES[m]} TP/FN, family right" for m in MODE_NAMES)
    lines += [f"| Family (instances) | {head} |", "|---|---|---|---|"]
    fams = list(FAMILIES_ORDER) + sorted(
        {f for m in s["modes"].values() for f in m["families"]} - set(FAMILIES_ORDER)
    )
    for fam in fams:
        cells, n = [], None
        for m in MODE_NAMES:
            row = s["modes"][m]["families"].get(fam)
            if row is None:
                cells.append("-")
                continue
            n = row["n"]
            cells.append(f"{row['tp']}/{row['n'] - row['tp']}, {row['right']}/{row['n']}")
        if n is not None:
            lines.append(f"| {fam} ({n}) | " + " | ".join(cells) + " |")
    fp = " | ".join(
        f"{s['modes'][m]['benign_fp']['fp']}/{s['modes'][m]['benign_fp']['n']}" for m in MODE_NAMES
    )
    lines.append(f"| Benign files with any alert (FP/n) | {fp} |")
    lines.append("")
    head = " | ".join(f"{MODE_NAMES[m]} alerts (as specified)" for m in MODE_NAMES)
    lines += [f"| Near-threshold file | Expected | {head} |", "|---|---|---|---|---|"]
    for name in [x["file"] for x in s["modes"]["rules"]["near"]]:
        cells, exp = [], ""
        for m in MODE_NAMES:
            x = next(v for v in s["modes"][m]["near"] if v["file"] == name)
            exp = " + ".join(x["expect"])
            cells.append(f"{', '.join(x['alerts']) or 'none'} ({x['as_specified']})")
        lines.append(f"| `{name}` | {exp} | " + " | ".join(cells) + " |")
    t = s["throughput"]
    lines += [
        "",
        f"Throughput ({s['label']}): {t['flows_per_s']} flows/s (scoring only, rules mode, "
        f"median of {s['repeat']} runs per file); per request p50 {t['request_ms_p50']} ms, "
        f"p95 {t['request_ms_p95']} ms (whole request, parse to verifier).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pcaps", required=True, help="directory with the pcaps and manifest.json")
    p.add_argument(
        "--model",
        default=str(REPO / "models" / "nids" / "student_cse_cic_ids2018_int8.onnx"),
        help="student ONNX file (the Pi copy is models/nids/student_cse_cic_ids2018_int8.onnx)",
    )
    p.add_argument("--flow-source", default="auto", choices=("auto", "nfstream", "stdlib"))
    p.add_argument("--repeat", type=int, default=3, help="runs per file for timing (rules mode)")
    p.add_argument("--label", default="laptop, not Pi", help="machine label for throughput")
    p.add_argument("--json", help="write the per-file and summary record to this path")
    args = p.parse_args(argv)

    pcaps = Path(args.pcaps)
    manifest = json.loads((pcaps / "manifest.json").read_text(encoding="utf-8"))
    model = Path(args.model)
    scorer = OnnxScorer(str(model))
    records = []
    with tempfile.TemporaryDirectory(prefix="holdout-scratch-") as tmp:
        scratch = Path(tmp)
        for row in manifest["cases"]:
            path = pcaps / row["file"]
            records.append(score_file(path, row, scorer, args.flow_source, scratch, args.repeat))
    summary = summarise(records, args.label, args.flow_source, model, args.repeat)
    print(markdown(summary, records))
    if args.json:
        doc = {"summary": summary, "files": records}
        Path(args.json).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
