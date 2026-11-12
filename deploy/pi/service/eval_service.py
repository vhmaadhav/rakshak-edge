#!/usr/bin/env python3
"""Evaluation client for the edge service: row selection, the request batch, and the summary.

    eval_service.py select   # writes rows.json (30 evidence rows + 10 benign LOW rows)
    eval_service.py run      # POSTs every row to the service over its Unix socket, in order
    eval_service.py report   # summary.json and the tables from responses.jsonl

    --rows / --out select another row file and output directory (the Hinglish set uses
    results/edge_eval_hinglish/); --lang sends the request field "lang" (en, hinglish or auto).

Rows come from the same seeded sampler as the rbench slot run (worker._ela_rows, seed 0, the
campaign SMS test split). The 10 benign rows are seeded non-scam rows (label_binary 0) whose
decision at selection time is LOW. The client talks to the service's Unix socket (run/edge.sock),
which works across the service's network namespace; it makes no network connection.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path.home() / "rakshak-bench"
sys.path[:0] = [
    str(HERE),
    str(REPO / "src"),
    str(REPO / "deploy" / "pi" / "bench"),
    str(REPO / "deploy" / "pi"),
]

EVAL_DIR = ROOT / "results" / "edge_eval"
ROWS = EVAL_DIR / "rows.json"
RESPONSES = EVAL_DIR / "responses.jsonl"
SUMMARY = EVAL_DIR / "summary.json"
SOCKET = ROOT / "run" / "edge.sock"
SEED = 0
N_EVIDENCE = 30
N_BENIGN = 10
START_MAX_TEMP_C = 65.0  # do not start a batch at or above this


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def _p(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "mean": round(sum(values) / len(values), 3) if values else None,
    }


def select() -> int:
    from edge_pipeline import scam_state_for
    from rbench.datasets import load_rows
    from rbench.registry import load_registry
    from rbench.runners.onnx import SmsRunner
    from rbench.worker import _ela_rows

    from rakshak.canonicalize import canonicalize
    from rakshak.evidence import build_package
    from rakshak.rules import run_rules

    reg = load_registry(ROOT / "deploy" / "pi" / "bench" / "registry.toml")
    ds = reg.dataset("sms_ela_campaign_test")
    detector = SmsRunner()
    detector.load(ROOT / "models" / "sms_student" / "best_student.onnx", 2, {})
    threshold = float(reg.model("sms_student").threshold)

    def decide(text: str) -> tuple[str, float, int]:
        canon = canonicalize(text)
        ev = run_rules(canon)
        prob, _ = detector.score(detector.run(detector.make_feed({"text": text})))
        state = scam_state_for(float(prob), threshold, ev)
        return build_package(canon, ev, state).risk.value, float(prob), len(ev)

    evidence_rows = _ela_rows(ROOT, ds, N_EVIDENCE, SEED).items
    base = load_rows(ROOT, ds, None, SEED)
    import numpy as np

    order = np.random.default_rng(SEED).permutation(len(base.items))
    benign: list[dict[str, Any]] = []
    scanned = 0
    for idx in order:
        if len(benign) >= N_BENIGN:
            break
        item = base.items[int(idx)]
        if item["label"] != 0:
            continue
        scanned += 1
        risk, prob, n_ev = decide(item["text"])
        if risk == "LOW":
            benign.append(
                {
                    **item,
                    "group": "benign_low",
                    "decision_at_selection": risk,
                    "score_at_selection": round(prob, 4),
                    "evidence_at_selection": n_ev,
                }
            )
    out_rows = []
    for item in evidence_rows:
        risk, prob, n_ev = decide(item["text"])
        out_rows.append(
            {
                **item,
                "group": "evidence",
                "decision_at_selection": risk,
                "score_at_selection": round(prob, 4),
                "evidence_at_selection": n_ev,
            }
        )
    out_rows += benign
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ROWS.write_text(
        json.dumps(
            {"seed": SEED, "threshold": threshold, "benign_scan": scanned, "rows": out_rows},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"select: {len(evidence_rows)} evidence rows, {len(benign)} benign LOW rows "
        f"(scanned {scanned} label-0 rows) -> {ROWS}"
    )
    return 0


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("edge", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def _call(method: str, path: str, body: bytes | None = None) -> tuple[int, dict[str, Any]]:
    conn = _UnixConnection(str(SOCKET), timeout=600)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    finally:
        conn.close()


def _post(sms: str, lang: str | None = None) -> tuple[int, dict[str, Any]]:
    doc: dict[str, Any] = {"sms": sms}
    if lang is not None:
        doc["lang"] = lang
    return _call("POST", "/analyze", json.dumps(doc).encode("utf-8"))


def _get(path: str) -> dict[str, Any]:
    status, doc = _call("GET", path)
    if status != 200:
        raise RuntimeError(f"GET {path} returned {status}")
    return doc


def run(rows_path: Path = ROWS, out_dir: Path = EVAL_DIR, lang: str | None = None) -> int:
    from edge_device import read_snapshot

    responses = out_dir / "responses.jsonl"
    doc = json.loads(rows_path.read_text(encoding="utf-8"))
    start = read_snapshot()
    if start["temp_c"] is None or start["temp_c"] >= START_MAX_TEMP_C:
        print(
            f"run: not starting, SoC at {start['temp_c']} C (limit {START_MAX_TEMP_C} C)",
            file=sys.stderr,
        )
        return 2
    health_before = _get("/health")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with responses.open("w", encoding="utf-8") as handle:
        for i, row in enumerate(doc["rows"]):
            status, body = _post(row["text"], lang)
            rec = {
                "i": i,
                "row_id": row["row_id"],
                "group": row["group"],
                "label": row["label"],
                "http_status": status,
                "body": body,
            }
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            if status == 200:
                print(
                    f"{i:2d} {row['group']:12s} {body['decision']['risk']:6s} "
                    f"{body['advisory']['mode']:8s} total={body['timings_ms']['total']:.0f}ms "
                    f"E={body['device']['energy']['energy_j']:.2f}J "
                    f"T={body['device']['end']['temp_c']}C",
                    flush=True,
                )
            else:
                print(f"{i:2d} HTTP {status}: {body.get('error')}", flush=True)
    wall = time.perf_counter() - t0
    health_after = _get("/health")
    meta = {
        "start_device": start,
        "health_before": health_before,
        "health_after": health_after,
        "batch_wall_s": round(wall, 2),
    }
    meta["request_lang"] = lang or "(not sent: service default)"
    meta["rows"] = str(rows_path)
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"run: {len(doc['rows'])} requests in {wall:.1f} s -> {responses}")
    return 0


def _records(out_dir: Path) -> list[dict[str, Any]]:
    text = (out_dir / "responses.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def report(out_dir: Path = EVAL_DIR) -> int:
    recs = _records(out_dir)
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    ok = [r for r in recs if r["http_status"] == 200]
    refused = [r for r in recs if r["http_status"] == 503]
    groups = {
        "evidence": [r for r in ok if r["group"] == "evidence"],
        "benign": [r for r in ok if r["group"].startswith("benign")],
        "all": ok,
    }
    summary: dict[str, Any] = {
        "requests": len(recs),
        "ok": len(ok),
        "refused_503": len(refused),
        "batch_wall_s": meta["batch_wall_s"],
        "groups": {},
    }
    for name, rs in groups.items():
        bodies = [r["body"] for r in rs]
        called = [b for b in bodies if not b["advisory"]["llm_skipped_low_risk"]]
        pts = [p for b in called for p in b["advisory"]["points"]]
        llm_pts = [p for p in pts if p["source"] == "llm"]
        temps = [b["device"]["end"]["temp_c"] for b in bodies if b["device"]["end"]["temp_c"]]
        temps += [b["device"]["start"]["temp_c"] for b in bodies if b["device"]["start"]["temp_c"]]
        energy_all = [b["device"]["energy"]["energy_j"] for b in bodies]
        energy_called = [b["device"]["energy"]["energy_j"] for b in called]
        stage = {
            "canonicalise_ms": _p([b["timings_ms"]["canonicalise"] for b in bodies]),
            "detect_ms": _p([b["timings_ms"]["detect"] for b in bodies]),
            "evidence_ms": _p([b["timings_ms"]["evidence"] for b in bodies]),
            "llm_per_advisory_ms": _p([b["timings_ms"]["llm_total"] for b in called]),
            "llm_per_call_ms": _p(
                [c["wall_ms"] for b in called for c in b["llm_calls"] if not c.get("cached")]
            ),
            "verify_ms": _p([b["timings_ms"]["verify"] for b in bodies]),
            "total_ms": _p([b["timings_ms"]["total"] for b in bodies]),
            "total_ms_llm_called": _p([b["timings_ms"]["total"] for b in called]),
        }
        summary["groups"][name] = {
            "n": len(bodies),
            "lang_detected": _count([b["decision"].get("lang_detected", "?") for b in bodies]),
            "advisory_lang": _count([b["advisory"].get("lang", "?") for b in bodies]),
            "decision_risk": _count([b["decision"]["risk"] for b in bodies]),
            "decision_action": _count([b["decision"]["action"] for b in bodies]),
            "advisory_mode": _count([b["advisory"]["mode"] for b in bodies]),
            "llm_skipped_low_risk": sum(b["advisory"]["llm_skipped_low_risk"] for b in bodies),
            "llm_advisories": len(called),
            "llm_calls_total": sum(len(b["llm_calls"]) for b in bodies),
            "llm_calls_cached": sum(
                1 for b in bodies for c in b["llm_calls"] if c.get("cached")
            ),
            "points_total": len(pts),
            "points_source_llm": len(llm_pts),
            # raw: the verifier on the model's own sentence, before the template fallback
            "points_raw_verifier_pass": sum(1 for p in pts if p["check"]["ok"]),
            "advisories_raw_verifier_pass": sum(
                1 for b in called if all(p["check"]["ok"] for p in b["advisory"]["points"])
            ),
            "g2_rejected_sentences": sum(len(b["verifier"]["g2_rejected"]) for b in bodies),
            "g2_distinct_sentences": sorted(
                {x["sentence"] for b in bodies for x in b["verifier"]["g2_rejected"]}
            ),
            "advisory_path": _count([b["advisory"].get("path", "?") for b in bodies]),
            "final_contract_ok": sum(b["verifier"]["final_contract_ok"] for b in bodies),
            "stages": stage,
            "energy_j_all": _p(energy_all),
            "energy_j_llm_called": _p(energy_called),
            "power_mean_w": _p([b["device"]["energy"]["mean_w"] for b in bodies]),
            "max_temp_c": max(temps) if temps else None,
            "rss_mb_max": max((b["device"]["rss_mb"] or 0.0) for b in bodies) if bodies else None,
        }
    summary["health_before"] = meta["health_before"]
    summary["health_after"] = meta["health_after"]
    summary["start_device"] = meta["start_device"]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=1))
    for name, g in summary["groups"].items():
        print(f"== {name} n={g['n']} risk={g['decision_risk']} mode={g['advisory_mode']}")
        print(
            f"   skipped={g['llm_skipped_low_risk']} llm_adv={g['llm_advisories']} "
            f"points={g['points_total']} llm_src={g['points_source_llm']} "
            f"raw_pass={g['points_raw_verifier_pass']}/{g['points_total']} "
            f"raw_adv={g['advisories_raw_verifier_pass']}/{g['llm_advisories']} "
            f"g2_rej={g['g2_rejected_sentences']} "
            f"final_ok={g['final_contract_ok']}"
        )
        for k, v in g["stages"].items():
            print(f"   {k:22s} n={v['n']:3d} p50={v['p50']} p95={v['p95']}")
        print(f"   energy_j all={g['energy_j_all']} llm_called={g['energy_j_llm_called']}")
        print(f"   max_temp={g['max_temp_c']} rss_max={g['rss_mb_max']}")
    return 0


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("select", "run", "report"))
    p.add_argument("--rows", type=Path, default=ROWS, help="row file (run)")
    p.add_argument("--out", type=Path, default=EVAL_DIR, help="output directory (run, report)")
    p.add_argument(
        "--lang",
        choices=("auto", "en", "hinglish"),
        default=None,
        help="send this request language with each row (run); default: field not sent",
    )
    args = p.parse_args(argv)
    if args.command == "select":
        return select()
    if args.command == "run":
        return run(args.rows, args.out, args.lang)
    return report(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
