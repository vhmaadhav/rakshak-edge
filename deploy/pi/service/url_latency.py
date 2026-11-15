"""U5 client for the Pi edge service: stdlib only, runs on the Pi against 127.0.0.1:8770.

Two modes (run on the Pi, the service must already be listening):

  analyze_url  POST /analyze_url once per URL of the parity file (300 requests). Reports the
               server-side URL time (``timings_ms.total`` in the response) and client wall time.
  analyze      POST /analyze once per SMS of a 100-message file (each SMS has one parity URL).
               Reports the server-side decision time: ``timings_ms.total`` minus
               ``timings_ms.llm_total`` (the explainer is excluded, so the URL cost is not hidden
               behind an LLM call), and the number of explainer calls.

Percentiles use nearest rank: p = sorted[ceil(q*n) - 1]. The first WARMUP requests are not recorded.

    python3 url_latency.py analyze_url --urls run/parity_urls.json --out run/u5_url.json
    python3 url_latency.py analyze --sms run/u5_sms.json --out run/u5_sms_verify.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8770"
WARMUP = 5


def _post(path: str, body: dict) -> tuple[dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        doc = json.loads(resp.read())
    return doc, round((time.perf_counter() - t0) * 1000.0, 3)


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(q * len(s)) - 1)]


def summary(values: list[float]) -> dict:
    return {"n": len(values), "p50": pct(values, 0.50), "p95": pct(values, 0.95),
            "mean": round(sum(values) / len(values), 3), "max": max(values)}


def run_analyze_url(urls: list[str]) -> dict:
    server, client, scorer = [], [], []
    for i, url in enumerate(urls):
        doc, wall = _post("/analyze_url", {"url": url})
        if i < WARMUP:
            continue
        server.append(float(doc["timings_ms"]["total"]))
        scorer.append(float(doc["timings_ms"].get("evidence", 0.0)))
        client.append(wall)
    return {"mode": "analyze_url", "server_total_ms": summary(server),
            "server_evidence_ms": summary(scorer), "client_wall_ms": summary(client),
            "url_model_fired": None}


def run_analyze(smss: list[str], warmup: int) -> dict:
    decision, llm_calls, client, fired = [], 0, [], 0
    for i, sms in enumerate(smss):
        doc, wall = _post("/analyze", {"sms": sms})
        if i < warmup:
            continue
        t = doc["timings_ms"]
        decision.append(round(float(t["total"]) - float(t.get("llm_total", 0.0)), 3))
        llm_calls += len(t.get("llm_calls", []) or [])
        client.append(wall)
        fired += any(e.get("type") == "phishing_url_model" for e in doc.get("evidence", []))
    return {"mode": "analyze", "decision_ms_excl_llm": summary(decision),
            "client_wall_ms": summary(client), "llm_calls": llm_calls,
            "url_model_evidence_items": fired}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=["analyze_url", "analyze"])
    p.add_argument("--urls", type=Path)
    p.add_argument("--sms", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=WARMUP,
                   help="requests not recorded at the start")
    args = p.parse_args()
    if args.mode == "analyze_url":
        urls = [r["url"] for r in json.loads(args.urls.read_text())]
        result = run_analyze_url(urls)
    else:
        result = run_analyze(json.loads(args.sms.read_text()), args.warmup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
