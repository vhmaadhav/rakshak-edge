"""Decision parity on the Pi: one process per (tree, mode). Writes one JSON line per message.

modes: old (baseline EdgeAnalyzer.analyse), new (analyse_verdict), newbatch (analyse_verdict_batch,
batch_onnx off), newbatch_onnx (batch_onnx on). The explainer is a stub; decisions do not use it.
"""
import os, argparse, json, random, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--mode", required=True, choices=["old", "new", "newbatch", "newbatch_onnx"])
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sample-n", type=int, default=5000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()

T = Path(a.tree)
sys.path[:0] = [str(T / "src"), str(T / "deploy/pi/service"), str(T / "deploy/pi/bench"), str(T / "deploy/pi")]
from rakshak_edge_service import _DetectorAdapter  # noqa
from edge_pipeline import EdgeAnalyzer  # noqa
from rbench.runners.onnx import SmsRunner  # noqa

THRESHOLD = 0.35127517580986023

class StubLLM:
    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        return " asks for action.", {"wall_ms": 0.0, "stop_hit": True}

# Message set: the two 40-row eval sets, then a seeded sample of eval_ok SMS from the data dir.
texts = []
for rows in ("results/edge_eval/rows.json", "results/edge_eval_hinglish/rows.json"):
    texts += [r["text"] for r in json.loads((T / rows).read_text())["rows"]]
import pandas as pd
df = pd.read_parquet(os.environ.get("SMS_PARQUET", "/home/hermoine/rakshak-bench/data/processed/sms.parquet"), columns=["text", "eval_ok"])
pool = df[df["eval_ok"] == True]["text"].astype(str).drop_duplicates().tolist()  # noqa: E712
random.Random(0).shuffle(pool)
texts += pool[: a.sample_n]

runner = SmsRunner()
runner.load(Path(a.onnx), a.threads, {})
if a.mode == "old":
    adapter = _DetectorAdapter(runner)
else:
    adapter = _DetectorAdapter(runner, batch_onnx=(a.mode == "newbatch_onnx"))
an = EdgeAnalyzer(adapter, StubLLM(), THRESHOLD)

def fields(r):
    d = r["decision"]
    return {"risk": d["risk"], "action": d["action"], "score": d["score"],
            "scam_state": d["scam_state"], "detector_only": d["detector_only"],
            "evidence": [(e["id"], e["type"], e["weight"]) for e in r["evidence"]],
            "remediation": r["remediation"]}

t0 = time.time()
out = []
if a.mode == "old":
    out = [fields(an.analyse(t)) for t in texts]
elif a.mode == "new":
    out = [fields(an.analyse_verdict(t)[0]) for t in texts]
else:
    for i in range(0, len(texts), a.batch):
        out += [fields(r) for r, _ in an.analyse_verdict_batch(texts[i:i + a.batch])]
with open(a.out, "w") as f:
    for i, (t, d) in enumerate(zip(texts, out)):
        f.write(json.dumps({"i": i, "text": t[:60], **d}, ensure_ascii=False, default=str) + "\n")
print(json.dumps({"mode": a.mode, "tree": a.tree, "n": len(out), "seconds": round(time.time() - t0, 1)}))
