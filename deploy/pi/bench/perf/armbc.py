"""Arms B and C (run on the Pi from the perf dir).

  armbc.py ttv PASSES          EN eval rows x PASSES, sequential /analyze (verdict_first + table).
                               Writes ttv_records.json (per request: text, status, id, times).
  armbc.py drain               GET /explanation/<id> for every pending id; fills wall_ms (LLM time).
  armbc.py http N HITS_FILE    sequential, batch 32/256, 4 HTTP clients over N table-hit messages.
  armbc.py mp N                4 processes, ONNX intra-op 1, analyse_verdict over N eval_ok messages.
"""
import os, concurrent.futures as cf, http.client, json, multiprocessing as mp, random, sys, time
from pathlib import Path
P = Path(os.environ.get("PERF_DIR", "/home/hermoine/rakshak-perf"))
TOOLS = Path(os.environ.get("TOOLS_DIR", "/home/hermoine/pf-tools"))
sys.path[:0] = [str(P/"src"), str(P/"deploy/pi/service"), str(P/"deploy/pi/bench"), str(P/"deploy/pi")]
ONNX = os.environ.get("ONNX_PATH", "/home/hermoine/rakshak-bench/models/sms_student/best_student.onnx")
THR = 0.35127517580986023
REC = TOOLS / "ttv_records.json"

def pct(v, q):
    s = sorted(v); return round(s[min(len(s)-1, int(round(q*(len(s)-1))))], 2)

def summ(v):
    return {"p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99), "n": len(v)}

def en_rows():
    return [r["text"] for r in json.loads((P/"results/edge_eval/rows.json").read_text())["rows"]]

def sample(n):
    import pandas as pd
    df = pd.read_parquet(os.environ.get("SMS_PARQUET", "/home/hermoine/rakshak-bench/data/processed/sms.parquet"), columns=["text", "eval_ok"])
    pool = df[df["eval_ok"] == True]["text"].astype(str).drop_duplicates().tolist()  # noqa: E712
    random.Random(0).shuffle(pool)
    return pool[:n]

def conn():
    return http.client.HTTPConnection("127.0.0.1", 8770, timeout=600)

def post(c, path, doc):
    body = json.dumps(doc).encode()
    t0 = time.perf_counter()
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse(); data = json.loads(r.read())
    return r.status, data, (time.perf_counter()-t0)*1000

def get(c, path):
    c.request("GET", path); r = c.getresponse(); return json.loads(r.read())

def ttv(passes):
    rows = en_rows(); c = conn(); recs = []
    for _ in range(passes):
        for t in rows:
            st, d, rtt = post(c, "/analyze", {"sms": t})
            assert st == 200, (st, str(d)[:200])
            ex = d["explanation"]
            recs.append({"text": t, "rtt": rtt, "inproc": d["timings_ms"]["total"],
                         "status": ex["status"], "id": ex.get("id")})
    REC.write_text(json.dumps(recs, ensure_ascii=False))
    hits = [r for r in recs if r["status"] == "ready"]
    hit_text = list(dict.fromkeys(r["text"] for r in hits))
    (TOOLS/"hits.json").write_text(json.dumps(hit_text, ensure_ascii=False))
    return {"n": len(recs), "table_hits": len(hits), "hit_rate": round(len(hits)/len(recs), 3),
            "distinct_hit_msgs": len(hit_text), "rtt_ms_all": summ([r["rtt"] for r in recs]),
            "inproc_ms_all": summ([r["inproc"] for r in recs]),
            "rtt_ms_hits": summ([r["rtt"] for r in hits]) if hits else None,
            "rtt_ms_misses": summ([r["rtt"] for r in recs if r["status"] != "ready"]),
            "inproc_ms_hits": summ([r["inproc"] for r in hits]) if hits else None}

def drain():
    recs = json.loads(REC.read_text()); c = conn(); walls = []; st = {}
    for r in recs:
        if r["id"]:
            doc = get(c, f"/explanation/{r['id']}")
            st[doc.get("status")] = st.get(doc.get("status"), 0) + 1
            if doc.get("status") == "ready":
                r["wall"] = doc["wall_ms"]; walls.append(doc["wall_ms"])
    REC.write_text(json.dumps(recs, ensure_ascii=False))
    miss = [r for r in recs if r["id"]]
    e2e_miss = [r["inproc"] + r.get("wall", 0) for r in miss]
    e2e_rtt_miss = [r["rtt"] + r.get("wall", 0) for r in miss]
    hits = [r for r in recs if not r["id"]]
    return {"explanation_status_counts": st, "llm_wall_ms": summ(walls) if walls else None,
            "e2e_miss_inproc_plus_llm_ms": summ(e2e_miss) if e2e_miss else None,
            "e2e_miss_rtt_plus_llm_ms": summ(e2e_rtt_miss) if e2e_rtt_miss else None,
            "e2e_hit_inproc_ms": summ([r["inproc"] for r in hits]) if hits else None,
            "e2e_hit_rtt_ms": summ([r["rtt"] for r in hits]) if hits else None}

def http_throughput(n, msgs):
    seq = [msgs[i % len(msgs)] for i in range(n)]
    out = {"messages": len(msgs), "n": n}
    c = conn(); t0 = time.perf_counter()
    for t in seq:
        st, d, _ = post(c, "/analyze", {"sms": t}); assert st == 200 and d["explanation"]["status"] == "ready"
    out["single_seq_msgs_s"] = round(n/(time.perf_counter()-t0), 1)
    for b in (32, 64):
        c = conn(); t0 = time.perf_counter(); done = 0
        while done < n:
            chunk = [msgs[(done+j) % len(msgs)] for j in range(min(b, n-done) if n-done >= b else b)]
            st, d, _ = post(c, "/analyze_batch", {"sms": chunk}); assert st == 200, str(d)[:200]
            done += len(chunk)
        out[f"batch{b}_msgs_s"] = round(done/(time.perf_counter()-t0), 1)
        out[f"batch{b}_n"] = done
    def worker(shard):
        c = conn()
        for t in shard:
            post(c, "/analyze", {"sms": t})
    shards = [[seq[i] for i in range(k, n, 4)] for k in range(4)]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(4) as ex:
        list(ex.map(worker, shards))
    out["concurrent4_msgs_s"] = round(n/(time.perf_counter()-t0), 1)
    return out

def mp_worker(args):
    shard, barrier, q = args
    sys.path[:0] = [str(P/"src"), str(P/"deploy/pi/service"), str(P/"deploy/pi/bench"), str(P/"deploy/pi")]
    from rakshak_edge_service import _DetectorAdapter
    from edge_pipeline import EdgeAnalyzer
    from rbench.runners.onnx import SmsRunner
    class Stub:
        def generate(self, prompt, max_new_tokens=64, stop_when=None):
            return " asks for action.", {"wall_ms": 0.0, "stop_hit": True}
    r = SmsRunner(); r.load(Path(ONNX), 1, {})
    an = EdgeAnalyzer(_DetectorAdapter(r), Stub(), THR)
    for t in shard[:5]:
        an.analyse_verdict(t)
    barrier.wait(); t0 = time.perf_counter()
    for t in shard:
        an.analyse_verdict(t)
    q.put((t0, time.perf_counter()))

def mp_throughput(n, procs=4):
    msgs = sample(n)
    barrier = mp.Barrier(procs); q = mp.Queue()
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=mp_worker, args=((msgs[i::procs], barrier, q),)) for i in range(procs)]
    for p in ps: p.start()
    spans = [q.get() for _ in ps]
    for p in ps: p.join()
    wall = max(e for _, e in spans) - min(s for s, _ in spans)
    return {"procs": procs, "intra_op_threads": 1, "n": n, "msgs_s": round(n/wall, 1)}

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ttv":
        print(json.dumps(ttv(int(sys.argv[2]))))
    elif mode == "drain":
        print(json.dumps(drain()))
    elif mode == "http":
        print(json.dumps(http_throughput(int(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()))))
    elif mode == "mp":
        print(json.dumps(mp_throughput(int(sys.argv[2]))))
