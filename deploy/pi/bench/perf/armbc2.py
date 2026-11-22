"""Round-2 arms for the verdict-first service (run on the Pi from /home/hermoine/pf-tools).

  armbc2.py ttv PASSES          EN eval rows x PASSES over the TCP relay (127.0.0.1:8770), /analyze.
                                Writes ttv_records.json and hits.json (table-hit messages).
  armbc2.py drain               GET /explanation/<id> for every pending id; LLM wall per entry.
  armbc2.py seq N HITS          N sequential /analyze over one keep-alive connection (TCP relay).
  armbc2.py batch B N HITS      N messages in /analyze_batch calls of B (keep-alive, TCP relay).
  armbc2.py clients C PER HITS  C keep-alive clients (4 processes), PER requests each, /analyze.
  armbc2.py uds N HITS          N sequential /analyze over the Unix socket run/edge.sock.
"""
import os, http.client, json, multiprocessing as mp, random, socket, sys, threading, time
from pathlib import Path

P = Path(os.environ.get("PERF_DIR", "/home/hermoine/rakshak-perf"))
TOOLS = Path(os.environ.get("TOOLS_DIR", "/home/hermoine/pf-tools"))
REC = TOOLS / "ttv_records.json"
SOCK = str(P / "run" / "edge.sock")


def sock_paths():
    """One socket per verdict worker (edge.sock.<n>) when there are several, else edge.sock."""
    import glob
    found = sorted(glob.glob(SOCK + ".*"), key=lambda q: (len(q), q))
    return found or [SOCK]


def pct(v, q):
    s = sorted(v)
    return round(s[min(len(s) - 1, int(round(q * (len(s) - 1))))], 2)


def summ(v):
    return {"p50": pct(v, .5), "p95": pct(v, .95), "p99": pct(v, .99), "n": len(v)} if v else None


def en_rows():
    return [r["text"] for r in json.loads((P / "results/edge_eval/rows.json").read_text())["rows"]]


def conn():
    return http.client.HTTPConnection("127.0.0.1", 8770, timeout=600)


def post(c, path, doc):
    body = json.dumps(doc).encode()
    t0 = time.perf_counter()
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    return r.status, data, (time.perf_counter() - t0) * 1000


def get(c, path):
    c.request("GET", path)
    r = c.getresponse()
    return json.loads(r.read())


def ttv(passes):
    rows = en_rows()
    c = conn()
    recs = []
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
    (TOOLS / "hits.json").write_text(json.dumps(hit_text, ensure_ascii=False))
    return {"n": len(recs), "table_hits": len(hits), "hit_rate": round(len(hits) / len(recs), 3),
            "distinct_hit_msgs": len(hit_text), "rtt_ms_all": summ([r["rtt"] for r in recs]),
            "inproc_ms_all": summ([r["inproc"] for r in recs]),
            "rtt_ms_hits": summ([r["rtt"] for r in hits]),
            "inproc_ms_hits": summ([r["inproc"] for r in hits]),
            "rtt_ms_pending": summ([r["rtt"] for r in recs if r["status"] == "pending"])}


def drain():
    """Waits for the explainer queue (FIFO: the last job done means all are done), then reads
    every explanation. e2e_* = verdict RTT + queue wait + LLM wall, from the service's own
    timestamps (queued_at, finished_at); the LLM wall is the job's wall_ms."""
    recs = json.loads(REC.read_text())
    c = conn()
    ids = [r["id"] for r in recs if r["id"]]
    if ids:
        get(c, f"/explanation/{ids[-1]}?wait=900")
    st, walls, e2e, e2e_inproc = {}, [], [], []
    for r in recs:
        if r["id"]:
            doc = get(c, f"/explanation/{r['id']}")
            st[doc.get("status")] = st.get(doc.get("status"), 0) + 1
            if doc.get("status") == "ready":
                r["wall"] = doc["wall_ms"]
                walls.append(doc["wall_ms"])
                q = (doc["finished_at"] - doc["queued_at"]) * 1000.0
                r["e2e"] = r["rtt"] + q
                e2e.append(r["e2e"])
    REC.write_text(json.dumps(recs, ensure_ascii=False))
    miss = [r for r in recs if r["id"]]
    hits = [r for r in recs if not r["id"]]
    return {"explanation_status_counts": st, "llm_wall_ms": summ(walls),
            "e2e_miss_rtt_plus_queue_llm_ms": summ(e2e),
            "e2e_hit_rtt_ms": summ([r["rtt"] for r in hits]),
            "e2e_hit_inproc_ms": summ([r["inproc"] for r in hits]),
            "e2e_all_rtt_ms": summ([r.get("e2e", r["rtt"]) for r in recs])}


def seq(n, msgs):
    c = conn()
    lat = []
    t0 = time.perf_counter()
    for i in range(n):
        st, d, rtt = post(c, "/analyze", {"sms": msgs[i % len(msgs)]})
        assert st == 200, st
        lat.append(rtt)
    el = time.perf_counter() - t0
    return {"n": n, "msgs_s": round(n / el, 1), "rtt_ms": summ(lat)}


def batch(b, n, msgs):
    c = conn()
    done = 0
    t0 = time.perf_counter()
    while done < n:
        chunk = [msgs[(done + j) % len(msgs)] for j in range(b)]
        st, d, _ = post(c, "/analyze_batch", {"sms": chunk})
        assert st == 200, str(d)[:200]
        assert d["count"] == b
        done += b
    el = time.perf_counter() - t0
    return {"batch": b, "n": done, "msgs_s": round(done / el, 1)}


def _client_proc(args):
    msgs, per, q = args
    c = conn()
    lat = []
    t0 = time.perf_counter()
    for i in range(per):
        st, d, rtt = post(c, "/analyze", {"sms": msgs[i % len(msgs)]})
        assert st == 200, st
        lat.append(rtt)
    q.put((t0, time.perf_counter(), lat))


def _client_group(shards, per, q):
    out_lat = []
    spans = []
    def worker(shard):
        c = conn()
        lat = []
        t0 = time.perf_counter()
        for i in range(per):
            st, d, rtt = post(c, "/analyze", {"sms": shard[i % len(shard)]})
            assert st == 200, st
            lat.append(rtt)
        spans.append((t0, time.perf_counter()))
        out_lat.extend(lat)
    ths = [threading.Thread(target=worker, args=(s,)) for s in shards]
    for t in ths: t.start()
    for t in ths: t.join()
    q.put((spans, out_lat))


def concurrent(nproc, threads_per, per, msgs):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    total = nproc * threads_per
    shards = [msgs[i::total] for i in range(total)]
    procs = []
    for k in range(nproc):
        grp = shards[k * threads_per:(k + 1) * threads_per]
        p = ctx.Process(target=_client_group, args=(grp, per, q))
        p.start()
        procs.append(p)
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()
    spans = [s for r in results for s in r[0]]
    lat = [x for r in results for x in r[1]]
    wall = max(e for _, e in spans) - min(s for s, _ in spans)
    n = total * per
    return {"clients": total, "per_client": per, "n": n, "msgs_s": round(n / wall, 1), "rtt_ms": summ(lat)}


def _batch_group(shards, calls, b, q):
    spans, counts = [], []
    def worker(shard):
        c = conn()
        t0 = time.perf_counter()
        n = 0
        for k in range(calls):
            chunk = [shard[(k * b + j) % len(shard)] for j in range(b)]
            st, d, _ = post(c, "/analyze_batch", {"sms": chunk})
            assert st == 200 and d["count"] == b, str(d)[:120]
            n += b
        spans.append((t0, time.perf_counter()))
        counts.append(n)
    ths = [threading.Thread(target=worker, args=(s,)) for s in shards]
    for t in ths: t.start()
    for t in ths: t.join()
    q.put((spans, sum(counts)))


def concurrent_batch(nproc, threads_per, calls, b, msgs):
    """16 keep-alive clients (4 processes x 4 threads), each sending /analyze_batch calls of b."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    total = nproc * threads_per
    shards = [msgs[i::total] for i in range(total)]
    procs = []
    for k in range(nproc):
        grp = shards[k * threads_per:(k + 1) * threads_per]
        pr = ctx.Process(target=_batch_group, args=(grp, calls, b, q))
        pr.start()
        procs.append(pr)
    results = [q.get() for _ in procs]
    for pr in procs:
        pr.join()
    spans = [sp for r in results for sp in r[0]]
    n = sum(r[1] for r in results)
    wall = max(e for _, e in spans) - min(st for st, _ in spans)
    return {"clients": total, "batch": b, "calls_per_client": calls, "n": n,
            "msgs_s": round(n / wall, 1)}


def _uds_batch_group(shards, calls, b, q, paths):
    """Keep-alive Unix-socket client that reads each batch reply by Content-Length, unparsed."""
    spans, counts = [], []
    def worker(shard, idx):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(paths[idx % len(paths)])
        t0 = time.perf_counter()
        n = 0
        for k in range(calls):
            chunk = [shard[(k * b + j) % len(shard)] for j in range(b)]
            body = json.dumps({"sms": chunk}).encode()
            req = (f"POST /analyze_batch HTTP/1.1\r\nHost: edge\r\nContent-Type: application/json\r\n"
                   f"Content-Length: {len(body)}\r\n\r\n").encode() + body
            s.sendall(req)
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += s.recv(1 << 20)
            head, rest = buf.split(b"\r\n\r\n", 1)
            clen = 0
            for line in head.split(b"\r\n")[1:]:
                kk, _, vv = line.partition(b":")
                if kk.strip().lower() == b"content-length":
                    clen = int(vv)
            while len(rest) < clen:
                rest += s.recv(1 << 20)
            assert head.startswith(b"HTTP/1.1 200"), head[:40]
            n += b
        s.close()
        spans.append((t0, time.perf_counter()))
        counts.append(n)
    ths = [threading.Thread(target=worker, args=(x, i)) for i, x in enumerate(shards)]
    for t in ths: t.start()
    for t in ths: t.join()
    q.put((spans, sum(counts)))


def concurrent_batch_uds(nproc, threads_per, calls, b, msgs):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    total = nproc * threads_per
    shards = [msgs[i::total] for i in range(total)]
    procs = []
    for k in range(nproc):
        grp = shards[k * threads_per:(k + 1) * threads_per]
        pr = ctx.Process(target=_uds_batch_group, args=(grp, calls, b, q, sock_paths()))
        pr.start()
        procs.append(pr)
    results = [q.get() for _ in procs]
    for pr in procs:
        pr.join()
    spans = [sp for r in results for sp in r[0]]
    n = sum(r[1] for r in results)
    wall = max(e for _, e in spans) - min(st for st, _ in spans)
    return {"transport": "uds", "clients": total, "batch": b, "n": n, "msgs_s": round(n / wall, 1)}


def uds(n, msgs):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_paths()[0])
    lat = []
    t0 = time.perf_counter()
    for i in range(n):
        body = json.dumps({"sms": msgs[i % len(msgs)]}).encode()
        req = (f"POST /analyze HTTP/1.1\r\nHost: edge\r\nContent-Type: application/json\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body
        t1 = time.perf_counter()
        s.sendall(req)
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += s.recv(65536)
        head, rest = buf.split(b"\r\n\r\n", 1)
        clen = 0
        for line in head.split(b"\r\n")[1:]:
            k, _, v = line.partition(b":")
            if k.strip().lower() == b"content-length":
                clen = int(v)
        while len(rest) < clen:
            rest += s.recv(65536)
        lat.append((time.perf_counter() - t1) * 1000)
        assert head.startswith(b"HTTP/1.1 200"), head[:40]
    el = time.perf_counter() - t0
    s.close()
    return {"n": n, "msgs_s": round(n / el, 1), "rtt_ms": summ(lat)}


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ttv":
        print(json.dumps(ttv(int(sys.argv[2]))))
    elif mode == "drain":
        print(json.dumps(drain()))
    elif mode == "seq":
        print(json.dumps(seq(int(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()))))
    elif mode == "batch":
        print(json.dumps(batch(int(sys.argv[2]), int(sys.argv[3]), json.loads(Path(sys.argv[4]).read_text()))))
    elif mode == "clients":
        msgs = json.loads(Path(sys.argv[4]).read_text())
        print(json.dumps(concurrent(4, 4, int(sys.argv[3]), msgs)))
    elif mode == "batchpar":
        msgs = json.loads(Path(sys.argv[4]).read_text())
        print(json.dumps(concurrent_batch(4, 4, int(sys.argv[3]), int(sys.argv[2]), msgs)))
    elif mode == "batchuds":
        msgs = json.loads(Path(sys.argv[4]).read_text())
        print(json.dumps(concurrent_batch_uds(4, 4, int(sys.argv[3]), int(sys.argv[2]), msgs)))
    elif mode == "uds":
        print(json.dumps(uds(int(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text()))))
