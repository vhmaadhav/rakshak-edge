"""One isolated explainer call on the NEW service (verdict_first, table off for this message). stdlib only."""
import http.client, json, sys, time
from pathlib import Path
text = json.loads(Path(sys.argv[1]).read_text())[0]
out = sys.argv[2]
c = http.client.HTTPConnection("127.0.0.1", 8770, timeout=900)
c.request("POST", "/analyze", json.dumps({"sms": text}).encode(), {"Content-Type": "application/json"})
d = json.loads(c.getresponse().read())
ex = d["explanation"]
res = {"text": text[:60], "verdict_status": ex["status"]}
if ex.get("id"):
    c.request("GET", f"/explanation/{ex["id"]}?wait=900")
    doc = json.loads(c.getresponse().read())
    res["doc"] = doc
    res["llm_wall_ms"] = doc.get("wall_ms")
    nt = doc.get("new_tokens") or (doc.get("stats") or {}).get("new_tokens")
    res["new_tokens"] = nt
    if nt and doc.get("wall_ms"):
        res["tok_per_s"] = round(nt / (doc["wall_ms"] / 1000.0), 2)
Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "doc"}))
