#!/usr/bin/env python3
"""UI-only mock of the Rakshak Edge Pi service, for previewing deploy/pi/service/index.html on a laptop.

Stdlib only. Serves the page at / (re-read on every request) and returns FAKE, hard-coded JSON shaped the
way the page reads it. No model runs and none of its numbers are measurements; never quote them.

Run:  python3 deploy/pi/service/ui_mock.py    (listens on 127.0.0.1:8799)
Hot-Pi banner: include the text "[hot]" in an SMS or link to get a 503.
Verdict-first: include "[vf]" in a scam SMS to get the verdict at once and a pending explanation that
turns ready about 2 s later (GET /explanation/<id>), as the real service does with --response-mode
verdict_first.
"""
import copy
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "127.0.0.1", 8799
PAGE = str(Path(__file__).resolve().parent / "index.html")
DELAY_S = 0.4
MAX_PCAP = 20 * 1024 * 1024


def snap(temp, throttled="0x0"):
    return {"temp_c": temp, "throttled": throttled}


LANGS = ("auto", "en", "hinglish")

# FAKE template text for the mock only. Hinglish wording is illustrative, not reviewed.
T = {
    "en": {
        "kyc_point": "The message asks you to update KYC through a link.",
        "block_point": "It warns that the account will be blocked today to create pressure.",
        "url_point": "It includes a link to follow.",
        "summary": "This SMS looks like a bank KYC scam: it creates urgency and asks you to open a link.",
        "action": "Do not tap links or reply. Verify with your bank through its official app or the number on your card.",
        "rem_title": "Do not open links in an unknown SMS",
        "rem_steps": ["Do not tap the link or reply to the message.",
                      "Open your bank's app or type the bank address yourself.",
                      "Forward the SMS to your mobile operator's spam reporting number."],
        "none_point": "No rule fired for this message.",
        "none_summary": "No scam pattern was matched in this message.",
        "low_action": "No known scam pattern was found. This does not make the message safe.",
    },
    "hinglish": {
        "kyc_point": "Message KYC update karne ke liye link kholne ko kehta hai.",
        "block_point": "Yeh aaj account block hone ka dar dikhata hai taaki aap jaldi karo.",
        "url_point": "Isme follow karne ke liye ek link hai.",
        "summary": "Yeh SMS bank KYC scam jaisa lagta hai: jaldi ka dabaav banata hai aur link kholne ko kehta hai.",
        "action": "Link par tap na karein, reply na karein. Bank ki official app ya card par diye number se khud verify karein.",
        "rem_title": "Anjaan SMS ke link na kholein",
        "rem_steps": ["Link par tap ya message ka reply na karein.",
                      "Bank ki app kholein ya bank ka address khud type karein.",
                      "SMS ko mobile operator ke spam report number par forward karein."],
        "none_point": "Is message par koi rule lagu nahi hua.",
        "none_summary": "Is message mein koi scam pattern nahi mila.",
        "low_action": "Koi known scam pattern nahi mila. Isse message safe nahi ho jaata.",
    },
}


def analyse_sms(text, lang="auto"):
    low = text.lower()
    if "[hot]" in low:
        return 503, {"error": "cooling", "temp_c": 81.4, "limit_c": 80.0}
    tx = T["hinglish"] if lang == "hinglish" else T["en"]
    high = "kyc" in low or "http" in low
    evidence, points, remediation = [], [], []
    if high:
        evidence = []
        if "kyc" in low:
            i = low.index("kyc")
            evidence.append({"id": "R01", "type": "kyc_block_urgency", "weight": 0.42,
                             "text": text[i:i + 3]})
        if "blocked" in low:
            i = low.index("blocked")
            evidence.append({"id": "R04", "type": "account_suspension", "weight": 0.31,
                             "text": text[i:i + len("blocked")]})
        if "http" in low:
            i = low.index("http")
            m = re.match(r"\S+", text[i:])
            evidence.append({"id": "R07", "type": "suspicious_url", "weight": 0.27,
                             "text": m.group(0) if m else text[i:]})
        points = [
            {"source": "llm", "evidence_id": "R01", "type": "kyc_block_urgency",
             "text": tx["kyc_point"],
             "llm_text": "It asks you to update your KYC through a link in the message.",
             "check": {"ok": True, "problems": []}},
            {"source": "template", "evidence_id": "R04", "type": "account_suspension",
             "text": tx["block_point"],
             "llm_text": None, "check": {"ok": None, "problems": []}},
            {"source": "template", "evidence_id": "R07", "type": "suspicious_url",
             "text": tx["url_point"],
             "llm_text": "Open the link on the page to finish the update within 24 hours.",
             "check": {"ok": False, "problems": ["sentence cites a deadline not in the evidence"]}},
        ]
        decision = {"risk": "HIGH", "action": tx["action"],
                    "score": 0.9412, "threshold": 0.5, "scam_state": "SCAM",
                    "evidence_count": len(evidence),
                    "lang_detected": "hinglish" if lang == "hinglish" else "en"}
        timings = {"canonicalise": 0.4, "detect": 18.6, "evidence": 0.9, "llm_total": 612.4,
                   "verify": 2.1, "total": 636.0, "llm_calls": [612.4], "llm_cached": 0}
        llm_calls = [{"cached": False}]
        advisory = {"mode": "llm+template", "llm_skipped_low_risk": False, "lang": lang if lang in LANGS else "en",
                    "summary": {"text": tx["summary"], "source": "template",
                                "check": "template, not an LLM sentence"},
                    "points": points}
        remediation = [{"code": "R01", "title": tx["rem_title"], "steps": list(tx["rem_steps"])}]
    else:
        decision = {"risk": "LOW", "action": tx["low_action"],
                    "score": 0.0821, "threshold": 0.5, "scam_state": "NO_PATTERN",
                    "evidence_count": 0, "lang_detected": "hinglish" if lang == "hinglish" else "en"}
        timings = {"canonicalise": 0.3, "detect": 17.9, "evidence": 0.7, "llm_total": 0.0,
                   "verify": 0.9, "total": 19.8, "llm_calls": [], "llm_cached": 0}
        llm_calls = []
        advisory = {"mode": "template", "llm_skipped_low_risk": True, "lang": lang if lang in LANGS else "en",
                    "summary": {"text": tx["none_summary"],
                                "source": "template", "check": "pass"},
                    "points": [{"source": "template", "evidence_id": "-", "type": "none",
                                "text": tx["none_point"], "llm_text": None,
                                "check": {"ok": None, "problems": []}}]}
    full = {
        "decision": decision,
        "timings_ms": timings,
        "llm_calls": llm_calls,
        "advisory": advisory,
        "evidence": evidence,
        "verifier": {"g2_rejected": ["sentence 2 cites a deadline not in the evidence"] if high else []},
        "remediation": remediation,
        "device": {"start": snap(52.3), "end": snap(54.0),
                   "energy": {"energy_j": 3.84, "method": "ina219", "mean_w": 4.6},
                   "rss_mb": 412},
    }
    if high and "[vf]" in low:
        return 200, verdict_first(full)
    return 200, full


# FAKE verdict-first store: id -> (created, full response). Ready 2 s after the verdict.
EXPLANATIONS = {}


def verdict_first(full):
    """The verdict at once, the template advisory, and a pending explanation (real shape)."""
    eid = uuid.uuid4().hex[:16]
    EXPLANATIONS[eid] = (time.time(), full)
    adv = copy.deepcopy(full["advisory"])
    adv.update(mode="template", path="pending")
    for point in adv["points"]:
        point.update(source="template", llm_text=None, check={"ok": None, "problems": []})
    timings = dict(full["timings_ms"], llm_total=0.0, llm_calls=[], total=22.0)
    return {
        "decision": full["decision"],
        "evidence": full["evidence"],
        "remediation": full["remediation"],
        "advisory": adv,
        "timings_ms": timings,
        "verifier": {"g2_rejected": []},
        "explanation": {"status": "pending", "id": eid, "url": "/explanation/" + eid,
                        "explanation_source": "llm"},
        "device": {"start": full["device"]["start"], "rss_mb": 412},
    }


def explanation(eid):
    entry = EXPLANATIONS.get(eid)
    if entry is None:
        return 404, {"error": "unknown or expired explanation id", "id": eid}
    created, full = entry
    if time.time() - created < 2.0:
        return 200, {"id": eid, "status": "pending", "queued_at": created}
    return 200, {"id": eid, "status": "ready", "explanation_source": "llm",
                 "advisory": full["advisory"], "verifier": full["verifier"],
                 "llm_calls": full["llm_calls"], "timings_ms": full["timings_ms"],
                 "wall_ms": 612.4, "queued_at": created}


def analyse_url(url):
    low = url.lower()
    if "[hot]" in low:
        return 503, {"error": "too hot", "temp_c": 80.9, "limit_c": 80.0}
    if "kyc" in low or "xyz" in low:
        evidence = [
            {"id": "U02", "type": "suspicious_tld", "weight": 0.20, "text": ".xyz"},
            {"id": "U05", "type": "credential_keyword", "weight": 0.35, "text": "kyc"},
        ]
        return 200, {
            "verdict": "phishing", "risk": "HIGH", "url_count": 1,
            "url_model": {"domain": "demo-bank-kyc-update.xyz", "enabled": True,
                          "score": 0.9731, "threshold": 0.6},
            "evidence": evidence,
            "remediation": [{"code": "U01", "title": "Do not enter details on this page",
                             "steps": ["Close the page without entering any details.",
                                       "Type your bank's address yourself."]}],
            "timings_ms": {"total": 6.8},
        }
    return 200, {
        "verdict": "no_signal", "risk": "LOW", "url_count": 1,
        "url_model": {"domain": "demo-shop.example", "enabled": True, "score": 0.0412,
                      "threshold": 0.6},
        "evidence": [], "remediation": [], "timings_ms": {"total": 5.1},
    }


def analyse_flows(body_len):
    return 200, {
        "decision": {"risk": "HIGH", "state": "ALERT", "alerts": 1, "hosts_seen": 4},
        "flows_processed": 1280, "flow_source": "pcap", "flows_per_s": 4120.5,
        "alerts": [{
            "src": "10.0.0.23",
            "decision": {"risk": "HIGH", "state": "ALERT", "family": "scan", "p_attack": 0.9931,
                         "flagged_flows": 412, "flows_from_host": 430, "distinct_dst_ports": 960},
            "verifier": {"ok": True, "reasons": []},
            "evidence": [{"id": "N01", "type": "many_dst_ports", "weight": 0.512},
                         {"id": "N03", "type": "syn_only_flows", "weight": 0.31}],
            "advisory": {"summary": "Host 10.0.0.23 opened connections to 960 ports in a short "
                                    "window, mostly SYN only.",
                         "action": "Isolate the host and check it for malware."},
            "examples": [{"dst": "10.0.0.5", "dst_port": 22, "family": "scan",
                          "p_attack": 0.9931, "state": "ALERT"}],
            "remediation": [{"code": "N01", "title": "Isolate the scanning host",
                             "steps": ["Move the host to a quarantine VLAN.",
                                       "Check its processes and recent logins."]}],
        }],
        "wall_ms": 84.2,
        "timings_ms": {"parse": 12.1, "score": 38.4, "per_flow_p50": 0.021,
                       "per_flow_p95": 0.066, "host_alerts": 2.3,
                       "remediation_and_verify": 1.7},
        "device": {"start": snap(52.1), "end": snap(53.0),
                   "energy": {"energy_j": 0.92, "method": "ina219"}, "rss_mb": 398},
    }


HEALTH = {
    "status": "ok", "pid": 4242, "uptime_s": 913.4, "threads": 4, "temp_limit_c": 80.0,
    "models": {}, "detector_only_mode": "quiet",
    "url_model": {"enabled": True, "threshold": 0.6, "evidence_visible": True},
    "hinglish_mode": "auto", "llm_runner": "executorch", "requests": 17, "refused_cooling": 0,
    "rss_mb": 402.1, "rss_peak_mb": 440.0, "device": snap(53.1),
    "network": {"interfaces": ["lo"], "routes": 0,
                "egress_probe": "failed: Network is unreachable"},
    "explainer_cache": {"capacity": 64, "size": 3, "hits": 1, "misses": 2},
}


class Handler(BaseHTTPRequestHandler):
    server_version = "MockPi/1.0"

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, status, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(PAGE, "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if self.path == "/health":
            return self._send(200, HEALTH)
        if self.path.startswith("/explanation/"):
            status, out = explanation(self.path[len("/explanation/"):].split("?")[0])
            return self._send(status, out)
        return self._send(404, {"error": "not found"})

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n > 0 else b""

    def do_POST(self):
        body = self._body()
        time.sleep(DELAY_S)  # show the loading state
        if self.path in ("/analyze", "/analyze_url"):
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except ValueError:
                return self._send(400, {"error": "body must be JSON"})
            if self.path == "/analyze":
                lang = data.get("lang", "auto")
                if lang not in LANGS:
                    return self._send(400, {"error": "field 'lang' must be one of %s" % list(LANGS)})
                status, out = analyse_sms(str(data.get("sms", "")), lang)
            else:
                status, out = analyse_url(str(data.get("url", "")))
            return self._send(status, out)
        if self.path == "/analyze_flows":
            if len(body) > MAX_PCAP:
                return self._send(413, {"error": "pcap larger than 20 MB"})
            status, out = analyse_flows(len(body))
            return self._send(status, out)
        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("mock Pi on http://%s:%d/" % (HOST, PORT))
    srv.serve_forever()
