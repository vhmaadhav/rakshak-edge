#!/usr/bin/env python3
"""Write parity_fixture.json: Python ground truth for the Kotlin port.

Usage (VM, repo on PYTHONPATH, data dir as argument):
    PYTHONPATH=<repo>/src python make_parity_fixture.py <data_dir> <out.json>

Rows: a seeded stratified sample of the SMS campaign TEST split (data/sms.parquet, partition
"test" of data/splits/campaign.json), the GroundLM and SmishX external sets (text present only),
and hand-made obfuscation edge cases. For every row this records what the Kotlin side must
reproduce exactly:

  canonical      canonicalize(text).normalized
  urls           canonicalize(text).urls
  evidence       run_rules(canon): id, type, weight, matched text (ids E1..En in order)
  flags_canon    16 side flags over the canonical text, in EVIDENCE_TYPES order
  flags_raw      16 side flags over the unfolded text (as encoding.featurize builds them)
  raw_ids        UTF-8 bytes of text, truncated to 256, padded with 256
  canon_ids      UTF-8 bytes of canonical text, same encoding
  url_info       per URL: registered domain and the url_features values the rules read
  lang           detect_lang(text)
  facts          fact_text per evidence item (prompts.fact_slot wording)
  prompts        Llama describe_kw prompt per evidence item (prefix and full ChatML text)
  packages       per state (scam, uncertain, benign): risk, allowed actions, chosen action,
                 template summary (English)

No model is needed. Text is cut at 400 characters for the fixture (text_truncated says so), but
the byte ids and canonical form are computed from the cut text, so the Kotlin side must cut the
same way before encoding.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

from rakshak.canonicalize import Canon, canonicalize, extract_urls
from rakshak.datasets.lang import detect_lang
from rakshak.evidence import Action, build_package, choose_action
from rakshak.models.encoding import MAX_LEN, PAD_ID
from rakshak.prompts import fact_slot
from rakshak.rules import EVIDENCE_TYPES, run_rules
from rakshak.url_features import brand_domain_mismatch, registered_domain, url_features
from rakshak.verifier import TYPE_PHRASE, _template_summary

SEED = 13
TEXT_CAP = 400
BOS, SOT, EOT_HDR, EOT = (
    "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
)
DESCRIBE_SYSTEM = (
    "You describe what a text message says or asks the reader to do. You are given one fact "
    "about the message. Write one short plain-English sentence that restates the fact. Do not "
    "judge the message or its sender. Use only the fact. Output only the sentence."
)
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_SLOT_LEAD = "This message"

HAND_MADE = [
    "Your SBI A/c is blocked, KYC​ pending. Update now: sbi​-kyc.in/upd",
    "Dear customer, your аccount will be suspended. Verify at sbi-login.com",
    "Y0ur 0TP is 4829. sh4re it with us 2 verify",
    "आपका OTP साझा करें, खाता बंद होगा",
    "Aapka khata band ho jayega, turant KYC karein: http://sbi-kyc.xn--80ak6aa92e.com/u",
    "Ｙｏｕ ｗｏｎ ａ ｌｏｔｔｅｒｙ! claim at ｈｔｔｐ://ｐｒｉｚｅ.ｃｏｍ",
    "\U0001D41A\U0001D41B\U0001D41C you won a prize, click here bit.ly/xyz",
    "K Y C update pending, visit s.b.i-secure[.]in",
    "hxxps://paytm[.]verify-kyc.top/login now",
    "Parcel held, pay Rs 49 customs fee to 9876543210",
    "Part-time job! earn 3000 per day, like youtube videos",
    "Guaranteed 20% daily returns on crypto trading group",
    "CBI digital arrest, video call now, money laundering case",
    "Install the AnyDesk app from http://192.168.4.7/a.apk",
    "Hello from user@example.com see http://user@bank.com/login",
    "",
    "a" * 300 + " OTP share karo",
    "ॐ" * 100,
    "\U0001F389\U0001F389 You have won a lottery \U0001F389 send OTP",
    "Bank‮no.elgoog",
    "Meeting at 5pm tomorrow, see you",
    "Zoom link: https://zoom.us/j/12345",
    "Your order #A12345 delivered. Track: amazon.in/track",
    "UPI collect request of Rs 500 from raju, approve to receive",
    "सरकारी योजना के लिए पंजीकरण करें, पैसे लेने के लिए UPI पिन डालें",
    "Turant bijli kat jayegi, KYC aaj hi update karein",
    "Abhi aa rahe hain, khana tayyar hai",
    "Dear customer ­SBI­ account suspended",
    "İ Istanbul",
    "sbi.co.in",
    "Visit www.google.com or go to https://g.co/x",
    "http://1.2.3.4/login verify your account",
    "Stock tips: double your money in 7 days, invest now",
    "Congratulations! Your KBC lottery winner prize 25 lakh, call 8888888888",
    "Do not share your OTP with anyone",
    "नमस्ते, आज मौसम अच्छा है",
    "Click: http://sbi.co.in.evil.xyz/verify",
    "Click: https://www.sbi.co.in/ to login",
    "Pay ₹ 1,200 to activate your card, verify now",
]


def sample_sms(data: Path) -> list[dict]:
    campaign = json.loads((data / "splits" / "campaign.json").read_text())
    test_ids = set(campaign["partitions"]["test"])
    df = pd.read_parquet(
        data / "sms.parquet", columns=["id", "text", "label_binary", "lang", "campaign_id"]
    )
    df = df[df["id"].isin(test_ids) & df["text"].notna()]
    picked = []
    for label in (0, 1):
        picked += df[df["label_binary"] == label].sample(n=75, random_state=SEED).to_dict("records")
    for lang in ("hinglish", "hi_deva"):
        picked += df[df["lang"] == lang].sample(n=15, random_state=SEED).to_dict("records")
    return [
        {"id": f"sms:{r['id']}", "source": "sms_campaign_test", "text": str(r["text"]),
         "label": int(r["label_binary"])}
        for r in picked
    ]


def sample_external(path: Path, tag: str, n: int) -> list[dict]:
    df = pd.read_parquet(path, columns=["id", "text", "label_binary"])
    df = df[df["text"].notna()]
    picked = df.sample(n=min(n, len(df)), random_state=SEED).to_dict("records")
    return [
        {"id": f"{tag}:{r['id']}", "source": tag, "text": str(r["text"]),
         "label": int(r["label_binary"])}
        for r in picked
    ]


def encode(text: str) -> list[int]:
    raw = text.encode("utf-8", errors="replace")[:MAX_LEN]
    out = [PAD_ID] * MAX_LEN
    for i, b in enumerate(raw):
        out[i] = b
    return out


def side_flags(canon: Canon) -> list[int]:
    fired = {ev.type for ev in run_rules(canon)}
    return [1 if t in fired else 0 for t in EVIDENCE_TYPES]


def fact_text(evidence) -> str:
    phrase = TYPE_PHRASE["en"][evidence.type]
    phrase = phrase[:1].upper() + phrase[1:]
    slot = fact_slot(evidence)
    return f'{phrase}. Matched text: "{slot}"' if slot else f"{phrase}."


def describe_kw_prompt(evidence) -> dict:
    clause = " ".join(_PAREN_RE.sub("", TYPE_PHRASE["en"][evidence.type]).split())
    prefix = f"{_SLOT_LEAD} {clause} and"
    user = f"FACT: {fact_text(evidence)}\n\nWrite the sentence for this fact."
    prompt = (
        f"{BOS}{SOT}system{EOT_HDR}\n\n{DESCRIBE_SYSTEM}{EOT}"
        f"{SOT}user{EOT_HDR}\n\n{user}{EOT}"
        f"{SOT}assistant{EOT_HDR}\n\n{prefix}"
    )
    return {"type": evidence.type, "prefix": prefix, "prompt": prompt}


def url_info(url: str, text: str) -> dict:
    f = url_features(url)
    return {
        "url": url,
        "registered_domain": registered_domain(url),
        "brand_domain_mismatch": brand_domain_mismatch(text, url),
        "tld_risky": f["tld_risky"] == 1.0,
        "punycode": f["punycode"] == 1.0,
        "ip_literal": f["ip_literal"] == 1.0,
        "has_at_sign": f["has_at_sign"] == 1.0,
        "brand_host_mismatch": f["brand_host_mismatch"] == 1.0,
        "suspicious_token_count": int(f["suspicious_token_count"]),
        "n_subdomains": int(f["n_subdomains"]),
    }


def build_row(item: dict) -> dict:
    text = item["text"]
    cut = text[:TEXT_CAP]
    canon = canonicalize(cut)
    raw_canon = Canon(original=cut, normalized=cut, urls=extract_urls(cut))
    evidence = run_rules(canon)
    packages = {}
    for state in ("scam", "uncertain", "benign"):
        pkg = build_package(canon, evidence, state)
        packages[state] = {
            "risk": pkg.risk.value,
            "allowed_actions": [a.value for a in pkg.allowed_actions],
            "action": choose_action(pkg).value,
            "summary": _template_summary(pkg, "en"),
        }
    return {
        **item,
        "text": cut,
        "text_truncated": len(text) > TEXT_CAP,
        "lang": detect_lang(cut),
        "canonical": canon.normalized,
        "urls": canon.urls,
        "evidence": [
            {"id": e.id, "type": e.type, "weight": e.weight, "text": e.text} for e in evidence
        ],
        "flags_canon": side_flags(canon),
        "flags_raw": side_flags(raw_canon),
        "raw_ids": encode(cut),
        "canon_ids": encode(canon.normalized),
        "url_info": [url_info(u, canon.normalized) for u in canon.urls],
        "facts": [fact_text(e) for e in evidence],
        "prompts": [describe_kw_prompt(e) for e in evidence],
        "packages": packages,
    }


def main() -> int:
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    items = []
    items += sample_sms(data)
    items += sample_external(data / "external" / "external_groundlm.parquet", "groundlm", 50)
    items += sample_external(data / "external" / "external_smishx.parquet", "smishx", 50)
    items += [
        {"id": f"hand:{i:02d}", "source": "hand_made", "text": t, "label": -1}
        for i, t in enumerate(HAND_MADE)
    ]
    seen, rows = set(), []
    for item in items:
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        rows.append(build_row(item))
    doc = {
        "schema": 1,
        "seed": SEED,
        "text_cap": TEXT_CAP,
        "max_len": MAX_LEN,
        "pad_id": PAD_ID,
        "evidence_types": list(EVIDENCE_TYPES),
        "actions": [a.value for a in Action],
        "rows": rows,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"rows={len(rows)} bytes={out.stat().st_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
