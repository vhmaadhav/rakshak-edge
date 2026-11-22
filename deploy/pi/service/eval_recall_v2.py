"""Recall V2 evaluation: arms R0-R4 of docs/RECALL_V2_PLAN.md (quiet mode, no LLM).

Order (the plan's honesty rules, enforced by the code path):
1. tau_url from the PhreshPhish 5 % grouped cal carve with the shipped rule, written to
   thresholds.json before any other set is scored.
2. tau_NP (arm R4) from the student scores on the pre-named Indian-legit negatives only, written to
   the same file before GroundLM / SmishX / campaign are scored.
3. Every arm on GroundLM, SmishX and campaign test; G5 on PhreshPhish official.test; url_domain
   test reported. Visible warning = action != IGNORE, exactly as eval_groundlm.py.

Run (repo root, data at data/processed):
  python deploy/pi/service/eval_recall_v2.py --student-onnx S.onnx --student-threshold T \
      --url-onnx U.onnx --url-scaler scaler.json --tranco top-1m.csv --out-dir OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi",
          REPO / "kaggle" / "jobs"):
    sys.path.insert(0, str(p))

import baseline_v0 as base  # noqa: E402
import eval_groundlm as EG  # noqa: E402
import url_threshold as UT  # noqa: E402
from edge_decision import add_url_evidence, decide_edge  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.eval import data as D  # noqa: E402
from rakshak.eval import phreshphish as P  # noqa: E402
from rakshak.evidence import Action  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402
from rakshak.url_model import UrlHit, add_url_model_evidence  # noqa: E402
from rakshak.url_reputation import TRANCO_TOP_N, ReputationGuard  # noqa: E402

ARMS = ("R0", "R1", "R2", "R3", "R4")
NP_ALPHA, NP_DELTA = 0.01, 0.05
INDIAN_LEGIT = ("cloveai_india_spam_sms", "finee_synth", "ridham115", "skyaara")
GATES = {"G1_groundlm_fp_max": 4, "G2_smishx_tp_min": 136, "G3_smishx_ham_fp_max": 34,
         "G4_campaign_fpr_max": 0.01, "G5_phresh_recall_min": 0.80}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k: int, n: int) -> list[float | None]:
    lo, hi = EG.wilson(k, n)
    return [lo, hi]


def np_order_threshold(neg_scores: np.ndarray, alpha: float, delta: float) -> dict:
    """Neyman-Pearson umbrella threshold (Tong, Feng, Li 2018): the k-th smallest negative score,
    k the smallest index with P(Bin(n, 1 - alpha) >= k) <= delta. Flag = score > threshold."""
    from scipy.stats import binom

    s = np.sort(np.asarray(neg_scores, dtype=np.float64))
    n = int(s.size)
    ks = np.arange(1, n + 1)
    viol = binom.sf(ks - 1, n, 1 - alpha)  # P(Bin >= k)
    ok = np.flatnonzero(viol <= delta)
    if ok.size == 0:
        raise ValueError("n too small for the NP order statistic at this alpha/delta")
    k = int(ks[ok[0]])
    return {"tau_np": float(s[k - 1]), "k": k, "n": n, "alpha": alpha, "delta": delta,
            "violation_prob": float(viol[ok[0]]), "flag": "score > tau_np",
            "empirical_fpr": float((s > s[k - 1]).mean())}


def url_scorer(onnx: Path, scaler_path: Path):
    scaler = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
    cols = list(scaler["columns"])
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    session = base._onnx_session(onnx)

    def score(urls):
        return UT.score_url_strings(session, list(urls), cols, mean, std)

    return score, cols


def best_hits(texts, score_urls) -> list[UrlHit | None]:
    url_lists = [list(canonicalize(str(t)).urls) for t in texts]
    uniq = sorted({u for urls in url_lists for u in urls})
    vals = dict(zip(uniq, (float(v) for v in score_urls(uniq)), strict=True)) if uniq else {}
    out = []
    for urls in url_lists:
        if not urls:
            out.append(None)
            continue
        best = max(range(len(urls)), key=lambda i: (vals[urls[i]], -i))
        out.append(UrlHit(urls[best], vals[urls[best]]))
    return out


def decide_arm(arm, text, prob, thr, hit, tau_url, guard, tau_np) -> dict:
    canon = canonicalize(text)
    ev = run_rules(canon)
    mode = "quiet"
    if arm == "R1":
        ev = add_url_evidence(ev, canon, hit, tau_url, "quiet", p_scam=prob, p_threshold=thr,
                              quiet_policy="corroborated")
    elif arm == "R2":
        ev = add_url_model_evidence(ev, canon, hit, tau_url)
    elif arm in ("R3", "R4"):
        ev = add_url_evidence(ev, canon, hit, tau_url, "quiet", quiet_policy="guarded",
                              guard=guard)
    if arm == "R4" and prob > tau_np:
        mode = "verify"  # detector-only rows become UNCERTAIN + VERIFY_INDEPENDENTLY
    dec = decide_edge(canon, ev, prob, thr, mode)
    return {"visible": dec.action != Action.IGNORE, "risk": dec.package.risk.value,
            "url_item": any(e.type == "phishing_url_model" for e in ev)}


def rate(flags: np.ndarray) -> dict:
    k, n = int(np.sum(flags)), int(np.size(flags))
    return {"k": k, "n": n, "rate": (k / n) if n else None, "wilson95": wilson(k, n)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Recall V2 arms R0-R4")
    ap.add_argument("--student-onnx", required=True)
    ap.add_argument("--student-threshold", type=float, required=True)
    ap.add_argument("--url-onnx", required=True)
    ap.add_argument("--url-scaler", required=True)
    ap.add_argument("--tranco", required=True, help="Tranco list V354N CSV (rank,domain)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = REPO / "data" / "processed"
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)  # noqa: E731

    pins = {"student_onnx": sha256(args.student_onnx), "url_onnx": sha256(args.url_onnx),
            "url_scaler": sha256(args.url_scaler), "tranco_csv": sha256(args.tranco)}
    guard = ReputationGuard.from_file(args.tranco, top_n=TRANCO_TOP_N)
    pins["tranco_top_n"] = TRANCO_TOP_N
    pins["tranco_domains_loaded"] = len(guard.domains)
    score_urls, _cols = url_scorer(Path(args.url_onnx), Path(args.url_scaler))

    # 1. tau_url (shipped rule) before anything else is scored
    data = P.load_phreshphish(root)
    keep, cal = UT.grouped_carve(data)
    urls_all = data.df["url"].astype(str).to_numpy()
    s_cal = score_urls(urls_all[cal])
    rec_url = UT.threshold_rule(data.labels[cal], s_cal)
    tau_url = rec_url["tau"]
    thresholds = {"student_threshold": args.student_threshold, "tau_url": rec_url, "pins": pins}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    log(f"tau_url {tau_url:.6f} (cal FPR {rec_url['cal_fpr']:.4f}, recall {rec_url['cal_recall']:.4f})")

    # 2. tau_NP from the Indian-legit negatives only
    student = EG.Student(Path(args.student_onnx), args.threads)
    legit = pd.concat([pd.read_parquet(root / "external" / "indian_legit" / f"{n}.parquet")
                       .assign(set=n) for n in INDIAN_LEGIT], ignore_index=True)
    legit = legit[legit["label"] == 0].reset_index(drop=True)
    legit_p = np.array([student.prob(str(t)) for t in legit["text"]])
    rec_np = np_order_threshold(legit_p, NP_ALPHA, NP_DELTA)
    rec_np["per_set_n"] = legit["set"].value_counts().to_dict()
    rec_np["fpr_at_student_threshold"] = float((legit_p >= args.student_threshold).mean())
    thresholds["tau_np"] = rec_np
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    log(f"tau_np {rec_np['tau_np']:.6f} (k {rec_np['k']} of {rec_np['n']})")

    # 3. the three message sets
    gl = EG.select_groundlm(pd.read_parquet(EG.GROUNDLM_PATH), 0).reset_index(drop=True)
    sx = EG.select_smishx(pd.read_parquet(EG.SMISHX_PATH), 0).reset_index(drop=True)
    cp = EG.select_campaign(0).reset_index(drop=True)
    sets = {"groundlm": gl, "smishx": sx, "campaign_test": cp}
    rows = {}
    for name, frame in sets.items():
        texts = frame["text"].astype(str).tolist()
        probs = np.array([student.prob(t) for t in texts])
        hits = best_hits(texts, score_urls)
        rec = {"prob": probs,
               "url_score": np.array([h.score if h else np.nan for h in hits]),
               "url_established": np.array([bool(h) and guard.established(h.url) for h in hits])}
        for arm in ARMS:
            ds = [decide_arm(arm, t, float(p), args.student_threshold, h, tau_url, guard,
                             rec_np["tau_np"]) for t, p, h in zip(texts, probs, hits, strict=True)]
            rec[f"{arm}_visible"] = np.array([d["visible"] for d in ds])
        rows[name] = rec
        log(f"scored {name}: {len(texts)} rows")

    neg = (gl["policy_negative"].fillna(False).astype(bool)
           & gl["eval_ok"].fillna(False).astype(bool)).to_numpy()
    gl_fraud = ((~gl["policy_negative"].fillna(False).astype(bool))
                & gl["eval_ok"].fillna(False).astype(bool)).to_numpy()
    sx_scam = (sx["label_fine"] == "smishing").to_numpy()
    sx_ham = (sx["label_fine"] == "legitimate").to_numpy()
    cp_pos = (cp["label_binary"] == 1).to_numpy()
    cp_neg = (cp["label_binary"] == 0).to_numpy()
    arms = {}
    for arm in ARMS:
        g, s, c = (rows["groundlm"][f"{arm}_visible"], rows["smishx"][f"{arm}_visible"],
                   rows["campaign_test"][f"{arm}_visible"])
        arms[arm] = {
            "groundlm_negatives_visible": rate(g[neg]),
            "groundlm_fraud_visible_recall": rate(g[gl_fraud]),
            "smishx_smishing_visible_recall": rate(s[sx_scam]),
            "smishx_legit_visible_fpr": rate(s[sx_ham]),
            "campaign_visible_recall": rate(c[cp_pos]),
            "campaign_visible_fpr": rate(c[cp_neg]),
        }

    # 4. G5 and url_domain, with and without the guard
    test = data.test
    s_test = score_urls(urls_all[test])
    y_test = data.labels[test]
    est_test = np.array([guard.established(u) for u in urls_all[test]])
    flag = s_test >= tau_url
    phresh = {
        "phish_recall_no_guard": rate(flag[y_test == 1]),
        "phish_recall_guard": rate((flag & ~est_test)[y_test == 1]),
        "benign_fpr_no_guard": rate(flag[y_test == 0]),
        "benign_fpr_guard": rate((flag & ~est_test)[y_test == 0]),
        "phish_established_share": rate(est_test[y_test == 1]),
        "benign_established_share": rate(est_test[y_test == 0]),
    }
    ud = D.load_urls(root)
    ud_labels = ud["label"].to_numpy().astype(np.int64)
    run_ud = next(r for r in D.load_runs(root, ud, patterns=["url_domain"], labels=ud_labels,
                                         mask=None) if r.name == "url_domain")
    ud_test = np.asarray(run_ud.test, dtype=np.int64)
    ud_urls = ud["url"].astype(str).to_numpy()[ud_test]
    s_ud, y_ud = score_urls(ud_urls), ud_labels[ud_test]
    est_ud = np.array([guard.established(u) for u in ud_urls])
    url_domain = {
        "benign_fpr_no_guard": rate((s_ud >= tau_url)[y_ud == 0]),
        "benign_fpr_guard": rate(((s_ud >= tau_url) & ~est_ud)[y_ud == 0]),
        "phish_recall_no_guard": rate((s_ud >= tau_url)[y_ud == 1]),
        "phish_recall_guard": rate(((s_ud >= tau_url) & ~est_ud)[y_ud == 1]),
        "benign_established_share": rate(est_ud[y_ud == 0]),
        "note": "circularity caveat: benign URLs may come from popularity lists (plan §4)",
    }

    r3 = arms["R3"]
    gates = {
        "G1": r3["groundlm_negatives_visible"]["k"] <= GATES["G1_groundlm_fp_max"],
        "G2": r3["smishx_smishing_visible_recall"]["k"] >= GATES["G2_smishx_tp_min"],
        "G3": r3["smishx_legit_visible_fpr"]["k"] <= GATES["G3_smishx_ham_fp_max"],
        "G4": (r3["campaign_visible_fpr"]["rate"] or 0) <= GATES["G4_campaign_fpr_max"],
        "G5": (phresh["phish_recall_guard"]["rate"] or 0) >= GATES["G5_phresh_recall_min"],
    }
    # False-positive buckets for R3 on GroundLM (domains only, no message text).
    fp_rows = np.flatnonzero(rows["groundlm"]["R3_visible"] & neg)
    fp_domains = []
    for i in fp_rows:
        urls = canonicalize(str(gl["text"].iloc[i])).urls
        fp_domains.append(guard.check(urls[0]).registrable if urls else "")
    metrics = {"plan": "docs/RECALL_V2_PLAN.md", "thresholds": thresholds, "gate_limits": GATES,
               "arms": arms, "phreshphish_test": phresh, "url_domain_test": url_domain,
               "R3_gates": gates, "R3_pass": all(gates.values()),
               "R3_groundlm_fp_first_url_domains": fp_domains,
               "elapsed_s": round(time.time() - t0, 1)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    pd.DataFrame({f"{n}_{k}": pd.Series(v) for n, rec in rows.items() for k, v in rec.items()}
                 ).to_parquet(out / "row_flags.parquet")
    for arm in ARMS:
        a = arms[arm]
        log(f"{arm}: GL FP {a['groundlm_negatives_visible']['k']}/{a['groundlm_negatives_visible']['n']}"
            f"  SmishX TP {a['smishx_smishing_visible_recall']['k']}/{a['smishx_smishing_visible_recall']['n']}"
            f"  SmishX ham FP {a['smishx_legit_visible_fpr']['k']}/{a['smishx_legit_visible_fpr']['n']}"
            f"  camp recall {a['campaign_visible_recall']['rate']:.4f} FPR {a['campaign_visible_fpr']['rate']:.4f}")
    log(f"G5 phresh recall guard {phresh['phish_recall_guard']['rate']:.4f} "
        f"(no guard {phresh['phish_recall_no_guard']['rate']:.4f}); gates {gates}")
    return 0 if math.isfinite(tau_url) else 1


if __name__ == "__main__":
    raise SystemExit(main())
