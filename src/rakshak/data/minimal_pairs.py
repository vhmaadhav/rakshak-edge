"""Minimal scam twins of legitimate SMS templates (docs/SMS_RETRAIN_PLAN.md, section 3, arm B).

Each selected legit template gets one deterministic scam edit (an appended sentence) from one
of six families. Family assignment is round-robin over the selection order, and the sentence choice
comes from a seeded RNG, so the same input and seed give identical output.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# family -> (rakshak.rules evidence type it should fire, candidate scam sentences)
FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "credential_otp_request": (
        "credential_request",
        (
            "Share your OTP to confirm this transaction.",
            "Please send your password and PIN to verify the account.",
            "Reply with your CVV and OTP to continue.",
        ),
    ),
    "kyc_block_urgency": (
        "kyc_block_urgency",
        (
            "Your account will be blocked today. Complete KYC now.",
            "KYC pending: account blocked within 24 hours, update immediately.",
        ),
    ),
    "fee_prize_claim": (
        "prize_lottery",
        (
            "Congratulations, you won a lottery prize of Rs 50000. Pay Rs 199 fee to claim.",
            "You are selected as a lucky winner. Pay processing fee to claim your reward.",
        ),
    ),
    "suspicious_short_link": (
        "shortened_url",
        (
            "Verify now at http://bit.ly/3xYz9Q",
            "Update details here: tinyurl.com/kyc-verify",
        ),
    ),
    "remote_access_install": (
        "remote_access_app",
        (
            "Install AnyDesk app to resolve this issue.",
            "Download TeamViewer quicksupport to continue.",
        ),
    ),
    "upi_collect_request": (
        "upi_collect_or_pay",
        (
            "Approve the UPI collect request of Rs 5000 to receive your refund.",
            "Accept the collect request in your UPI app to get the cashback.",
        ),
    ),
}
FAMILY_NAMES = tuple(FAMILIES)


def _sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def make_minimal_pairs(legit: pd.DataFrame, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Scam twins for up to ``n`` legit templates, one per cluster first.

    ``legit`` needs columns text, label (0 rows are used), cluster and optionally id.
    Output columns: id, text, label (1), source ("minimal_pair"), cluster (the twin's cluster),
    pair_of (the twin's id), edit_family. The twins must be in the training set.
    """
    cols = ["id", "text", "label", "source", "cluster", "pair_of", "edit_family"]
    src = legit[legit["label"] == 0].copy()
    if "id" not in src.columns:
        src["id"] = [_sha(t) for t in src["text"]]
    src = src.sort_values(["id", "text"], kind="stable").reset_index(drop=True)
    if n <= 0 or src.empty:
        return pd.DataFrame(columns=cols)

    rng = np.random.default_rng(seed)
    # round-robin over clusters: one row per cluster per round, cluster order shuffled by seed
    clusters = sorted(src["cluster"].unique().tolist())
    order = rng.permutation(len(clusters))
    per_cluster = {c: [] for c in clusters}
    for idx, c in enumerate(src["cluster"].tolist()):
        per_cluster[c].append(idx)
    for c in clusters:
        per_cluster[c] = [per_cluster[c][k] for k in rng.permutation(len(per_cluster[c]))]
    chosen: list[int] = []
    depth = 0
    while len(chosen) < n:
        advanced = False
        for k in order:
            members = per_cluster[clusters[k]]
            if depth < len(members):
                chosen.append(members[depth])
                advanced = True
                if len(chosen) >= n:
                    break
        if not advanced:
            break
        depth += 1

    rows = []
    fam_names = FAMILY_NAMES
    for pos, idx in enumerate(chosen):
        twin = src.iloc[idx]
        family = fam_names[pos % len(fam_names)]
        sentences = FAMILIES[family][1]
        sentence = sentences[int(rng.integers(len(sentences)))]
        text = f"{twin['text'].rstrip()} {sentence}"
        rows.append(
            {
                "id": _sha(f"minimal_pair|{twin['id']}|{family}"),
                "text": text,
                "label": 1,
                "source": "minimal_pair",
                "cluster": twin["cluster"],
                "pair_of": twin["id"],
                "edit_family": family,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def evidence_fire_rates(pairs: pd.DataFrame) -> dict[str, dict]:
    """Share of each family's edited texts whose rakshak.rules output has the intended type."""
    from rakshak.canonicalize import canonicalize
    from rakshak.rules import run_rules

    out: dict[str, dict] = {}
    for family, (etype, _) in FAMILIES.items():
        sub = pairs[pairs["edit_family"] == family]
        hits = 0
        for text in sub["text"]:
            types = {ev.type for ev in run_rules(canonicalize(text))}
            hits += int(etype in types)
        out[family] = {
            "evidence_type": etype,
            "n": int(len(sub)),
            "fired": hits,
            "fire_rate": (hits / len(sub)) if len(sub) else None,
        }
    return out
