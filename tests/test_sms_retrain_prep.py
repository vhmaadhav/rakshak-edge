"""Tests for the SMS retrain data prep (docs/SMS_RETRAIN_PLAN.md, sections 2-4). Synthetic only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rakshak.data.minimal_pairs import evidence_fire_rates, make_minimal_pairs
from rakshak.data.retrain_prep import (
    PER_GROUP_CAP,
    build_pools,
    normalize,
    threshold_rule,
    two_level_labels,
)


def _row(text: str, label: int, source: str) -> dict:
    return {"id": f"{source}:{text}", "text": text, "label": label, "source": source}


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("Int64")
    return df


TEMPLATE = "Your A/c XX1234 debited Rs 500 on 12-08-26 at the grocery shop for the weekly groceries"
NEAR = "Your A/c XX1234 debited Rs 500 on 12-08-26 at the grocery store for the weekly groceries"


def test_normalize_urls_digits_and_whitespace():
    assert (
        normalize("  HDFC  Bank:  Rs 500\n\nvisit HTTPS://Example.com/x?a=1 ")
        == "hdfc bank: rs 000 visit <url>"
    )
    assert normalize("Pay at www.bank.in now") == "pay at <url> now"
    assert normalize("Call 18002586161") == "call 00000000000"


def test_near_duplicate_removal_drops_whole_cluster():
    finee = _frame(
        [
            _row(TEMPLATE, 0, "finee_synth"),
            _row(NEAR, 0, "finee_synth"),
            _row(
                "Flight booking confirmed for Mumbai on Friday, check in online",
                0,
                "finee_synth",
            ),
        ]
    )
    # the test text equals TEMPLATE after normalisation, so the whole TEMPLATE/NEAR cluster goes
    res = build_pools("A", {"finee_synth": finee}, [normalize(TEMPLATE)], 10_000, seed=0)
    texts = set(res.train_extra["text"]) | set(res.legit_val["text"])
    assert TEMPLATE not in texts and NEAR not in texts
    assert "Flight booking confirmed for Mumbai on Friday, check in online" in texts
    assert res.report["stages"]["split_groups_dropped_test_contaminated"] == 1


def test_test_only_texts_never_survive():
    rows = [
        _row(f"Parcel {i} delivered at your door step number {i} thank you", 0, "finee_synth")
        for i in range(30)
    ]
    finee = _frame(rows)
    test_texts = [rows[k]["text"] for k in (0, 7, 19)]
    res = build_pools("A", {"finee_synth": finee}, test_texts, 10_000, seed=3)
    survivors = set(res.train_extra["text"]) | set(res.legit_val["text"])
    assert not survivors & set(test_texts)


def test_per_cluster_cap_and_total_cap():
    # 12 rows that normalise to one template (digits differ), one cluster
    cluster_rows = [
        _row(
            f"OTP for txn of Rs.{100 + i} at AMAZON is {200000 + i}. Do not share", 0, "finee_synth"
        )
        for i in range(12)
    ]
    # 60 unrelated templates, one row each, distinct clusters
    others = [
        _row(
            f"Unique notice {i} about {chr(65 + i % 26)}{chr(97 + (i * 7) % 26)} item {i * 13}",
            0,
            "finee_synth",
        )
        for i in range(60)
    ]
    finee = _frame(cluster_rows + others)
    res = build_pools("A", {"finee_synth": finee}, [], 40, seed=1)  # budget = floor(0.25 * 40) = 10
    rows_per_cluster = pd.concat([res.train_extra, res.legit_val]).groupby("cap_group").size()
    assert rows_per_cluster.max() <= PER_GROUP_CAP
    assert len(res.train_extra[res.train_extra["label"] == 0]) <= 10
    assert res.report["stages"]["total_cap_budget"] == 10


def test_group_disjoint_train_and_validation():
    rows = [
        _row(
            f"Mail {i % 9} from bank branch {i} regarding account service request {i * 3}",
            0,
            "finee_synth",
        )
        for i in range(120)
    ]
    rows += [
        _row("Skyaara seed sentence about a refund credited to your wallet balance", 0, "skyaara"),
        _row("Ridham safe message about a loan EMI due on the first day", 0, "ridham115"),
        _row("Ridham scam message asking to verify KYC within a day", 1, "ridham115"),
    ]
    sources = {
        "finee_synth": _frame(rows[:120]),
        "skyaara": _frame([rows[120]]),
        "ridham115": _frame(rows[121:]),
    }
    res = build_pools("A", sources, [], 10_000, seed=5)
    train_clusters = set(res.train_extra["cluster"])
    assert train_clusters.isdisjoint(set(res.legit_val["cluster"]))
    assert train_clusters.isdisjoint(set(res.val_scam["cluster"]))
    assert res.report["stages"]["group_disjoint_train_vs_val"] is True
    assert set(res.val_scam["label"]) == {1}
    assert set(res.legit_val["label"]) == {0}


def test_arm_c_drops_groundlm_touching_clusters():
    clove = _frame(
        [
            _row(
                "Congratulations you won a prize claim now at example", 1, "cloveai_india_spam_sms"
            ),
            _row(
                "Congratulations you won a prize claim now at example site",
                1,
                "cloveai_india_spam_sms",
            ),
            _row(
                "Meeting moved to 3 PM in conference room B today please",
                0,
                "cloveai_india_spam_sms",
            ),
        ]
    )
    finee = _frame(
        [
            _row(
                "Passport appointment confirmed for the morning slot at the centre",
                0,
                "finee_synth",
            )
        ]
    )
    groundlm = ["Congratulations you won a prize claim now at example"]
    res = build_pools(
        "C",
        {"finee_synth": finee, "cloveai_india_spam_sms": clove},
        [],
        10_000,
        seed=2,
        groundlm_texts=groundlm,
    )
    texts = set(res.train_extra["text"])
    assert "Congratulations you won a prize claim now at example" not in texts
    assert "Congratulations you won a prize claim now at example site" not in texts
    assert "Meeting moved to 3 PM in conference room B today please" in texts
    assert res.report["stages"]["arm_c_clove_groundlm_split_groups_dropped"] == 1


def test_threshold_rule_meets_both_limits_and_takes_no_test_input():
    rng = np.random.default_rng(0)
    s_val = rng.uniform(0.0, 0.2, 500)
    s_cal = np.concatenate([rng.uniform(0.0, 0.3, 400), rng.uniform(0.7, 1.0, 100)])
    y_cal = np.concatenate([np.zeros(400, int), np.ones(100, int)])
    tau, fallback = threshold_rule(s_val, s_cal, y_cal)
    assert fallback is False
    assert np.mean(s_val >= tau) <= 0.01
    assert np.mean(s_cal[y_cal == 0] >= tau) <= 0.02
    # smallest: the previous candidate score (largest observed score below tau) must break one limit
    candidates = np.unique(np.concatenate([s_val, s_cal]))
    lower = candidates[candidates < tau].max()
    assert np.mean(s_val >= lower) > 0.01 or np.mean(s_cal[y_cal == 0] >= lower) > 0.02


def test_threshold_rule_fallback_flag_when_no_threshold_passes():
    s_val = np.full(100, 0.995)
    s_cal = np.concatenate([np.full(50, 0.99), np.full(5, 0.99)])
    y_cal = np.concatenate([np.zeros(50, int), np.ones(5, int)])
    tau, fallback = threshold_rule(s_val, s_cal, y_cal)
    assert fallback is True
    assert tau == 0.99


def test_minimal_pairs_determinism_and_pairing():
    legit = _frame(
        [
            _row(
                f"Bill {i} of Rs {i * 11} due on the {i + 1} day of the month, pay at the office",
                0,
                "finee_synth",
            )
            for i in range(12)
        ]
    )
    legit["id"] = [f"id{i}" for i in range(12)]
    legit["cluster"] = list(range(12))
    a = make_minimal_pairs(legit, n=12, seed=4)
    b = make_minimal_pairs(legit, n=12, seed=4)
    assert a.equals(b)
    assert set(a["label"]) == {1} and set(a["source"]) == {"minimal_pair"}
    twins = legit.set_index("id")
    for _, r in a.iterrows():
        assert r["pair_of"] in twins.index
        assert r["cluster"] == twins.loc[r["pair_of"], "cluster"]
        assert r["text"].startswith(twins.loc[r["pair_of"], "text"].rstrip())
        assert r["text"] != twins.loc[r["pair_of"], "text"]
    assert set(a["edit_family"]) == {
        "credential_otp_request",
        "kyc_block_urgency",
        "fee_prize_claim",
        "suspicious_short_link",
        "remote_access_install",
        "upi_collect_request",
    }


def test_minimal_pair_edits_fire_intended_evidence():
    legit = _frame(
        [
            _row(
                f"Your order {i} for Rs {i * 7} from the shop is out for delivery today",
                0,
                "finee_synth",
            )
            for i in range(12)
        ]
    )
    legit["id"] = [f"id{i}" for i in range(12)]
    legit["cluster"] = list(range(12))
    pairs = make_minimal_pairs(legit, n=12, seed=0)
    rates = evidence_fire_rates(pairs)
    assert all(v["fire_rate"] == 1.0 for v in rates.values())


def _variant_texts(n_templates: int = 8, per_template: int = 10) -> list[str]:
    """Templates with a few word swaps each, so split (0.7) and cap (0.9) groups differ."""
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    out = []
    for t in range(n_templates):
        base = [f"notice{t}", "your", "account", "statement", "for", "period", "ends", "soon"]
        for v in range(per_template):
            words_v = list(base)
            words_v[(v % len(base))] = words[(v + t) % len(words)]
            if v >= len(base):
                words_v[(v * 3) % len(base)] = words[(v * 5) % len(words)]
            out.append(" ".join(words_v) + f" reference {t} {v}")
    return out


def test_cap_groups_nest_inside_split_groups():
    texts = [normalize(t) for t in _variant_texts()]
    stats: dict = {}
    split, cap = two_level_labels(texts, stats)
    assert stats["cap_groups_spanning_split_groups"] == 0
    for c in set(cap.tolist()):
        assert len(set(split[cap == c].tolist())) == 1
    assert len(set(cap.tolist())) >= len(set(split.tolist()))


def test_split_groups_disjoint_between_train_and_legit_val():
    rows = [
        _row(
            f"Notice {i}: item {chr(65 + i % 26)}{chr(97 + i // 26)} at site {i * 37} today",
            0,
            "finee_synth",
        )
        for i in range(60)
    ]
    sources = {"finee_synth": _frame(rows)}
    res = build_pools("A", sources, [], 10_000, seed=7, split_jaccard=0.7, cap_jaccard=0.9)
    assert res.report["stages"]["finee_split_groups_held_out"] >= 1
    assert res.report["stages"]["group_disjoint_train_vs_val"] is True
    assert set(res.train_extra["cluster"]).isdisjoint(set(res.legit_val["cluster"]))
    assert len(res.legit_val) > 0


def test_cap_applied_per_cap_group_including_validation():
    rows = [_row(t, 0, "finee_synth") for t in _variant_texts(6, 12)]
    sources = {"finee_synth": _frame(rows)}
    res = build_pools("A", sources, [], 10_000, seed=3, split_jaccard=0.7, cap_jaccard=0.9)
    both = pd.concat([res.train_extra, res.legit_val, res.val_scam])
    assert both.groupby("cap_group").size().max() <= PER_GROUP_CAP
    assert res.report["stages"]["max_rows_per_cap_group"] <= PER_GROUP_CAP
