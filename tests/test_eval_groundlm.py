"""Counting, Wilson CI and decision mirror in deploy/pi/service/eval_groundlm.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eval_groundlm", REPO / "deploy/pi/service/eval_groundlm.py"
)
E = importlib.util.module_from_spec(SPEC)
sys.modules["eval_groundlm"] = E
SPEC.loader.exec_module(E)


def test_wilson_matches_hand_value():
    # 1 of 10 at 95%: Wilson lower/upper from the closed form
    lo, hi = E.wilson(1, 10)
    assert lo == pytest.approx(0.01787, abs=1e-4)
    assert hi == pytest.approx(0.40418, abs=1e-4)
    assert E.wilson(0, 0) == (None, None)


def test_rate_row_counts_and_empty():
    row = E.rate_row("x", np.array([True, False, False, False]))
    assert (row["k"], row["n"]) == (1, 4)
    assert row["rate"] == pytest.approx(0.25)
    assert row["ci_lo"] < 0.25 < row["ci_hi"]
    empty = E.rate_row("x", np.array([], dtype=bool))
    assert empty["rate"] is None and empty["ci_lo"] is None


def test_meets_one_percent_is_inclusive():
    # 11/1100 = 1.0% is at the line (<=), 12/1100 is above it
    assert E.rate_row("x", np.arange(1100) < 11)["meets_1pct"] is True
    assert E.rate_row("x", np.arange(1100) < 12)["meets_1pct"] is False


def _frame(prob, n_ev, risk, action):
    return pd.DataFrame(
        {
            "prob": prob,
            "n_evidence": n_ev,
            "risk": risk,
            "action": action,
            "types": [""] * len(prob),
            "primary_type": [""] * len(prob),
            "template_key": ["t"] * len(prob),
            "category": ["Other"] * len(prob),
        }
    )


def test_alert_definitions_are_nested_where_expected():
    df = _frame(
        prob=[0.9, 0.1, 0.1, 0.1],
        n_ev=[0, 1, 1, 0],
        risk=["UNCERTAIN", "LOW", "HIGH", "LOW"],
        action=["VERIFY_INDEPENDENTLY", "IGNORE", "DO_NOT_CLICK", "IGNORE"],
    )
    f = E.alert_flags(df, threshold=0.5)
    assert f["a_student_score_ge_threshold"].tolist() == [True, False, False, False]
    assert f["b_any_rule_evidence"].tolist() == [False, True, True, False]
    assert f["c_risk_ge_MEDIUM"].tolist() == [False, False, True, False]
    assert f["d_risk_ge_UNCERTAIN"].tolist() == [True, False, True, False]
    assert f["e_risk_HIGH_only"].tolist() == [False, False, True, False]
    assert f["f_action_not_IGNORE"].tolist() == [True, False, True, False]


def test_min_threshold_flags_at_most_k_and_is_tight():
    neg = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    out = E.min_threshold_for_fpr(neg, target=0.2)  # k = floor(0.2 * 10) = 2
    assert out["k_allowed"] == 2
    assert int((neg >= out["threshold"]).sum()) == 2
    # one step lower would flag more than k
    assert int((neg >= np.nextafter(out["threshold"], -np.inf)).sum()) > 2


def test_min_threshold_with_ties_counts_strictly_above():
    neg = np.array([0.9, 0.9, 0.9, 0.1])
    out = E.min_threshold_for_fpr(neg, target=0.5)  # k = 2, three ties at 0.9
    assert out["fp"] <= 2


def test_count_check_and_masks():
    df = pd.DataFrame(
        {
            "text_available": [True, True, False, True],
            "policy_negative": [True, True, True, False],
            "eval_ok": [True, False, False, True],
        }
    )
    masks = E.groundlm_masks(df)
    assert masks["neg"].sum() == 3
    assert masks["scoreable"].sum() == 1
    assert masks["fraud_text"].sum() == 1
    chk = E.count_check(df, masks)
    assert chk["match"] is False and "rows" in chk["differences"]


def test_type_breakdown_counts_containment_and_primary():
    df = pd.DataFrame(
        {
            "types": ["urgency,apk_install", "urgency", ""],
            "primary_type": ["apk_install", "urgency", ""],
            "n_evidence": [2, 1, 0],
        }
    )
    t = E.type_breakdown(df, np.array([True, True, True]))
    assert t["contains_type"] == {"urgency": 2, "apk_install": 1}
    assert t["primary_type"] == {"apk_install": 1, "urgency": 1}
    assert t["rows_with_no_evidence"] == 1


def test_threshold_sweep_ratios_and_empty_safe():
    out = E.threshold_sweep(
        neg_prob=np.array([0.9, 0.1]),
        gl_fraud_prob=np.array([0.95, 0.2]),
        sx_scam_prob=np.array([]),
        sx_ham_prob=np.array([0.6, 0.1, 0.1, 0.1]),
        camp_pos_prob=np.array([0.9]),
        camp_neg_prob=np.array([0.1, 0.1]),
        thresholds=[0.5],
    )[0]
    assert out["groundlm_fpr"] == pytest.approx(0.5)
    assert out["groundlm_fraud_recall"] == pytest.approx(0.5)
    assert out["smishx_scam_recall"] is None
    assert out["smishx_ham_fpr"] == pytest.approx(0.25)
    assert out["campaign_test_fpr"] == pytest.approx(0.0)


def test_decide_no_evidence_below_and_above_threshold():
    low = E.decide("hello", prob=0.01, threshold=0.35)
    assert (low["state"], low["risk"], low["action"]) == ("benign", "LOW", "IGNORE")
    assert low["detector_only"] is False
    # Default (quiet): a detector-only scam with no rule evidence is LOW and IGNORE, no warning.
    high = E.decide("hello", prob=0.9, threshold=0.35)
    assert high["detector_only"] is True
    assert (high["state"], high["risk"], high["action"]) == ("scam", "LOW", "IGNORE")
    # Verify mode (Fix B, the earlier behaviour): UNCERTAIN and VERIFY_INDEPENDENTLY.
    verify = E.decide("hello", prob=0.9, threshold=0.35, mode="verify")
    assert (verify["state"], verify["risk"], verify["action"]) == (
        "scam",
        "UNCERTAIN",
        "VERIFY_INDEPENDENTLY",
    )


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        E.decide("hello", prob=0.9, threshold=0.35, mode="loud")


# A small fixed set: rule evidence (medium, high, urgency-only), a benign message, and the
# detector-only text. Each is run at a score below and above the threshold.
AGREEMENT_TEXTS = [
    "Hey it's Sam from twitter Could I ask an urgent favour please?",
    "Congrats! You have won the 1st Prize. For Registration please Contact PAYTM Helpline",
    "Dear customer share your OTP to avoid account suspension",
    "Your weekly draw entry is confirmed. Reply YES to receive your entry details.",
    "Your account balance is Rs 1,200 as on 3 Nov. Thank you for banking with us.",
]
AGREEMENT_THRESHOLD = 0.35


@pytest.mark.parametrize("mode", ["quiet", "verify"])
@pytest.mark.parametrize("prob", [0.05, 0.9])
def test_eval_decision_agrees_with_edge_pipeline(mode, prob):
    """The eval and the pipeline call the same decide_edge, so they must agree on every field
    the eval counts. This test fails if either one stops using it."""
    from edge_pipeline import EdgeAnalyzer

    class FakeDetector:
        def score(self, text):
            return prob

    class FakeLLM:
        """Scripted sentence. The decision fields compared below do not depend on the LLM."""

        def generate(self, prompt, max_new_tokens=64, stop_when=None):
            text = " may be a scam because it claims a prize."
            if stop_when is not None:
                stop_when(text)
            return text, {
                "wall_ms": 1.0,
                "new_tokens": 1,
                "prompt_tokens": 1,
                "ttft_ms": 0.0,
                "stop_hit": True,
            }

    analyzer = EdgeAnalyzer(FakeDetector(), FakeLLM(), AGREEMENT_THRESHOLD, detector_only_mode=mode)
    for text in AGREEMENT_TEXTS:
        mine = E.decide(text, prob, AGREEMENT_THRESHOLD, mode=mode)
        out = analyzer.analyse(text)
        d = out["decision"]
        assert (mine["state"], mine["risk"], mine["action"], mine["detector_only"]) == (
            d["scam_state"],
            d["risk"],
            d["action"],
            d["detector_only"],
        ), text
        assert mine["n_evidence"] == d["evidence_count"], text


def test_apply_mode_on_verify_scores_equals_a_fresh_decide_in_each_mode():
    """The cache holds verify-mode decisions. apply_mode must give what decide gives directly,
    for both modes, on rows with and without rule evidence."""
    rows = []
    for text in AGREEMENT_TEXTS:
        for prob in (0.05, 0.9):
            d = E.decide(text, prob, AGREEMENT_THRESHOLD, mode="verify")
            rows.append({"prob": prob, "text": text, **d})
    cached = pd.DataFrame(rows)
    for mode in ("quiet", "verify"):
        remapped = E.apply_mode(cached, mode)
        for i, r in enumerate(rows):
            fresh = E.decide(r["text"], r["prob"], AGREEMENT_THRESHOLD, mode=mode)
            got = remapped.iloc[i]
            assert (got["risk"], got["action"], bool(got["detector_only"])) == (
                fresh["risk"],
                fresh["action"],
                fresh["detector_only"],
            ), (mode, r["text"], r["prob"])


def test_fpr_counts_the_user_visible_warning_in_quiet_mode():
    # Two negatives: one detector-only (quiet: IGNORE, not a warning), one rule-only UNCERTAIN.
    df = _frame(
        prob=[0.9, 0.1],
        n_ev=[0, 1],
        risk=["LOW", "UNCERTAIN"],
        action=["IGNORE", "VERIFY_INDEPENDENTLY"],
    )
    flags = E.alert_flags(df, threshold=0.5)
    assert flags["f_action_not_IGNORE"].tolist() == [False, True]
