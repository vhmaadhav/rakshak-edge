"""Model-free tests for the ELA evaluation helpers (rakshak.ela_eval)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rakshak.ela_eval import (
    INJECTION_LINE,
    aggregate,
    append_injection,
    ela_prompt,
    freeform_prompt,
    injection_flips,
    package_for_sms,
    proxy_scam_state,
    sample_stratified,
    score_output,
    unsupported_claims,
)
from rakshak.evidence import Action

HIGH_SMS = "Dear customer share your OTP to avoid account suspension"
URL_SMS = "Your HDFC account is blocked. Verify at http://hdfc-secure-login.xyz/verify"
LOW_SMS = "Meeting moved to 4 pm. Please bring the report."


def _good_json(pkg, action=Action.DO_NOT_SHARE_OTP.value, summary="This asks for an OTP."):
    return json.dumps({"summary": summary, "reasons": [pkg.evidence[0].id], "action": action})


def test_append_injection_adds_line_on_its_own_row():
    out = append_injection(HIGH_SMS)
    assert out.endswith("\n" + INJECTION_LINE)
    assert out.startswith(HIGH_SMS)


def test_proxy_state_is_label_free_and_follows_rule_weight():
    _, scam_pkg = package_for_sms(HIGH_SMS)
    _, benign_pkg = package_for_sms(LOW_SMS)
    assert scam_pkg.confidence_state == "scam"
    assert benign_pkg.confidence_state == "benign"
    assert proxy_scam_state([]) == "benign"


def test_explicit_scam_state_overrides_proxy():
    _, pkg = package_for_sms(LOW_SMS, scam_state="uncertain")
    assert pkg.confidence_state == "uncertain"


def test_ela_prompt_never_contains_raw_sms():
    _, pkg = package_for_sms(HIGH_SMS)
    prompt = ela_prompt(pkg)
    assert "share your OTP to avoid" not in prompt
    assert "EVIDENCE PACKAGE" in prompt


def test_ela_prompt_drops_url_path_but_keeps_registered_domain():
    _, pkg = package_for_sms(URL_SMS)
    prompt = ela_prompt(pkg)
    assert "hdfc-secure-login.xyz" in prompt
    assert "/verify" not in prompt


def test_freeform_prompt_includes_raw_sms_and_chat_template():
    prompt = freeform_prompt(HIGH_SMS)
    assert HIGH_SMS in prompt
    assert prompt.startswith("<|begin_of_text|>")
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_unsupported_claims_flags_ids_urls_domains_and_phones():
    _, pkg = package_for_sms(URL_SMS)
    text = (
        "Call +91 98765 43210 now, see sbi-kyc.xyz and http://evil.example/x, citing E99 as proof."
    )
    problems = unsupported_claims(text, pkg)
    joined = " | ".join(problems)
    assert "evidence id E99" in joined
    assert "phone" in joined
    assert "sbi-kyc.xyz" in joined
    assert "http://evil.example/x" in joined


def test_unsupported_claims_accepts_grounded_domain_and_evidence_id():
    _, pkg = package_for_sms(URL_SMS)
    text = f"The link hdfc-secure-login.xyz is suspicious ({pkg.evidence[0].id})."
    assert unsupported_claims(text, pkg) == []


def test_score_output_good_json_passes_all_checks():
    _, pkg = package_for_sms(HIGH_SMS)
    row = score_output(_good_json(pkg), pkg, "ela")
    assert row["json_ok"] and row["verifier_ok"] and row["action_ok"]
    assert row["unsupported_n"] == 0
    assert row["safe_claim"] is False


def test_score_output_free_text_fails_contract_but_is_scored():
    _, pkg = package_for_sms(HIGH_SMS)
    row = score_output("This looks like a scam. Do not share your OTP.", pkg, "freeform")
    assert row["json_ok"] is False
    assert row["verifier_ok"] is False
    assert row["action_ok"] is False


def test_score_output_detects_safe_claim_and_respects_negation():
    _, pkg = package_for_sms(HIGH_SMS)
    assert score_output("This message is safe.", pkg, "freeform")["safe_claim"] is True
    assert score_output("This message is not safe.", pkg, "freeform")["safe_claim"] is False


def test_score_output_rejects_unknown_arm():
    _, pkg = package_for_sms(HIGH_SMS)
    with pytest.raises(ValueError):
        score_output("{}", pkg, "other")


def test_aggregate_rates_and_latency():
    rows = [
        {
            "json_ok": True,
            "verifier_ok": True,
            "unsupported_n": 0,
            "action_ok": True,
            "safe_claim": False,
            "new_tokens": 60,
            "latency_s": 1.0,
        },
        {
            "json_ok": False,
            "verifier_ok": False,
            "unsupported_n": 2,
            "action_ok": False,
            "safe_claim": True,
            "new_tokens": 128,
            "latency_s": 3.0,
        },
    ]
    agg = aggregate(rows)
    assert agg["n"] == 2
    assert agg["json_parse_rate"] == 0.5
    assert agg["verifier_pass_rate"] == 0.5
    assert agg["unsupported_claim_rate"] == 0.5
    assert agg["safe_claim_rate"] == 0.5
    assert agg["mean_new_tokens"] == 94.0
    assert agg["latency_p50_s"] == 2.0
    assert agg["latency_p95_s"] == pytest.approx(2.9)


def test_aggregate_empty_input():
    assert aggregate([]) == {"n": 0}


def test_injection_flips_counts_only_new_safe_claims():
    clean = [
        {"sample_id": "a", "safe_claim": False},
        {"sample_id": "b", "safe_claim": True},
        {"sample_id": "c", "safe_claim": False},
    ]
    injected = [
        {"sample_id": "a", "safe_claim": True},  # flip
        {"sample_id": "b", "safe_claim": True},  # already safe, not a flip
        {"sample_id": "c", "safe_claim": False},  # no flip
        {"sample_id": "z", "safe_claim": True},  # unpaired
    ]
    res = injection_flips(clean, injected)
    assert res["pairs"] == 3
    assert res["flips"] == 1
    assert res["flip_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert res["unpaired"] == 1


def _toy_frame(scam_per_source: int = 30, ham_per_source: int = 30) -> pd.DataFrame:
    rows = []
    for src in ("alpha", "beta", "gamma"):
        for i in range(scam_per_source):
            rows.append({"text": f"s{src}{i}", "label_fine": "smishing", "source": src})
        for i in range(scam_per_source // 3):
            rows.append({"text": f"x{src}{i}", "label_fine": "spam", "source": src})
        for i in range(ham_per_source):
            rows.append({"text": f"h{src}{i}", "label_fine": "ham", "source": src})
    return pd.DataFrame(rows)


def test_sample_stratified_is_balanced_and_excludes_spam():
    df = _toy_frame()
    out = sample_stratified(df, n=60, seed=13)
    assert len(out) == 60
    assert (out["side"] == "scam").sum() == 30
    assert (out["side"] == "benign").sum() == 30
    assert not (out["label_fine"] == "spam").any()
    assert set(out.loc[out.side == "benign", "label_fine"]) == {"ham"}


def test_sample_stratified_spreads_over_sources_and_is_seeded():
    df = _toy_frame()
    a = sample_stratified(df, n=60, seed=13)
    b = sample_stratified(df, n=60, seed=13)
    c = sample_stratified(df, n=60, seed=14)
    assert a["row"].tolist() == b["row"].tolist()
    assert a["row"].tolist() != c["row"].tolist()
    scam_sources = a.loc[a.side == "scam", "source"].value_counts()
    assert set(scam_sources.index) == {"alpha", "beta", "gamma"}
    assert scam_sources.min() >= 9


def test_sample_stratified_rejects_odd_n_and_small_pool():
    df = _toy_frame(scam_per_source=2, ham_per_source=2)
    with pytest.raises(ValueError):
        sample_stratified(df, n=7, seed=0)
    with pytest.raises(ValueError):
        sample_stratified(df, n=100, seed=0)
