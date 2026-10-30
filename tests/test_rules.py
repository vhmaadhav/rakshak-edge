import pytest
from conftest import load_fixture

from rakshak.canonicalize import canonicalize
from rakshak.rules import EVIDENCE_TYPES, Evidence, run_rules

CASES = load_fixture("rule_cases.json")


@pytest.mark.parametrize("case", CASES, ids=[c["text"][:45] for c in CASES])
def test_rule_case(case):
    evidence = run_rules(canonicalize(case["text"]))
    types = {e.type for e in evidence}
    for expected in case["expect"]:
        assert expected in types, f"{expected} did not fire: {types}"
    for forbidden in case["forbid"]:
        assert forbidden not in types, f"{forbidden} fired on benign text: {types}"


def test_at_least_25_hand_written_cases():
    assert len(CASES) >= 25
    assert sum(1 for c in CASES if not c["expect"]) >= 8


def test_evidence_ids_are_sequential_and_typed():
    ev = run_rules(canonicalize(CASES[0]["text"]))
    assert ev
    assert [e.id for e in ev] == [f"E{i}" for i in range(1, len(ev) + 1)]
    for e in ev:
        assert isinstance(e, Evidence)
        assert e.type in EVIDENCE_TYPES
        assert 0.0 < e.weight <= 1.0


def test_benign_bank_alert_has_no_evidence_at_all():
    text = "Rs 500 debited from a/c XX1234 on 12-10-26. Not you? Call 1800 123 4567"
    assert run_rules(canonicalize(text)) == []


def test_negated_otp_warning_does_not_fire_credential_request():
    text = "Your OTP is 482913. Do not share it with anyone. -SBI"
    types = {e.type for e in run_rules(canonicalize(text))}
    assert "credential_request" not in types


def test_hindi_negated_request_does_not_fire():
    types = {e.type for e in run_rules(canonicalize("OTP mat batao kisi ko"))}
    assert "credential_request" not in types


def test_homoglyph_brand_still_detected_via_canonicalizer():
    text = "Ваш SBI аккаунт blocked. Update at https://sbi-kyc.in/upd"
    types = {e.type for e in run_rules(canonicalize(text))}
    assert "brand_domain_mismatch" in types


def test_urgency_alone_never_sets_an_action_or_a_high_risk():
    from conftest import package_for

    from rakshak.evidence import Action, Risk
    from rakshak.verifier import template_explanation

    pkg = package_for("Meeting ke liye turant aao, sab wait kar rahe hain", state="benign")
    assert {e.type for e in pkg.evidence} == {"urgency"}
    assert pkg.risk is Risk.LOW
    assert pkg.allowed_actions == [Action.IGNORE]
    # the template explanation must cover urgency in both languages
    assert "time pressure" in template_explanation(pkg, "en")["summary"]
    assert "dabav" in template_explanation(pkg, "hinglish")["summary"]


def test_devanagari_and_hinglish_scam_fires_the_new_types():
    cases = {
        "electricity_disconnect": "आपका बिजली कनेक्शन आज रात 9:30 बजे काट दिया जाएगा।",
        "digital_arrest": "आपके खिलाफ साइबर क्राइम में शिकायत दर्ज हुई है।",
        "investment_returns": "निवेश करें और दोगुना मुनाफा पाएं",
        "credential_request": "ओटीपी बताएं, तभी आपका खाता सुरक्षित रहेगा",
    }
    for expected, text in cases.items():
        types = {e.type for e in run_rules(canonicalize(text))}
        assert expected in types, (expected, types)


def test_devanagari_negated_otp_warning_stays_benign():
    types = {e.type for e in run_rules(canonicalize("ओटीपी किसी को न बताएं"))}
    assert "credential_request" not in types
