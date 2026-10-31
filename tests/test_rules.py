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


# ---------------------------------------------------------------------------
# Clause-scoped windows. A flat character window misses a scam when the subject and the
# payment demand land in different clauses of the same sentence, which is the commonest
# phrasing in the real corpus ("parcel ... due to an unpaid shipping fee").
# ---------------------------------------------------------------------------


def _types(text):
    return {e.type for e in run_rules(canonicalize(text))}


def test_clause_window_fires_across_a_sentence_phrase():
    """Subject and demand inside one sentence but further apart than the old 50-char window."""
    text = (
        "Post Office Your parcel have be redirect to your local Post Office branch due to an "
        "unpaid shipping fee To reschedule a delivery please visit "
        "http://postoffice-tracking updates.com"
    )
    assert "parcel_customs" in _types(text)


def test_clause_window_does_not_span_sentences():
    """A parcel mention in one sentence and an unrelated fee in the next must not pair up."""
    text = "Your parcel arrived at the depot. The coffee here costs a small fee."
    assert "parcel_customs" not in _types(text)


def test_plain_delivery_notification_stays_clean():
    """Delivery confirmations with no money demand must stay clean."""
    for text in (
        "Your Flipkart package has been delivered to the reception desk. Track it in the app.",
        "Your parcel is out for delivery and will arrive between 2pm and 4pm.",
    ):
        assert "parcel_customs" not in _types(text), text


def test_legit_order_confirmation_with_amount_stays_clean():
    """An ordinary order confirmation can mention an amount payable without being a scam."""
    text = (
        "Your Snapdeal order will be delivered by Blue Dart. Amount payable is Rs 439. "
        "Track on http://bluedart.in/?79125099782"
    )
    assert "parcel_customs" not in _types(text)


def test_job_rule_fires_on_daily_salary_recruitment():
    """Recruit + promised daily salary in one sentence is the Telegram-job signature."""
    text = "Part-time daily salary of 3,000 rupee no time limit no fee, contact wa.me/919999999999"
    assert "job_task_scam" in _types(text)


def test_job_rule_does_not_join_separate_sentences():
    """The pairing is sentence-scoped on purpose: a job ad and a later salary line stay apart."""
    text = "We are hiring part time staff. Daily salary 3000 pesos. Apply on wa.me/919999999999"
    assert "job_task_scam" not in _types(text)


def test_job_rule_ignores_personal_job_talk():
    """Genuine chat about a temporary job with an ordinary hourly rate must stay clean."""
    text = (
        "I got this temp job 2 promote IT stuff at NUS fro 28 jul-1 aug "
        "pay ard 5.50-7.50 u interested?"
    )
    assert "job_task_scam" not in _types(text)


def test_job_rule_ignores_ordinary_job_ads():
    for text in (
        "Looking for a new job? A good CV should be short and relevant.",
        "My job is somehow related to machine learning and I like it.",
    ):
        assert "job_task_scam" not in _types(text), text


def test_prize_rule_fires_on_plural_prizes():
    """"prizes" (plural) is an unambiguous prize claim and used to be missed."""
    assert "prize_lottery" in _types("Win exciting prizes worth Rs 1 crore, call 1800 now")


def test_prize_rule_ignores_ordinary_use_of_lucky():
    assert "prize_lottery" not in _types("I swear today is my lucky day")


def test_otp_warning_stays_clean_after_widening():
    """A genuine, negated OTP warning must never become credential_request."""
    text = "Your OTP is 482913. Do not share it with anyone. -SBI"
    assert "credential_request" not in _types(text)
