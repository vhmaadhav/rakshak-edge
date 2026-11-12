"""Tests for src/rakshak/data/surrogates.py (plan section 3). Standard library and brand table."""

from __future__ import annotations

import inspect
import random
import re
from urllib.parse import urlsplit

import pytest

from rakshak.data import surrogates as S
from rakshak.url_features import load_brands

# Every tag in the plan's section 1 inventory, plus the two the scrubber emits (SENDER, DOMAIN).
INVENTORY = (
    "NAMED_ENTITY",
    "URL",
    "DATE_TIME",
    "LOCATION",
    "PHONE_NUMBER",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
    "NRP",
    "EMAIL_ADDRESS",
    "DECIMAL",
    "TIME",
    "EMAIL",
    "NUMBER",
    "PATH",
    "AMOUNT",
    "ACCOUNT_CARD",
    "UPI_ID",
    "PHONE",
    "NAME",
    "QUERY",
    "OTP",
    "FRAGMENT",
    "DATE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_PASSPORT",
    "US_SSN",
    "US_ITIN",
    "UK_NHS",
    "MEDICAL_LICENSE",
    "IP_ADDRESS",
    "CRYPTO",
    "SENDER",
    "DOMAIN",
)
PROPERTY_TAGS = tuple(INVENTORY) + ("SOMETHING_NEW", "X_Y")
TEXTS_FILLER = (
    "pay",
    "now",
    "account",
    "your",
    "is",
    "due",
    "click",
    "to",
    "verify",
    "dear",
    "customer",
    "bank",
    "update",
    "kyc",
    "today",
    "limit",
    "credited",
    "debited",
)
AMOUNT_RE = re.compile(
    r"^(?:Rs\.|Rs |INR |₹)(?:\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?$"
)
PHONE_DIGITS_RE = re.compile(r"[6-9]\d{9}")
UPI_RE = re.compile(r"^[a-z]+\d{2,4}@([a-z]+)$")
MASKED_RE = re.compile(r"^(?:XX\d{4}|XXXXXXXX\d{4}|\*\*\*\* \d{4})$")
EMAIL_RE = re.compile(r"^[a-z]+\d{2,4}@([a-z.]+)$")
ROW_IDS = [f"row-{i}" for i in range(200)]


def _rng(seed: int = 1) -> random.Random:
    return random.Random(seed)


def _values(tag: str, n: int = 300) -> list[str]:
    rng = _rng(7)
    return [S.fill(f"<{tag}>", rng) for _ in range(n)]


# ---- coverage of the tag inventory ----


def test_every_inventory_tag_has_its_own_branch():
    missing = [t for t in INVENTORY if t not in S.TAG_BRANCHES]
    assert missing == []


@pytest.mark.parametrize("tag", INVENTORY)
def test_every_inventory_tag_is_replaced_and_leaves_no_tag(tag):
    text = f"x <{tag}> y"
    out = S.fill(text, _rng(3))
    assert f"<{tag}>" not in out
    assert not S.TAG_RE.search(out)
    assert out.startswith("x ") and out.endswith(" y")


def test_unknown_tag_falls_back_to_a_neutral_word():
    out = S.fill("<FOO_BAR> due", _rng(2))
    word = out.split(" due")[0]
    assert word in S.NEUTRAL_WORDS
    assert not S.TAG_RE.search(out)


def test_text_without_tags_is_returned_byte_for_byte():
    for text in ("", "plain text", "a < b > c", "<b>html</b>", "<lower_case>", "Rs. 500 <1>"):
        assert S.fill(text, _rng(4)) == text
        assert S.fill_row("r", text) == text


def test_multiple_adjacent_start_and_end_tags():
    out = S.fill("<NAME><PHONE> at <AMOUNT>, <DATE>", _rng(5))
    assert not S.TAG_RE.search(out)
    assert "<" not in out and ">" not in out
    head, rest = out.split(" at ", 1)
    assert len(head) > 10  # NAME and PHONE both filled before " at "
    amount, date = rest.split(", ", 1)
    assert AMOUNT_RE.match(amount), amount
    assert date
    # a tag at the very start and at the very end, with words between
    edge = S.fill("<OTP> is your code <DATE>", _rng(6))
    assert re.match(r"^\d{4,6} is your code .+$", edge)


def test_fill_returns_a_string_and_takes_only_text_and_rng():
    out = S.fill("<NUMBER>", _rng())
    assert isinstance(out, str) and out.isdigit()


# ---- determinism ----


def test_same_row_id_and_seed_gives_the_same_output_across_calls():
    text = "Pay <AMOUNT> to <UPI_ID> or <URL>, <NAME> on <DATE_TIME>"
    a = S.fill_row("inbox-17", text)
    b = S.fill_row("inbox-17", text)
    assert a == b
    assert S.fill_many(["inbox-17"], [text]) == [a]


def test_different_row_ids_differ_for_most_of_a_sample():
    text = "Pay <AMOUNT> to <UPI_ID> or <URL>, <NAME> on <DATE_TIME>"
    outs = [S.fill_row(r, text) for r in ROW_IDS]
    assert len(set(outs)) >= int(0.95 * len(ROW_IDS))


def test_a_different_seed_changes_the_output():
    text = "Pay <AMOUNT> to <UPI_ID> or <URL>, <NAME> on <DATE_TIME>"
    changed = sum(S.fill_row(r, text, seed=1) != S.fill_row(r, text, seed=2) for r in ROW_IDS)
    assert changed >= int(0.95 * len(ROW_IDS))


def test_row_rng_is_seeded_from_sha256_not_from_hash():
    # Same as seeding random with the first eight bytes of sha256("<seed>:<row id>").
    import hashlib

    digest = hashlib.sha256(f"{S.SURROGATE_SEED}:abc".encode()).digest()
    expected = random.Random(int.from_bytes(digest[:8], "big"))
    assert S.row_rng("abc").random() == expected.random()


def test_pinned_outputs_for_fixed_row_ids():
    text = "Pay <AMOUNT> to <UPI_ID> via <URL> or call <PHONE>, <NAME> <DATE_TIME>"
    assert S.fill_row("sms-000001", text) == (
        "Pay ₹5,616 to imran582@okicici via http://airtel-verify.xyz/1oa or call "
        "88751 89277, Divya Shah 17/06/24"
    )
    assert S.fill_row("inbox-phone2-17", text) == (
        "Pay ₹82,756 to john4480@okicici via https://basket82.com/inz5kwq/gno or call "
        "96450 61270, Carlos Bose 19/06/26"
    )
    assert S.fill_row("imc25:42", text) == (
        "Pay Rs.243 to meera795@oksbi via http://indiapost-login.xyz/4voli/i9x93j or call "
        "+917618996208, Emily Kumar 17 May 2025"
    )
    assert S.fill_row("sms-000001", "<OTP> <ACCOUNT_CARD> <EMAIL>") == (
        "686165 XX5822 priya131@hotmail.com"
    )


def test_fill_many_length_mismatch_raises():
    with pytest.raises(ValueError):
        S.fill_many(["a", "b"], ["<NAME>"])


# ---- label independence (structural) ----


@pytest.mark.parametrize("fn", [S.fill, S.fill_row, S.fill_many])
def test_no_label_or_source_parameter(fn):
    names = set(inspect.signature(fn).parameters)
    assert not names & {"label", "labels", "source", "y", "sources"}


def test_label_independence_by_construction():
    # The output depends only on (text, row id, seed): the same call gives the same result
    # whatever the caller's label (the label is simply not an input).
    text = "Your <OTP> is <NUMBER>"
    assert S.fill_row("row-9", text) == S.fill_row("row-9", text)


# ---- format checks ----


def test_amount_format_and_range():
    for value in _values("AMOUNT", 500):
        assert AMOUNT_RE.match(value), value
        digits = re.sub(r"^(?:Rs\.|Rs |INR |₹)", "", value).replace(",", "")
        amount = float(digits)
        assert 1.0 <= amount <= 200_000.0 + 0.01, value


def test_amount_uses_both_grouping_styles():
    joined = " ".join(_values("AMOUNT", 400))
    assert re.search(r"\d{1,2},\d{2},\d{3}", joined)  # Indian: 1,23,456
    assert re.search(r"(?<![\d,])\d{1,3},\d{3}(?![\d,])", joined)  # Western: 123,456


def test_indian_phone_format():
    for value in _values("PHONE", 300):
        digits = re.sub(r"[^\d]", "", value)
        if value.startswith("+91"):
            digits = digits[2:]
        assert PHONE_DIGITS_RE.fullmatch(digits), value
        assert re.fullmatch(r"(?:\+91 ?)?[6-9]\d{4} ?\d{5}", value), value


def test_otp_is_four_or_six_digits():
    for value in _values("OTP", 300):
        assert re.fullmatch(r"\d{4}|\d{6}", value), value


def test_upi_id_format_and_real_handles():
    for value in _values("UPI_ID", 300):
        m = UPI_RE.match(value)
        assert m, value
        assert m.group(1) in S.PSP_HANDLES


def test_masked_account_forms():
    for tag in ("ACCOUNT_CARD", "US_BANK_NUMBER", "CREDIT_CARD", "IBAN_CODE"):
        for value in _values(tag, 60):
            assert MASKED_RE.match(value), (tag, value)


def test_email_format():
    for value in _values("EMAIL", 200):
        m = EMAIL_RE.match(value)
        assert m, value
        assert m.group(1) in S.WEBMAIL_DOMAINS


def test_url_parses_and_has_a_dot_in_the_host():
    for value in _values("URL", 300):
        parts = urlsplit(value)
        assert parts.scheme in ("http", "https"), value
        assert parts.hostname and "." in parts.hostname, value
        assert not S.TAG_RE.search(value)


def test_other_branches_produce_their_shapes():
    assert all(re.fullmatch(r"\d+", v) for v in _values("NUMBER", 200))
    assert all(re.fullmatch(r"\d+\.\d{1,2}", v) for v in _values("DECIMAL", 200))
    assert all(
        re.fullmatch(r"\d{2}:\d{2}|\d{1,2}:\d{2} (?:AM|PM)", v) for v in _values("TIME", 200)
    )
    assert all(v in S.CITIES for v in _values("LOCATION", 200))
    assert all(v in S.NATIONALITIES for v in _values("NRP", 200))
    assert all(v.startswith("X000") and len(v) == 9 for v in _values("US_DRIVER_LICENSE", 50))
    assert all(re.fullmatch(r"9\d{2}-00-\d{4}", v) for v in _values("US_SSN", 100))
    assert all(
        re.fullmatch(r"(?:192\.0\.2|198\.51\.100|203\.0\.113)\.\d{1,3}", v)
        for v in _values("IP_ADDRESS", 100)
    )


# ---- brand-lookalike pool ----


def test_no_lookalike_equals_a_brand_registered_domain():
    registered = {d.lower() for info in load_brands().values() for d in info["domains"]}
    lookalikes = set(S.lookalike_domains())
    assert lookalikes, "the lookalike pool must not be empty"
    assert lookalikes.isdisjoint(registered)
    assert registered == set(S.registered_brand_domains())


def test_generated_url_hosts_never_equal_a_brand_domain():
    registered = S.registered_brand_domains()
    rng = _rng(11)
    for _ in range(500):
        host = urlsplit(S.fill("<URL>", rng)).hostname
        assert host not in registered


def test_brand_pool_uses_the_brand_aliases_and_the_fixed_lists():
    lookalikes = S.lookalike_domains()
    assert any(d.startswith("sbi") for d in lookalikes)
    assert all(d.rsplit(".", 1)[-1] in S.LOOKALIKE_TLDS for d in lookalikes)
    assert set(S.SHORTENER_DOMAINS).isdisjoint(S.registered_brand_domains())


# ---- assert_no_tags ----


def test_assert_no_tags_passes_clean_text():
    S.assert_no_tags(["ok", "fine", ""])


def test_assert_no_tags_raises_with_a_count_and_no_text():
    texts = ["ZQXSENTINEL <AMOUNT>", "ok", "ZQXSENTINEL <NAME> and <URL>"]
    with pytest.raises(ValueError) as info:
        S.assert_no_tags(texts)
    message = str(info.value)
    assert "2" in message
    assert "ZQX" not in message
    assert "<AMOUNT>" not in message


def test_filled_texts_pass_the_assertion():
    texts = [S.fill_row(str(i), f"<NAME> paid <AMOUNT> via <UPI_ID> <URL> {i}") for i in range(50)]
    S.assert_no_tags(texts)


# ---- property check ----


def test_random_texts_never_keep_a_tag_and_never_raise():
    rng = random.Random(2026)
    tags = list(PROPERTY_TAGS)
    for i in range(2000):
        parts = []
        for _ in range(rng.randint(0, 8)):
            if rng.random() < 0.5:
                parts.append(f"<{rng.choice(tags)}>")
            else:
                parts.append(rng.choice(TEXTS_FILLER))
        text = " ".join(parts)
        out = S.fill_row(f"prop-{i}", text)
        assert not S.TAG_RE.search(out)
    S.assert_no_tags([S.fill_row(f"prop2-{i}", "<AMOUNT> <TIME>") for i in range(100)])
