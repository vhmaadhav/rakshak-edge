from rakshak.canonicalize import canonicalize, extract_urls


def test_zero_width_and_bidi_stripped_and_counted():
    c = canonicalize("Dear​ cust‍omer ‮abc")
    assert "​" not in c.normalized and "‍" not in c.normalized
    assert "‮" not in c.normalized
    assert c.normalized.startswith("Dear customer")
    assert c.flags.zero_width_count == 3
    assert c.original == "Dear​ cust‍omer ‮abc"


def test_cyrillic_and_greek_lookalikes_fold_to_ascii():
    c = canonicalize("Ваш ЅВI аккаунт")
    assert "SBI" in c.normalized
    assert c.flags.homoglyph_count >= 3


def test_fullwidth_and_math_letters_fold():
    c = canonicalize("ＳＢＩ and \U0001D412\U0001D401\U0001D408 alert")
    assert "SBI and SBI alert" in c.normalized


def test_spaced_letters_collapse_to_word():
    c = canonicalize("Update your K Y C now")
    assert "KYC" in c.normalized
    assert c.flags.spacing_fold >= 1


def test_dotted_acronym_collapses():
    c = canonicalize("Call S.B.I helpline")
    assert "SBI helpline" in c.normalized


def test_spaced_fold_keeps_word_boundaries():
    c = canonicalize("your S.B.I KYC ok")
    assert c.normalized == "your SBI KYC ok"


def test_leet_folds_inside_alphabetic_token():
    c = canonicalize("upd4te your acc")
    assert "update" in c.normalized
    assert c.flags.leet_fold >= 1


def test_leet_does_not_touch_otp_amounts_or_numbers():
    c = canonicalize("Your OTP is 482913, Rs500 debited, call 1800 123 4567")
    assert "482913" in c.normalized
    assert "Rs500" in c.normalized
    assert "1800 123 4567" in c.normalized


def test_devanagari_preserved():
    text = "आपका केवाईसी आज बंद हो जाएगा, तुरंत अपडेट करें"
    c = canonicalize(text)
    assert c.normalized == text
    assert c.flags.homoglyph_count == 0


def test_url_extraction_scheme_less():
    urls = extract_urls("Go to bit.ly/x or sbi-kyc.in/upd now")
    assert urls == ["bit.ly/x", "sbi-kyc.in/upd"]


def test_url_extraction_obfuscated_hxxp_and_brackets():
    c = canonicalize("Visit hxxp://sbi-kyc[.]in/upd now")
    assert c.urls == ["http://sbi-kyc.in/upd"]
    assert "sbi-kyc.in/upd" in c.normalized


def test_url_extraction_dot_word_obfuscation():
    c = canonicalize("Paytm (dot) com/kyc")
    assert c.urls == ["Paytm.com/kyc"]


def test_non_urls_are_not_extracted():
    assert canonicalize("Dr.Sharma will call Rs.500 today").urls == []


def test_mixed_script_flag():
    # Mixed means one word that mixes Latin letters with Cyrillic/Greek lookalikes.
    assert canonicalize("Your SBІ account").flags.mixed_script is True
    assert canonicalize("Ваш SBI").flags.mixed_script is False
    assert canonicalize("Plain English text").flags.mixed_script is False


def test_canonicalize_is_idempotent_on_plain_text():
    text = "Your OTP is 482913. Do not share it."
    once = canonicalize(text).normalized
    assert canonicalize(once).normalized == once
