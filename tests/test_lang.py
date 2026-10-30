"""Hand-labelled language cases for the SMS language detector (tests/fixtures/lang_cases.json).

Gold labels are en, hinglish, hi_deva or mixed. ``mixed`` (English-dominant code-mix with a real
Hindi share) is scored in the hinglish family, because the detector has no separate bucket for
it. Agreement is computed on canonicalised text, as the dataset build does.
"""

from __future__ import annotations

from collections import Counter

import pytest
from conftest import load_fixture

from rakshak.canonicalize import canonicalize
from rakshak.datasets.lang import HINGLISH_RATIO, detect_lang

CASES = load_fixture("lang_cases.json")
FAMILY = {"en": "en", "hinglish": "hinglish", "mixed": "hinglish", "hi_deva": "hi_deva"}
# Agreement on the hand-labelled set after tuning (see lang.py). The bar is set below it so an
# honest small regression does not fail CI, but a collapse does.
MIN_FAMILY_AGREEMENT = 0.95


def _predict(text: str) -> str:
    return detect_lang(canonicalize(text).normalized)


def test_fixture_is_large_and_covers_every_gold_class():
    assert len(CASES) >= 80
    assert {c["gold"] for c in CASES} == {"en", "hinglish", "mixed", "hi_deva"}
    assert sum(c["gold"] == "hinglish" for c in CASES) >= 20


def test_family_agreement_on_hand_labelled_set():
    hits = sum(_predict(c["text"]) == FAMILY[c["gold"]] for c in CASES)
    assert hits / len(CASES) >= MIN_FAMILY_AGREEMENT


def test_hinglish_recall_on_hand_labelled_set():
    gold = [c for c in CASES if c["gold"] in ("hinglish", "mixed")]
    hits = sum(_predict(c["text"]) == "hinglish" for c in gold)
    assert hits / len(gold) >= 0.9


def test_english_with_indian_names_and_places_stays_english():
    names = [c for c in CASES if c["gold"] == "en" and any(
        w in c["text"] for w in ("Ramesh", "Kavitha", "Mohan", "Kamal", "Sunny", "Anand", "Ravi"))]
    assert names
    assert all(_predict(c["text"]) == "en" for c in names)


def test_per_class_confusion_is_reported_for_debugging():
    conf = Counter((c["gold"], _predict(c["text"])) for c in CASES)
    assert sum(conf.values()) == len(CASES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("aapka KYC pending hai, jaldi karo warna account band ho jayega", "hinglish"),
        ("SHADI MUBARAK HO !!!", "hinglish"),
        ("Bhai paise bhej diye, check karo", "hinglish"),
        ("Kolkata Knight Riders beat Punjab Kings by 5 wickets.", "en"),
        ("Tentamos entregar a sua encomenda hoje", "other"),
        ("", "other"),
    ],
)
def test_detector_examples(text, expected):
    assert _predict(text) == expected


def test_ratio_is_a_keyword_option():
    text = "Your account ka KYC is pending, jaldi karo"
    assert detect_lang(text, hinglish_ratio=0.99) != "hinglish"
    assert detect_lang(text, hinglish_ratio=HINGLISH_RATIO) in {"hinglish", "en"}
