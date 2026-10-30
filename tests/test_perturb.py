import random

import pytest
from conftest import load_fixture

from rakshak.canonicalize import canonicalize
from rakshak.perturb import FAMILIES, perturb
from rakshak.rules import run_rules

SEEDS = load_fixture("scam_seeds.json")
SEED_NUMBERS = [1, 2, 3]


def test_registry_has_all_nine_families():
    assert set(FAMILIES) == {
        "zero_width_insert", "homoglyph_swap", "char_spacing", "punctuation_split",
        "leetspeak", "brand_misspell", "benign_sentence_insert",
        "urgency_paraphrase", "url_obfuscate",
    }


def test_perturb_is_deterministic_per_seed():
    text = SEEDS[0]["text"]
    for family in FAMILIES:
        assert perturb(text, family, 11) == perturb(text, family, 11)


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        perturb("hello", "no_such_family", 0)


def test_seeds_fire_expected_rules_clean():
    for seed in SEEDS:
        types = {e.type for e in run_rules(canonicalize(seed["text"]))}
        assert types & set(seed["rules"]), seed["text"]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_rule_still_fires_after_perturbation(family):
    failures = []
    for seed in SEEDS:
        for n in SEED_NUMBERS:
            perturbed = perturb(seed["text"], family, n)
            types = {e.type for e in run_rules(canonicalize(perturbed))}
            if not (types & set(seed["rules"])):
                failures.append((seed["text"][:40], n, sorted(types), perturbed[:80]))
    assert not failures, failures


# Inputs too short or too odd for a family to mangle in the usual way. "sbi" and "Sbi" are the
# case-insensitive brand matches that used to raise in brand_misspell.
EDGE_TEXTS = [
    "", " ", "Vi", "SBI", "sbi", "Sbi", "OK", "a", "ab", "SBI SBI", "Paytm", "🙂", "🙂🎉 😀",
    "नमस्ते", "आपका खाता बंद होगा", "ॐ", "http://", "x.in", "hxxp://sbi.in",
]
PROPERTY_SEEDS = random.Random(20261030).sample(range(1_000_000), 200)
ALL_TEXTS = (
    [case["text"] for case in SEEDS]
    + [case["text"] for case in load_fixture("rule_cases.json")]
    + [case["text"] for case in load_fixture("lang_cases.json")]
    + EDGE_TEXTS
)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_is_total(family):
    """Every family returns a string for every fixture text, edge string and seed."""
    for text in ALL_TEXTS:
        for seed in PROPERTY_SEEDS:
            out = perturb(text, family, seed)
            assert isinstance(out, str), (family, text, seed)
