"""URL helpers must be total: hostile strings from real SMS never raise."""

import pytest

from rakshak.canonicalize import canonicalize
from rakshak.rules import run_rules
from rakshak.url_features import (
    _best_effort_split,
    brand_domain_mismatch,
    registered_domain,
    url_features,
)

HOSTILE = [
    "[http://x",
    "http://x]",
    "example[.]com/[a",
    "hxxp://[.]",
    "hxxps://paypal[.]com[/]login",
    "http://[::1",
    "http://[::1]/",
    "http://[2001:db8::1]:8080/a?b#c",
    "http://[2001:db8::1/p",
    "http://[zz]/",
    "http://[]/",
    "http://[::ffff:1.2.3.4]/",
    "http://[12345::1]/",
    "http://[fe80::1%25eth0]/",
    "[::1]",
    "[",
    "]",
    "[]",
    "[[[[]]]]",
    "http://a]b.com",
    "http://ex[ample.com",
    "//[bad",
    "http://:80",
    "http://user@[::1]:",
    "http://пример.рф/вход",
    "http://ｅｘａｍｐｌｅ．ｃｏｍ",
    "http://‮example.com",
    "http://xn--",
    "http://xn--a.com",
    "http://a..b..c",
    "http://" + "a" * 5000 + ".com",
    "https://" + "a." * 2000 + "com/" + "x" * 4000,
    "http://%41%42%43.com",
    "http://%zz%.com/%",
    "%",
    "",
    " ",
    "\x00",
    "http://exa\x00mple.com",
    "http://\ud83dexample.com",
    "hxxp://evil[.]xyz/login?user=a@b",
    "http://1.2.3.4:99999999/",
    "http://[::1]:notaport",
    "www.sbi.co.in(dot)kyc-update.xyz",
    "http://xn--80ak6aa92e.com",
    "https://sbi­-kyc.in",
    "http://İ.example",
    "http://a b c.com/x y z",
    "http://<script>",
    "ftp://[::]",
    "http://[::1]]]",
    "http://[v1.fe]/",
    "http://%5B%3A%3A1%5D/",
]


@pytest.mark.parametrize("s", HOSTILE)
def test_url_helpers_never_raise(s):
    feats = url_features(s)
    assert all(isinstance(v, float) for v in feats.values())
    assert isinstance(registered_domain(s), str)
    assert isinstance(brand_domain_mismatch("verify your SBI account", s), bool)


@pytest.mark.parametrize("s", HOSTILE)
def test_canonicalize_and_rules_never_raise(s):
    run_rules(canonicalize(s))
    run_rules(canonicalize(f"Dear customer verify your SBI account now {s} click"))


def test_bracketed_host_falls_back_to_best_effort_host():
    assert url_features("[http://x")["url_parse_error"] == 1.0
    assert _best_effort_split("[http://x")[1] == "x"
    assert _best_effort_split("example[.]com/[a")[1] == "example.com"
    assert registered_domain("example[.]com/[a") == "example.com"


def test_parse_error_flag_is_zero_for_valid_urls():
    assert url_features("https://sbi.co.in/")["url_parse_error"] == 0.0
    assert url_features("http://[2001:db8::1]/p")["url_parse_error"] == 0.0


def test_hopeless_input_gives_empty_host():
    assert registered_domain("[") == ""
    assert registered_domain("") == ""
