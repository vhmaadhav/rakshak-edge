import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "pi" / "service"))

from rakshak.url_reputation import ReputationGuard, private_registrable  # noqa: E402


def _guard():
    return ReputationGuard({"google.com", "sbi.co.in", "github.io", "amazon.in", "bit.ly"})


def test_private_suffix_splits_hosting_tenants():
    assert private_registrable("evil.github.io") == "evil.github.io"
    assert private_registrable("www.onlinesbi.sbi.co.in") == "sbi.co.in"


def test_popular_domain_is_established():
    r = _guard().check("https://www.sbi.co.in/web/personal-banking")
    assert r.established and r.reason == "popular" and r.registrable == "sbi.co.in"


def test_hosting_tenant_not_established_even_if_platform_listed():
    r = _guard().check("https://sbi-kyc-update.github.io/login")
    assert not r.established and r.reason == "not_popular"


def test_shortener_never_established():
    r = _guard().check("http://bit.ly/3abcd")
    assert not r.established and r.reason == "shortener"


def test_user_content_host_on_popular_parent():
    r = _guard().check("https://docs.google.com/forms/d/e/xyz/viewform")
    assert not r.established and r.reason == "user_content"


def test_ip_and_userinfo():
    assert _guard().check("http://192.168.1.10/pay").reason == "ip"
    assert _guard().check("http://amazon.in@evil.example/x").reason in {"userinfo", "not_popular"}
    assert not _guard().check("http://amazon.in@evil.example/x").established


def test_unknown_domain():
    r = _guard().check("https://sbi-rewards-claim.xyz/")
    assert not r.established and r.reason == "not_popular"


def test_from_file_reads_tranco_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("1,google.com\n2,sbi.co.in\n3,example.org\n")
    g = ReputationGuard.from_file(p, top_n=2)
    assert g.domains == {"google.com", "sbi.co.in"} and len(g.list_sha256) == 64


def test_guarded_quiet_policy_adds_only_unestablished_urls():
    from edge_decision import add_url_evidence

    from rakshak.canonicalize import canonicalize
    from rakshak.url_model import UrlHit

    g = _guard()
    bad = canonicalize("Your SBI KYC expires today, update at https://sbi-kyc.xyz/u now")
    ok = canonicalize("Rs 500 debited. Not you? Visit https://www.sbi.co.in/help")
    ev_bad = add_url_evidence([], bad, UrlHit(bad.urls[0], 0.97), 0.63, "quiet",
                              quiet_policy="guarded", guard=g)
    ev_ok = add_url_evidence([], ok, UrlHit(ok.urls[0], 0.97), 0.63, "quiet",
                             quiet_policy="guarded", guard=g)
    below = add_url_evidence([], bad, UrlHit(bad.urls[0], 0.10), 0.63, "quiet",
                             quiet_policy="guarded", guard=g)
    off = add_url_evidence([], bad, UrlHit(bad.urls[0], 0.97), 0.63, "quiet")
    assert [e.type for e in ev_bad] == ["phishing_url_model"]
    assert ev_ok == [] and below == [] and off == []
