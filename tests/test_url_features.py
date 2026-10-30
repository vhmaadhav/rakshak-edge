import json
import socket
from importlib import resources

from rakshak.url_features import (
    brand_domain_mismatch,
    load_brands,
    mentioned_brand_domains,
    registered_domain,
    url_features,
)


def test_features_are_floats_and_cover_core_keys():
    f = url_features("https://sbi-kyc.in/upd?x=1")
    assert all(isinstance(v, float) for v in f.values())
    for key in ("url_len", "host_entropy", "n_subdomains", "shortener", "tld_risky",
                "punycode", "ip_literal", "suspicious_token_count", "brand_host_mismatch"):
        assert key in f


def test_shortener_and_risky_tld_flags():
    assert url_features("bit.ly/x")["shortener"] == 1.0
    assert url_features("http://deal-now.xyz/go")["tld_risky"] == 1.0
    assert url_features("https://www.sbi.co.in/")["tld_common"] == 1.0


def test_punycode_ip_and_at_sign():
    assert url_features("https://xn--sbi-kyc-xog.in/upd")["punycode"] == 1.0
    assert url_features("http://192.168.4.10/login")["ip_literal"] == 1.0
    assert url_features("https://user@example.com/")["has_at_sign"] == 1.0


def test_subdomain_padding_counted():
    assert url_features("https://a.b.c.example.com/")["n_subdomains"] == 3.0
    assert url_features("https://example.com/")["n_subdomains"] == 0.0


def test_brand_host_mismatch_feature():
    assert url_features("https://sbi-kyc.in/upd")["brand_host_mismatch"] == 1.0
    assert url_features("https://www.sbi.co.in/")["brand_host_mismatch"] == 0.0


def test_registered_domain_uses_public_suffix_snapshot():
    assert registered_domain("https://www.onlinesbi.sbi/login") == "onlinesbi.sbi"
    assert registered_domain("https://shop.example.co.in/x") == "example.co.in"


def test_registered_domain_works_offline(monkeypatch):
    def _no_net(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _no_net)
    monkeypatch.setattr(socket, "create_connection", _no_net)
    assert registered_domain("https://sbi-kyc.in/upd") == "sbi-kyc.in"


def test_brand_domain_mismatch_detects_lookalike():
    assert brand_domain_mismatch("Your SBI KYC is blocked", "https://sbi-kyc.in/upd") is True


def test_brand_domain_mismatch_accepts_official_domains():
    assert brand_domain_mismatch("SBI alert", "https://sbi.co.in/x") is False
    assert brand_domain_mismatch("Your order", "https://www.amazon.in/track") is False
    assert brand_domain_mismatch("Recharge at paytm", "https://paytm.com/recharge") is False


def test_brand_domain_mismatch_needs_a_brand_mention():
    assert brand_domain_mismatch("Meeting at 4 pm", "https://random-site.in/x") is False


def test_mentioned_brand_domains_maps_to_official_list():
    doms = mentioned_brand_domains("Your SBI account")
    assert "sbi.co.in" in doms["SBI"]


def test_brands_json_is_well_formed():
    brands = load_brands()
    assert len(brands) >= 20
    for name, info in brands.items():
        assert info["domains"], name
        assert isinstance(info["verified"], bool), name
        assert info["aliases"] and info["host_aliases"], name
    raw = resources.files("rakshak").joinpath("data/brands.json").read_text(encoding="utf-8")
    assert json.loads(raw)["version"] == 1
