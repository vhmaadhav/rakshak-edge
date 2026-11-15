"""URL model path: phishing_url_model evidence, POST /analyze_url, the registry switch and the
eval_groundlm URL option. The URL student is a fake scorer; the registry threshold is a test
fixture (URL_TAU). No model file or Pi state is read."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SERVICE = REPO / "deploy" / "pi" / "service"
for p in (SERVICE, REPO / "src", REPO / "deploy" / "pi" / "bench", REPO / "deploy" / "pi"):
    sys.path.insert(0, str(p))

import rakshak_edge_service as svc  # noqa: E402
from edge_pipeline import EdgeAnalyzer, check_url  # noqa: E402

from rakshak.canonicalize import canonicalize  # noqa: E402
from rakshak.evidence import Action, build_package, choose_action  # noqa: E402
from rakshak.prompts import facts_block  # noqa: E402
from rakshak.remediation import remediation_for, remediation_problems  # noqa: E402
from rakshak.rules import run_rules  # noqa: E402
from rakshak.url_model import (  # noqa: E402
    URL_MODEL_TYPE,
    UrlHit,
    add_url_model_evidence,
    best_url_hit,
)
from rakshak.verifier import check_points, template_only_advisory  # noqa: E402

URL_TAU = 0.5  # test fixture for the registry threshold (the real one comes from Kaggle)
THRESHOLD = 0.35  # the SMS student threshold used by these tests
SBI_TEXT = "Your SBI KYC is blocked. Update now at https://sbi-kyc.in/upd/verify?acct=123"
# A link with no rule hit: the URL model is the only thing that can raise it.
PLAIN_LINK_TEXT = "Read the article here: https://news-update.example.org/article/42"
LOW_TEXT = "Hey it's Sam from twitter Could I ask an urgent favour please?"


class FakeScorer:
    """Scores every URL with one value (or a per-URL map) and records the calls."""

    def __init__(self, score=0.9, by_url=None):
        self.score = score
        self.by_url = by_url or {}
        self.calls = 0

    def score_urls(self, urls):
        self.calls += 1
        return [self.by_url.get(u, self.score) for u in urls]


class FakeDetector:
    def __init__(self, p):
        self.p = p

    def score(self, text):
        return self.p


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, max_new_tokens=64, stop_when=None):
        self.calls += 1
        text = " may be a scam because it asks for details."
        if stop_when is not None:
            stop_when(text)
        return text, {
            "wall_ms": 10.0,
            "new_tokens": 5,
            "prompt_tokens": 50,
            "ttft_ms": 5.0,
            "stop_hit": True,
        }


REGISTERED_TAU = 0.6308487409530461  # the url_student threshold in deploy/pi/bench/registry.toml
THRESHOLD_LINE = "threshold = 0.6308487409530461\n"


def _analyzer(p=0.1, scorer=None, threshold=URL_TAU, mode="quiet"):
    return EdgeAnalyzer(
        FakeDetector(p),
        FakeLLM(),
        THRESHOLD,
        url_scorer=scorer,
        url_threshold=threshold,
        detector_only_mode=mode,
    )


def _strip_timings(doc):
    doc = dict(doc)
    doc.pop("timings_ms", None)
    return doc


# --- disabled by default: byte-identical ------------------------------------------------------


@pytest.mark.parametrize("text", [SBI_TEXT, PLAIN_LINK_TEXT, LOW_TEXT])
def test_url_path_off_gives_the_same_result_as_before_it_existed(text):
    baseline = EdgeAnalyzer(FakeDetector(0.2), FakeLLM(), THRESHOLD).analyse(text)
    # Scorer present but no threshold (url_student not registered): still off.
    no_threshold = _analyzer(0.2, FakeScorer(0.99), threshold=None).analyse(text)
    # No scorer, threshold present: still off.
    no_scorer = _analyzer(0.2, None, threshold=URL_TAU).analyse(text)
    # Quiet mode (the default) with the URL model on and the score at or above tau: unchanged.
    quiet_url_on = _analyzer(0.2, FakeScorer(0.99), threshold=URL_TAU).analyse(text)
    for other in (no_threshold, no_scorer, quiet_url_on):
        assert json.dumps(_strip_timings(baseline), sort_keys=True) == json.dumps(
            _strip_timings(other), sort_keys=True
        )
    assert all(e["type"] != URL_MODEL_TYPE for e in baseline["evidence"])


def test_no_scorer_call_when_the_url_path_is_off():
    scorer = FakeScorer(0.99)
    _analyzer(0.2, scorer, threshold=None).analyse(SBI_TEXT)
    assert scorer.calls == 0


def _registry_text(without_threshold=False):
    text = (REPO / "deploy/pi/bench/registry.toml").read_text(encoding="utf-8")
    assert text.count(THRESHOLD_LINE) == 1
    return text.replace(THRESHOLD_LINE, "") if without_threshold else text


def test_registry_without_threshold_disables_the_url_path(tmp_path, monkeypatch):
    registry = _registry_text(without_threshold=True)
    root = tmp_path / "root"
    (root / "deploy/pi/bench").mkdir(parents=True)
    (root / "deploy/pi/bench/registry.toml").write_text(registry, encoding="utf-8")
    from rakshak.fusion import url_expert

    def boom(*a, **k):
        raise AssertionError("the model must not be opened without a threshold")

    monkeypatch.setattr(url_expert, "FrozenURLExpert", boom)
    scorer, threshold, info = svc.load_url_model(root)
    assert scorer is None and threshold is None
    assert info["enabled"] is False
    assert "no threshold" in info["reason"]


def test_registry_threshold_and_ready_model_enable_the_url_path(tmp_path, monkeypatch):
    registry = _registry_text()
    root = tmp_path / "root"
    (root / "deploy/pi/bench").mkdir(parents=True)
    (root / "deploy/pi/bench/registry.toml").write_text(registry, encoding="utf-8")
    import rbench.registry as rreg

    from rakshak.fusion import url_expert

    seen = {}
    monkeypatch.setattr(rreg, "model_state", lambda r, m, hash_it=True: {"status": "ready"})

    class Expert:
        def __init__(self, folder):
            seen["folder"] = Path(folder)

    monkeypatch.setattr(url_expert, "FrozenURLExpert", Expert)
    scorer, threshold, info = svc.load_url_model(root)
    assert isinstance(scorer, Expert) and threshold == REGISTERED_TAU
    assert info["enabled"] is True and info["threshold"] == REGISTERED_TAU
    assert seen["folder"] == root / "models" / "url_student"


def test_model_not_ready_disables_the_url_path_without_stopping_the_service(tmp_path, monkeypatch):
    registry = _registry_text()
    root = tmp_path / "root"
    (root / "deploy/pi/bench").mkdir(parents=True)
    (root / "deploy/pi/bench/registry.toml").write_text(registry, encoding="utf-8")
    import rbench.registry as rreg

    monkeypatch.setattr(rreg, "model_state", lambda r, m, hash_it=True: {"status": "sha mismatch"})
    scorer, threshold, info = svc.load_url_model(root)
    assert scorer is None and threshold is None
    assert info["status"] == "sha mismatch" and "not ready" in info["reason"]


# --- evidence at or above the threshold -----------------------------------------------------


def test_evidence_only_at_or_above_tau_and_the_text_has_domain_not_path():
    above = _analyzer(0.1, FakeScorer(0.7), mode="verify").analyse(SBI_TEXT)
    items = [e for e in above["evidence"] if e["type"] == URL_MODEL_TYPE]
    assert len(items) == 1
    assert items[0]["text"].startswith("sbi-kyc.in score=0.700")
    assert "/upd" not in items[0]["text"] and "acct" not in items[0]["text"]
    assert items[0]["weight"] == 0.5

    below = _analyzer(0.1, FakeScorer(0.4), mode="verify").analyse(SBI_TEXT)
    assert all(e["type"] != URL_MODEL_TYPE for e in below["evidence"])

    exactly = _analyzer(0.1, FakeScorer(URL_TAU), mode="verify").analyse(SBI_TEXT)
    assert any(e["type"] == URL_MODEL_TYPE for e in exactly["evidence"])


def test_message_score_is_the_max_over_its_urls():
    text = "Visit https://example-parcel.net/a or https://sbi-kyc.in/upd today"
    canon = canonicalize(text)
    assert len(canon.urls) == 2
    scorer = FakeScorer(by_url={canon.urls[0]: 0.2, canon.urls[1]: 0.8})
    hit = best_url_hit(canon.urls, scorer)
    assert hit is not None and hit.url == canon.urls[1] and hit.score == 0.8
    assert best_url_hit([], scorer) is None and scorer.calls == 1


def test_no_url_means_no_scorer_call_and_no_item():
    scorer = FakeScorer(0.99)
    out = _analyzer(0.1, scorer).analyse(LOW_TEXT)
    assert scorer.calls == 0
    assert all(e["type"] != URL_MODEL_TYPE for e in out["evidence"])


def test_item_ids_stay_unique_after_a_filter():
    canon = canonicalize(SBI_TEXT)
    rules = [e for e in run_rules(canon) if e.type == "brand_domain_mismatch"]
    hit = UrlHit(canon.urls[0], 0.9)
    out = add_url_model_evidence(rules, canon, hit, URL_TAU)
    ids = [e.id for e in out]
    assert len(ids) == len(set(ids)) and ids[-1] == "E" + str(int(rules[-1].id[1:]) + 1)


# --- mode rule: quiet ignores URL evidence, verify includes it ----------------------------


def test_quiet_mode_ignores_url_evidence_even_at_or_above_tau():
    scorer = FakeScorer(0.9)
    quiet = _analyzer(0.05, scorer, mode="quiet").analyse(PLAIN_LINK_TEXT)
    none = _analyzer(0.05, None, mode="quiet").analyse(PLAIN_LINK_TEXT)
    assert scorer.calls == 0
    assert all(e["type"] != URL_MODEL_TYPE for e in quiet["evidence"])
    assert _strip_timings(quiet) == _strip_timings(none)
    assert quiet["decision"]["scam_state"] == "benign" and quiet["decision"]["risk"] == "LOW"


def test_verify_mode_includes_url_evidence_at_or_above_tau():
    scorer = FakeScorer(0.9)
    out = _analyzer(0.05, scorer, mode="verify").analyse(PLAIN_LINK_TEXT)
    assert scorer.calls == 1
    assert [e["type"] for e in out["evidence"]] == [URL_MODEL_TYPE]
    assert out["decision"]["scam_state"] == "uncertain" and out["decision"]["risk"] == "UNCERTAIN"
    below = _analyzer(0.05, FakeScorer(URL_TAU - 0.01), mode="verify").analyse(PLAIN_LINK_TEXT)
    assert below["evidence"] == []


# --- verify-mode visibility and the HIGH combination -----------------------------------------


def test_url_item_is_visible_in_verify_mode_as_at_least_uncertain():
    out = _analyzer(0.05, FakeScorer(0.9), mode="verify").analyse(PLAIN_LINK_TEXT)
    d = out["decision"]
    assert d["detector_only"] is False
    assert d["scam_state"] == "uncertain"
    assert d["risk"] == "UNCERTAIN"
    assert d["action"] == Action.DO_NOT_OPEN_LINK.value
    assert out["advisory"]["path"] == "explainer"
    assert [r["code"] for r in out["remediation"]][0] == Action.DO_NOT_OPEN_LINK.value


def test_url_item_with_the_sms_student_over_threshold_is_high():
    out = _analyzer(0.9, FakeScorer(0.9), mode="verify").analyse(SBI_TEXT)
    assert out["decision"]["scam_state"] == "scam"
    assert out["decision"]["risk"] == "HIGH"
    assert any(e["type"] == URL_MODEL_TYPE for e in out["evidence"])


def test_remediation_text_matches_the_plan():
    from rakshak.remediation import SMS_EN

    title, steps = SMS_EN[Action.DO_NOT_OPEN_LINK.value]
    assert title == "Do not open the link; check the sender through the official app or number"
    assert steps[0] == "Do not open the link in this message."


# --- /analyze_url (service layer) -------------------------------------------------------------


def _check(scorer=None, threshold=URL_TAU, text=""):
    return check_url(text, scorer, threshold)


def test_phishing_by_the_url_model():
    out = _check(FakeScorer(0.9), text=PLAIN_LINK_TEXT)
    assert out["verdict"] == "phishing"
    assert out["url_model"] == {
        "enabled": True,
        "threshold": URL_TAU,
        "score": 0.9,
        "domain": "example.org",
        "fired": True,
    }
    assert out["risk"] == "UNCERTAIN"
    assert out["remediation"][0]["code"] == Action.DO_NOT_OPEN_LINK.value
    assert out["remediation_ok"] is True


def test_phishing_by_a_brand_domain_mismatch_without_the_model():
    out = _check(None, text=SBI_TEXT)
    assert out["verdict"] == "phishing"
    assert out["url_model"]["enabled"] is False and out["url_model"]["score"] is None
    assert out["risk"] == "HIGH"
    assert any(e["type"] == "brand_domain_mismatch" for e in out["evidence"])


def test_suspicious_for_a_shortened_link_below_tau():
    out = _check(FakeScorer(0.1), text="Tap https://bit.ly/3xYz to claim")
    assert out["verdict"] == "suspicious"
    assert out["url_model"]["fired"] is False
    assert any(e["type"] == "shortened_url" for e in out["evidence"])


def test_no_signal_for_text_without_a_link():
    out = _check(FakeScorer(0.99), text="Hey it's Sam, are we still on for lunch?")
    assert out["verdict"] == "no_signal"
    assert out["url_count"] == 0 and out["evidence"] == []
    assert out["risk"] == "LOW" and out["action"] == "IGNORE"
    assert out["url_model"]["score"] is None


def test_check_url_never_puts_the_path_in_the_model_evidence():
    out = _check(FakeScorer(0.9), text=SBI_TEXT)
    item = next(e for e in out["evidence"] if e["type"] == URL_MODEL_TYPE)
    assert "/upd" not in item["text"] and "acct" not in item["text"]


# --- /analyze_url (HTTP surface) --------------------------------------------------------------


def _device(temp=50.0):
    return lambda: {"temp_c": temp, "throttled": "0x0", "power_w": 4.0, "ts": 0.0}


def _energy_stub():
    class Stub:
        def start(self):
            pass

        def stop(self):
            return {
                "energy_j": 1.5,
                "mean_w": 3.0,
                "samples": 2,
                "duration_s": 0.5,
                "method": "trapezoid",
            }

    return Stub()


class CountingCheck:
    def __init__(self, scorer=None):
        self.calls = 0
        self.scorer = scorer

    def __call__(self, text):
        self.calls += 1
        return check_url(text, self.scorer, URL_TAU)


def _service(temp=50.0, check=None):
    return svc.Service(
        FakeAnalyzer(),
        threads=2,
        temp_limit=75.0,
        log_path=None,
        models={},
        device_fn=_device(temp),
        energy_factory=_energy_stub,
        url_check=check if check is not None else CountingCheck(FakeScorer(0.9)),
    )


class FakeAnalyzer:
    def analyse(self, text, lang="auto"):
        raise AssertionError("/analyze_url must not run the SMS analyser")


@pytest.fixture
def url_server():
    service = _service()
    httpd = svc.serve(service, "127.0.0.1", 0, (svc.HERE / "index.html").read_bytes())
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}", service
    httpd.shutdown()
    httpd.server_close()


def _post(url, body: bytes, ctype="application/json"):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_analyze_url_over_http_phishing(url_server):
    base, _ = url_server
    status, doc = _post(base + "/analyze_url", json.dumps({"url": SBI_TEXT}).encode())
    assert status == 200 and doc["verdict"] == "phishing"
    assert doc["url_model"]["domain"] == "sbi-kyc.in"
    assert doc["device"]["start"]["temp_c"] == 50.0
    assert doc["wall_ms"] >= 0 and doc["request_id"]


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b"{}",
        json.dumps({"url": ""}).encode(),
        json.dumps({"url": "   "}).encode(),
        json.dumps({"url": 5}).encode(),
        json.dumps({"url": "a" * 2001}).encode(),
    ],
)
def test_analyze_url_bad_input_is_400_and_never_checks(url_server, body):
    base, service = url_server
    status, doc = _post(base + "/analyze_url", body)
    assert status == 400 and "error" in doc
    assert service.url_check.calls == 0


def test_analyze_url_no_link_is_a_200_no_signal(url_server):
    base, _ = url_server
    status, doc = _post(base + "/analyze_url", json.dumps({"url": "hello there"}).encode())
    assert status == 200 and doc["verdict"] == "no_signal"


def test_analyze_url_thermal_refusal_runs_nothing():
    check = CountingCheck(FakeScorer(0.9))
    service = _service(temp=80.0, check=check)
    status, body = service.analyse_url(SBI_TEXT)
    assert status == 503 and body["error"] == "cooling"
    assert body["limit_c"] == 75.0
    assert check.calls == 0 and service.refused == 1


def test_analyze_url_at_the_limit_is_refused():
    check = CountingCheck(FakeScorer(0.9))
    service = _service(temp=75.0, check=check)
    assert service.analyse_url(SBI_TEXT)[0] == 503
    assert check.calls == 0


def test_analyze_url_without_a_check_is_503_not_a_crash():
    service = _service(check=None)
    service.url_check = None
    status, body = service.analyse_url(SBI_TEXT)
    assert status == 503 and "not loaded" in body["error"]


def test_page_has_the_check_a_link_box(url_server):
    base, _ = url_server
    with urllib.request.urlopen(base + "/", timeout=10) as r:
        html = r.read().decode("utf-8")
    assert "Check a link" in html and 'id="linkText"' in html and "/analyze_url" in html


# --- verifier and explainer ---------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "hinglish"])
def test_verifier_accepts_the_new_type_in_both_languages(lang):
    canon = canonicalize(SBI_TEXT)
    ev = add_url_model_evidence([], canon, UrlHit(canon.urls[0], 0.9), URL_TAU)
    pkg = build_package(canon, ev, "uncertain")
    advisory = template_only_advisory(pkg, lang)
    verdict = check_points(json.dumps(advisory, ensure_ascii=False), pkg, lang)
    assert verdict.ok, verdict.points[0].problems


def test_verifier_still_rejects_an_unknown_domain_in_the_new_type_summary():
    canon = canonicalize(SBI_TEXT)
    ev = add_url_model_evidence([], canon, UrlHit(canon.urls[0], 0.9), URL_TAU)
    pkg = build_package(canon, ev, "uncertain")
    out = {
        "summary": "This link goes to other-bank.com, do not open it.",
        "points": ["This message contains a link that the URL model scores as high risk."],
    }
    verdict = check_points(json.dumps(out), pkg, "en")
    assert verdict.summary_ok is False


def test_prompt_facts_have_the_domain_and_score_but_not_the_path():
    canon = canonicalize(SBI_TEXT)
    ev = add_url_model_evidence([], canon, UrlHit(canon.urls[0], 0.9), URL_TAU)
    pkg = build_package(canon, ev, "uncertain")
    facts = facts_block(pkg, "en")
    assert "sbi-kyc.in score 0.90" in facts
    assert "/upd" not in facts and "acct" not in facts
    view = pkg.to_prompt_json()
    assert "sbi-kyc.in" in view and "/upd" not in view and "acct" not in view


def test_remediation_is_valid_for_the_package():
    canon = canonicalize(SBI_TEXT)
    ev = add_url_model_evidence([], canon, UrlHit(canon.urls[0], 0.9), URL_TAU)
    pkg = build_package(canon, ev, "uncertain")
    action = choose_action(pkg)
    items = remediation_for(pkg, action.value, "en")
    assert items and items[0]["code"] == Action.DO_NOT_OPEN_LINK.value
    assert remediation_problems(items, pkg) == []


# --- eval_groundlm ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def E():
    spec = importlib.util.spec_from_file_location(
        "eval_groundlm", REPO / "deploy/pi/service/eval_groundlm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_groundlm"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_eval_decide_without_url_is_unchanged(E):
    plain = E.decide(SBI_TEXT, 0.1, THRESHOLD, mode="verify")
    none = E.decide(SBI_TEXT, 0.1, THRESHOLD, mode="verify", url_score=None, url_threshold=None)
    assert plain == none
    assert "phishing_url_model" not in plain["types"]


def test_eval_decide_adds_the_item_through_the_shared_path(E):
    below = E.decide(
        PLAIN_LINK_TEXT, 0.1, THRESHOLD, "verify", url_score=0.4, url_threshold=URL_TAU
    )
    at = E.decide(
        PLAIN_LINK_TEXT, 0.1, THRESHOLD, "verify", url_score=URL_TAU, url_threshold=URL_TAU
    )
    assert "phishing_url_model" not in below["types"]
    assert at["n_evidence"] == below["n_evidence"] + 1
    assert "phishing_url_model" in at["types"]
    assert at["state"] == "uncertain" and at["risk"] == "UNCERTAIN"
    assert at["action"] == Action.DO_NOT_OPEN_LINK.value
    # The rules-only cross-check stays rules-only.
    assert at["rules_only_risk"] == below["rules_only_risk"]


def test_eval_decide_quiet_ignores_the_url_item_at_or_above_tau(E):
    plain = E.decide(PLAIN_LINK_TEXT, 0.1, THRESHOLD, "quiet")
    with_url = E.decide(
        PLAIN_LINK_TEXT, 0.1, THRESHOLD, "quiet", url_score=0.9, url_threshold=URL_TAU
    )
    assert with_url == plain
    assert "phishing_url_model" not in with_url["types"]
    verify = E.decide(
        PLAIN_LINK_TEXT, 0.1, THRESHOLD, "verify", url_score=0.9, url_threshold=URL_TAU
    )
    assert "phishing_url_model" in verify["types"]


def test_eval_decide_high_when_the_student_is_over_threshold(E):
    out = E.decide(SBI_TEXT, 0.9, THRESHOLD, "verify", url_score=0.9, url_threshold=URL_TAU)
    assert out["state"] == "scam" and out["risk"] == "HIGH"


def test_url_scores_file_is_read_and_validated(E, tmp_path):
    good = tmp_path / "url_scores.parquet"
    pd.DataFrame(
        {
            "id": ["1", "2", "3"],
            "set": ["groundlm", "smishx", "campaign_test"],
            "score": [0.9, float("nan"), 0.2],
            "has_url": [True, False, True],
        }
    ).to_parquet(good, index=False)
    m = E.url_score_map(good)
    assert m == {("groundlm", "1"): 0.9, ("smishx", "2"): None, ("campaign_test", "3"): 0.2}

    bad_set = tmp_path / "bad_set.parquet"
    pd.DataFrame({"id": ["1"], "set": ["other"], "score": [0.9], "has_url": [True]}).to_parquet(
        bad_set, index=False
    )
    with pytest.raises(SystemExit, match="unknown set values"):
        E.url_score_map(bad_set)

    dup = tmp_path / "dup.parquet"
    pd.DataFrame(
        {"id": ["1", "1"], "set": ["groundlm"] * 2, "score": [0.9, 0.1], "has_url": [True, True]}
    ).to_parquet(dup, index=False)
    with pytest.raises(SystemExit, match="duplicate"):
        E.url_score_map(dup)


def test_apply_url_model_changes_only_the_rows_with_a_url_score(E):
    texts = pd.DataFrame(
        {"dataset": ["groundlm", "groundlm"], "id": ["1", "2"], "text": [PLAIN_LINK_TEXT, LOW_TEXT]}
    )
    probs = [0.1, 0.1]
    rows = []
    for ds, rid, text, p in zip(texts["dataset"], texts["id"], texts["text"], probs, strict=True):
        rows.append(
            {
                "dataset": ds,
                "id": rid,
                "label_binary": 0,
                "prob": p,
                **E.decide(text, p, THRESHOLD, "verify"),
            }
        )
    scored = pd.DataFrame(rows)
    before_rules = scored["rules_only_risk"].tolist()
    url_map = {("groundlm", "1"): 0.9, ("groundlm", "2"): None}
    out = E.apply_url_model(scored, texts, url_map, URL_TAU, THRESHOLD, mode="verify")
    assert out["types"].iloc[0].count("phishing_url_model") == 1
    quiet = E.apply_url_model(scored, texts, url_map, URL_TAU, THRESHOLD, mode="quiet")
    assert not quiet["types"].str.contains("phishing_url_model").any()
    assert quiet["risk"].tolist() == scored["risk"].tolist()
    assert out["types"].iloc[1] == scored["types"].iloc[1]
    assert out["risk"].iloc[0] == "UNCERTAIN" and scored["risk"].iloc[0] == "LOW"
    assert out["rules_only_risk"].tolist() == before_rules
    assert out["prob"].tolist() == probs


def test_apply_url_model_refuses_misaligned_rows(E):
    scored = pd.DataFrame({"dataset": ["groundlm"], "id": ["9"], "prob": [0.1]})
    texts = pd.DataFrame({"dataset": ["groundlm"], "id": ["1"], "text": [SBI_TEXT]})
    with pytest.raises(SystemExit, match="rescore"):
        E.apply_url_model(scored, texts, {}, URL_TAU, THRESHOLD)


def test_url_threshold_and_scores_must_come_together(E, tmp_path):
    with pytest.raises(SystemExit):
        E.main(["run", "--out-dir", str(tmp_path), "--url-threshold", "0.5"])
    with pytest.raises(SystemExit):
        E.main(["run", "--out-dir", str(tmp_path), "--url-scores", "x.parquet"])
    with pytest.raises(SystemExit):
        E.main(
            [
                "run",
                "--out-dir",
                str(tmp_path),
                "--url-scores",
                "x.parquet",
                "--url-threshold",
                "1.5",
            ]
        )


@pytest.mark.parametrize("mode", ["quiet", "verify"])
def test_analyze_url_is_the_same_in_every_mode(mode):
    service = svc.Service(
        FakeAnalyzer(),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={},
        detector_only_mode=mode,
        device_fn=_device(50.0),
        energy_factory=_energy_stub,
        url_check=CountingCheck(FakeScorer(0.9)),
    )
    status, body = service.analyse_url(SBI_TEXT)
    assert status == 200 and body["verdict"] == "phishing"
    assert body["url_model"]["fired"] is True
    assert any(e["type"] == URL_MODEL_TYPE for e in body["evidence"])


@pytest.mark.parametrize("mode", ["quiet", "verify"])
def test_health_reports_whether_url_evidence_is_visible(mode):
    service = svc.Service(
        FakeAnalyzer(),
        threads=1,
        temp_limit=75.0,
        log_path=None,
        models={svc.URL_MODEL_ID: {"enabled": True, "threshold": REGISTERED_TAU}},
        detector_only_mode=mode,
        device_fn=_device(50.0),
        energy_factory=_energy_stub,
    )
    assert service.health()["url_model"] == {
        "enabled": True,
        "threshold": REGISTERED_TAU,
        "evidence_visible": mode == "verify",
    }
