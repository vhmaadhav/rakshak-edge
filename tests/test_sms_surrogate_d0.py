"""Tests for kaggle/jobs/sms_surrogate_d0.py (plan section 4). Synthetic inputs only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_surrogate_d0 as job  # noqa: E402

from rakshak.data import surrogates as S  # noqa: E402

SENTINEL = "ZQXSENTINEL"
SENDER_SENTINEL = "ZQXBANK"
R0 = (0.5, 0.6, 0.7)


def _fpr(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": [0.0, 1.0]}


def _at(fprs: dict[str, tuple[int, int]], taus=(0.5, 0.6, 0.7)) -> dict:
    out = {}
    for i, (name, (k, n)) in enumerate(fprs.items()):
        out[name] = {"tau": taus[i], "legit_fpr": _fpr(k, n), "scam_recall": _fpr(0, 0)}
    return out


# ---- the H1 rule ----


def test_h1_holds_when_filled_is_a_tenth_or_less_at_every_tau():
    scrub = _at({"r0_1": (90, 100), "r0_2": (95, 100), "r0_3": (99, 100)})
    fill = _at({"r0_1": (5, 100), "r0_2": (6, 100), "r0_3": (9, 100)})
    h1 = job.h1_rule(scrub, fill)
    assert h1["holds"] is True
    assert h1["verdict_tau_name"] == "r0_2"
    assert h1["verdict_tau"] == pytest.approx(0.6)
    assert "not triggered" in h1["stop_rule_1"]
    assert all(r["holds"] for r in h1["per_threshold"])


def test_h1_fails_when_the_filled_fpr_is_not_ten_times_lower():
    scrub = _at({"r0_1": (90, 100), "r0_2": (90, 100), "r0_3": (90, 100)})
    fill = _at({"r0_1": (20, 100), "r0_2": (20, 100), "r0_3": (20, 100)})
    h1 = job.h1_rule(scrub, fill)
    assert h1["holds"] is False
    assert "stop rule 1 triggered" in h1["stop_rule_1"]
    assert "S0 to S2 do not run" in h1["stop_rule_1"]


def test_h1_boundary_is_inclusive_at_exactly_one_tenth():
    scrub = _at({"r0_1": (50, 100), "r0_2": (50, 100), "r0_3": (50, 100)})
    fill = _at({"r0_1": (5, 100), "r0_2": (5, 100), "r0_3": (5, 100)})
    assert job.h1_rule(scrub, fill)["holds"] is True
    fill_over = _at({"r0_1": (6, 100), "r0_2": (6, 100), "r0_3": (6, 100)})
    assert job.h1_rule(scrub, fill_over)["holds"] is False


def test_h1_zero_scrubbed_fpr_does_not_hold():
    scrub = _at({"r0_1": (0, 100), "r0_2": (0, 100), "r0_3": (0, 100)})
    fill = _at({"r0_1": (0, 100), "r0_2": (0, 100), "r0_3": (0, 100)})
    h1 = job.h1_rule(scrub, fill)
    assert h1["holds"] is False
    assert all("nothing to reduce" in r["reason"] for r in h1["per_threshold"])


def test_h1_zero_filled_fpr_with_positive_scrubbed_holds():
    scrub = _at({"r0_1": (3, 100), "r0_2": (3, 100), "r0_3": (3, 100)})
    fill = _at({"r0_1": (0, 100), "r0_2": (0, 100), "r0_3": (0, 100)})
    h1 = job.h1_rule(scrub, fill)
    assert h1["holds"] is True
    assert all("unbounded" in r["reason"] for r in h1["per_threshold"])


def test_h1_verdict_is_the_median_tau_not_the_first_listed():
    # Listed out of order: sorted by tau the middle is r0_3 (0.5), not r0_1 (0.9).
    taus = (0.9, 0.2, 0.5)
    scrub = _at({"r0_1": (90, 100), "r0_2": (90, 100), "r0_3": (90, 100)}, taus)
    fill_r3_fails = _at({"r0_1": (1, 100), "r0_2": (1, 100), "r0_3": (20, 100)}, taus)
    h1 = job.h1_rule(scrub, fill_r3_fails)
    assert h1["verdict_tau_name"] == "r0_3"
    assert h1["holds"] is False
    fill_r3_holds = _at({"r0_1": (90, 100), "r0_2": (90, 100), "r0_3": (1, 100)}, taus)
    assert job.h1_rule(scrub, fill_r3_holds)["holds"] is True


def test_h1_needs_three_taus():
    with pytest.raises(ValueError):
        job.h1_rule(_at({"r0_1": (1, 2)}), _at({"r0_1": (1, 2)}))


def test_h1_rule_text_is_written_into_the_output():
    h1 = job.h1_rule(
        _at({"r0_1": (9, 10), "r0_2": (9, 10), "r0_3": (9, 10)}),
        _at({"r0_1": (0, 10), "r0_2": (0, 10), "r0_3": (0, 10)}),
    )
    assert h1["rule"] == job.H1_RULE
    assert "median" in h1["rule"]


# ---- tables and helpers ----


def test_rates_and_table_on_a_tiny_frame():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.8])
    rates = job.rates_at(labels, scores, 0.5)
    assert rates["legit_fpr"]["k"] == 2 and rates["legit_fpr"]["n"] == 4
    assert rates["legit_fpr"]["rate"] == pytest.approx(0.5)
    assert rates["scam_recall"]["k"] == 1 and rates["scam_recall"]["n"] == 2
    lo, hi = rates["legit_fpr"]["wilson95"]
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    table = job.version_table(labels, scores, {"shipped": 0.35, "r0_1": 0.5})
    assert table["legit_scores"]["n"] == 4
    assert table["legit_scores"]["min"] == pytest.approx(0.1)
    assert table["legit_scores"]["max"] == pytest.approx(0.9)
    assert table["scam_scores"]["n"] == 2
    assert table["at"]["shipped"]["tau"] == 0.35
    assert table["at"]["r0_1"]["legit_fpr"]["k"] == 2


def test_empty_class_gives_none_rates_not_nan():
    rates = job.rates_at(np.array([1, 1]), np.array([0.9, 0.1]), 0.5)
    assert rates["legit_fpr"]["rate"] is None
    assert rates["legit_fpr"]["wilson95"] == [None, None]
    q = job.quantiles(np.array([]))
    assert q["n"] == 0 and q["p50"] is None


def test_tag_share_counts_rows_with_a_tag_before_filling():
    share = job.tag_share(["a <NAME>", "plain", "<OTP> and <URL>", "x < y"])
    assert share == {"rows": 4, "with_tag": 2, "share": 0.5}
    assert job.tag_share([]) == {"rows": 0, "with_tag": 0, "share": None}


def test_tau_map_names_the_shipped_and_the_r0_taus():
    taus = job.tau_map(R0)
    assert list(taus) == ["shipped", "r0_1", "r0_2", "r0_3"]
    assert taus["shipped"] == job.SHIPPED_TAU
    with pytest.raises(ValueError):
        job.tau_map((0.5, 0.6))


def test_parse_args_requires_three_taus_in_range(capsys):
    args = job.parse_args(["--r0-taus", "0.5", "0.6", "0.7"])
    assert args.v0_onnx == "/kaggle/input"
    assert args.r0_taus == [0.5, 0.6, 0.7]
    with pytest.raises(SystemExit):
        job.parse_args([])
    with pytest.raises(SystemExit):
        job.parse_args(["--r0-taus", "0.5", "0.6", "1.5"])


def test_build_sets_refuses_legit_with_scam_label_and_empty_sets():
    phone1 = pd.DataFrame({"id": ["a", "b"], "text": ["x", "y"], "label": [0, 1]})
    legit = pd.DataFrame({"id": ["c"], "text": ["z"], "label": [0]})
    imc = pd.DataFrame({"id": ["d"], "text": ["w"], "label": [1]})
    sets = job.build_sets(phone1, legit, imc)
    assert set(sets) == set(job.SET_NAMES)
    assert list(sets[job.SET_PHONE1].columns) == ["id", "text", "label"]
    bad = legit.assign(label=[1])
    with pytest.raises(ValueError):
        job.build_sets(phone1, bad, imc)
    with pytest.raises(ValueError):
        job.build_sets(phone1, legit, imc.iloc[0:0])


def test_attach_row_ids_matches_by_text_and_sender_in_source_order():
    source = pd.DataFrame(
        {
            "id": ["r1", "r2", "r3", "r4"],
            "text": ["same", "other", "same", "third"],
            "sender": ["s1", "s2", "s1", "s3"],
        }
    )
    sub = pd.DataFrame({"text": ["same", "third", "same"], "sender": ["s1", "s3", "s1"]})
    assert job.attach_row_ids(sub, source) == ["r1", "r4", "r3"]
    missing = pd.DataFrame({"text": ["nope"], "sender": ["s9"]})
    with pytest.raises(ValueError):
        job.attach_row_ids(missing, source)


# ---- the v0 sha check ----


def test_sha_mismatch_raises(tmp_path):
    (tmp_path / "best_student.onnx").write_bytes(b"not the shipped model")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        job.locate_v0(tmp_path)


def test_missing_file_and_missing_folder_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        job.locate_v0(tmp_path)
    with pytest.raises(FileNotFoundError):
        job.locate_v0(tmp_path / "absent")


def test_a_matching_sha_is_located(tmp_path, monkeypatch):
    payload = b"a stand-in model"
    path = tmp_path / "kernel" / "best_student.onnx"
    path.parent.mkdir()
    path.write_bytes(payload)
    monkeypatch.setattr(job, "V0_SHA", hashlib.sha256(payload).hexdigest())
    assert job.locate_v0(tmp_path) == path


def test_pinned_v0_sha_is_the_registry_pin():
    assert job.V0_SHA == "5341e14ca02a523bd746a71dd67903439e9e6c6f6fc3193fa1c6f786953f2466"
    assert job.SHIPPED_TAU == 0.35127517580986023


# ---- summary, report and the leak scan ----


def _scored(sentinel: str) -> dict:
    rng = np.random.default_rng(0)
    n_legit, n_scam = 100, 20
    labels_p1 = np.array([0] * n_legit + [1] * n_scam)
    texts_p1 = [f"{sentinel} legit {i} <AMOUNT>" for i in range(n_legit)] + [
        f"{sentinel} scam {i} <URL>" for i in range(n_scam)
    ]
    scrub = np.concatenate([rng.uniform(0.9, 1.0, n_legit), rng.uniform(0.9, 1.0, n_scam)])
    filled = np.concatenate([rng.uniform(0.0, 0.2, n_legit), rng.uniform(0.9, 1.0, n_scam)])
    texts_legit = [f"{sentinel} held {i}" for i in range(30)]
    texts_imc = [f"{sentinel} imc {i} <URL>" for i in range(40)]
    return {
        job.SET_PHONE1: {
            "labels": labels_p1,
            "texts": texts_p1,
            "scores": {"scrubbed": scrub, "filled": filled},
        },
        job.SET_LEGIT: {
            "labels": np.zeros(30, dtype=np.int64),
            "texts": texts_legit,
            "scores": {"scrubbed": np.full(30, 0.95), "filled": rng.uniform(0.0, 0.1, 30)},
        },
        job.SET_IMC25: {
            "labels": np.ones(40, dtype=np.int64),
            "texts": texts_imc,
            "scores": {"scrubbed": np.full(40, 0.99), "filled": rng.uniform(0.8, 1.0, 40)},
        },
    }


def test_summarise_reports_the_sets_and_a_holding_h1():
    result = job.summarise(_scored(SENTINEL), R0)
    assert result["h1"]["holds"] is True
    s = result["sets"][job.SET_PHONE1]
    assert s["n"] == 120 and s["n_legit"] == 100 and s["n_scam"] == 20
    assert s["tag_share_before_fill"]["with_tag"] == 120
    assert set(s["scrubbed"]["at"]) == {"shipped", "r0_1", "r0_2", "r0_3"}
    assert result["sets"][job.SET_IMC25]["n_legit"] == 0


def test_leak_scan_no_input_text_or_sender_in_metrics_or_report():
    scored = _scored(SENTINEL)
    payload = {
        "job": job.JOB,
        "status": "scored",
        "args": {"r0_taus": list(R0)},
        **job.summarise(scored, R0),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    report = job.render_report({**payload, "v0": {"sha256": job.V0_SHA}, "surrogate_seed": 1})
    for text in [t for entry in scored.values() for t in entry["texts"]]:
        assert text not in blob
        assert text not in report
    assert SENTINEL not in blob and SENTINEL not in report
    assert SENDER_SENTINEL not in blob
    assert '"text"' not in blob
    assert '"id"' not in blob


def test_filled_text_never_reaches_the_output_either():
    scored = _scored(SENTINEL)
    ids = [f"row-{i}" for i in range(len(scored[job.SET_PHONE1]["texts"]))]
    filled = S.fill_many(ids, scored[job.SET_PHONE1]["texts"])
    S.assert_no_tags(filled)
    blob = json.dumps(job.summarise(scored, R0), default=str)
    for text in filled:
        assert text not in blob


def test_render_report_states_the_h1_verdict_and_the_stop_rule():
    payload = {
        "job": job.JOB,
        "status": "scored",
        "v0": {"sha256": job.V0_SHA},
        "surrogate_seed": S.SURROGATE_SEED,
        **job.summarise(_scored(SENTINEL), R0),
    }
    text = job.render_report(payload)
    assert "H1 holds: True" in text
    assert "H1 (filled phone-1 legit FPR" in text
    assert "stop rule 1" in text.lower() or "H1 holds" in text


def test_stopped_status_renders_without_tables():
    text = job.render_report({"status": job.inbox.STOP_STATUS, "reason": "not enough rows"})
    assert job.inbox.STOP_STATUS in text
    assert "H1" not in text


def test_job_declares_redaction_and_no_row_level_output():
    assert job.REDACT_ERRORS is True
    source = (JOBS / "sms_surrogate_d0.py").read_text()
    assert "probs" not in source.split("def _write")[0].split("def run")[1]
    assert "to_parquet" not in source
    assert "write_thresholds" not in source
