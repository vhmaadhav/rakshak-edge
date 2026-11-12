"""Tests for kaggle/jobs/sms_inbox_retrain.py. Synthetic fixtures only; no torch, Kaggle or data.

The job imports torch only inside run(), so the pure helpers, the pools, the stop rule, the
threshold-before-test order (with fake callbacks), the outputs and the privacy scan run here.
Sentinel tokens stand in for inbox text and sender strings, and the scan fails if one reaches an
output file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import sms_inbox_retrain as job  # noqa: E402
import sms_retrain as base  # noqa: E402

from rakshak.data import retrain_prep as RP  # noqa: E402
from rakshak.models.encoding import N_RULES  # noqa: E402
from rakshak.models.features import Featurized, with_side_extra  # noqa: E402
from rakshak.models.sender_features import (  # noqa: E402
    SENDER_FEATURE_NAMES,
    sender_features,
    sender_matrix,
)
from tools.inbox_export.export import COLUMNS  # noqa: E402

SENTINEL = "ZQXSENTINEL"
SENDER_SENTINEL = "ZQXBANK"


def _word_text(rng: np.random.Generator, n_words: int = 9) -> str:
    """Random letters: near-duplicates occur only when a template is reused on purpose."""
    letters = list("abcdefghijklmnopqrstuvwxyz")
    words = ["".join(rng.choice(letters, size=7)) for _ in range(n_words)]
    return " ".join(words)


def _inbox_rows(seed: int, n_senders: int, per_sender: int, label_policy: str, prefix: str) -> list:
    """Each sender gets its own template family; messages in a family differ only in digits."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_senders):
        sender = f"AX-{SENDER_SENTINEL}{prefix}{s:02d}-S"
        base_text = _word_text(rng)
        for k in range(per_sender):
            text = f"{SENTINEL} {base_text} order {k * 7 + s} ref {1000 + k}"
            rows.append({"id": f"{prefix}{s}-{k}", "text": text, "sender": sender,
                         "sender_style": "alphanumeric", "label_policy": label_policy,
                         "synthetic": False, "human_reviewed": True,
                         "scrubber_version": "t"})
    return rows


def _donor_df(legit_senders: int = 30, per_sender: int = 6, scam_senders: int = 8) -> pd.DataFrame:
    rows = _inbox_rows(11, legit_senders, per_sender, "promotion", "L")
    rows += _inbox_rows(12, scam_senders, 3, "scam", "S")
    return pd.DataFrame(rows)


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    out, _ = job.label_inbox(df)
    return out


# ---- arguments, widths, label map ----


def test_arguments_and_side_widths():
    args = job.parse_args(["--arm", "R2"])
    assert args.arm == "R2" and args.seeds == [1, 2, 3]
    with pytest.raises(SystemExit):
        job.parse_args([])
    with pytest.raises(SystemExit):
        job.parse_args(["--arm", "A"])
    with pytest.raises(SystemExit):
        job.parse_args(["--arm", "R1", "--seeds", "1", "1"])
    assert N_RULES == 16
    assert (job.side_width("R0"), job.side_width("R1"), job.side_width("R2")) == (16, 16, 21)
    assert len(SENDER_FEATURE_NAMES) == 5


def test_inbox_columns_match_the_exporter():
    assert tuple(job.INBOX_COLUMNS) == tuple(COLUMNS)


def test_label_map_and_drops():
    rows = []
    for policy in ("benign_transactional", "promotion", "scam", "benign_personal", "unclear"):
        r = _inbox_rows(3, 1, 1, policy, "M")[0]
        rows.append(r)
    unreviewed = dict(_inbox_rows(4, 1, 1, "scam", "U")[0], human_reviewed=False)
    rows.append(unreviewed)
    out, counts = job.label_inbox(pd.DataFrame(rows))
    assert sorted(out["label"].tolist()) == [0, 0, 1]
    assert counts["dropped_personal_or_unclear"] == 2
    assert counts["not_human_reviewed"] == 1
    assert counts["kept_legit"] == 2 and counts["kept_scam"] == 1
    assert "sender" in out.columns and out["text"].str.contains(SENTINEL).all()


def test_codex_labelled_rows_kept_other_sources_dropped():
    codex = dict(_inbox_rows(6, 1, 1, "scam", "C")[0], human_reviewed=False, label_source="codex")
    other = dict(_inbox_rows(7, 1, 1, "scam", "O")[0], human_reviewed=False, label_source="regex")
    human = dict(_inbox_rows(8, 1, 1, "promotion", "H")[0], label_source="human")
    out, counts = job.label_inbox(pd.DataFrame([codex, other, human]))
    assert sorted(out["label"].tolist()) == [0, 1]
    assert counts["codex_labelled"] == 1 and counts["not_human_reviewed"] == 1


def test_label_inbox_refuses_synthetic_and_missing_columns():
    df = pd.DataFrame(_inbox_rows(5, 1, 2, "scam", "X"))
    with pytest.raises(ValueError):
        job.label_inbox(df.assign(synthetic=True))
    with pytest.raises(ValueError):
        job.label_inbox(df.drop(columns=["sender"]))


# ---- sender features ----


def test_sender_features_encoding_and_zero_rows():
    assert sender_features(None).sum() == 0
    assert sender_features("").sum() == 0
    assert sender_features("9876543210").sum() == 0  # person-to-person number
    assert sender_features("AX-HDFCBK-T").tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    m = sender_matrix([None, "AX-HDFCBK-T", ""])
    assert m.shape == (3, 5) and m[0].sum() == 0 and m[2].sum() == 0
    fz = Featurized(texts=["a", "b"], raw_bytes=np.zeros((2, 4), np.uint16),
                    canon_bytes=np.zeros((2, 4), np.uint16),
                    side_raw=np.ones((2, N_RULES), np.float32),
                    side_canon=np.ones((2, N_RULES), np.float32))
    out = with_side_extra(fz, sender_matrix([None, "AX-HDFCBK-T"]))
    assert out.side_raw.shape == (2, 21) and out.side_canon.shape == (2, 21)
    assert out.side_raw[0, N_RULES:].sum() == 0
    assert out.side_canon[1, N_RULES + 3] == 1.0


def test_model_side_width_when_torch_is_present():
    pytest.importorskip("torch")
    from rakshak.models.student import ModelSpec
    from rakshak.models.student import side_width as model_side_width

    assert model_side_width(ModelSpec("dual", "base", "dual", True)) == 16
    assert model_side_width(ModelSpec("dual", "base", "dual", True, side_extra=5)) == 21
    with pytest.raises(ValueError):
        ModelSpec("dual", "base", "dual", False, side_extra=5)


# ---- sender-group split, dedup and pools ----


def test_sender_split_is_disjoint_and_test_donor_never_reaches_train_or_val():
    donor = _labelled(_donor_df())
    test_raw = _inbox_rows(21, 6, 3, "promotion", "T")
    test_texts = [r["text"] for r in test_raw]
    # a training-donor row that repeats a test-donor text must be removed with its cluster
    leak = donor.iloc[[0]].copy()
    leak["text"] = test_texts[0]
    donor = pd.concat([donor, leak], ignore_index=True)
    pools = job.build_inbox_pools(donor, test_texts + ["campaign test text " * 5], 1000)
    st = pools["report"]["stages"]
    train_codes = set(pools["train_extra"]["sender_code"].tolist())
    val_codes = set(pools["val_legit"]["sender_code"].tolist()) | set(
        pools["val_scam"]["sender_code"].tolist())
    assert train_codes.isdisjoint(val_codes)
    assert st["sender_ids_disjoint"] is True
    assert st["sender_split"]["sender_ids_held_out"] >= 1
    train_clusters = set(pools["train_extra"]["cluster"].tolist())
    val_clusters = set(pools["val_legit"]["cluster"].tolist()) | set(
        pools["val_scam"]["cluster"].tolist())
    assert train_clusters.isdisjoint(val_clusters)
    banned = {RP.normalize(t) for t in test_texts}
    for frame in (pools["train_extra"], pools["val_legit"], pools["val_scam"]):
        assert not {RP.normalize(t) for t in frame["text"]} & banned
    assert st["rows_dropped_test_contaminated"] >= 1


def test_pools_legit_and_scam_labels_and_caps():
    donor = _labelled(_donor_df())
    pools = job.build_inbox_pools(donor, ["unrelated sentence for the test set"] * 2, 1000)
    assert set(pools["val_legit"]["label"].tolist()) <= {0}
    assert set(pools["val_scam"]["label"].tolist()) <= {1}
    per_group = pools["train_extra"].groupby("cap_group").size()
    assert int(per_group.max()) <= RP.PER_GROUP_CAP
    assert pools["report"]["stages"]["total_cap_budget"] == min(RP.TOTAL_CAP,
                                                                int(RP.TOTAL_CAP_FRACTION * 1000))


def test_template_novel_mask():
    train = ["alpha bravo charlie delta echo foxtrot golf hotel india juliet"]
    test = [train[0], "completely different words that share nothing with the others at all"]
    novel = job.novel_mask(test, train)
    assert novel.tolist() == [False, True]


def test_pool_frames_keep_sender_in_memory_only():
    pools = job.build_inbox_pools(_labelled(_donor_df()), ["campaign"], 1000)
    # the sender string is kept in memory for the R2 features; the outputs carry codes only
    assert "sender" in pools["train_extra"].columns
    report_blob = repr(pools["report"])
    assert SENDER_SENTINEL not in report_blob and SENTINEL not in report_blob


# ---- stop rule ----


@pytest.mark.parametrize(
    "donor,val,stops",
    [(499, 300, True), (500, 199, True), (500, 200, False), (800, 250, False)],
)
def test_stop_rule(donor, val, stops):
    report = {"stages": {"after_dedup_legit": donor, "val_legit_after_dedup": val}}
    reason = job.stop_reason(report)
    assert (reason is not None) == stops
    if stops:
        assert str(donor) in reason or str(val) in reason


def test_stop_status_string():
    assert job.STOP_STATUS == "stopped: insufficient data"


# ---- threshold before any test set ----


def test_threshold_written_before_any_test_set_is_scored(tmp_path):
    events_seen = []
    written = {}

    def fit():
        return "model"

    def score(model, name):
        if name in ("legit_val", "campaign_cal"):
            n = 300 if name == "legit_val" else 200
            return np.linspace(0.0, 0.2, n)
        # a post-threshold set: the threshold file must already exist
        assert (tmp_path / "thresholds.json").exists()
        events_seen.append(name)
        return np.linspace(0.0, 1.0, 50)

    def on_threshold(tau, fallback):
        written["tau"] = tau
        job.write_thresholds(tmp_path, {"seeds": {"1": {"tau": tau}}})

    y_cal = np.array([0, 1] * 100)
    pipe = base.run_seed_pipeline(
        fit=fit,
        score=score,
        rule=RP.threshold_rule,
        y_cal=y_cal,
        post_sets=["test_donor", "campaign_test"],
        on_threshold=on_threshold,
    )
    assert pipe["events"].index("threshold") < pipe["events"].index("score:test_donor")
    assert events_seen == ["test_donor", "campaign_test"]
    assert "tau" in written


# ---- metrics helpers ----


def test_donor_rates_and_median_seed():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.3, 0.8, 0.1])
    novel = np.array([True, False, True, True, False, True])
    out = job.donor_rates(labels, scores, 0.5, novel)
    assert out["legit_fpr"]["k"] == 1 and out["legit_fpr"]["n"] == 4
    assert out["scam_recall"]["k"] == 1 and out["scam_recall"]["n"] == 2
    assert out["legit_fpr_template_novel"]["n"] == 3
    assert job.median_seed({1: 0.02, 2: 0.005, 3: 0.01}) == 3
    assert job.median_seed({1: 0.01, 2: 0.01, 3: 0.01}) == 2  # ties sort by seed


def test_gates_g5_and_majority():
    rows = [{"seed": s, "groundlm_fp": 5, "smishx_tp": 200, "campaign_f1": 0.99,
             "onnx_bytes": 400_000, "donor_fpr": f}
            for s, f in ((1, 0.004), (2, 0.02), (3, 0.006))]
    out = job.gate_eval_inbox(rows)
    assert out["G5_test_donor_legit_fpr"]["passes"] is True
    assert out["arm_pass"] is True
    rows[1]["donor_fpr"] = 0.03
    rows[2]["donor_fpr"] = 0.03
    assert job.gate_eval_inbox(rows)["G5_test_donor_legit_fpr"]["passes"] is False


def _probs_for_seed(seed, tau, scores_by_group, groups, flags=None):
    n = len(groups)
    ids = [f"test_donor_{i}" for i in range(n)]
    return job.probs_frame(seed, tau, {
        "test_donor": {"ids": ids, "labels": np.zeros(n), "scores": scores_by_group,
                       "extra": {"group": np.asarray(groups), "novel": np.ones(n, bool)}},
    })


def test_paired_bootstrap_identical_models_and_clear_shift():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 5)
    base_scores = rng.uniform(0, 1, len(groups))
    r1 = _probs_for_seed(1, 0.5, base_scores, groups)
    same = job.paired_donor_fpr_diff(r1, r1, 1, 1, 200)
    assert same["diff_b_minus_a"] == 0.0
    assert same["excludes_zero"] is False
    # R2 lowers the scores of half the groups, which removes false positives
    lower = np.where(groups < 20, base_scores * 0.1, base_scores)
    r2 = _probs_for_seed(1, 0.5, lower, groups)
    diff = job.paired_donor_fpr_diff(r1, r2, 1, 1, 400)
    assert diff["diff_b_minus_a"] < 0
    assert diff["excludes_zero"] is True
    assert diff["ci95"][1] < 0


def test_paired_bootstrap_refuses_mismatched_rows():
    groups = np.repeat(np.arange(5), 2)
    a = _probs_for_seed(1, 0.5, np.zeros(10), groups)
    b = _probs_for_seed(1, 0.5, np.zeros(10), groups[::-1].copy())
    with pytest.raises(ValueError):
        job.paired_donor_fpr_diff(a, b, 1, 1, 10)


# ---- inputs and the v0 comparator ----


def test_find_one_requires_exactly_one_copy(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "train_donor.parquet").write_bytes(b"x")
    assert job.find_one("train_donor.parquet", [tmp_path]).name == "train_donor.parquet"
    (tmp_path / "b" / "train_donor.parquet").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        job.find_one("train_donor.parquet", [tmp_path])
    with pytest.raises(FileNotFoundError):
        job.find_one("missing.parquet", [tmp_path])


def test_v0_is_skipped_when_not_attached(tmp_path):
    donor = _labelled(_donor_df(2, 2, 1))
    out = job.score_v0_donors([tmp_path], donor, donor, np.ones(len(donor), bool))
    assert "skipped" in out


def test_job_dependencies_include_baseline_v0():
    text = (Path(__file__).resolve().parents[1] / "kaggle" / "build_kernel.py").read_text()
    expected = "'sms_inbox_retrain': ('kaggle/jobs/baseline_v0.py', 'kaggle/jobs/sms_retrain.py')"
    assert expected in text


# ---- privacy: a fake run writes every output; no sentinel text or sender may appear ----


def _fake_block(seed: int, tau: float, donor_rate: float) -> dict:
    return {
        "seed": seed, "tau": tau, "fallback_flag": False,
        "gate_row": {"groundlm_fp": 5, "smishx_tp": 200, "campaign_f1": 0.98},
        "groundlm_paired": {"negatives": {"mcnemar_exact_p": 0.5}},
        "legit_val": {"k": 1, "n": 200, "rate": 0.005, "wilson95": [0.0, 0.02]},
        "test_donor": {
            "legit_fpr": {"k": 1, "n": 100, "rate": donor_rate, "wilson95": [0.0, 0.05]},
            "legit_fpr_template_novel": {"k": 0, "n": 40, "rate": 0.0, "wilson95": [0.0, 0.09]},
            "scam_recall": {"k": 1, "n": 2, "rate": 0.5, "wilson95": [0.1, 0.9]},
        },
    }


def test_privacy_scan_of_every_output(tmp_path):
    donor = _labelled(_donor_df())
    test_raw = _inbox_rows(21, 6, 3, "promotion", "T")
    pools = job.build_inbox_pools(donor, [r["text"] for r in test_raw], 1000)
    test_df = _labelled(pd.DataFrame(test_raw))
    prep = {"arm": "R2", "test_donor": {"kept": len(test_df)},
            "train_donor": {"kept": len(donor)}, **pools["report"]}
    rng = np.random.default_rng(3)
    frames = []
    for seed in (1, 2, 3):
        tau = 0.5
        sets = {
            "legit_val": {"ids": [f"legit_val_{i}" for i in range(len(pools["val_legit"]))],
                          "labels": np.zeros(len(pools["val_legit"])),
                          "scores": rng.uniform(0, 1, len(pools["val_legit"]))},
            "test_donor": {"ids": [f"test_donor_{i}" for i in range(len(test_df))],
                           "labels": test_df["label"].to_numpy(),
                           "scores": rng.uniform(0, 1, len(test_df)),
                           "extra": {"group": np.arange(len(test_df)),
                                     "novel": np.ones(len(test_df), bool)}},
        }
        frames.append(job.probs_frame(seed, tau, sets))
    probs = pd.concat(frames, ignore_index=True)
    rows = [dict(_fake_block(s, 0.5, 0.01)["gate_row"], seed=s, donor_fpr=0.01) for s in (1, 2, 3)]
    payload = {
        "job": "sms_inbox_retrain", "arm": "R2", "status": "scored", "seeds": [1, 2, 3],
        "side_width": 21, "gates": job.gate_eval_inbox(rows),
        "per_seed": {str(s): _fake_block(s, 0.5, 0.01) for s in (1, 2, 3)},
        "v0": {}, "paired_r2_vs_r1": {"skipped": "not attached"}, "prep": prep,
    }
    job.write_thresholds(tmp_path, {"arm": "R2", "seeds": {"1": {"tau": 0.5}}})
    job.write_outputs(tmp_path, payload, probs, prep)
    # a stopped run writes the same kinds of files
    stop_dir = tmp_path / "stopped"
    job.write_outputs(stop_dir, {"job": "sms_inbox_retrain", "arm": "R1",
                                 "status": job.STOP_STATUS, "reason": "counts only",
                                 "counts": prep}, None, prep)

    # a unit that raises with fixture text in its message: the log and results carry no text
    ctx = _Ctx()
    units = job.RedactedUnits(ctx)
    units.run({"seed": 9}, _leaky, [test_df["text"].iloc[0]])
    (tmp_path / "run.log").write_text("\n".join(ctx.lines) + "\n", encoding="utf-8")
    (tmp_path / "results.json").write_text(
        json.dumps(ctx.results, default=str), encoding="utf-8")
    assert ctx.results["failures"] and ctx.lines

    scanned = 0
    for path in list(tmp_path.rglob("*")) + list(stop_dir.rglob("*")):
        if not path.is_file():
            continue
        scanned += 1
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
            cells = [str(v) for v in df.to_numpy().ravel().tolist()]
            blob = " ".join(cells + [str(c) for c in df.columns])
        else:
            blob = path.read_text(encoding="utf-8")
        assert SENTINEL not in blob, path.name
        assert SENDER_SENTINEL not in blob, path.name
        for text in donor["text"].tolist()[:5] + test_df["text"].tolist()[:5]:
            assert text not in blob, path.name
    assert scanned >= 9
    assert "stopped: insufficient data" in (stop_dir / "report.md").read_text()


# ---- error redaction: a raised message can carry inbox text or sender strings ----


class _Ctx:
    """Stands in for the runner's context: captures log lines and results."""

    def __init__(self):
        self.lines: list[str] = []
        self.results: dict = {}

    def log(self, msg):
        self.lines.append(str(msg))

    def save(self):
        pass


def _leaky(texts):
    raise ValueError(f"could not parse {texts[0]} from {SENDER_SENTINEL}")


def test_job_declares_redaction():
    assert job.REDACT_ERRORS is True


def test_redacted_units_drop_the_message_and_keep_frames():
    ctx = _Ctx()
    units = job.RedactedUnits(ctx)
    donor = _labelled(_donor_df(2, 2, 1))
    out = units.run({"seed": 1}, _leaky, [donor["text"].iloc[0]])
    assert out is None and units.failed == 1
    blob = repr(ctx.results) + "\n".join(ctx.lines)
    assert SENTINEL not in blob and SENDER_SENTINEL not in blob
    failure = ctx.results["failures"][0]
    assert failure["error"] == "ValueError (message redacted)"
    assert any("_leaky" in frame for frame in failure["frames"])


def test_run_wrapper_raises_without_the_message(monkeypatch):
    def boom(ctx):
        raise ValueError(f"row text {SENTINEL} from {SENDER_SENTINEL}")

    monkeypatch.setattr(job, "_run", boom)
    ctx = _Ctx()
    with pytest.raises(job.RedactedError) as info:
        job.run(ctx)
    assert str(info.value) == "ValueError (message redacted)"
    assert info.value.__cause__ is None and info.value.__suppress_context__
    blob = str(info.value) + "\n".join(ctx.lines)
    assert SENTINEL not in blob and SENDER_SENTINEL not in blob
    assert any(line.startswith("  at ") and "boom" in line for line in ctx.lines)


def test_runner_describe_error_honours_redaction():
    import traceback

    sys.path.insert(0, str(JOBS.parent))
    import build_kernel as bk

    template = bk.RUN_TEMPLATE
    start, end = template.index("def describe_error"), template.index("def main():")
    ns = {"traceback": traceback, "REDACT": False}
    exec(template[start:end], ns)
    try:
        _leaky([SENTINEL])
    except ValueError as exc:
        plain_error, plain_trace = ns["describe_error"](exc)
        assert SENTINEL in plain_error  # without the job opt-in the message is kept
        ns["REDACT"] = True
        error, trace = ns["describe_error"](exc)
    assert error == "ValueError (message redacted)"
    assert SENTINEL not in error and SENTINEL not in trace
    assert "_leaky" in trace and "line" in trace


# ---- S arms: arguments, the surrogate fill, rule 5, stop rule 2 (plan sections 5 to 7) ----

from rakshak.data import surrogates as S  # noqa: E402


def _gate_row(seed: int, smishx: float | None = 200, groundlm: int = 5, f1: float = 0.98,
              onnx: int = 1_000_000, donor: float = 0.005) -> dict:
    return {"seed": seed, "groundlm_fp": groundlm, "smishx_tp": smishx, "campaign_f1": f1,
            "onnx_bytes": onnx, "donor_fpr": donor}


def test_s_arm_arguments_and_recipe_mapping():
    a = job.parse_args(["--arm", "S0"])
    assert a.arm == "S0" and a.fill_surrogates is True
    assert job.parse_args(["--arm", "S2", "--r1-probs-dir", "x"]).fill_surrogates is True
    assert job.parse_args(["--arm", "R2"]).fill_surrogates is False
    with pytest.raises(SystemExit):
        job.parse_args(["--arm", "R0", "--fill-surrogates"])
    with pytest.raises(SystemExit):
        job.parse_args(["--arm", "S3"])
    assert [job.recipe_of(x) for x in ("S0", "S1", "S2", "R0", "R2")] == [
        "R0", "R1", "R2", "R0", "R2"]
    assert (job.side_width("S0"), job.side_width("S1"), job.side_width("S2")) == (16, 16, 21)
    assert "fill_surrogates" not in job.args_record(job.parse_args(["--arm", "R1"]))
    assert job.args_record(job.parse_args(["--arm", "S1"]))["fill_surrogates"] is True
    assert job.paired_key("R2") == "paired_r2_vs_r1" and job.paired_key("R0") == "paired_r2_vs_r1"
    assert job.paired_key("S2") == "paired_s2_vs_s1"
    assert job.fill_fields("R0", False) == {}
    assert job.fill_fields("S1", True) == {
        "fill_surrogates": True, "surrogate_seed": S.SURROGATE_SEED, "recipe": "R1"}


def test_fill_sms_frame_replaces_tags_and_keeps_untagged_text():
    sms = pd.DataFrame({
        "id": ["10", "11", "12"],
        "text": ["Pay <AMOUNT> to <NAME> now", "plain words, no tag", "OTP <OTP> expires"],
        "source": ["imc25_fishing_for_smishing", "scamshield", "scamshield"],
        "label_binary": [1, 0, 1],
    })
    before = sms.copy()
    filled, stats, scrubbed = job.fill_sms_frame(sms)
    job.assert_filled(filled["text"].tolist(), "sms.parquet")
    assert filled["text"].iloc[1] == "plain words, no tag"
    assert filled["text"].iloc[0] != before["text"].iloc[0]
    assert scrubbed == before["text"].tolist()
    assert sms.equals(before), "the input frame must not be changed"
    assert filled["label_binary"].tolist() == [1, 0, 1]
    assert stats == {
        "imc25_fishing_for_smishing": {"rows": 1, "with_tag": 1},
        "scamshield": {"rows": 2, "with_tag": 1},
    }


def test_fill_is_deterministic_and_keyed_by_id():
    frame = pd.DataFrame({"id": ["a", "b"], "text": ["<NUMBER> units", "<NUMBER> units"],
                          "source": ["scamshield", "scamshield"]})
    first, _, _ = job.fill_sms_frame(frame)
    second, _, _ = job.fill_sms_frame(frame)
    assert first["text"].tolist() == second["text"].tolist()
    assert first["text"].tolist() == S.fill_many(["a", "b"], ["<NUMBER> units"] * 2)


def test_fill_inbox_frame_covers_every_row_and_keeps_the_rest():
    rows = _inbox_rows(41, 3, 2, "scam", "Q")
    rows[0]["text"] += " <AMOUNT>"
    raw = pd.DataFrame(rows)
    filled, stats = job.fill_inbox_frame(raw)
    job.assert_filled(filled["text"].tolist(), "inbox")
    assert len(filled) == len(raw)
    assert stats == {"rows": len(raw), "with_tag": 1}
    assert filled["id"].tolist() == raw["id"].tolist()
    assert filled["sender"].tolist() == raw["sender"].tolist()
    for old, new in zip(raw["text"], filled["text"], strict=True):
        if "<" not in old:
            assert old == new
    assert job.label_inbox(filled)[1]["kept"] == job.label_inbox(raw)[1]["kept"]


def test_fill_inbox_frame_requires_id_and_text():
    with pytest.raises(ValueError):
        job.fill_inbox_frame(pd.DataFrame({"text": ["a <NAME>"]}))


def test_only_sms_and_inbox_texts_reach_the_fill(monkeypatch):
    """GroundLM, SmishX and the shift files are not filled: the fill sees only the sms and inbox
    ids, so no external row is ever replaced (plan rule 4)."""
    seen: list[list[str]] = []
    real = S.fill_many

    def spy(ids, texts, seed=S.SURROGATE_SEED):
        seen.append(list(ids))
        return real(ids, texts, seed)

    monkeypatch.setattr(S, "fill_many", spy)
    sms = pd.DataFrame({"id": ["m1", "m2"], "text": ["<NAME> x", "plain"],
                        "source": ["scamshield", "scamshield"], "label_binary": [1, 0]})
    raw = pd.DataFrame(_inbox_rows(43, 2, 1, "scam", "W"))
    job.fill_sms_frame(sms)
    job.fill_inbox_frame(raw)
    assert seen == [["m1", "m2"], list(raw["id"].astype(str))]
    external_ids = {"groundlm_1", "smishx_1", "shift_1"}
    assert external_ids.isdisjoint({i for call in seen for i in call})


def test_assert_filled_stops_with_a_count_and_no_text():
    with pytest.raises(ValueError) as exc:
        job.assert_filled(["clean text", f"{SENTINEL} <NAME> left over"], "sms.parquet")
    message = str(exc.value)
    assert "1 row(s)" in message and "sms.parquet" in message
    assert SENTINEL not in message and "<NAME>" not in message
    job.assert_filled(["clean", "also clean"], "sms.parquet")


def test_clopper_pearson_known_values():
    lo, hi = job.clopper_pearson(0, 14)
    assert lo == 0.0 and round(hi, 4) == 0.2316
    lo, hi = job.clopper_pearson(14, 14)
    assert round(lo, 4) == 0.7684 and hi == 1.0
    lo, hi = job.clopper_pearson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.1871, 0.8129)
    lo, hi = job.clopper_pearson(1, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.0025, 0.4450)
    assert job.clopper_pearson(0, 0) is None
    with pytest.raises(ValueError):
        job.clopper_pearson(3, 2)


def test_section7_pass_rule_on_seed_rows():
    g = job.gate_eval_inbox([_gate_row(s) for s in (1, 2, 3)])
    assert g["arm_pass"] is True
    # median passes and 2 of 3 seeds pass
    g = job.gate_eval_inbox([_gate_row(1, smishx=160), _gate_row(2, smishx=150),
                             _gate_row(3, smishx=200)])
    g2 = g["G2_smishx_smishing_tp"]
    assert g2["passes"] is True and g2["median"] == 160.0 and g2["seeds_pass"] == 2
    # median fails and only 1 of 3 passes
    g = job.gate_eval_inbox([_gate_row(1, smishx=200), _gate_row(2, smishx=150),
                             _gate_row(3, smishx=140)])
    assert g["G2_smishx_smishing_tp"]["passes"] is False
    assert g["G2_smishx_smishing_tp"]["seeds_pass"] == 1
    assert g["arm_pass"] is False
    # a failed seed (no value) fails the gate
    g = job.gate_eval_inbox([_gate_row(1), {"seed": 2}, _gate_row(3)])
    assert g["G2_smishx_smishing_tp"]["passes"] is False


def test_median_pass_with_one_seed_passing_cannot_occur_for_one_sided_gates():
    """The case 'median passes but only 1 of 3 seeds passes' cannot happen for the one-sided gates:
    with three seeds, a passing median means at least two values pass. The test checks that
    property on random values for every gate, so the 2-of-3 rule adds no further constraint here."""
    rng = np.random.default_rng(0)
    scale = {"groundlm_fp": 30, "smishx_tp": 300, "campaign_f1": 1.0, "onnx_bytes": 2e7,
             "donor_fpr": 0.03}
    for _ in range(1000):
        for _name, key, ok in job.INBOX_GATES:
            vals = [float(v) for v in rng.uniform(0, scale[key], 3)]
            res = base._gate(vals, ok)
            assert not (res["median_pass"] and res["seeds_pass"] < 2), (key, vals)
            assert res["passes"] == (res["median_pass"] and res["seeds_pass"] >= 2)


def test_stop_rule_2_and_its_payload_field():
    fail_all = [_gate_row(s, smishx=150) for s in (1, 2, 3)]
    assert job.g2_failed_all_seeds(fail_all) is True
    assert job.g2_failed_all_seeds(
        [_gate_row(1, smishx=150), _gate_row(2, smishx=160), _gate_row(3, smishx=150)]) is False
    assert job.g2_failed_all_seeds(
        [_gate_row(1, smishx=150), {"seed": 2}, _gate_row(3, smishx=150)]) is False
    assert job.g2_failed_all_seeds([]) is False
    seed_out = {s: _fake_block(s, 0.5, 0.01) for s in (1, 2, 3)}
    v0 = {"test_donor": {"scam_recall": {"k": 12, "n": 14, "rate": 12 / 14, "wilson95": [0, 1]}}}
    out0 = job.fill_result_fields("S0", fail_all, seed_out, v0, None, None)
    assert out0["stop_rule_2"] == {"g2_failed_all_seeds": True}
    assert "stop_rule_2" not in job.fill_result_fields("S1", fail_all, seed_out, v0, None, None)
    cp = out0["test_donor_scam_recall_cp95"]
    assert cp["seeds"]["1"] == {"k": 1, "n": 2,
                                "clopper_pearson95": [round(x, 4) for x in
                                                      job.clopper_pearson(1, 2)]}
    assert cp["v0"]["k"] == 12 and cp["v0"]["n"] == 14
    skipped = job.fill_result_fields("S0", fail_all, seed_out, {"skipped": "x"}, None, None)
    assert skipped["test_donor_scam_recall_cp95"]["v0"] == {"skipped": "x"}


def test_fill_outputs_carry_no_synthetic_text(tmp_path):
    raw_inbox = pd.DataFrame(_inbox_rows(51, 3, 2, "scam", "V"))
    raw_inbox.loc[0, "text"] += " <AMOUNT>"
    filled_inbox, inbox_stats = job.fill_inbox_frame(raw_inbox)
    sms = pd.DataFrame({"id": ["m1", "m2"],
                        "text": [f"{SENTINEL} <NAME> pay", "hello <URL> now"],
                        "source": ["scamshield", "scamshield"], "label_binary": [1, 0]})
    filled_sms, sms_stats, scrubbed = job.fill_sms_frame(sms)
    seed_out = {s: _fake_block(s, 0.5, 0.01) for s in (1, 2, 3)}
    rows = [_gate_row(s) for s in (1, 2, 3)]
    payload = {"job": "sms_inbox_retrain", "arm": "S0", "status": "scored", "seeds": [1, 2, 3],
               "side_width": 16, "gates": job.gate_eval_inbox(rows),
               "per_seed": {str(s): seed_out[s] for s in seed_out},
               "v0": {"skipped": "not attached"}}
    payload.update(job.fill_fields("S0", True))
    payload.update(job.fill_result_fields(
        "S0", rows, seed_out, {"skipped": "not attached"},
        {"threshold": 0.35, "n_test": 2, "f1": 0.9},
        {"scrubbed": {"skipped": "n"}, "filled": {"skipped": "n"}}))
    prep = {"arm": "S0", "fill": {"sms_by_source_before_fill": sms_stats,
                                  "train_donor_before_fill": inbox_stats}}
    job.write_outputs(tmp_path, payload, None, prep)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    blob = "\n".join(p.read_text(encoding="utf-8") for p in files)
    texts = [SENTINEL, SENDER_SENTINEL, *scrubbed, *filled_sms["text"], *filled_inbox["text"]]
    for text in texts:
        if text:
            assert text not in blob
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["fill_surrogates"] is True and metrics["arm"] == "S0"
    assert metrics["surrogate_seed"] == S.SURROGATE_SEED
    report = (tmp_path / "report.md").read_text()
    assert "sms_inbox_retrain arm S0" in report and "Surrogate-filled text" in report


def test_r_arm_reports_have_no_fill_lines_and_s_paired_block_is_labelled():
    base_payload = {"job": "sms_inbox_retrain", "arm": "R2", "status": "scored",
                    "seeds": [1, 2, 3], "side_width": 21,
                    "gates": job.gate_eval_inbox([_gate_row(s) for s in (1, 2, 3)]),
                    "per_seed": {}, "v0": {"skipped": "not attached"},
                    "paired_r2_vs_r1": {"skipped": "not attached: pass --r1-probs-dir with "
                                                   "the R1 kernel output"}}
    report = job.render_report(base_payload)
    assert "Surrogate" not in report and "R2 vs R1: not attached" in report
    s0 = {**base_payload, "arm": "S0", "side_width": 16,
          "paired_s2_vs_s1": {"skipped": "x"}, **job.fill_fields("S0", True)}
    assert "S0 vs" not in job.render_report(s0)
    s2 = {**base_payload, "arm": "S2", "side_width": 21,
          "paired_s2_vs_s1": {"skipped": "not attached"}, **job.fill_fields("S2", True)}
    assert "S2 vs S1: not attached" in job.render_report(s2)
