"""Tests for the URL student job and the PhreshPhish protocol helpers it shares with url_baselines.

Tiny in-memory or tmp_path fixtures only. The torch tests (ByteTransformer, the char-CNN side
input, the training loop and the job's dataset builder) are skipped when torch is not importable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rakshak.eval import data as D
from rakshak.eval import report
from rakshak.eval.phreshphish import (
    PAPER_V2_GTE,
    PHRESH_FAMILY,
    PHRESH_FILE,
    grouped_cal,
    load_phreshphish,
    paper_reference,
    protocol_section,
    protocol_tables,
    subsample_rows,
)
from rakshak.models.encoding import (
    PAD_ID,
    apply_standardizer,
    encode_batch,
    fit_standardizer,
)

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"


# ---------------------------------------------------------------------------
# shared protocol helpers (numpy, pandas, sklearn only)
# ---------------------------------------------------------------------------


def test_grouped_cal_keeps_every_group_on_one_side_and_both_classes():
    rng = np.random.default_rng(0)
    rows = np.arange(400)
    labels = rng.integers(0, 2, size=400)
    groups = np.array([f"d{i // 4}" for i in range(400)], dtype=object)  # 100 groups of 4
    keep, cal = grouped_cal(rows, labels, groups[rows], 0.2, seed=1)
    assert len(np.intersect1d(keep, cal)) == 0
    assert len(keep) + len(cal) == len(rows)
    cal_groups = set(groups[cal])
    keep_groups = set(groups[keep])
    assert cal_groups.isdisjoint(keep_groups)
    assert len(np.unique(labels[cal])) == 2


def test_grouped_cal_is_deterministic_for_a_seed():
    labels = np.tile([0, 1], 50)
    groups = np.array([f"g{i}" for i in range(100)], dtype=object)
    a = grouped_cal(np.arange(100), labels, groups, 0.1, seed=3)
    b = grouped_cal(np.arange(100), labels, groups, 0.1, seed=3)
    assert np.array_equal(a[1], b[1])


def test_grouped_cal_rejects_a_one_class_cal_side_and_bad_inputs():
    labels = np.array([0] * 9 + [1])
    groups = np.array([f"g{i}" for i in range(10)], dtype=object)
    with pytest.raises(ValueError, match="one class"):
        grouped_cal(np.arange(10), np.zeros(10, dtype=int), groups, 0.5, seed=0)
    with pytest.raises(ValueError, match="one group key per row"):
        grouped_cal(np.arange(10), labels, groups[:5], 0.5, seed=0)
    with pytest.raises(ValueError, match="fraction"):
        grouped_cal(np.arange(10), labels, groups, 1.0, seed=0)


def test_subsample_rows_is_sorted_seeded_and_bounded():
    rows = np.arange(1000, 1100)
    assert np.array_equal(subsample_rows(rows, None, 0), rows)
    assert np.array_equal(subsample_rows(rows, 500, 0), rows)  # not smaller: all rows
    a = subsample_rows(rows, 10, seed=7)
    assert len(a) == 10 and np.all(np.diff(a) > 0)
    assert np.array_equal(a, subsample_rows(rows, 10, seed=7))
    assert not np.array_equal(a, subsample_rows(rows, 10, seed=8))
    with pytest.raises(ValueError):
        subsample_rows(rows, 1, seed=0)


def test_standardizer_fits_on_rows_only_and_handles_constant_and_nonfinite():
    x = np.array([[1.0, 5.0], [3.0, 5.0], [100.0, 5.0], [np.nan, 5.0]], dtype=np.float32)
    mean, std = fit_standardizer(x, rows=np.array([0, 1]))
    assert mean[0] == pytest.approx(2.0)
    assert std[1] == pytest.approx(1.0)  # constant column: std set to 1
    z = apply_standardizer(x, mean, std)
    assert z.dtype == np.float32
    assert z[2, 0] == pytest.approx(10.0)  # (100 - 2) / 1 clipped to +10
    assert z[3, 0] == 0.0  # NaN becomes 0
    assert np.all(np.abs(z) <= 10.0)


def test_encode_batch_pads_and_truncates_urls():
    ids = encode_batch(["ab", "", "x" * 300], 200)
    assert ids.shape == (3, 200)
    assert ids[0, 2] == PAD_ID and ids[0, 1] == ord("b")
    assert np.all(ids[1] == PAD_ID)
    assert np.all(ids[2] == ord("x"))


def test_paper_reference_matches_the_v2_gte_row():
    ref = paper_reference()
    assert ref["0.05%"] == {"ap": 0.6050, "p_at_r90": 0.0891}
    assert ref["24% (full)"]["ap"] == pytest.approx(PAPER_V2_GTE["24% (full)"][0])


def _section(models: dict[str, float]) -> dict:
    per_rate = {
        "0.05%": {"n_subsets": 2, "ap_mean": 0.5, "ap_std": 0.1, "p_at_r90_mean": 0.2,
                  "p_at_r90_std": 0.05, "base_rate_mean": 0.0005, "base_rate_std": 0.0,
                  "subset_size_mean": 100.0},
        "1%": {"n_subsets": 1, "ap_mean": 0.9, "ap_std": float("nan"), "p_at_r90_mean": 0.7,
               "p_at_r90_std": float("nan"), "base_rate_mean": 0.01, "base_rate_std": float("nan"),
               "subset_size_mean": 100.0},
    }
    full = {"n": 10, "positives": 3, "base_rate": 0.3, "ap": 0.97, "p_at_r90": 0.95}
    out = {}
    for name in models:
        out[name] = {"per_rate": per_rate, "full_benchmark_24pct": full,
                     "full_test_split": {**full, "ap": 0.99}}
    return {"paper_v2_gte_reference": paper_reference(), "models": out}


def test_protocol_tables_one_per_model_with_paper_column_and_full_rows():
    tables = protocol_tables(_section({"charcnn-wide/s0": 0, "transformer/s0": 0}))
    assert [t["title"] for t in tables] == [
        "PhreshPhish protocol: charcnn-wide/s0",
        "PhreshPhish protocol: transformer/s0",
    ]
    table = tables[0]
    assert len(table["header"]) == 6
    assert "Paper GTE AP" in table["header"][4]
    rate_row = table["rows"][0]
    assert rate_row[0] == "0.05%" and rate_row[1] == "2"
    assert rate_row[2] == "0.5000 ± 0.1000"
    assert rate_row[4] == "0.6050" and rate_row[5] == "0.0891"
    one_subset = table["rows"][1]
    assert one_subset[2] == "0.9000"  # std is NaN for a single subset: no +- part
    assert table["rows"][-2][0] == "24% (full benchmark)"
    assert table["rows"][-2][4] == "0.9779"
    assert table["rows"][-1][0] == "full test split"
    assert table["rows"][-1][4] == "n/a"


def test_protocol_section_scores_only_the_test_rows_and_benchmark():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    scores = np.full(8, np.nan)
    test = np.array([4, 5, 6, 7])
    scores[test] = [0.1, 0.9, 0.2, 0.8]
    bench = np.array([4, 5, 6, 7])
    subsets = [("1%", np.array([4, 5, 6, 7]))]
    out = protocol_section(labels, scores, test, bench, subsets)
    assert out["per_rate"]["1%"]["ap_mean"] == pytest.approx(1.0)
    assert out["full_benchmark_24pct"]["n"] == 4
    assert out["full_test_split"]["p_at_r90"] == pytest.approx(1.0)


def _phresh_root(tmp_path: Path) -> Path:
    """A four-row PhreshPhish table with benchmark columns and the official split file."""
    root = tmp_path / "data"
    (root / D.SPLIT_DIR).mkdir(parents=True)
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "url": ["https://x.test/a", "https://y.test/b", "https://x.test/c", "https://z.test/d"],
        "label": [0, 1, 0, 1],
        "reg_domain": ["x.test", "y.test", "x.test", "z.test"],
        "lang": ["en"] * 4,
        "split": ["train", "train", "test", "test"],
        "bench_benign": [False, False, True, False],
        "bench_full": [False, True, True, True],
        "bench_subsets": [[], ["benchmark-005/0"], [], ["benchmark-005/0"]],
        "len": [1.0, 2.0, 3.0, 4.0],
    })
    df.to_parquet(root / PHRESH_FILE, index=False)
    split = {"family": PHRESH_FAMILY, "seed": 0,
             "partitions": {"official": {"train": ["a", "b"], "test": ["c", "d"]}}}
    (root / D.SPLIT_DIR / f"{PHRESH_FAMILY}.json").write_text(json.dumps(split))
    return root


def test_load_phreshphish_reads_official_split_subsets_and_features(tmp_path):
    data = load_phreshphish(_phresh_root(tmp_path))
    assert data.train.tolist() == [0, 1]
    assert data.test.tolist() == [2, 3]
    assert data.bench_full.tolist() == [1, 2, 3]
    assert "len" in data.feature_cols
    assert "reg_domain" not in data.feature_cols
    rates = dict(data.subsets)
    assert "0.05%" in rates and rates["0.05%"].tolist() == [2, 1, 3]


def test_load_phreshphish_rejects_split_ids_missing_from_the_table(tmp_path):
    root = _phresh_root(tmp_path)
    split_path = root / D.SPLIT_DIR / f"{PHRESH_FAMILY}.json"
    split = json.loads(split_path.read_text())
    split["partitions"]["official"]["test"].append("zz")
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="missing"):
        load_phreshphish(root)


def test_report_renders_tables_after_splits(tmp_path):
    payload = {"title": "T", "meta": {}, "splits": {},
               "tables": [{"title": "Table one", "note": "a note", "header": ["A", "B"],
                           "rows": [["x", "1.0000"], ["y", "n/a"]]}]}
    text = report.render_markdown(payload)
    assert "## Table one" in text
    assert "a note" in text
    assert "| x | 1.0000 |" in text
    paths = report.write_report(payload, tmp_path)
    assert paths["report"].read_text().count("## Table one") == 1


# ---------------------------------------------------------------------------
# torch: model shapes, the training loop, the job's dataset builder
# ---------------------------------------------------------------------------


def _torch():
    return pytest.importorskip("torch")


def test_byte_transformer_shapes_padding_and_length_guard():
    torch = _torch()
    from rakshak.models.url_nets import ByteTransformer

    net = ByteTransformer(max_len=32, d_model=16, layers=2, heads=2).eval()
    ids = torch.as_tensor(encode_batch(["https://a.test/x", "", "b" * 40], 32).astype("int64"))
    out = net(ids)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()  # the all-padding row stays finite
    with pytest.raises(ValueError, match="max_len"):
        net(torch.zeros(1, 33, dtype=torch.long))
    with pytest.raises(ValueError, match="divisible"):
        ByteTransformer(d_model=10, heads=4)


def test_charcnn_takes_the_url_side_input_and_rejects_a_missing_one():
    torch = _torch()
    from rakshak.models.charcnn import BASE, CharCNN

    net = CharCNN(BASE, views=("raw",), n_side=5).eval()
    ids = torch.as_tensor(encode_batch(["https://a.test/x", "u"], 40).astype("int64"))
    side = torch.zeros(2, 5)
    assert net({"raw": ids}, side).shape == (2,)
    with pytest.raises(ValueError, match="side"):
        net({"raw": ids}, None)


def _tiny_dataset():
    from url_student import Dataset  # noqa: PLC0415  (needs the job path, set by the fixture)

    rng = np.random.default_rng(0)
    n = 80
    labels = np.tile([0, 1], n // 2).astype(np.int64)
    urls = [f"https://site{i % 7}.test/login/{i}" if labels[i] else f"https://n{i}.example/p"
            for i in range(n)]
    ids = encode_batch(urls, 24)
    side = rng.normal(size=(n, 3)).astype(np.float32)
    rows = np.arange(n)
    return Dataset(
        name="tiny", ids=ids, labels=labels, row_ids=np.array([str(i) for i in range(n)],
                                                               dtype=object),
        fit=rows[:48], cal=rows[48:64], test=rows[64:], side=side, side_cols=["a", "b", "c"],
        side_stats=(np.zeros(3, np.float32), np.ones(3, np.float32)), urls=np.array(urls,
                                                                                   dtype=object),
        protocol={"bench_full": rows[64:], "subsets": [("5%", rows[64:])]},
    )


@pytest.fixture()
def job(monkeypatch):
    _torch()
    monkeypatch.syspath_prepend(str(JOBS))
    monkeypatch.delitem(sys.modules, "url_student", raising=False)
    import url_student

    return url_student


def test_train_model_stops_at_an_expired_budget_and_still_scores_cal(job):
    torch = _torch()
    from rakshak.models.charcnn import BASE, CharCNN

    ds = _tiny_dataset()
    cfg = job.TrainCfg(epochs=5, batch_size=16, lr=1e-3, weight_decay=0.0, patience=3, seed=0,
                       kd_alpha=0.5, kd_temp=2.0, amp=False)
    model = CharCNN(BASE, views=("raw",), n_side=3)
    info = job.train_model(model, "charcnn", ds, cfg, torch.device("cpu"), True,
                           deadline=0.0, teacher_logits=None, log=lambda m: None)
    assert info["stopped"] == "budget"
    assert info["epochs_run"] == 1
    assert info["history"][0]["cal_pr_auc"] == info["history"][0]["cal_pr_auc"]


def test_train_model_learns_the_tiny_task_without_a_budget(job):
    torch = _torch()
    from rakshak.models.charcnn import BASE, CharCNN

    ds = _tiny_dataset()
    cfg = job.TrainCfg(epochs=3, batch_size=16, lr=5e-3, weight_decay=0.0, patience=3, seed=0,
                       kd_alpha=0.5, kd_temp=2.0, amp=False)
    model = CharCNN(BASE, views=("raw",), n_side=3)
    info = job.train_model(model, "charcnn", ds, cfg, torch.device("cpu"), True, deadline=None,
                           teacher_logits=None, log=lambda m: None)
    assert info["stopped"] in ("max_epochs", "early_stop")
    assert info["best_epoch"] >= 0
    assert info["best_cal_pr_auc"] is not None


def test_knowledge_distillation_unit_needs_teacher_logits_for_its_split(job):
    spec = job.Spec("charcnn", "charcnn-wide-kd", True, True)
    args = job.parse_args([])
    ds = _tiny_dataset()
    with pytest.raises(RuntimeError, match="no transformer teacher"):
        job.run_unit(ds, spec, 0, args, _torch().device("cpu"), None, {"teacher": {}},
                     lambda m: None)


def test_unit_specs_follow_the_flags(job):
    specs = job.unit_specs(job.parse_args(["--kd", "--models", "charcnn"]))
    assert [s.key for s in specs] == ["transformer-L4-d256", "charcnn-wide", "charcnn-wide-kd"]
    assert specs[0].kind == "transformer" and specs[1].kind == "charcnn"
    assert specs[2].kd and specs[2].side and specs[2].kind == "charcnn"
    none_side = job.unit_specs(job.parse_args(["--no-side", "--models", "charcnn"]))
    assert none_side[0].key == "charcnn-wide-noside" and not none_side[0].side


def test_parse_args_rejects_bad_values(job):
    with pytest.raises(SystemExit):
        job.parse_args(["--tf-layers", "0"])
    with pytest.raises(SystemExit):
        job.parse_args(["--budget-h", "0"])
    with pytest.raises(SystemExit):
        job.parse_args(["--cal-fraction", "1.5"])
