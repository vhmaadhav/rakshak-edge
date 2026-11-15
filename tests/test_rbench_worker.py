"""rbench worker on the real Rakshak artifacts (skipped when the repo files are not present).

The repo's kaggle outputs and data are symlinked into a temporary bench root at the paths the
registry uses, so the sha256 pins are checked for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rbench import registry as R
from rbench import worker as W

REPO = Path(__file__).resolve().parents[1]
REG = R.load_registry(R.DEFAULT_REGISTRY)
SMS_ONNX = "kaggle/out/sms_student/output/best_student.onnx"
URL_ONNX = "kaggle/out/url_student/output/best_url_student.onnx"
URL_SCALER = "kaggle/out/url_student/output/url_side_scaler.json"


def _have(*rel: str) -> bool:
    return all((REPO / r).exists() for r in rel)


@pytest.fixture()
def real_root(tmp_path: Path):
    links = {
        "models/sms_student/best_student.onnx": SMS_ONNX,
        "models/url_student/best_url_student.onnx": URL_ONNX,
        "models/url_student/url_side_scaler.json": URL_SCALER,
    }
    needed = [
        *links.values(),
        "data/processed/sms.parquet",
        "data/processed/urls.parquet",
        "data/processed/urls_phreshphish.parquet",
        "data/processed/external/external_groundlm.parquet",
        "data/processed/splits/campaign.json",
        "data/processed/splits/url_domain.json",
    ]
    if not _have(*needed):
        pytest.skip("repo artifacts not present")
    for dst, src in links.items():
        target = tmp_path / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(REPO / src)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "processed").symlink_to(REPO / "data" / "processed")
    return tmp_path


def _execute(root: Path, out: Path, *argv: str) -> dict:
    args = W.build_parser().parse_args(["--root", str(root), "--out", str(out), *argv])
    return W.execute(args)


def test_sms_student_on_campaign_test_quality_and_sha(real_root, tmp_path):
    doc = _execute(
        real_root,
        tmp_path / "o1",
        "--model",
        "sms_student",
        "--dataset",
        "sms_campaign_test",
        "--limit",
        "5",
        "--warmup",
        "0",
    )
    assert doc["status"] == "ok"
    assert doc["model"]["files"][0]["sha256"] == REG.model("sms_student").sha256
    assert doc["dataset"]["n_used"] == 5
    assert doc["quality"]["at_threshold"]["threshold"] == pytest.approx(0.35127517580986023)
    assert doc["perf"]["latency_ms"]["n"] == 5
    assert doc["perf"]["latency_scope"].startswith("featurise")
    assert (tmp_path / "o1" / "items.jsonl").is_file()


def test_groundlm_denominator_is_1133_scoreable_policy_negatives(real_root, tmp_path):
    doc = _execute(
        real_root,
        tmp_path / "o2",
        "--model",
        "sms_student",
        "--dataset",
        "groundlm_fpr",
        "--limit",
        "20",
        "--warmup",
        "0",
    )
    assert "policy-negatives with text available: 1133 rows" in doc["dataset"]["denominator_notes"]
    assert doc["quality"]["fpr"]["denominator"] == 20
    assert doc["quality"]["n_scored"] == 20


def test_url_student_phreshphish_threshold_free_quality(real_root, tmp_path):
    doc = _execute(
        real_root,
        tmp_path / "o3",
        "--model",
        "url_student",
        "--dataset",
        "phreshphish_test",
        "--limit",
        "30",
        "--warmup",
        "0",
    )
    assert doc["status"] == "ok"
    assert doc["quality"]["average_precision"] is not None
    assert doc["quality"]["threshold"] == pytest.approx(0.6308487409530461)  # tau_url
    # with a registered threshold the key metric is thresholded (F1); AP is still reported
    assert doc["quality"]["key_metric"]["name"] == "f1"


def test_incompatible_pair_raises_usage_exit(real_root, tmp_path):
    from rbench.registry import IncompatibleError

    with pytest.raises(IncompatibleError):
        _execute(
            real_root, tmp_path / "o4", "--model", "sms_student", "--dataset", "url_domain_test"
        )


def test_worker_main_writes_error_document_for_usage(real_root, tmp_path, capsys):
    rc = W.main(
        [
            "--root",
            str(real_root),
            "--out",
            str(tmp_path / "o5"),
            "--model",
            "sms_student",
            "--dataset",
            "url_domain_test",
            "--limit",
            "3",
        ]
    )
    assert rc == 2
    doc = json.loads((tmp_path / "o5" / "worker.json").read_text())
    assert doc["status"] == "error" and doc["error_category"] == "usage"
