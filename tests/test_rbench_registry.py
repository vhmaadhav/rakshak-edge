"""rbench registry: parsing, validation, compatibility and artifact checks (no network)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from rbench import registry as R

REAL = R.DEFAULT_REGISTRY


def _doc(**overrides):
    base = {
        "schema": "rbench-registry/1",
        "models": [
            {
                "id": "m1",
                "kind": "onnx_nids",
                "task": "nids_latency",
                "path": "models/m1.onnx",
                "sha256": "",
                "notes": "x",
            },
        ],
        "datasets": [
            {"id": "d1", "task": "nids_latency", "synthetic": True, "path": "", "notes": "y"},
        ],
    }
    base.update(overrides)
    return base


def test_real_registry_loads_and_has_expected_entries():
    reg = R.load_registry(REAL)
    assert reg.model("sms_student").task == "sms_binary"
    assert reg.model("sms_student").threshold == pytest.approx(0.35127517580986023)
    # tau_url, registered 2026-11-12 (docs/URL_SERVICE_RESULTS.md)
    assert reg.model("url_student").threshold == pytest.approx(0.6308487409530461)
    assert reg.model("llama32_1b_spinquant").sha256.startswith("8715cdba")
    assert reg.model("llama32_1b_spinquant").hf["repo"].startswith("executorch-community/")
    assert not reg.model("llama32_1b_qlora").enabled
    assert reg.dataset("groundlm_fpr").row_filter == "policy_negative_scoreable"
    nids = [m for m in reg.models if m.startswith("nids_")]
    assert len(nids) == 8
    assert all(reg.model(m).kind == "onnx_nids" for m in nids)


def test_real_registry_disabled_entries_have_reasons():
    reg = R.load_registry(REAL)
    for m in reg.models.values():
        if not m.enabled:
            assert m.notes.strip()
    for d in reg.datasets.values():
        if not d.enabled:
            assert d.notes.strip()


def test_compatibility_matrix():
    reg = R.load_registry(REAL)
    sms = reg.model("sms_student")
    R.check_compatible(sms, reg.dataset("sms_campaign_test"))
    R.check_compatible(sms, reg.dataset("groundlm_fpr"))
    with pytest.raises(R.IncompatibleError, match="cannot run on dataset url_domain_test"):
        R.check_compatible(sms, reg.dataset("url_domain_test"))
    assert "groundlm_fpr" in R.compatible_datasets(reg, sms)
    assert R.compatible_datasets(reg, reg.model("url_student")) == [
        "url_domain_test",
        "phreshphish_test",
    ]


def test_parse_minimal_ok_and_defaults():
    reg = R.parse_registry(_doc())
    assert reg.model("m1").enabled is True
    assert reg.dataset("d1").synthetic is True
    assert reg.model("m1").files()[0].path == "models/m1.onnx"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(schema="nope"), "registry schema"),
        (lambda d: d["models"][0].update(kind="weird"), "unknown kind"),
        (lambda d: d["models"][0].update(task="url_binary"), "must use task"),
        (lambda d: d["models"][0].update(path="/abs/m.onnx"), "relative"),
        (lambda d: d["models"][0].update(path="../escape.onnx"), "relative"),
        (lambda d: d["models"][0].update(sha256="ABC"), "sha256"),
        (lambda d: d["models"][0].update(threshold=1.5), "threshold"),
        (lambda d: d["models"][0].update(enabled=False, notes=""), "say why"),
        (lambda d: d["datasets"][0].update(task="other"), "unknown task"),
        (lambda d: d["datasets"][0].update(synthetic=False, path=""), "needs a path"),
        (lambda d: d["datasets"][0].update(row_filter="magic"), "row_filter"),
        (lambda d: d["models"].append(dict(d["models"][0])), "duplicate model"),
    ],
)
def test_parse_rejects_bad_entries(mutate, message):
    doc = _doc()
    mutate(doc)
    with pytest.raises(R.RegistryError, match=message):
        R.parse_registry(doc)


def test_check_file_and_model_state(tmp_path: Path):
    payload = b"hello onnx"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m1.onnx").write_bytes(payload)
    good = hashlib.sha256(payload).hexdigest()
    state = R.check_file(tmp_path, "models/m1.onnx", good)
    assert state["present"] and state["ok"] is True and state["sha256"] == good
    bad = R.check_file(tmp_path, "models/m1.onnx", "0" * 64)
    assert bad["ok"] is False
    unpinned = R.check_file(tmp_path, "models/m1.onnx", "")
    assert unpinned["ok"] is None
    missing = R.check_file(tmp_path, "models/none.onnx", good)
    assert missing["present"] is False and missing["ok"] is None
    # no hashing when not asked and not present on disk
    lazy = R.check_file(tmp_path, "models/m1.onnx", good, hash_it=False)
    assert lazy["sha256"] is None and lazy["ok"] is None

    reg = R.parse_registry(
        _doc(
            models=[
                {
                    "id": "m1",
                    "kind": "onnx_nids",
                    "task": "nids_latency",
                    "path": "models/m1.onnx",
                    "sha256": good,
                    "notes": "",
                }
            ]
        )
    )
    assert R.model_state(tmp_path, reg.model("m1"))["status"] == "ready"
    assert R.model_state(tmp_path / "nowhere", reg.model("m1"))["status"] == "missing"


def test_model_state_disabled_and_not_downloaded(tmp_path: Path):
    reg = R.load_registry(REAL)
    assert (
        R.model_state(tmp_path, reg.model("llama32_1b_qlora"), hash_it=False)["status"]
        == "disabled"
    )
    assert (
        R.model_state(tmp_path, reg.model("llama32_1b_spinquant"), hash_it=False)["status"]
        == "not downloaded"
    )


def test_sync_manifest_lists_sidecar_sources_and_skips_llama():
    reg = R.load_registry(REAL)
    rows = R.sync_manifest(reg)
    by_path = {r["path"]: r for r in rows}
    assert len(by_path) == len(rows)  # no duplicate destinations
    scaler = by_path["models/url_student/url_side_scaler.json"]
    assert scaler["source"] == "kaggle/out/url_student/output/url_side_scaler.json"
    assert scaler["sha256"] == reg.model("url_student").side_scaler_sha256
    split = by_path["data/processed/splits/campaign.json"]
    assert split["role"] == "split" and split["owner"] == "dataset:sms_campaign_test"
    assert all(not r["path"].startswith("models/llama32_1b_spinquant/") for r in rows)
    assert all(not r["path"].startswith("data/ela/") for r in rows)  # disabled dataset
    assert all(r["source"] for r in rows)


def test_sidecar_source_fields_parse_and_default_blank():
    doc = _doc(
        models=[
            {
                "id": "m1",
                "kind": "onnx_nids",
                "task": "nids_latency",
                "path": "models/m1.onnx",
                "source": "a/m1.onnx",
                "parity_logits": "models/m1_logits.parquet",
                "parity_logits_source": "a/logits.parquet",
                "parity_column": "logit",
                "extras": [{"role": "tok", "path": "models/tok.model", "sha256": ""}],
                "notes": "x",
            }
        ]
    )
    reg = R.parse_registry(doc)
    files = {f.role: f for f in reg.model("m1").files()}
    assert files["model"].source == "a/m1.onnx"
    assert files["parity_logits"].source == "a/logits.parquet"
    assert files["tok"].source == ""
    assert [r["path"] for r in R.sync_manifest(reg)] == [
        "models/m1.onnx",
        "models/m1_logits.parquet",
    ]
    bad = _doc(
        models=[
            {
                "id": "m1",
                "kind": "onnx_nids",
                "task": "nids_latency",
                "path": "models/m1.onnx",
                "source": "../escape.onnx",
                "notes": "x",
            }
        ]
    )
    with pytest.raises(R.RegistryError, match="relative"):
        R.parse_registry(bad)


def test_status_labels_pinned_unverified_incomplete_and_ready(tmp_path: Path):
    payload = b"0123456789"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.bin").write_bytes(payload)
    good = hashlib.sha256(payload).hexdigest()
    base = {
        "id": "m",
        "kind": "onnx_nids",
        "task": "nids_latency",
        "path": "models/m.bin",
        "notes": "",
    }
    pinned = R.parse_registry(_doc(models=[base | {"sha256": good, "bytes": 10}])).model("m")
    # pinned and hashed, matching size: ready
    assert R.model_state(tmp_path, pinned)["status"] == "ready"
    # pinned but not hashed (--no-hash): must not claim "sha not pinned"
    assert R.model_state(tmp_path, pinned, hash_it=False)["status"] == "present (not verified)"
    # partial file (wrong size): incomplete, whatever the hash mode
    partial = R.parse_registry(_doc(models=[base | {"sha256": good, "bytes": 99}])).model("m")
    assert R.model_state(tmp_path, partial)["status"] == "incomplete"
    assert R.model_state(tmp_path, partial, hash_it=False)["status"] == "incomplete"
    # no pin at all: keeps the honest label
    unpinned = R.parse_registry(_doc(models=[base | {"sha256": ""}])).model("m")
    assert R.model_state(tmp_path, unpinned)["status"] == "present (sha not pinned)"
    # wrong bytes pinned with a matching sha but a different file: sha mismatch wins over ready
    wrong = R.parse_registry(_doc(models=[base | {"sha256": "0" * 64}])).model("m")
    assert R.model_state(tmp_path, wrong)["status"] == "sha mismatch"


def test_real_llama_model_pins_sha_and_size():
    reg = R.load_registry(REAL)
    model = reg.model("llama32_1b_spinquant")
    assert model.sha256 == "8715cdba9e91f6bede00cc5f2d6b12397b95225da5630c4972a8da03001cda3b"
    assert model.bytes == 1135951488
    tok = next(e for e in model.extras if e.role == "tokenizer")
    assert tok.sha256 == "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"
    assert tok.bytes == 2183982


def test_bytes_must_be_positive_int():
    with pytest.raises(R.RegistryError, match="bytes must be a positive integer"):
        R.parse_registry(
            _doc(
                models=[
                    {
                        "id": "m",
                        "kind": "onnx_nids",
                        "task": "nids_latency",
                        "path": "models/m.bin",
                        "bytes": 0,
                        "notes": "",
                    }
                ]
            )
        )
