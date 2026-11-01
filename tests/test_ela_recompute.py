"""Synthetic clustered point-count fixtures for the Kaggle-only statistics supplement."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rakshak.eval.audit import clustered_ratio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle/jobs"))
import ela_recompute as job  # noqa: E402


def test_cluster_ratio_pools_points_instead_of_averaging_message_rates():
    res = clustered_ratio([1, 0, 0], [1, 3, 0], ["a", "b", "c"], n_boot=30)
    assert res["rate"] == 0.25 and res["events"] == 1 and res["total"] == 4
    assert res["n_clusters"] == 3
    assert res["ci95"] == [0.0, 1.0]
    assert clustered_ratio([0], [0], ["a"])["rate"] is None
    with pytest.raises(ValueError):
        clustered_ratio([2], [1], ["a"])


def test_recompute_binds_to_frozen_source_and_never_regenerates(tmp_path, monkeypatch):
    source = tmp_path / "input/source"
    source.mkdir(parents=True)
    rows = [{"arm": arm, "injected": False, "sample_id": "a", "campaign_id": "c1",
             "side": "scam", "json_ok": True, "verifier_ok": False, "unsupported_n": 0,
             "safe_claim": False, "points_n": 2, "points_pass_n": 1}
            for arm in ("slot", "slot_hinglish")]
    (source / "generations.jsonl").write_text("\n".join(map(json.dumps, rows)))
    (source / "results.json").write_text(json.dumps({
        "tree_sha256": "frozen", "status": "ok", "job": "llama_ela_eval",
    }))
    (source / "sample_manifest.json").write_text(json.dumps({
        "rows": [{"sample_id": "a", "lang_subset": True}],
    }))
    monkeypatch.setenv("RAKSHAK_WORK", str(tmp_path / "out"))
    ctx = SimpleNamespace(input_root=tmp_path / "input", argv=["--expected-tree", "frozen"],
                          results={})
    assert job.run(ctx)
    result = json.loads((tmp_path / "out/metrics.json").read_text())
    assert result["source_tree"] == "frozen"
    assert result["lang_compare"]["slot"]["point_pass"]["rate"] == 0.5
    ctx.argv = ["--expected-tree", "wrong"]
    with pytest.raises(ValueError, match="found 0"):
        job.run(ctx)
