"""rbench CLI exit codes and an end-to-end run through the thermal guard with a tiny ONNX model.

Nothing here touches the Pi or the real artifacts: the model is built with the onnx helper and
the registry is written into a temporary bench root.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import pytest
from rbench import EXIT_MISSING, EXIT_RUNTIME, EXIT_THERMAL, EXIT_USAGE, cli, store
from rbench import guard as G
from rbench import schema as S


def make_tiny_onnx(path: Path) -> str:
    import numpy as np

    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    weights = numpy_helper.from_array(rng.normal(size=(19, 8)).astype(np.float32), name="W")
    nodes = [
        helper.make_node("MatMul", ["features", "W"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probs"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", 19])],
        [helper.make_tensor_value_info("probs", TensorProto.FLOAT, ["batch", 8])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # onnx 1.23 writes IR 14; the Pi's onnxruntime reads up to 13
    onnx.save(model, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def bench(tmp_path: Path, monkeypatch):
    """A temporary bench root with a tiny NIDS model and a registry that uses it."""
    import thermal_run

    monkeypatch.setenv("PYTHONPATH", "")
    # the guard reads this file; a fixed cool value keeps the laptop's own heat out of the tests
    temp_file = tmp_path / "temp"
    temp_file.write_text("42000\n")
    monkeypatch.setattr(thermal_run, "TEMP_PATH", temp_file)
    root = tmp_path / "bench"
    (root / "models").mkdir(parents=True)
    sha = make_tiny_onnx(root / "models" / "tiny.onnx")
    registry = tmp_path / "registry.toml"
    registry.write_text(f'''
schema = "rbench-registry/1"

[[models]]
id = "tiny_nids"
kind = "onnx_nids"
task = "nids_latency"
path = "models/tiny.onnx"
sha256 = "{sha}"
notes = "tiny test model"

[[models]]
id = "off_model"
kind = "onnx_nids"
task = "nids_latency"
enabled = false
path = "models/tiny.onnx"
notes = "disabled on purpose"

[[models]]
id = "absent_model"
kind = "onnx_nids"
task = "nids_latency"
path = "models/absent.onnx"
notes = "file not present"

[[models]]
id = "wrong_sha"
kind = "onnx_nids"
task = "nids_latency"
path = "models/tiny.onnx"
sha256 = "{"0" * 64}"
notes = "pin does not match"

[[datasets]]
id = "tiny_synth"
task = "nids_latency"
synthetic = true
notes = "synthetic test rows"

[[datasets]]
id = "sms_fake"
task = "sms_binary"
path = "data/none.parquet"
notes = "not present"
''')
    return {"root": root, "registry": registry, "tmp": tmp_path, "sha": sha}


def _run(bench, *extra: str) -> list[str]:
    return [
        "run",
        "--root",
        str(bench["root"]),
        "--registry",
        str(bench["registry"]),
        "--job-run-dir",
        str(bench["tmp"] / "jobrun"),
        "--warmup",
        "2",
        *extra,
    ]


def test_list_models_and_datasets_json(bench, capsys):
    assert (
        cli.main(
            [
                "list",
                "models",
                "--json",
                "--no-hash",
                "--root",
                str(bench["root"]),
                "--registry",
                str(bench["registry"]),
            ]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    ids = {m["id"]: m for m in doc["models"]}
    assert ids["tiny_nids"]["status"] == "ready" or ids["tiny_nids"]["status"].startswith("present")
    assert ids["off_model"]["status"] == "disabled"
    assert ids["absent_model"]["status"] == "missing"
    assert ids["tiny_nids"]["compatible_datasets"] == ["tiny_synth"]


def test_usage_errors_exit_2(bench, capsys):
    assert cli.main(_run(bench, "--model", "nope", "--dataset", "tiny_synth")) == EXIT_USAGE
    assert cli.main(_run(bench, "--model", "tiny_nids", "--dataset", "sms_fake")) == EXIT_USAGE
    assert cli.main(_run(bench, "--model", "off_model", "--dataset", "tiny_synth")) == EXIT_USAGE
    assert (
        cli.main(_run(bench, "--model", "tiny_nids", "--dataset", "tiny_synth", "--limit", "0"))
        == EXIT_USAGE
    )
    err = capsys.readouterr().err
    assert "cannot run on dataset sms_fake" in err and "disabled" in err


def test_missing_artifact_exit_3(bench):
    assert (
        cli.main(_run(bench, "--model", "absent_model", "--dataset", "tiny_synth")) == EXIT_MISSING
    )
    assert cli.main(_run(bench, "--model", "tiny_nids", "--dataset", "sms_fake")) == EXIT_USAGE


def test_sha_mismatch_is_a_worker_missing_error_exit_3(bench, capsys):
    code = cli.main(
        _run(bench, "--model", "wrong_sha", "--dataset", "tiny_synth", "--json", "--limit", "5")
    )
    assert code == EXIT_MISSING
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "error" and "sha256 mismatch" in doc["error"]


def test_end_to_end_run_writes_result_telemetry_and_index(bench, capsys):
    code = cli.main(
        _run(
            bench,
            "--model",
            "tiny_nids",
            "--dataset",
            "tiny_synth",
            "--limit",
            "40",
            "--threads",
            "1",
            "--json",
        )
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0, out
    assert out["status"] == "ok" and out["error"] is None
    assert out["dataset"]["n_used"] == 40 and out["dataset"]["synthetic"] is True
    assert out["quality"]["status"] == "not measured"
    assert out["perf"]["latency_ms"]["n"] == 40
    assert out["perf"]["latency_ms"]["p50"] is not None
    assert out["perf"]["throughput_items_s"] > 0
    assert out["model"]["sha256"] == bench["sha"]
    run_dir = bench["root"] / "results" / out["run_id"]
    on_disk = json.loads((run_dir / "result.json").read_text())
    assert S.validate_result(on_disk) == []
    assert (run_dir / "telemetry.csv").is_file() and (run_dir / "run.log").is_file()
    items = (run_dir / "items.jsonl").read_text().splitlines()
    assert len(items) == 40 and "latency_ms" in json.loads(items[0])
    index = store.read_index(bench["root"] / "results")
    assert [row["run_id"] for row in index] == [out["run_id"]]
    assert index[0]["n"] == 40 and index[0]["status"] == "ok"


def test_results_show_and_compare_round_trip(bench, capsys):
    ids = []
    for _ in range(2):
        cli.main(
            _run(
                bench, "--model", "tiny_nids", "--dataset", "tiny_synth", "--limit", "20", "--json"
            )
        )
        ids.append(json.loads(capsys.readouterr().out)["run_id"])
    assert cli.main(["results", "--json", "--root", str(bench["root"]), "--last", "5"]) == 0
    listed = json.loads(capsys.readouterr().out)["runs"]
    assert {r["run_id"] for r in listed} == set(ids)
    assert cli.main(["show", ids[0], "--json", "--root", str(bench["root"])]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == ids[0]
    assert cli.main(["compare", ids[0], ids[1], "--json", "--root", str(bench["root"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["comparable"] is True
    assert "perf.latency_ms.p50" in report["metrics"]
    assert cli.main(["show", "20260101T000000Z-000000", "--root", str(bench["root"])]) == EXIT_USAGE


def test_busy_lock_exit_5(bench, capsys):
    results = bench["root"] / "results"
    results.mkdir(parents=True)
    with (results / G.LOCK_NAME).open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = cli.main(_run(bench, "--model", "tiny_nids", "--dataset", "tiny_synth"))
    assert code == EXIT_RUNTIME
    assert "another benchmark is running" in capsys.readouterr().err


class _FakeJob:
    """Stands in for thermal_run.Job: the guard killed the run at 80 C."""

    def __init__(self, *, name, cmd, guard, run_dir, **_):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = run_dir / "fake.csv"
        self.log_path = run_dir / "fake.log"
        self.csv_path.write_text("ts,temp,cap,freq,throttled,power_w\n1,80.5,1500,1500,0x8,4\n")
        self.log_path.write_text("killed\n")
        self.state = "killed"
        self.throttled_before = "0x0"
        self.throttled_after = "0x8"
        self.pauses = 2
        self.paused_s = 6.5
        self.exit_code = 3
        self._rc = 3

    def run(self):
        return self._rc


def test_thermal_kill_maps_to_exit_4(bench):
    from rbench.guard import RunSpec, run_guarded

    doc = run_guarded(
        bench["root"],
        bench["registry"],
        RunSpec(
            model="tiny_nids", dataset="tiny_synth", limit=5, threads=1, warmup=0, seed=0, name="t"
        ),
        job_run_dir=bench["tmp"] / "jobrun",
        job_factory=_FakeJob,
    )
    assert doc["exit_code"] == EXIT_THERMAL
    assert doc["status"] == "thermal_killed"
    assert doc["thermal"]["pauses"] == 2 and doc["thermal"]["throttled_after"] == "0x8"
    assert doc["thermal"]["guard_kill_temp_c"] == 80.0
    assert S.validate_result(doc) == []


def test_fetch_refuses_disabled_and_is_noop_when_present(bench, tmp_path):
    registry = tmp_path / "fetch_registry.toml"
    payload = b"pte bytes"
    (bench["root"] / "models" / "llama").mkdir()
    (bench["root"] / "models" / "llama" / "m.pte").write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    registry.write_text(f'''
schema = "rbench-registry/1"

[[models]]
id = "llama_pinned"
kind = "pte_llm"
task = "ela_json"
path = "models/llama/m.pte"
sha256 = "{sha}"
hf = {{ repo = "x/y", revision = "abc", files = ["m.pte"] }}
notes = "test"

[[models]]
id = "llama_unpinned"
kind = "pte_llm"
task = "ela_json"
enabled = false
path = "models/llama/u.pte"
hf = {{ repo = "x/y", revision = "abc", files = ["u.pte"] }}
notes = "not pinned"
''')

    assert (
        cli.main(
            [
                "fetch",
                "llama_pinned",
                "--json",
                "--root",
                str(bench["root"]),
                "--registry",
                str(registry),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            ["fetch", "llama_unpinned", "--root", str(bench["root"]), "--registry", str(registry)]
        )
        == EXIT_USAGE
    )


def test_doctor_reports_imports_and_exit_code(bench, capsys):
    code = cli.main(
        [
            "doctor",
            "--json",
            "--no-hash",
            "--root",
            str(bench["root"]),
            "--registry",
            str(bench["registry"]),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code in (0, 3)
    assert report["imports"]["onnxruntime"] is True
    assert {a["id"] for a in report["artifacts"]} >= {"tiny_nids", "tiny_synth"}


def test_run_rejects_threads_and_warmup_usage(bench):
    assert (
        cli.main(_run(bench, "--model", "tiny_nids", "--dataset", "tiny_synth", "--threads", "0"))
        == EXIT_USAGE
    )
