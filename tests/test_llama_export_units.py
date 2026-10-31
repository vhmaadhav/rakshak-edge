"""Step independence of kaggle/jobs/llama_export.py: the prebuilt unit and the export unit each
run on their own, and a failure in one does not skip the other.

Downloads, pip, the export subprocess and the load-check subprocess are all monkeypatched, so no
network, no install and no model compute run here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import llama_export as job  # noqa: E402

SPINQUANT = "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
QLORA = "Llama-3.2-1B-Instruct-QLORA_INT4_EO8.pte"
PREBUILT_FILES = {SPINQUANT, QLORA}


class FakeCtx:
    def __init__(self, root: Path):
        self.work = root / "work"
        self.scratch = root / "scratch"
        self.work.mkdir()
        self.scratch.mkdir()
        self.results: dict = {}
        self.lines: list[str] = []

    def log(self, msg):
        self.lines.append(str(msg))

    def save(self):
        pass


class Fakes:
    """Knobs for the fakes. Each test sets the ones it needs."""

    def __init__(self):
        self.pip_calls: list[str] = []
        self.pip_fail: set[str] = set()
        self.snapshot_fail: set[str] = set()
        self.convert_error: Exception | None = None
        self.export_rc = 0
        self.export_calls = 0
        self.broken: set[str] = set()  # file names whose load check reports load=False
        self.sanity_calls: list[str] = []


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setenv("RAKSHAK_WORK", str(tmp_path / "work"))
    knobs = Fakes()
    hub = tmp_path / "hub"

    def fake_snapshot(repo, revision, allow_patterns=None, snapshot_download=None):
        if repo in knobs.snapshot_fail:
            raise OSError(f"no network for {repo}")
        folder = hub / repo.replace("/", "__") / revision
        folder.mkdir(parents=True, exist_ok=True)
        if repo == job.UNSLOTH_REPO:
            (folder / "config.json").write_text("{}")
            (folder / "tokenizer.json").write_text("{}")
        else:
            fname = next(f for (r, _rev, f) in job.PREBUILT_PTE.values() if r == repo)
            (folder / fname).write_bytes(b"prebuilt-" + fname.encode())
        return folder

    def fake_pip(ctx, args, label):
        knobs.pip_calls.append(label)
        if label in knobs.pip_fail:
            raise RuntimeError(f"{label} failed: simulated")

    def fake_convert(ctx, unsloth, scratch):
        if knobs.convert_error is not None:
            raise knobs.convert_error
        ckpt = scratch / "llama32_1b_meta.pth"
        ckpt.write_bytes(b"ckpt")
        params = scratch / "params.json"
        params.write_text("{}")
        return ckpt, params, {"layers": 16, "hidden_dim": 8192, "dtype": "bfloat16",
                              "checkpoint_bytes": 4}

    def fake_run_export(ctx, export_dir, config, log_path):
        knobs.export_calls += 1
        log_path.write_text("delegation: xnnpack partitions 1\n")
        if knobs.export_rc == 0:
            (export_dir / job.PTE_NAME).write_bytes(b"our-pte")
        return knobs.export_rc, 0.1, 2048

    def fake_sanity(scratch, pte, tokenizer):
        knobs.sanity_calls.append(pte.name)
        assert tokenizer.name == "tokenizer.json"
        ok = pte.name not in knobs.broken
        return {"load": ok, "tokens_emitted": 16 if ok else 0, "text": "ok" if ok else None,
                **({} if ok else {"error": "simulated load failure"})}

    monkeypatch.setattr(job, "snapshot", fake_snapshot)
    monkeypatch.setattr(job, "_pip", fake_pip)
    monkeypatch.setattr(job, "_convert_to_meta", fake_convert)
    monkeypatch.setattr(job, "_run_export", fake_run_export)
    monkeypatch.setattr(job, "_sanity", fake_sanity)
    monkeypatch.setattr(job, "_resources",
                        lambda ctx: {"ram_gb": 31.3, "free_disk_gb": 19.5, "cpus": 4})
    knobs.ctx = FakeCtx(tmp_path)
    return knobs


def _output_names(ctx: FakeCtx) -> set[str]:
    return {p.name for p in ctx.work.iterdir()}


def test_both_units_pass_and_only_our_pte_reaches_output(fakes):
    ctx = fakes.ctx
    status = job.run(ctx)

    assert status == "ok"
    units = json.loads((ctx.work / "export_metrics.json").read_text())["units"]
    assert units["prebuilt"]["ok"] and units["prebuilt"]["verified"] == 2
    assert units["export"]["ok"]
    assert _output_names(ctx) >= {job.PTE_NAME, "tokenizer.json", "config.json",
                                  "export_metrics.json", job.YAML_NAME, job.LOG_NAME}
    assert not (_output_names(ctx) & PREBUILT_FILES), "prebuilt .pte must not reach the output"
    assert sum(1 for n in _output_names(ctx) if n.endswith(".pte")) == 1


def test_prebuilt_entries_record_sha_and_size_without_copying(fakes):
    ctx = fakes.ctx
    job.run(ctx)
    prebuilt = json.loads((ctx.work / "export_metrics.json").read_text())["units"]["prebuilt"]
    for entry in prebuilt["entries"]:
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
        assert entry["sanity"]["load"] is True
        assert entry["tokenizer_source"] == "unsloth_base", "fake repo ships no tokenizer.json"
    assert not (ctx.work / SPINQUANT).exists() and not (ctx.work / QLORA).exists()


def test_tokenizer_prefers_the_pte_repo_and_falls_back_to_the_base_model(tmp_path, fakes):
    own = tmp_path / "repo"
    own.mkdir()
    pte = own / "model.pte"
    pte.write_bytes(b"x")
    assert job._tokenizer_for(pte)[1] == "unsloth_base"
    (own / "tokenizer.json").write_text("{}")
    path, source = job._tokenizer_for(pte)
    assert source == "pte_repo" and path == own / "tokenizer.json"


def test_sanity_gate_needs_load_and_at_least_one_token():
    assert job._sanity_ok({"load": True, "tokens_emitted": 16})
    assert not job._sanity_ok({"load": True, "tokens_emitted": 0})
    assert not job._sanity_ok({"load": True, "tokens_emitted": 4, "generate_error": "0x14"})
    assert not job._sanity_ok({"load": False, "tokens_emitted": 16})


def test_install_failure_still_downloads_and_hashes_prebuilt(fakes):
    fakes.pip_fail = {"executorch"}
    ctx = fakes.ctx
    status = job.run(ctx)

    metrics = json.loads((ctx.work / "export_metrics.json").read_text())
    assert status == "fail"
    # pip stopped at the first failing step, and the tokenizers step was never tried
    assert fakes.pip_calls == ["torch", "executorch"]
    steps = [(s["step"], s["ok"]) for s in metrics["install"]]
    assert steps == [("torch", True), ("executorch", False)]

    prebuilt = metrics["units"]["prebuilt"]
    assert not prebuilt["ok"]
    for entry in prebuilt["entries"]:
        assert len(entry["sha256"]) == 64, "download and sha256 must not need the install"
        assert entry["sanity"]["load"] is False
        assert "install failed" in entry["sanity"]["error"]
    assert fakes.sanity_calls == [], "no load check runs without an installed executorch"

    export = metrics["units"]["export"]
    assert not export["ok"] and export["reason"].startswith("install failed")
    assert fakes.export_calls == 0


def test_tokenizers_step_failure_is_recorded_after_executorch_ok(fakes):
    fakes.pip_fail = {"pytorch-tokenizers"}
    ctx = fakes.ctx
    job.run(ctx)
    metrics = json.loads((ctx.work / "export_metrics.json").read_text())
    assert fakes.pip_calls == ["torch", "executorch", "pytorch-tokenizers"]
    assert [s["step"] for s in metrics["install"]] == ["torch", "executorch", "pytorch-tokenizers"]
    assert metrics["install"][-1]["ok"] is False
    assert metrics["units"]["prebuilt"]["ok"] is False
    assert metrics["units"]["export"]["ok"] is False


def test_export_exception_does_not_skip_prebuilt(fakes):
    fakes.convert_error = ValueError("bad shapes")
    ctx = fakes.ctx
    status = job.run(ctx)

    metrics = json.loads((ctx.work / "export_metrics.json").read_text())
    assert status == "partial"
    assert metrics["units"]["prebuilt"]["ok"] is True
    assert metrics["units"]["prebuilt"]["verified"] == 2
    export = metrics["units"]["export"]
    assert not export["ok"] and export["reason"].startswith("HF to Meta conversion")
    assert fakes.export_calls == 0
    assert not (ctx.work / job.PTE_NAME).exists()


def test_prebuilt_download_failure_does_not_skip_export(fakes):
    fakes.snapshot_fail = {"executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET"}
    ctx = fakes.ctx
    status = job.run(ctx)

    metrics = json.loads((ctx.work / "export_metrics.json").read_text())
    assert status == "partial"
    prebuilt = metrics["units"]["prebuilt"]
    assert not prebuilt["ok"] and prebuilt["verified"] == 1
    failed = next(e for e in prebuilt["entries"] if e["key"] == "spinquant_int4_eo8")
    passed = next(e for e in prebuilt["entries"] if e["key"] == "qlora_int4_eo8")
    assert failed["reason"].startswith("download") and "sha256" not in failed
    assert passed["ok"] and len(passed["sha256"]) == 64
    assert metrics["units"]["export"]["ok"] is True
    assert fakes.export_calls == 1
    assert (ctx.work / job.PTE_NAME).exists()


def test_failed_export_saves_nothing_and_prebuilt_still_checked(fakes):
    fakes.export_rc = 1
    ctx = fakes.ctx
    status = job.run(ctx)

    metrics = json.loads((ctx.work / "export_metrics.json").read_text())
    assert status == "partial"
    assert metrics["units"]["prebuilt"]["ok"] is True
    export = metrics["units"]["export"]
    assert not export["ok"] and export["exit_code"] == 1
    assert export["reason"] == f"export_llm exited 1; see {job.LOG_NAME}"
    assert not (ctx.work / job.PTE_NAME).exists()
    assert not (ctx.work / "tokenizer.json").exists()


def test_export_that_does_not_load_is_not_saved(fakes):
    fakes.broken = {job.PTE_NAME}
    ctx = fakes.ctx
    status = job.run(ctx)

    metrics = json.loads((ctx.work / "export_metrics.json").read_text())
    assert status == "partial"
    export = metrics["units"]["export"]
    assert not export["ok"] and export["reason"].startswith("exported .pte did not load")
    assert not (ctx.work / job.PTE_NAME).exists()
    assert metrics["units"]["prebuilt"]["ok"] is True


def test_unexpected_exception_in_prebuilt_unit_still_runs_export(fakes, monkeypatch):
    def boom(ctx, scratch, install):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(job, "_unit_prebuilt", boom)
    ctx = fakes.ctx
    status = job.run(ctx)

    metrics = json.loads((ctx.work / "export_metrics.json").read_text())
    assert status == "partial"
    assert metrics["units"]["prebuilt"]["ok"] is False
    assert metrics["units"]["prebuilt"]["reason"].startswith("RuntimeError: unexpected")
    assert metrics["units"]["export"]["ok"] is True
    assert (ctx.work / job.PTE_NAME).exists()


def test_both_units_fail_gives_fail_status(fakes):
    fakes.pip_fail = {"torch"}
    fakes.export_rc = 2
    ctx = fakes.ctx
    assert job.run(ctx) == "fail"
    assert fakes.export_calls == 0  # install failed first, so no export was attempted
