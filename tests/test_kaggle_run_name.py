"""--run-name: one job as several Kaggle kernels (build, wait and launch/fetch naming).

The shell tests run kaggle/launch.sh and kaggle/fetch.sh in a scratch copy of the launcher, with a
fake kaggle CLI on PATH that only logs its calls, so nothing is pushed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "kaggle"
TOPIC = "rakshak-test-topic-0001"
# Generated run.py: the comment sits on its own line above SLUG, and the SLUG line ends there.
PLAIN_SLUG = (
    "# notification slug\n"
    "SLUG = ('colab-' if PLATFORM == 'colab' else '') + JOB.replace('_', '-')\n"
)
RUN_B_SLUG = (
    "# notification slug\n"
    "SLUG = ('colab-' if PLATFORM == 'colab' else '') + JOB.replace('_', '-') + \"-b\"\n"
)
JOB_SOURCE = "def run(ctx):\n    return True\n"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, KAGGLE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(root: Path) -> Path:
    """A git repository with the files the build bundles and no commits (label no-commit)."""
    repo = root
    (repo / "src" / "rakshak").mkdir(parents=True)
    (repo / "src" / "rakshak" / "__init__.py").write_text("")
    (repo / "kaggle" / "jobs").mkdir(parents=True)
    (repo / "kaggle" / "jobs" / "_common.py").write_text("")
    (repo / "kaggle" / "jobs" / "dc_job.py").write_text(JOB_SOURCE)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


@pytest.fixture
def builder(tmp_path, monkeypatch):
    bk = load("build_kernel")
    pins = tmp_path / "requirements-kaggle.txt"
    pins.write_text("numpy==2.0.0\n")
    monkeypatch.setattr(bk, "REPO", make_repo(tmp_path / "repo"))
    monkeypatch.setattr(bk, "OUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(bk, "PINS_FILE", pins)
    monkeypatch.setenv("RAKSHAK_NTFY_TOPIC", TOPIC)
    return bk, tmp_path


def test_default_build_keeps_plain_slug_and_folder(builder):
    bk, tmp = builder
    bk.main(["--job", "dc_job"])
    kernel = tmp / "out" / "dc_job" / "kernel"
    assert PLAIN_SLUG in (kernel / "run.py").read_text()
    meta = json.loads((kernel / "kernel-metadata.json").read_text())
    assert meta["id"] == "maadhavvh/rakshak-dc-job"
    assert not (tmp / "out" / "dc_job-b").exists()


def test_run_name_builds_its_own_kernel(builder):
    bk, tmp = builder
    bk.main(["--job", "dc_job", "--run-name", "b", "--internet"])
    kernel = tmp / "out" / "dc_job-b" / "kernel"
    meta = json.loads((kernel / "kernel-metadata.json").read_text())
    assert meta["id"] == "maadhavvh/rakshak-dc-job-b"
    assert meta["title"] == "rakshak dc-job-b"
    assert not (tmp / "out" / "dc_job").exists()
    run_py = kernel / "run.py"
    assert RUN_B_SLUG in run_py.read_text()
    ns = {"__name__": "rakshak_kernel_test"}
    exec(compile(run_py.read_text(), str(run_py), "exec"), ns)
    try:
        assert ns["SLUG"] == "dc-job-b"
        assert ns["NOTIFY_URL"] == f"https://ntfy.sh/{TOPIC}"
    finally:
        shutil.rmtree(ns["SCRATCH"], ignore_errors=True)


def test_run_name_accepts_twenty_characters(builder):
    bk, tmp = builder
    name = "a" * 20
    bk.main(["--job", "dc_job", "--run-name", name])
    assert (tmp / "out" / f"dc_job-{name}" / "kernel" / "run.py").is_file()


@pytest.mark.parametrize("name", ["", "B", "a_b", "a b", "x" * 21, "../x", "b.c"])
def test_run_name_is_validated_before_anything_is_written(builder, name):
    bk, tmp = builder
    with pytest.raises(SystemExit):
        bk.main(["--job", "dc_job", "--run-name", name])
    assert not (tmp / "out").exists()


def test_run_name_is_refused_for_colab(builder):
    bk, tmp = builder
    with pytest.raises(SystemExit):
        bk.main(["--job", "dc_job", "--platform", "colab", "--run-name", "b"])
    assert not (tmp / "colab").exists()


@pytest.fixture
def waiter(builder, monkeypatch):
    bk, tmp = builder
    wait = load("wait_event")
    monkeypatch.setattr(wait, "OUT", tmp / "out")
    monkeypatch.setattr(wait, "COLAB_OUT", tmp / "colab_out")
    seen = []

    class Stream:
        events = []

        def __enter__(self):
            return [json.dumps(event).encode() + b"\n" for event in self.events]

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=None):
        seen.append(url)
        return Stream()

    monkeypatch.setattr(wait.urllib.request, "urlopen", fake_urlopen)
    return wait, bk, tmp, Stream, seen


def test_wait_names_the_run_either_way(waiter, monkeypatch):
    wait, bk, tmp, stream, seen = waiter
    bk.main(["--job", "dc_job", "--run-name", "b", "--internet"])
    folder = tmp / "out" / "dc_job-b"
    (folder / "launch_epoch").write_text("100\n")
    stream.events = [
        # the default run of the same job: a different kernel, must not complete this wait
        {"event": "message", "time": 150, "message": "rakshak-dc-job ok git=no-commit tree=abc"},
        {"event": "message", "time": 200, "message": "rakshak-dc-job-b ok git=no-commit tree=abc"},
    ]
    for argv in (["dc_job-b"], ["dc_job", "--run-name", "b"]):
        monkeypatch.setattr(sys, "argv", ["wait_event.py", *argv])
        assert wait.main() == 0
        receipt = json.loads((folder / "completion.json").read_text())
        assert receipt["job"] == "dc_job"
        assert receipt["run_name"] == "b"
        assert receipt["message"].startswith("rakshak-dc-job-b ok")
    assert all("since=100" in url for url in seen)


def test_wait_without_arguments_follows_the_latest_named_launch(waiter, monkeypatch):
    wait, bk, tmp, stream, _seen = waiter
    bk.main(["--job", "dc_job", "--run-name", "b", "--internet"])
    folder = tmp / "out" / "dc_job-b"
    (folder / "launch_epoch").write_text("100\n")
    (tmp / "out" / "latest_launch.json").write_text(
        json.dumps({"job": "dc_job", "run_name": "b"}) + "\n")
    stream.events = [
        {"event": "message", "time": 200, "message": "rakshak-dc-job-b fail git=no-commit"}]
    monkeypatch.setattr(sys, "argv", ["wait_event.py"])
    assert wait.main() == 0
    assert json.loads((folder / "completion.json").read_text())["run_name"] == "b"


@pytest.fixture
def launcher_copy(tmp_path):
    """A scratch copy of the launcher scripts in its own repository, with a fake kaggle CLI on
    PATH."""
    root = make_repo(tmp_path / "copy")
    kag = root / "kaggle"
    for name in ("launch.sh", "fetch.sh", "wait_event.sh", "wait_event.py", "build_kernel.py",
                 "requirements-kaggle.txt"):
        shutil.copy(KAGGLE / name, kag / name)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "kaggle_calls.log"
    fake = bindir / "kaggle"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$@\" >> '{calls}'\n"
        "if [ \"$1 $2\" = 'kernels output' ]; then\n"
        "  printf '%s' '{\"job\": \"dc_job\", \"status\": \"ok\"}' > \"${@: -1}/results.json\"\n"
        "fi\n"
    )
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
           "RAKSHAK_NTFY_TOPIC": TOPIC}
    return kag, env, calls


def run_bash(script: Path, args, env):
    return subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True)


def test_launch_default_writes_the_same_latest_record(launcher_copy):
    kag, env, calls = launcher_copy
    done = run_bash(kag / "launch.sh", ["dc_job"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (kag / "out" / "latest_launch.json").read_text() == '{"job": "dc_job"}\n'
    assert (kag / "out" / "dc_job" / "kernel" / "kernel-metadata.json").is_file()
    assert not (kag / "out" / "dc_job-b").exists()
    assert "kernels push -p " + str(kag / "out" / "dc_job" / "kernel") in calls.read_text()
    assert "kaggle/wait_event.sh dc_job, then kaggle/fetch.sh dc_job." in done.stdout


def test_launch_run_name_uses_its_own_folder_and_record(launcher_copy):
    kag, env, calls = launcher_copy
    done = run_bash(kag / "launch.sh", ["dc_job", "--run-name", "b", "--internet"], env)
    assert done.returncode == 0, done.stdout + done.stderr
    folder = kag / "out" / "dc_job-b"
    assert (folder / "kernel" / "run.py").is_file()
    assert (folder / "launch_epoch").is_file()
    assert json.loads((kag / "out" / "latest_launch.json").read_text()) == {
        "job": "dc_job", "run_name": "b"}
    assert "kernels push -p " + str(folder / "kernel") in calls.read_text()
    assert "kaggle/wait_event.sh dc_job-b, then kaggle/fetch.sh dc_job-b." in done.stdout
    assert not (kag / "out" / "dc_job").exists()


def test_launch_rejects_a_bad_run_name_before_building(launcher_copy):
    kag, env, _calls = launcher_copy
    done = run_bash(kag / "launch.sh", ["dc_job", "--run-name", "B"], env)
    assert done.returncode == 2
    assert not (kag / "out").exists()


def test_fetch_takes_either_form_and_checks_the_job(launcher_copy):
    kag, env, calls = launcher_copy
    for args in (["dc_job-b"], ["dc_job", "--run-name", "b"]):
        done = run_bash(kag / "fetch.sh", args, env)
        assert done.returncode == 0, done.stdout + done.stderr
    assert calls.read_text().count("kernels output maadhavvh/rakshak-dc-job-b -p ") == 2
    assert (kag / "out" / "dc_job-b" / "output" / "results.json").is_file()
    # results.json says dc_job, so fetching a different job's run must be refused.
    done = run_bash(kag / "fetch.sh", ["other-b"], env)
    assert done.returncode != 0
    assert "results.json is for job 'dc_job'" in done.stderr
