"""Colab launcher (colab/launch.sh) and its shared pieces: the runner rendered by
kaggle/build_kernel.py --platform colab, the rakshak-colab-<job> completion matcher in
kaggle/wait_event.py, the /content/rakshak-data candidate in rakshak.eval.data.

launch.sh is exercised in a temporary copy of the repository with a fake ``colab`` binary first on
PATH. The fake records every call, so no Colab session, no network and no real CLI is involved.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

JOB_SOURCE = """\
from __future__ import annotations


def run(ctx):
    return True
"""

FAKE_COLAB = """\
#!/bin/sh
echo "$*" >> "$FAKE_COLAB_LOG"
case "$1" in
  new) [ -z "${FAKE_FAIL_NEW:-}" ] || exit 1 ;;
  download)
    for last in "$@"; do :; done
    printf 'tarball' > "$last" ;;
esac
exit 0
"""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exec_runner(path: Path) -> dict:
    ns = {"__name__": "rakshak_runner_test"}
    exec(compile(path.read_text(), str(path), "exec"), ns)
    return ns


@pytest.fixture
def build_repo(tmp_path, monkeypatch):
    """A temporary repo that build_kernel.py can bundle, with its output roots redirected."""
    bk = load_module("build_kernel", REPO / "kaggle" / "build_kernel.py")
    repo = tmp_path / "repo"
    (repo / "src" / "rakshak").mkdir(parents=True)
    (repo / "src" / "rakshak" / "__init__.py").write_text("")
    (repo / "kaggle" / "jobs").mkdir(parents=True)
    (repo / "kaggle" / "jobs" / "_common.py").write_text("")
    (repo / "kaggle" / "jobs" / "dc_job.py").write_text(JOB_SOURCE)
    (repo / "kaggle" / "requirements-kaggle.txt").write_text("numpy==2.0.0\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setattr(bk, "REPO", repo)
    monkeypatch.setattr(bk, "OUT_ROOT", repo / "kaggle" / "out")
    monkeypatch.setattr(bk, "COLAB_OUT_ROOT", repo / "colab" / "out")
    monkeypatch.setattr(bk, "PINS_FILE", repo / "kaggle" / "requirements-kaggle.txt")
    monkeypatch.setenv("RAKSHAK_NTFY_TOPIC", "colabtesttopic1")
    return bk, repo


def test_colab_runner_renders_colab_paths(build_repo):
    bk, repo = build_repo
    bk.main(["--platform", "colab", "--job", "dc_job", "--job-args=--x 1"])
    out = repo / "colab" / "out" / "dc_job" / "kernel"
    assert not (out / "kernel-metadata.json").exists()
    assert not (repo / "kaggle" / "out").exists()
    ns = exec_runner(out / "run.py")
    try:
        assert ns["PLATFORM"] == "colab"
        assert ns["WORK"] == Path("/content/out")
        assert ns["INPUT_ROOT"] == Path("/content")
        assert ns["SLUG"] == "colab-dc-job"
        assert ns["ENABLE_INTERNET"] is True
        assert ns["NOTIFY_URL"] == "https://ntfy.sh/colabtesttopic1"
        assert ns["JOB_ARGV"] == ["--x", "1"]
        assert ns["PIP_PINS"] == ["numpy==2.0.0"]
    finally:
        shutil.rmtree(ns["SCRATCH"], ignore_errors=True)


def test_kaggle_runner_keeps_kaggle_paths(build_repo):
    bk, repo = build_repo
    bk.main(["--job", "dc_job"])
    out = repo / "kaggle" / "out" / "dc_job" / "kernel"
    assert (out / "kernel-metadata.json").is_file()
    assert not (repo / "colab" / "out").exists()
    ns = exec_runner(out / "run.py")
    try:
        assert ns["PLATFORM"] == "kaggle"
        assert ns["WORK"] == Path("/kaggle/working")
        assert ns["INPUT_ROOT"] == Path("/kaggle/input")
        assert ns["SLUG"] == "dc-job"
        assert ns["ENABLE_INTERNET"] is False
        assert ns["NOTIFY_URL"] == ""
    finally:
        shutil.rmtree(ns["SCRATCH"], ignore_errors=True)


def test_colab_environment_reports_gpu_key(build_repo):
    bk, repo = build_repo
    bk.main(["--platform", "colab", "--job", "dc_job"])
    ns = exec_runner(repo / "colab" / "out" / "dc_job" / "kernel" / "run.py")
    try:
        ns["environment"]()
        info = ns["STATE"]["environment"]
        # None on a machine without nvidia-smi; a list of "name, memory" strings otherwise.
        assert "gpu" in info
        assert info["gpu"] is None or all(isinstance(row, str) for row in info["gpu"])
    finally:
        shutil.rmtree(ns["SCRATCH"], ignore_errors=True)


def test_colab_completion_line_has_colab_prefix(build_repo, monkeypatch):
    bk, repo = build_repo
    bk.main(["--platform", "colab", "--job", "dc_job"])
    ns = exec_runner(repo / "colab" / "out" / "dc_job" / "kernel" / "run.py")
    sent: list[bytes] = []

    class Reply:
        def read(self):
            return b""

    def fake_urlopen(req, timeout=None):
        sent.append(req.data)
        return Reply()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    try:
        ns["STATE"]["status"] = "ok"
        ns["STATE"]["summary"] = "done"
        ns["notify"]()
        assert sent, "notify sent nothing"
        assert sent[0].decode().startswith(f"rakshak-colab-dc-job ok git={ns['GIT_LABEL']} ")
    finally:
        shutil.rmtree(ns["SCRATCH"], ignore_errors=True)


@pytest.mark.parametrize("extra", [["--gpu", "t4"], ["--datasets", "a/b"], ["--models", "a/b/c"]])
def test_colab_refuses_kaggle_only_options(build_repo, extra):
    bk, _ = build_repo
    with pytest.raises(SystemExit) as info:
        bk.main(["--platform", "colab", "--job", "dc_job", *extra])
    assert info.value.code == 2


def test_wait_event_matches_platform_prefix():
    we = load_module("wait_event", REPO / "kaggle" / "wait_event.py")
    colab_msg = {"event": "message", "time": 150,
                 "message": "rakshak-colab-sms-teacher ok git=abc123 tree=1 done"}
    kaggle_msg = {"event": "message", "time": 150,
                  "message": "rakshak-sms-teacher ok git=abc123 tree=1 done"}
    assert we.matches(colab_msg, "sms-teacher", "abc123", 100, "colab")
    assert not we.matches(colab_msg, "sms-teacher", "abc123", 100)
    assert we.matches(kaggle_msg, "sms-teacher", "abc123", 100)
    assert not we.matches(kaggle_msg, "sms-teacher", "abc123", 100, "colab")
    assert not we.matches(colab_msg, "sms-teacher", "abc123", 200, "colab")  # stale
    assert not we.matches(colab_msg, "sms-teacher", "other", 100, "colab")  # other build


def test_data_root_checks_colab_folder(tmp_path, monkeypatch):
    from rakshak.eval import data as D

    assert D.COLAB_DATA == Path("/content/rakshak-data")
    colab_data = tmp_path / "colab-data"
    colab_data.mkdir()
    (colab_data / D.SMS_FILE).write_text("")
    monkeypatch.delenv(D.ENV_VAR, raising=False)
    monkeypatch.setattr(D, "KAGGLE_DATA", tmp_path / "no-kaggle")
    monkeypatch.setattr(D, "COLAB_DATA", colab_data)
    monkeypatch.setattr(D, "REPO_DATA", tmp_path / "no-repo")
    assert D.resolve_data_root() == colab_data


def _run_wrapper(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(REPO / "colab" / "remote_run_job.py")],
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "RAKSHAK_REMOTE_ROOT": str(root)})


def _fake_runner(root: Path, body: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "rakshak_run.py").write_text(body)


def test_remote_run_job_streams_output_and_logs_it(tmp_path):
    root = tmp_path / "content"
    _fake_runner(root, "import sys\nprint('out line')\nprint('err line', file=sys.stderr)\n")
    res = _run_wrapper(root)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "out line" in res.stdout
    assert "err line" in res.stdout  # stderr is merged into the same stream
    log = (root / "out" / "runner.stdout.log").read_text()
    assert "out line\n" in log and "err line\n" in log
    assert "runner exited 0" in res.stdout


def test_remote_run_job_passes_the_runner_exit_code_on(tmp_path):
    root = tmp_path / "content"
    _fake_runner(root, "print('before failing')\nraise SystemExit(3)\n")
    res = _run_wrapper(root)
    assert res.returncode == 3, res.stdout + res.stderr
    assert "before failing" in res.stdout
    assert "runner exited 3" in res.stdout
    assert "before failing" in (root / "out" / "runner.stdout.log").read_text()


def test_remote_run_job_runs_in_a_fresh_interpreter(tmp_path):
    root = tmp_path / "content"
    _fake_runner(root, "import os\nprint(f'pids {os.getpid()} {os.getppid()}')\n")
    env = {**os.environ, "RAKSHAK_REMOTE_ROOT": str(root)}
    proc = subprocess.Popen([sys.executable, "-I", str(REPO / "colab" / "remote_run_job.py")],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, out + err
    line = next(line for line in out.splitlines() if line.startswith("pids "))
    runner_pid, parent_pid = (int(x) for x in line.split()[1:])
    # The runner is a child of the wrapper process, never the wrapper process itself.
    assert parent_pid == proc.pid
    assert runner_pid != proc.pid


def test_remote_run_job_refuses_a_missing_runner(tmp_path):
    res = _run_wrapper(tmp_path / "content")
    assert res.returncode != 0
    assert "rakshak_run.py is missing" in res.stderr


def test_remote_helpers_compile():
    for name in ("remote_unpack_data.py", "remote_pack_out.py", "remote_run_job.py"):
        path = REPO / "colab" / name
        compile(path.read_text(), str(path), "exec")


def test_launch_script_syntax():
    subprocess.run(["bash", "-n", str(REPO / "colab" / "launch.sh")], check=True)


def test_launch_stops_vm_on_every_exit_path():
    text = (REPO / "colab" / "launch.sh").read_text()
    assert re.search(r"^trap on_exit EXIT$", text, re.M)
    assert "trap 'exit 130' INT" in text
    assert "trap 'exit 143' TERM" in text
    assert "stop_vm || true" in text
    # The stop is armed before colab new is requested, so a failed allocation is still stopped.
    assert text.index("STOP_PENDING=1") < text.index('run "${COLAB[@]}" "${NEW_ARGS[@]}"')


@pytest.fixture
def launch_repo(tmp_path, monkeypatch):
    """A temporary copy of what launch.sh needs, with a fake colab binary first on PATH."""
    repo = tmp_path / "repo"
    for sub in ("colab", "kaggle/jobs", "src/rakshak", "data/processed"):
        (repo / sub).mkdir(parents=True)
    for name in ("launch.sh", "data_bundle.py", "remote_unpack_data.py", "remote_pack_out.py",
                 "remote_run_job.py"):
        shutil.copy2(REPO / "colab" / name, repo / "colab" / name)
    (repo / "colab" / "launch.sh").chmod(0o755)
    shutil.copy2(REPO / "kaggle" / "build_kernel.py", repo / "kaggle" / "build_kernel.py")
    (repo / "src" / "rakshak" / "__init__.py").write_text("")
    (repo / "kaggle" / "jobs" / "_common.py").write_text("")
    (repo / "kaggle" / "jobs" / "dc_job.py").write_text(JOB_SOURCE)
    (repo / "kaggle" / "requirements-kaggle.txt").write_text("numpy==2.0.0\n")
    write_data(repo / "data" / "processed")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    fake = fakebin / "colab"
    fake.write_text(FAKE_COLAB)
    fake.chmod(0o755)
    monkeypatch.setenv("FAKE_COLAB_LOG", str(calls_file(repo)))
    monkeypatch.delenv("FAKE_FAIL_NEW", raising=False)
    monkeypatch.setenv("RAKSHAK_NTFY_TOPIC", "colabtesttopic1")
    monkeypatch.setenv("PATH", f"{fakebin}{os.pathsep}{os.environ['PATH']}")
    return repo


DATA_FILES = {
    "sms.parquet": 300_000,
    "urls.parquet": 1_000,
    "urls_phreshphish.parquet": 1_000,
    "build_report.json": 100,
    "README.txt": 100,
    "splits/campaign.json": 1_000,
    "splits/temporal.json": 1_000,
    "splits/url_domain.json": 1_000,
    "splits/url_phreshphish.json": 1_000,
    "external/external_smishx.parquet": 1_000,
}


def write_data(root: Path, sizes: dict[str, int] = DATA_FILES) -> None:
    """A small data/processed tree. Random bytes, so sizes are exact and nothing compresses away."""
    for rel, size in sizes.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(os.urandom(size))


def add_job(repo: Path, name: str) -> None:
    (repo / "kaggle" / "jobs" / f"{name}.py").write_text(JOB_SOURCE)


def manifest_of(repo: Path, job: str) -> dict:
    return json.loads((repo / "colab" / "out" / job / "data" / "manifest.json").read_text())


def launch(repo: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(["bash", str(repo / "colab" / "launch.sh"), *args],
                          capture_output=True, text=True, cwd=repo, timeout=120, env=run_env)




def calls_file(repo: Path) -> Path:
    return repo.parent / "colab-calls.txt"


def calls_of(repo: Path) -> list[str]:
    path = calls_file(repo)
    return path.read_text().splitlines() if path.exists() else []


def test_dry_run_prints_colab_commands_and_runs_none(launch_repo):
    res = launch(launch_repo, "dc_job", "--gpu", "T4", "--data",
                 "--job-args=--models x --no-fp16", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    out = res.stdout
    session = re.search(r"colab new -s (rakshak-dc-job-\d+) --gpu T4", out)
    assert session, out
    name = session.group(1)
    assert f"[dry-run] colab upload -s {name} " in out
    assert "/content/rakshak-data.manifest.json" in out
    assert re.search(rf"\[dry-run\] colab upload -s {name} \S+/data/rakshak-data\.part-000 "
                     r"/content/rakshak-data\.part-000", out)
    assert re.search(rf"\[dry-run\] colab exec -s {name} -f \S+/colab/remote_unpack_data\.py "
                     r"--timeout 900", out)
    assert re.search(rf"\[dry-run\] colab upload -s {name} \S+/colab/out/dc_job/kernel/run\.py "
                     r"/content/rakshak_run\.py", out)
    assert re.search(rf"\[dry-run\] colab exec -s {name} -f \S+/colab/remote_run_job\.py "
                     r"--timeout 43200", out)
    assert "kernel/run.py --timeout" not in out  # the runner is never exec'd in the kernel
    assert f"[dry-run] colab download -s {name} /content/rakshak-out.tar.gz" in out
    assert f"[dry-run] colab stop -s {name}" in out
    assert "dry run complete" in out
    assert (launch_repo / "colab" / "out" / "dc_job" / "kernel" / "run.py").is_file()
    # The bundle is built locally even in a dry run; no colab command runs.
    assert (launch_repo / "colab" / "out" / "dc_job" / "data" / "manifest.json").is_file()
    assert calls_of(launch_repo) == []


@pytest.mark.parametrize("gpu", ["A100", "L4", "H100"])
def test_paid_gpu_refused_without_yes_paid(launch_repo, gpu):
    res = launch(launch_repo, "dc_job", "--gpu", gpu, "--dry-run")
    assert res.returncode == 2
    assert f"refusing --gpu {gpu}" in res.stderr
    assert "[dry-run]" not in res.stdout
    assert calls_of(launch_repo) == []


def test_high_mem_refused_without_yes_paid(launch_repo):
    res = launch(launch_repo, "dc_job", "--high-mem", "--dry-run")
    assert res.returncode == 2
    assert "refusing --high-mem" in res.stderr


def test_cpu_allocates_without_an_accelerator(launch_repo):
    res = launch(launch_repo, "dc_job", "--cpu", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    assert re.search(r"\[dry-run\] colab new -s rakshak-dc-job-\d+$", res.stdout, re.M)
    assert "cpu=1" in (launch_repo / "colab" / "out" / "dc_job" / "launch.log").read_text()


def test_cpu_and_gpu_are_exclusive(launch_repo):
    res = launch(launch_repo, "dc_job", "--cpu", "--gpu", "T4", "--dry-run")
    assert res.returncode == 2
    assert "cannot be combined" in res.stderr
    assert calls_of(launch_repo) == []


def test_cpu_is_not_a_paid_run(launch_repo):
    res = launch(launch_repo, "dc_job", "--cpu", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "refusing" not in res.stderr


def test_paid_gpu_allowed_with_yes_paid_in_dry_run(launch_repo):
    res = launch(launch_repo, "dc_job", "--gpu", "a100", "--high-mem", "--yes-paid", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    assert re.search(r"colab new -s rakshak-dc-job-\d+ --gpu A100 --high-mem", res.stdout)


def test_unknown_gpu_refused(launch_repo):
    res = launch(launch_repo, "dc_job", "--gpu", "V100", "--yes-paid", "--dry-run")
    assert res.returncode == 2
    assert "unknown GPU 'V100'" in res.stderr


def test_failed_allocation_still_stops_vm(launch_repo, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_NEW", "1")
    res = launch(launch_repo, "dc_job")
    calls = calls_of(launch_repo)
    assert res.returncode != 0
    assert any(c.startswith("new -s rakshak-dc-job-") for c in calls)
    assert any(c.startswith("stop -s rakshak-dc-job-") for c in calls)
    assert not any(c.startswith("exec ") for c in calls)


def test_full_run_order_and_stop_after_download(launch_repo):
    res = launch(launch_repo, "dc_job", "--data", "--job-args=--x 1")
    assert res.returncode == 0, res.stdout + res.stderr
    calls = calls_of(launch_repo)
    name = next(c.split()[2] for c in calls if c.startswith("new -s "))

    def index(prefix: str) -> int:
        matches = [i for i, c in enumerate(calls) if c.startswith(prefix)]
        assert matches, (prefix, calls)
        return matches[0]

    order = [
        index("usage"),
        index(f"new -s {name} --gpu T4"),
        index(f"upload -s {name} {launch_repo}/colab/out/dc_job/data/manifest.json "
              "/content/rakshak-data.manifest.json"),
        index(f"upload -s {name} {launch_repo}/colab/out/dc_job/data/rakshak-data.part-000 "
              "/content/rakshak-data.part-000"),
        index(f"exec -s {name} -f {launch_repo}/colab/remote_unpack_data.py"),
        index(f"upload -s {name} {launch_repo}/colab/out/dc_job/kernel/run.py "
              "/content/rakshak_run.py"),
        index(f"exec -s {name} -f {launch_repo}/colab/remote_run_job.py --timeout 43200"),
        index(f"exec -s {name} -f {launch_repo}/colab/remote_pack_out.py"),
        index(f"download -s {name} /content/rakshak-out.tar.gz"),
        index(f"stop -s {name}"),
    ]
    assert order == sorted(order)
    assert calls[-1] == "usage"
    tarball = launch_repo / "colab" / "out" / "dc_job" / "output" / "out.tar.gz"
    assert tarball.read_text() == "tarball"
    assert "launch: done" in res.stdout
    assert (launch_repo / "colab" / "out" / "dc_job" / "launch.log").is_file()


def test_every_upload_uses_an_absolute_content_path(launch_repo):
    res = launch(launch_repo, "dc_job", "--data", "--dry-run", env={"RAKSHAK_COLAB_CHUNK_MB": "1"})
    assert res.returncode == 0, res.stdout + res.stderr
    uploads = re.findall(r"colab upload -s \S+ \S+ (\S+)", res.stdout)
    assert uploads and all(path.startswith("/content/") for path in uploads), uploads


def test_data_files_uploads_only_selected_files(launch_repo):
    res = launch(launch_repo, "dc_job", "--data-files", "splits/url_*.json", "urls.parquet",
                 "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    files = [entry["path"] for entry in manifest_of(launch_repo, "dc_job")["files"]]
    assert files == ["splits/url_domain.json", "splits/url_phreshphish.json", "urls.parquet"]
    assert "patterns from --data-files: splits/url_*.json urls.parquet" in res.stdout


def test_data_files_stops_at_next_option(launch_repo):
    res = launch(launch_repo, "dc_job", "--data-files", "sms.parquet", "--gpu", "T4", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    files = [entry["path"] for entry in manifest_of(launch_repo, "dc_job")["files"]]
    assert files == ["sms.parquet"]
    assert "colab new -s rakshak-dc-job-" in res.stdout


def test_data_files_without_pattern_refused(launch_repo):
    res = launch(launch_repo, "dc_job", "--data-files", "--dry-run")
    assert res.returncode == 2
    assert "--data-files needs at least one PATTERN" in res.stderr
    assert calls_of(launch_repo) == []


def test_data_files_pattern_matching_nothing_refused_before_colab_new(launch_repo):
    res = launch(launch_repo, "dc_job", "--data-files", "no_such_file.parquet")
    assert res.returncode != 0
    assert "matched no file" in res.stderr + res.stdout
    assert calls_of(launch_repo) == []


@pytest.mark.parametrize("job, expected", [
    ("sms_probe", ["external/external_smishx.parquet", "sms.parquet", "splits/campaign.json",
                   "splits/temporal.json"]),
    ("url_probe", ["splits/url_domain.json", "splits/url_phreshphish.json", "urls.parquet",
                   "urls_phreshphish.parquet"]),
    ("llama_probe", ["sms.parquet", "splits/campaign.json"]),
])
def test_default_data_per_job_family(launch_repo, job, expected):
    add_job(launch_repo, job)
    res = launch(launch_repo, job, "--data", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    files = [entry["path"] for entry in manifest_of(launch_repo, job)["files"]]
    assert files == expected


def test_default_data_for_other_jobs_is_everything(launch_repo):
    res = launch(launch_repo, "dc_job", "--data", "--dry-run")
    assert res.returncode == 0, res.stdout + res.stderr
    files = [entry["path"] for entry in manifest_of(launch_repo, "dc_job")["files"]]
    assert files == sorted(DATA_FILES)


def test_large_bundle_is_split_and_every_part_checksummed(launch_repo):
    # Random bytes do not compress, so 2.6 MB of sms.parquet gives an archive of 3 parts at 1 MiB.
    write_data(launch_repo / "data" / "processed", {"sms.parquet": 2_600_000})
    res = launch(launch_repo, "dc_job", "--data-files", "sms.parquet", "--dry-run",
                 env={"RAKSHAK_COLAB_CHUNK_MB": "1"})
    assert res.returncode == 0, res.stdout + res.stderr
    manifest = manifest_of(launch_repo, "dc_job")
    data_dir = launch_repo / "colab" / "out" / "dc_job" / "data"
    parts = manifest["parts"]
    assert len(parts) >= 2
    assert [p["name"] for p in parts] == [f"rakshak-data.part-{i:03d}" for i in range(len(parts))]
    joined = b"".join((data_dir / p["name"]).read_bytes() for p in parts)
    assert len(joined) == manifest["archive"]["bytes"]
    assert hashlib.sha256(joined).hexdigest() == manifest["archive"]["sha256"]
    for part in parts:
        assert part["bytes"] <= 1024 * 1024
        assert hashlib.sha256((data_dir / part["name"]).read_bytes()).hexdigest() == part["sha256"]
    uploads = re.findall(r"colab upload -s \S+ \S+ (/content/\S+)", res.stdout)
    assert uploads == ["/content/rakshak-data.manifest.json"] + [
        f"/content/{p['name']}" for p in parts] + ["/content/rakshak_run.py"]


def test_stale_parts_are_not_uploaded(launch_repo):
    write_data(launch_repo / "data" / "processed", {"sms.parquet": 2_600_000})
    env = {"RAKSHAK_COLAB_CHUNK_MB": "1"}
    assert launch(launch_repo, "dc_job", "--data-files", "sms.parquet", "--dry-run",
                  env=env).returncode == 0
    stale = launch_repo / "colab" / "out" / "dc_job" / "data" / "rakshak-data.part-099"
    stale.write_bytes(b"old")
    assert launch(launch_repo, "dc_job", "--data-files", "urls.parquet", "--dry-run",
                  env=env).returncode == 0
    assert not stale.exists()


@pytest.mark.parametrize("value", ["0", "51", "abc", "-1"])
def test_chunk_size_must_be_a_safe_whole_number(launch_repo, value):
    res = launch(launch_repo, "dc_job", "--data", "--dry-run",
                 env={"RAKSHAK_COLAB_CHUNK_MB": value})
    assert res.returncode == 2
    assert "RAKSHAK_COLAB_CHUNK_MB must be a whole number of MiB from 1 to 50" in res.stderr
    assert calls_of(launch_repo) == []


def _run_unpack(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(REPO / "colab" / "remote_unpack_data.py")],
                          capture_output=True, text=True, timeout=120,
                          env={**os.environ, "RAKSHAK_REMOTE_ROOT": str(root)})


def _stage_remote(repo: Path, job: str, root: Path) -> None:
    """Copy the manifest and parts the laptop built into a fake /content folder."""
    data_dir = repo / "colab" / "out" / job / "data"
    root.mkdir(parents=True, exist_ok=True)
    for src in [data_dir / "manifest.json", *data_dir.glob("rakshak-data.part-*")]:
        dst = root / ("rakshak-data.manifest.json" if src.name == "manifest.json" else src.name)
        dst.write_bytes(src.read_bytes())


def test_remote_unpack_joins_verifies_and_extracts(launch_repo, tmp_path):
    write_data(launch_repo / "data" / "processed", {"sms.parquet": 2_600_000})
    assert launch(launch_repo, "dc_job", "--data-files", "sms.parquet", "splits/campaign.json",
                  "--dry-run", env={"RAKSHAK_COLAB_CHUNK_MB": "1"}).returncode == 0
    root = tmp_path / "content"
    _stage_remote(launch_repo, "dc_job", root)
    assert len(list(root.glob("rakshak-data.part-*"))) >= 2
    res = _run_unpack(root)
    assert res.returncode == 0, res.stdout + res.stderr
    dest = root / "rakshak-data"
    assert (dest / "sms.parquet").read_bytes() == (
        launch_repo / "data" / "processed" / "sms.parquet").read_bytes()
    assert (dest / "splits" / "campaign.json").is_file()
    assert not list(root.glob("rakshak-data.part-*"))
    assert not (root / "rakshak-data.tar.gz").exists()
    assert not (root / "rakshak-data.manifest.json").exists()


def test_remote_unpack_refuses_a_corrupt_part(launch_repo, tmp_path):
    write_data(launch_repo / "data" / "processed", {"sms.parquet": 2_600_000})
    assert launch(launch_repo, "dc_job", "--data-files", "sms.parquet", "--dry-run",
                  env={"RAKSHAK_COLAB_CHUNK_MB": "1"}).returncode == 0
    root = tmp_path / "content"
    _stage_remote(launch_repo, "dc_job", root)
    part = sorted(root.glob("rakshak-data.part-*"))[1]
    data = bytearray(part.read_bytes())
    data[100] ^= 0xFF
    part.write_bytes(bytes(data))
    res = _run_unpack(root)
    assert res.returncode != 0
    assert "fails its sha256" in res.stderr
    assert not (root / "rakshak-data").exists()


def test_remote_unpack_refuses_a_missing_part(launch_repo, tmp_path):
    write_data(launch_repo / "data" / "processed", {"sms.parquet": 2_600_000})
    assert launch(launch_repo, "dc_job", "--data-files", "sms.parquet", "--dry-run",
                  env={"RAKSHAK_COLAB_CHUNK_MB": "1"}).returncode == 0
    root = tmp_path / "content"
    _stage_remote(launch_repo, "dc_job", root)
    sorted(root.glob("rakshak-data.part-*"))[-1].unlink()
    res = _run_unpack(root)
    assert res.returncode != 0
    assert "missing part" in res.stderr
    assert not (root / "rakshak-data").exists()


def test_bundle_selection_rules(launch_repo):
    sys.path.insert(0, str(REPO / "colab"))
    try:
        import data_bundle as db
    finally:
        sys.path.pop(0)
    root = launch_repo / "data" / "processed"
    assert db.select(root, ["splits/*.json", "!splits/url_*.json"]) == [
        "splits/campaign.json", "splits/temporal.json"]
    with pytest.raises(db.BundleError, match="matched no file"):
        db.select(root, ["nothing_here.parquet"])
    with pytest.raises(db.BundleError, match="stay inside"):
        db.select(root, ["../etc/passwd"])
    with pytest.raises(db.BundleError, match="include PATTERN"):
        db.select(root, ["!sms.parquet"])
    with pytest.raises(db.BundleError, match="at least"):
        db.build(root, launch_repo / "bundle", 10, ["sms.parquet"])

