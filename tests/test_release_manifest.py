"""Tests for scripts/release_manifest.py. Everything runs in temporary directories."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_manifest.py"
spec = importlib.util.spec_from_file_location("release_manifest", SCRIPT)
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sha256_and_size_match_hashlib(tmp_path: Path) -> None:
    f = write(tmp_path / "a.bin", b"x" * (3 * (1 << 20) + 7))  # spans several chunks
    digest, size = rm.sha256_file(f)
    assert digest == sha(f.read_bytes())
    assert size == f.stat().st_size


def test_generate_records_git_state_and_sha256sums(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q", "-b", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "t@example.invalid"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    tracked = write(repo / "tracked.txt", b"hello")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    apk = write(repo / "build" / "app.apk", b"apk-bytes")
    write(repo / "untracked.txt", b"dirty")  # makes the tree dirty

    out = tmp_path / "out"
    code = rm.main(
        [
            "--repo",
            str(repo),
            "--no-discover",
            "--out-dir",
            str(out),
            "--artifact",
            f"{apk}@android:app",
        ]
    )
    assert code == 0
    manifest = json.loads((out / "release_manifest.json").read_text())
    assert manifest["schema"] == rm.SCHEMA
    assert manifest["git"]["commit"] == head and len(head) == 40
    assert manifest["git"]["branch"] == "main"
    assert manifest["git"]["dirty"] is True
    assert manifest["created_utc"].endswith("Z")
    (art,) = manifest["artifacts"]
    assert art["device"] == "android" and art["modality"] == "app"
    assert art["path"] == "build/app.apk"  # repo-relative in the manifest
    assert art["sha256"] == sha(b"apk-bytes") and art["status"] == "ok"
    sums = (out / "SHA256SUMS").read_text().splitlines()
    assert sums == [f"{sha(b'apk-bytes')}  build/app.apk"]
    assert tracked.exists()


def test_git_info_outside_a_repository_is_none(tmp_path: Path) -> None:
    info = rm.git_info(tmp_path)  # fields are None, not an exception
    assert info == {"commit": None, "branch": None, "dirty": None, "dirty_paths": None}


def test_missing_artifact_fails_unless_allowed(tmp_path: Path) -> None:
    out = tmp_path / "out"
    args = [
        "--repo",
        str(tmp_path),
        "--no-discover",
        "--out-dir",
        str(out),
        "--artifact",
        str(tmp_path / "nope.apk") + "@android:app",
    ]
    assert rm.main(args) == 1
    manifest = json.loads((out / "release_manifest.json").read_text())
    assert manifest["artifacts"][0]["status"] == "missing"
    assert manifest["summary"]["missing"] == 1
    assert (out / "SHA256SUMS").read_text() == ""  # missing files are not listed as hashed
    assert rm.main(args + ["--allow-missing"]) == 0


def test_verify_passes_then_detects_change_and_removal(tmp_path: Path, capsys) -> None:
    f = write(tmp_path / "model.onnx", b"weights-v1")
    out = tmp_path / "out"
    assert (
        rm.main(
            [
                "--repo",
                str(tmp_path),
                "--no-discover",
                "--out-dir",
                str(out),
                "--artifact",
                str(f) + "@pi:sms",
            ]
        )
        == 0
    )
    manifest = out / "release_manifest.json"

    assert rm.verify(manifest, tmp_path) == 0

    f.write_bytes(b"weights-v2")
    assert rm.verify(manifest, tmp_path) == 1
    assert "FAIL" in capsys.readouterr().out

    f.unlink()
    assert rm.verify(manifest, tmp_path) == 1
    assert "MISSING" in capsys.readouterr().out


def test_verify_rejects_non_manifest(tmp_path: Path) -> None:
    bad = write(tmp_path / "release_manifest.json", b'{"hello": 1}')
    assert rm.verify(bad, tmp_path) == 2
    assert rm.verify(tmp_path / "absent.json", tmp_path) == 2


def test_verify_detects_sha256sums_disagreement(tmp_path: Path, capsys) -> None:
    f = write(tmp_path / "a.txt", b"one")
    out = tmp_path / "out"
    rm.main(
        [
            "--repo",
            str(tmp_path),
            "--no-discover",
            "--out-dir",
            str(out),
            "--artifact",
            str(f) + "@x:y",
        ]
    )
    (out / "SHA256SUMS").write_text(f"{sha(b'other')}  {f}\n")
    assert rm.verify(out / "release_manifest.json", tmp_path) == 1
    assert "SHA256SUMS" in capsys.readouterr().out


def test_registry_discovery_uses_repo_copy_bench_fallback_and_pins(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    bench = tmp_path / "bench"
    good = write(repo / "kaggle" / "good.onnx", b"good-model")
    write(bench / "models" / "fallback.onnx", b"fallback-model")
    write(repo / "kaggle" / "side.json", b"scaler")
    write(bench / "models" / "tok.model", b"tok")
    registry = write(
        repo / "registry.toml",
        f"""
schema = "rbench-registry/1"

[[models]]
id = "good"
kind = "onnx_sms"
enabled = true
source = "kaggle/good.onnx"
path = "models/good.onnx"
sha256 = "{sha(b"good-model")}"

[[models]]
id = "pinned_wrong"
kind = "onnx_url"
enabled = true
source = "kaggle/good.onnx"
path = "models/wrong.onnx"
sha256 = "{"0" * 64}"
side_scaler = "models/side.json"
side_scaler_source = "kaggle/side.json"
side_scaler_sha256 = "{sha(b"scaler")}"

[[models]]
id = "fallback"
kind = "onnx_nids"
enabled = true
source = "kaggle/not_here.onnx"
path = "models/fallback.onnx"
sha256 = "{sha(b"fallback-model")}"

[[models]]
id = "llama"
kind = "pte_llm"
enabled = true
path = "models/llama.pte"
sha256 = "{"1" * 64}"
extras = [{{ role = "tokenizer", path = "models/tok.model", sha256 = "{sha(b"tok")}", bytes = 3 }}]

[[models]]
id = "off"
kind = "onnx_sms"
enabled = false
path = "models/off.onnx"
""".encode(),
    )
    assert good.exists()

    arts = {a["name"]: a for a in rm.registry_artifacts(registry, repo, bench)}
    assert "off" not in arts  # disabled entries are not shipped

    assert arts["good"]["status"] == "ok"
    assert arts["good"]["modality"] == "sms" and arts["good"]["device"] == "pi"
    assert arts["good"]["path"] == "kaggle/good.onnx"

    assert arts["pinned_wrong"]["status"] == "pin_mismatch"
    assert arts["pinned_wrong:side_scaler"]["status"] == "ok"
    assert arts["pinned_wrong:side_scaler"]["role"] == "side_scaler"

    assert arts["fallback"]["status"] == "ok"  # repo source absent, bench copy used
    assert arts["fallback"]["path"].startswith(str(bench.resolve()))

    assert arts["llama"]["status"] == "missing"  # no repo source, no bench copy
    assert arts["llama"]["pinned_sha256"] == "1" * 64  # the pin is still recorded
    assert arts["llama:tokenizer"]["status"] == "ok"
    assert arts["llama:tokenizer"]["modality"] == "explainer"


def test_artifact_spec_parsing(tmp_path: Path) -> None:
    rec = rm.parse_artifact_spec(str(tmp_path / "x.apk") + "@android:app", tmp_path)
    assert (rec["device"], rec["modality"], rec["name"]) == ("android", "app", "x.apk")
    plain = rm.parse_artifact_spec(str(tmp_path / "y.bin"), tmp_path)
    assert (plain["device"], plain["modality"]) == ("unspecified", "unspecified")
    with pytest.raises(ValueError):
        rm.parse_artifact_spec("a@:app", tmp_path)
    with pytest.raises(ValueError):
        rm.parse_artifact_spec("@android:app", tmp_path)


def test_usage_error_when_nothing_to_record(tmp_path: Path) -> None:
    assert (
        rm.main(["--repo", str(tmp_path), "--no-discover", "--out-dir", str(tmp_path / "o")]) == 2
    )


def test_android_artifacts_read_the_repo_that_was_asked_for(tmp_path: Path) -> None:
    """--repo must point the Android asset lookup at that checkout, not at this one."""
    assets = tmp_path / "android" / "app" / "src" / "main" / "assets"
    model = write(assets / "sms_student.onnx", b"model-bytes")
    data = write(assets / "rakshak" / "brands.json", b"{}")
    records = rm.android_artifacts(tmp_path / "no_registry.toml", tmp_path)
    by_name = {r["name"]: r for r in records}
    assert by_name["android_sms_student"]["sha256"] == sha(b"model-bytes")
    assert by_name["android_sms_student"]["path"] == model.relative_to(tmp_path).as_posix()
    assert by_name["android_brands"]["sha256"] == sha(b"{}")
    assert by_name["android_brands"]["path"] == data.relative_to(tmp_path).as_posix()
