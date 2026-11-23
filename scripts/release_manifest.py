#!/usr/bin/env python
"""Release manifest and SHA256SUMS for Rakshak-Edge artifacts.

Writes ``release_manifest.json`` and ``SHA256SUMS`` for a set of files, and re-hashes an existing
manifest with ``--verify``.

Artifacts come from three places:

* the Pi model registry (``deploy/pi/bench/registry.toml``): every enabled model, its side scaler
  and its extra files, with the sha256 and byte size pinned there. The repo copy (``source``) is
  used when it exists. ``--bench-root`` adds a second lookup in a bench tree (for example a
  ``~/rakshak-bench`` copy) for files whose repo source is absent, such as the Llama ``.pte``;
* the Android assets that are not in git (``sms_student.onnx``, pinned by the registry entry
  ``sms_student``) and the bundled JSON assets under ``android/app/src/main/assets/rakshak/``;
* ``--artifact PATH[@DEVICE:MODALITY]`` for anything else, such as a built APK.

A file that is expected but not present is recorded with status ``missing`` and makes the run fail
(exit 1), unless ``--allow-missing`` is given. A file whose hash differs from its pin is recorded as
``pin_mismatch`` and also fails. Nothing is skipped silently.

Usage
-----
    python scripts/release_manifest.py --out-dir release
    python scripts/release_manifest.py --artifact \
        android/app/build/outputs/apk/debug/app-debug.apk@android:app
    python scripts/release_manifest.py --bench-root ~/rakshak-bench --out-dir release
    python scripts/release_manifest.py --verify release/release_manifest.json

Exit codes: 0 all artifacts present and matching; 1 a missing, mismatched or changed file; 2 a
usage or manifest-format error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "rakshak-release-manifest/1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO / "deploy" / "pi" / "bench" / "registry.toml"
ANDROID_ASSETS = REPO / "android" / "app" / "src" / "main" / "assets"
ANDROID_SMS_ASSET = ANDROID_ASSETS / "sms_student.onnx"
ANDROID_JSON_DIR = ANDROID_ASSETS / "rakshak"
# Registry model whose pin the Android sms_student.onnx asset must match.
ANDROID_SMS_REGISTRY_ID = "sms_student"
# Registry `kind` -> modality label. Unknown kinds are recorded under their own name.
MODALITY_BY_KIND = {
    "onnx_sms": "sms",
    "onnx_url": "url",
    "onnx_nids": "network",
    "pte_llm": "explainer",
}
CHUNK = 1 << 20


def sha256_file(path: Path) -> tuple[str, int]:
    """Return (hex sha256, size in bytes) of a file, read in chunks."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def git_info(repo: Path) -> dict[str, Any]:
    """Commit, branch and dirty state of ``repo``. Fields are None when git is unavailable."""

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None, "dirty_paths": None}
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain") or ""
    dirty_paths = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit.strip(),
        "branch": branch.strip() if branch else None,
        "dirty": bool(dirty_paths),
        "dirty_paths": len(dirty_paths),
    }


def display_path(path: Path, repo: Path) -> str:
    """Repo-relative POSIX path when the file is under ``repo``, else the absolute path."""
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def artifact_record(
    *,
    name: str,
    path: Path,
    repo: Path,
    device: str,
    modality: str,
    pinned_sha256: str | None = None,
    pinned_bytes: int | None = None,
    role: str = "primary",
) -> dict[str, Any]:
    """Hash one file and compare it with its pin, if it has one."""
    rec: dict[str, Any] = {
        "name": name,
        "role": role,
        "device": device,
        "modality": modality,
        "path": display_path(path, repo),
        "pinned_sha256": pinned_sha256 or None,
        "pinned_bytes": pinned_bytes,
    }
    if not path.is_file():
        rec.update(status="missing", sha256=None, bytes=None)
        return rec
    digest, size = sha256_file(path)
    rec.update(sha256=digest, bytes=size)
    if pinned_sha256 and digest != pinned_sha256:
        rec["status"] = "pin_mismatch"
    elif pinned_bytes is not None and size != pinned_bytes:
        rec["status"] = "pin_mismatch"
    else:
        rec["status"] = "ok"
    return rec


def registry_artifacts(registry: Path, repo: Path, bench_root: Path | None) -> list[dict[str, Any]]:
    """Every enabled registry model and its side files, with pins from the registry."""
    if not registry.is_file():
        return [
            {
                "name": "registry",
                "role": "registry",
                "device": "pi",
                "modality": "config",
                "path": display_path(registry, repo),
                "status": "missing",
                "sha256": None,
                "bytes": None,
                "pinned_sha256": None,
                "pinned_bytes": None,
            }
        ]
    with registry.open("rb") as f:
        data = tomllib.load(f)

    def locate(source: str | None, path: str | None) -> Path:
        # Prefer the repo copy; fall back to the bench tree for the file's Pi path.
        if source:
            candidate = repo / source
            if candidate.is_file() or bench_root is None:
                return candidate
        if bench_root is not None and path:
            return bench_root / path
        return repo / (source or path or "")

    out: list[dict[str, Any]] = []
    for model in data.get("models", []):
        if not model.get("enabled", False):
            continue  # disabled entries are not shipped; their notes say why
        mid = model["id"]
        modality = MODALITY_BY_KIND.get(model.get("kind", ""), model.get("kind", "unknown"))
        out.append(
            artifact_record(
                name=mid,
                path=locate(model.get("source"), model.get("path")),
                repo=repo,
                device="pi",
                modality=modality,
                pinned_sha256=model.get("sha256"),
                pinned_bytes=model.get("bytes"),
            )
        )
        if model.get("side_scaler"):
            out.append(
                artifact_record(
                    name=f"{mid}:side_scaler",
                    path=locate(model.get("side_scaler_source"), model["side_scaler"]),
                    repo=repo,
                    device="pi",
                    modality=modality,
                    pinned_sha256=model.get("side_scaler_sha256"),
                    role="side_scaler",
                )
            )
        for extra in model.get("extras", []):
            out.append(
                artifact_record(
                    name=f"{mid}:{extra.get('role', 'extra')}",
                    path=locate(extra.get("source"), extra.get("path")),
                    repo=repo,
                    device="pi",
                    modality=modality,
                    pinned_sha256=extra.get("sha256"),
                    pinned_bytes=extra.get("bytes"),
                    role=extra.get("role", "extra"),
                )
            )
    return out


def android_artifacts(registry: Path, repo: Path) -> list[dict[str, Any]]:
    """Android files that are not in git: the SMS student asset and the bundled JSON data."""
    pins: dict[str, Any] = {}
    if registry.is_file():
        with registry.open("rb") as f:
            for model in tomllib.load(f).get("models", []):
                pins[model["id"]] = model
    sms_pin = pins.get(ANDROID_SMS_REGISTRY_ID, {})
    # Resolve the asset paths under the repo that was asked for, not the checkout this script is in.
    assets = repo / ANDROID_ASSETS.relative_to(REPO)
    sms_asset = assets / ANDROID_SMS_ASSET.name
    json_dir = assets / ANDROID_JSON_DIR.name
    out = [
        artifact_record(
            name="android_sms_student",
            path=sms_asset,
            repo=repo,
            device="android",
            modality="sms",
            pinned_sha256=sms_pin.get("sha256"),
            pinned_bytes=sms_pin.get("bytes"),
        )
    ]
    json_files = sorted(json_dir.glob("*.json")) if json_dir.is_dir() else []
    if not json_files:
        out.append(
            {
                "name": "android_rakshak_assets",
                "role": "data",
                "device": "android",
                "modality": "data",
                "path": display_path(json_dir, repo),
                "status": "missing",
                "sha256": None,
                "bytes": None,
                "pinned_sha256": None,
                "pinned_bytes": None,
            }
        )
    for jf in json_files:
        out.append(
            artifact_record(
                name=f"android_{jf.stem}",
                path=jf,
                repo=repo,
                device="android",
                modality="data",
                role="data",
            )
        )
    return out


def parse_artifact_spec(spec: str, repo: Path) -> dict[str, Any]:
    """``PATH`` or ``PATH@DEVICE:MODALITY``. The path is relative to the current directory."""
    path_text, sep, serves = spec.rpartition("@")
    if not sep or ":" not in serves:
        path_text, device, modality = spec, "unspecified", "unspecified"
    else:
        device, _, modality = serves.partition(":")
        if not device or not modality:
            raise ValueError(f"bad --artifact {spec!r}: expected PATH@DEVICE:MODALITY")
    if not path_text:
        raise ValueError(f"bad --artifact {spec!r}: empty path")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return artifact_record(
        name=path.name,
        path=path,
        repo=repo,
        device=device,
        modality=modality,
        role="given",
    )


def build_manifest(artifacts: list[dict[str, Any]], repo: Path, registry: Path | None) -> dict:
    counts: dict[str, int] = {}
    for a in artifacts:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_info(repo),
        "registry": display_path(registry, repo) if registry and registry.exists() else None,
        "artifacts": artifacts,
        "summary": {"total": len(artifacts), **dict(sorted(counts.items()))},
    }


def write_outputs(manifest: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "release_manifest.json"
    sums_path = out_dir / "SHA256SUMS"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"{a['sha256']}  {a['path']}" for a in manifest["artifacts"] if a.get("sha256") is not None
    ]
    sums_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return manifest_path, sums_path


def verify(manifest_path: Path, repo: Path) -> int:
    """Re-hash every artifact in a manifest.

    Returns 0 when all match, 1 on any failure and 2 on bad input.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"verify: cannot read {manifest_path}: {exc}", file=sys.stderr)
        return 2
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("artifacts"), list):
        print(f"verify: {manifest_path} is not a {SCHEMA} manifest", file=sys.stderr)
        return 2

    ok = 0
    failures = 0
    for a in manifest["artifacts"]:
        raw = a.get("path")
        if not raw:
            print(f"FAIL  (no path) {a.get('name')}")
            failures += 1
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = repo / raw
        if not path.is_file():
            print(f"MISSING  {raw}")
            failures += 1
            continue
        digest, size = sha256_file(path)
        if digest != a.get("sha256") or size != a.get("bytes"):
            print(
                f"FAIL  {raw}: recorded {a.get('sha256')} ({a.get('bytes')} B), "
                f"now {digest} ({size} B)"
            )
            failures += 1
        else:
            print(f"OK  {raw}")
            ok += 1

    sums = manifest_path.parent / "SHA256SUMS"
    if sums.is_file():
        recorded = {a["path"]: a.get("sha256") for a in manifest["artifacts"] if a.get("sha256")}
        seen_paths: set[str] = set()
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, sep, raw = line.partition("  ")
            if not sep or recorded.get(raw) != digest:
                print(f"FAIL  SHA256SUMS line does not match the manifest: {raw or line}")
                failures += 1
            else:
                seen_paths.add(raw)
        for missing_path in sorted(set(recorded) - seen_paths):
            print(f"FAIL  SHA256SUMS is missing manifest entry: {missing_path}")
            failures += 1
    total = len(manifest["artifacts"])
    print(f"verify: {ok} of {total} ok, {failures} failing")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", metavar="MANIFEST", type=Path, help="re-hash a manifest and exit")
    p.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="PATH[@DEVICE:MODALITY]",
        help="extra file to include, e.g. app-debug.apk@android:app (repeatable)",
    )
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="model registry TOML")
    p.add_argument("--bench-root", type=Path, default=None, help="Pi bench tree for registry files")
    p.add_argument("--no-discover", action="store_true", help="only the --artifact files")
    p.add_argument(
        "--repo", type=Path, default=REPO, help="repository root (git state, relative paths)"
    )
    p.add_argument("--out-dir", type=Path, default=REPO / "release", help="where to write outputs")
    p.add_argument("--allow-missing", action="store_true", help="exit 0 even if files are missing")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()

    if args.verify is not None:
        return verify(args.verify, repo)

    artifacts: list[dict[str, Any]] = []
    try:
        for spec in args.artifact:
            artifacts.append(parse_artifact_spec(spec, repo))
    except ValueError as exc:
        print(f"release_manifest: {exc}", file=sys.stderr)
        return 2
    if not args.no_discover:
        artifacts += registry_artifacts(args.registry, repo, args.bench_root)
        artifacts += android_artifacts(args.registry, repo)
    if not artifacts:
        print("release_manifest: nothing to record (no --artifact, discovery off)", file=sys.stderr)
        return 2

    manifest = build_manifest(artifacts, repo, args.registry)
    manifest_path, sums_path = write_outputs(manifest, args.out_dir.resolve())
    bad = [a for a in artifacts if a["status"] != "ok"]
    for a in bad:
        label = f"{a['name']}, {a['device']}/{a['modality']}"
        print(f"{a['status'].upper()}  {a['path']}  ({label})", file=sys.stderr)
    print(f"wrote {manifest_path}")
    print(f"wrote {sums_path}")
    print(f"summary: {manifest['summary']}")
    non_missing_bad = [a for a in bad if a["status"] != "missing"]
    if non_missing_bad or (bad and not args.allow_missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
