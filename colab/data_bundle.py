"""Select, pack and describe the data bundle that colab/launch.sh uploads to a Colab VM.

Runs on the laptop, standard library only. The VM side is colab/remote_unpack_data.py, which reads
the manifest written here and checks every part, the whole archive and every extracted file.

  data_bundle.py select DATA_DIR PATTERN...
      Print the selected files (relative POSIX paths, sorted) and a total line.
  data_bundle.py build DATA_DIR OUT_DIR CHUNK_BYTES PATTERN...
      Write OUT_DIR/rakshak-data.tar.gz, OUT_DIR/rakshak-data.part-NNN (each at most CHUNK_BYTES
      bytes, in order) and OUT_DIR/manifest.json. Earlier parts in OUT_DIR are removed first.

PATTERN is a glob relative to DATA_DIR: '**' crosses folders. A pattern starting with '!' removes
what it matches from the selection. Every include pattern must match at least one file, so a typo
fails here, before any Colab session is requested. Quote patterns in the shell (for example
'splits/*.json') so the shell does not expand them first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

FORMAT = 1
ARCHIVE_NAME = "rakshak-data.tar.gz"
PART_PREFIX = "rakshak-data.part-"
MANIFEST_NAME = "manifest.json"
MIB = 1024 * 1024
MIN_CHUNK_BYTES = MIB


class BundleError(Exception):
    """A pattern, folder or chunk size that cannot make a bundle."""


def _matches(data_dir: Path, pattern: str) -> set[str]:
    if not pattern or pattern.startswith("/") or ".." in Path(pattern).parts:
        raise BundleError(f"pattern {pattern!r} must be relative and stay inside the data folder")
    return {p.relative_to(data_dir).as_posix() for p in data_dir.glob(pattern) if p.is_file()}


def select(data_dir: str | Path, patterns: list[str]) -> list[str]:
    """Return the sorted relative paths of the files the patterns select."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise BundleError(f"{data_dir} is not a folder")
    includes = [p for p in patterns if not p.startswith("!")]
    excludes = [p[1:] for p in patterns if p.startswith("!")]
    if not includes:
        raise BundleError("give at least one include PATTERN (one that does not start with '!')")
    chosen: set[str] = set()
    for pattern in includes:
        hits = _matches(data_dir, pattern)
        if not hits:
            raise BundleError(f"pattern {pattern!r} matched no file under {data_dir}")
        chosen |= hits
    for pattern in excludes:
        chosen -= _matches(data_dir, pattern)
    if not chosen:
        raise BundleError("the patterns select no file")
    return sorted(chosen)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def build(data_dir: str | Path, out_dir: str | Path, chunk_bytes: int,
          patterns: list[str]) -> dict:
    """Pack the selected files, split the archive into parts and write the manifest."""
    if chunk_bytes < MIN_CHUNK_BYTES:
        raise BundleError(f"chunk size must be at least {MIN_CHUNK_BYTES} bytes")
    data_dir = Path(data_dir)
    files = select(data_dir, patterns)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob(PART_PREFIX + "*"), out / ARCHIVE_NAME, out / MANIFEST_NAME]:
        stale.unlink(missing_ok=True)

    archive = out / ARCHIVE_NAME
    with tarfile.open(archive, mode="w:gz") as tar:
        for rel in files:
            tar.add(data_dir / rel, arcname=rel, recursive=False)
    archive_bytes = archive.stat().st_size
    archive_sha = _sha256_file(archive)

    parts = []
    with open(archive, "rb") as source:
        index = 0
        while True:
            block = source.read(chunk_bytes)
            if not block:
                break
            name = f"{PART_PREFIX}{index:03d}"
            (out / name).write_bytes(block)
            parts.append({"name": name, "bytes": len(block),
                          "sha256": hashlib.sha256(block).hexdigest()})
            index += 1

    manifest = {
        "format": FORMAT,
        "archive": {"name": ARCHIVE_NAME, "bytes": archive_bytes, "sha256": archive_sha},
        "chunk_bytes": chunk_bytes,
        "parts": parts,
        "files": [{"path": rel, "bytes": (data_dir / rel).stat().st_size} for rel in files],
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _mib(count: int) -> str:
    return f"{count / MIB:.1f} MiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_select = sub.add_parser("select")
    p_select.add_argument("data_dir")
    p_select.add_argument("patterns", nargs="+")
    p_build = sub.add_parser("build")
    p_build.add_argument("data_dir")
    p_build.add_argument("out_dir")
    p_build.add_argument("chunk_bytes", type=int)
    p_build.add_argument("patterns", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "select":
            data_dir = Path(args.data_dir)
            files = select(data_dir, args.patterns)
            total = 0
            for rel in files:
                size = (data_dir / rel).stat().st_size
                total += size
                print(rel)
            print(f"total: {len(files)} files, {_mib(total)} raw")
            return 0
        manifest = build(args.data_dir, args.out_dir, args.chunk_bytes, args.patterns)
    except BundleError as exc:
        print(f"data_bundle: {exc}", file=sys.stderr)
        return 2
    raw = sum(entry["bytes"] for entry in manifest["files"])
    archive = manifest["archive"]
    print(f"bundle: {len(manifest['files'])} files, {_mib(raw)} raw, archive "
          f"{_mib(archive['bytes'])} ({archive['sha256'][:12]}), {len(manifest['parts'])} parts "
          f"of at most {_mib(manifest['chunk_bytes'])}")
    for entry in manifest["files"]:
        print(f"  {entry['path']} {entry['bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
