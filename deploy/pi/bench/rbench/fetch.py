"""`rbench fetch <model>`: download a pinned prebuilt .pte from Hugging Face into models/.

It is a no-op when every file is already present and matches its sha256 pin. It never
overwrites a file whose sha256 differs; it reports the mismatch and stops.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rbench.errors import ArtifactError, MissingDependency
from rbench.registry import Model, check_file


class FetchRefused(ValueError):
    """The model cannot be fetched (not a downloadable model, disabled, or unpinned)."""


def fetch_model(root: Path, model: Model, *, download: bool = True) -> dict[str, Any]:
    if model.hf is None:
        raise FetchRefused(f"{model.id} has no hf entry; it is not downloaded by rbench")
    if not model.enabled:
        raise FetchRefused(f"{model.id} is disabled: {model.notes}")
    if not model.sha256:
        raise FetchRefused(f"{model.id} has no pinned sha256; pin it in registry.toml first")
    states = []
    for f in model.files():
        states.append(check_file(root, f.path, f.sha256, hash_it=True) | {"role": f.role})
    if all(s["present"] and s["ok"] is not False for s in states) and all(
        s["ok"] for s in states if s["expected_sha256"]
    ):
        return {"id": model.id, "status": "present", "downloaded": [], "files": states}
    if any(s["ok"] is False for s in states):
        bad = [s["path"] for s in states if s["ok"] is False]
        raise ArtifactError(
            f"sha256 mismatch (a partial or corrupt file), not overwriting: {', '.join(bad)}; "
            "move it aside by hand and rerun"
        )
    if not download:
        return {"id": model.id, "status": "not downloaded", "downloaded": [], "files": states}
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MissingDependency(
            "huggingface_hub is not installed (pip install huggingface_hub)"
        ) from exc
    targets = {Path(f.path).name: f.path for f in model.files()}
    downloaded: list[str] = []
    for name in model.hf["files"]:
        dest_rel = targets.get(Path(name).name)
        if dest_rel is None:
            continue
        dest = root / dest_rel
        if dest.is_file() and any(s["path"] == dest_rel and s["ok"] for s in states):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(
            repo_id=model.hf["repo"],
            filename=name,
            revision=model.hf["revision"],
            local_dir=str(dest.parent),
        )
        if Path(cached) != dest:
            shutil.move(cached, dest)
        downloaded.append(dest_rel)
    states = [
        check_file(root, f.path, f.sha256, hash_it=True) | {"role": f.role} for f in model.files()
    ]
    if any(s["ok"] is False for s in states):
        raise ArtifactError(
            "downloaded file fails its sha256 pin; delete it and retry after checking the revision"
        )
    status = "downloaded" if downloaded else "present"
    return {"id": model.id, "status": status, "downloaded": downloaded, "files": states}
