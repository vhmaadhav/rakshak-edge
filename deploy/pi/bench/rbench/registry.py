"""Registry of benchmark models and datasets (registry.toml), with validation and checks.

Stdlib only. Paths in the registry are relative to the bench root; the root is the directory that
holds src/rakshak and deploy/pi on the Pi (~/rakshak-bench) or the repository root on the laptop.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rbench import REGISTRY_SCHEMA

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PACKAGE_DIR / "registry.toml"

# model kind -> task of the models of that kind
KIND_TASK: dict[str, str] = {
    "onnx_sms": "sms_binary",
    "onnx_url": "url_binary",
    "onnx_nids": "nids_latency",
    "pte_llm": "ela_json",
}
DATASET_TASKS: tuple[str, ...] = (
    "sms_binary",
    "url_binary",
    "groundlm_fpr",
    "nids_latency",
    "ela_prompts",
)
# model task -> dataset tasks it can run on
COMPATIBLE: dict[str, tuple[str, ...]] = {
    "sms_binary": ("sms_binary", "groundlm_fpr"),
    "url_binary": ("url_binary",),
    "nids_latency": ("nids_latency",),
    "ela_json": ("ela_prompts",),
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


class RegistryError(ValueError):
    """The registry file is malformed."""


class IncompatibleError(ValueError):
    """The model and the dataset cannot be used together."""


@dataclass(frozen=True)
class Extra:
    role: str
    path: str
    sha256: str
    source: str = ""
    bytes: int | None = None


@dataclass(frozen=True)
class Model:
    id: str
    kind: str
    task: str
    enabled: bool
    path: str
    sha256: str
    notes: str
    source: str = ""
    threshold: float | None = None
    extras: tuple[Extra, ...] = ()
    hf: dict[str, Any] | None = None
    bytes: int | None = None
    side_scaler: str = ""
    side_scaler_sha256: str = ""
    side_scaler_source: str = ""
    parity_logits: str = ""
    parity_logits_sha256: str = ""
    parity_column: str = ""
    parity_logits_source: str = ""

    @property
    def compatible_tasks(self) -> tuple[str, ...]:
        return COMPATIBLE[self.task]

    def files(self) -> list[Extra]:
        """Every file of the model: the primary artifact, then the extras (with sidecars)."""
        out = [Extra("model", self.path, self.sha256, self.source, self.bytes)]
        if self.side_scaler:
            out.append(
                Extra(
                    "side_scaler",
                    self.side_scaler,
                    self.side_scaler_sha256,
                    self.side_scaler_source,
                )
            )
        if self.parity_logits:
            out.append(
                Extra(
                    "parity_logits",
                    self.parity_logits,
                    self.parity_logits_sha256,
                    self.parity_logits_source,
                )
            )
        out.extend(self.extras)
        return out


@dataclass(frozen=True)
class Dataset:
    id: str
    task: str
    enabled: bool
    path: str
    sha256: str
    notes: str
    source: str = ""
    synthetic: bool = False
    split_file: str = ""
    split_sha256: str = ""
    split_part: str = ""
    split_column: str = ""
    split_value: str = ""
    split_source: str = ""
    row_filter: str = "none"
    columns: tuple[str, ...] = ()


@dataclass
class Registry:
    models: dict[str, Model] = field(default_factory=dict)
    datasets: dict[str, Dataset] = field(default_factory=dict)
    source: str = ""

    def model(self, model_id: str) -> Model:
        if model_id not in self.models:
            raise KeyError(f"unknown model {model_id!r}; see `rbench list models`")
        return self.models[model_id]

    def dataset(self, dataset_id: str) -> Dataset:
        if dataset_id not in self.datasets:
            raise KeyError(f"unknown dataset {dataset_id!r}; see `rbench list datasets`")
        return self.datasets[dataset_id]


# -- validation -------------------------------------------------------------


def _check_rel(path: str, where: str, allow_empty: bool = False) -> str:
    if path == "" and allow_empty:
        return path
    if not path or path.startswith("/") or ".." in Path(path).parts:
        raise RegistryError(f"{where}: path must be relative with no '..': {path!r}")
    return path


def _check_sha(value: str, where: str) -> str:
    if value and not SHA_RE.match(value):
        raise RegistryError(f"{where}: sha256 must be 64 lowercase hex chars or blank")
    return value


def _str(entry: dict[str, Any], key: str, where: str, default: str | None = None) -> str:
    if key not in entry:
        if default is None:
            raise RegistryError(f"{where}: missing '{key}'")
        return default
    value = entry[key]
    if not isinstance(value, str):
        raise RegistryError(f"{where}: '{key}' must be a string")
    return value


def _parse_bytes(entry: dict[str, Any], where: str) -> int | None:
    """Optional expected size in bytes. A present file of another size is "incomplete"."""
    if "bytes" not in entry:
        return None
    value = entry["bytes"]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RegistryError(f"{where}: bytes must be a positive integer")
    return value


def _parse_extras(entry: dict[str, Any], where: str) -> tuple[Extra, ...]:
    out = []
    for raw in entry.get("extras", []):
        role = _str(raw, "role", where)
        path = _check_rel(_str(raw, "path", where), f"{where} extra {role}")
        sha = _check_sha(_str(raw, "sha256", where, ""), f"{where} extra {role}")
        source = _check_rel(
            _str(raw, "source", where, ""), f"{where} extra {role}", allow_empty=True
        )
        out.append(Extra(role, path, sha, source, _parse_bytes(raw, f"{where} extra {role}")))
    return tuple(out)


def _parse_model(entry: dict[str, Any], index: int) -> Model:
    where = f"models[{index}]"
    model_id = _str(entry, "id", where)
    where = f"model {model_id}"
    if not _ID_RE.match(model_id):
        raise RegistryError(f"{where}: bad id")
    kind = _str(entry, "kind", where)
    if kind not in KIND_TASK:
        raise RegistryError(f"{where}: unknown kind {kind!r}")
    task = _str(entry, "task", where, KIND_TASK[kind])
    if task != KIND_TASK[kind]:
        raise RegistryError(f"{where}: kind {kind} must use task {KIND_TASK[kind]}")
    enabled = bool(entry.get("enabled", True))
    notes = _str(entry, "notes", where, "")
    if not enabled and not notes.strip():
        raise RegistryError(f"{where}: disabled entries must say why in notes")
    path = _check_rel(_str(entry, "path", where), where)
    sha = _check_sha(_str(entry, "sha256", where, ""), where)
    threshold = entry.get("threshold")
    if threshold is not None:
        if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
            raise RegistryError(f"{where}: threshold must be in (0, 1)")
        threshold = float(threshold)
    hf = entry.get("hf")
    if hf is not None:
        if not isinstance(hf, dict) or not {"repo", "revision", "files"} <= set(hf):
            raise RegistryError(f"{where}: hf needs repo, revision and files")
    side_scaler = _check_rel(_str(entry, "side_scaler", where, ""), where, allow_empty=True)
    parity = _check_rel(_str(entry, "parity_logits", where, ""), where, allow_empty=True)
    return Model(
        id=model_id,
        kind=kind,
        task=task,
        enabled=enabled,
        path=path,
        sha256=sha,
        notes=notes,
        source=_check_rel(_str(entry, "source", where, ""), where, allow_empty=True),
        threshold=threshold,
        extras=_parse_extras(entry, where),
        hf=hf,
        bytes=_parse_bytes(entry, where),
        side_scaler=side_scaler,
        side_scaler_sha256=_check_sha(_str(entry, "side_scaler_sha256", where, ""), where),
        side_scaler_source=_check_rel(
            _str(entry, "side_scaler_source", where, ""), where, allow_empty=True
        ),
        parity_logits=parity,
        parity_logits_sha256=_check_sha(_str(entry, "parity_logits_sha256", where, ""), where),
        parity_column=_str(entry, "parity_column", where, ""),
        parity_logits_source=_check_rel(
            _str(entry, "parity_logits_source", where, ""), where, allow_empty=True
        ),
    )


def _parse_dataset(entry: dict[str, Any], index: int) -> Dataset:
    where = f"datasets[{index}]"
    dataset_id = _str(entry, "id", where)
    where = f"dataset {dataset_id}"
    if not _ID_RE.match(dataset_id):
        raise RegistryError(f"{where}: bad id")
    task = _str(entry, "task", where)
    if task not in DATASET_TASKS:
        raise RegistryError(f"{where}: unknown task {task!r}")
    enabled = bool(entry.get("enabled", True))
    notes = _str(entry, "notes", where, "")
    if not enabled and not notes.strip():
        raise RegistryError(f"{where}: disabled entries must say why in notes")
    synthetic = bool(entry.get("synthetic", False))
    path = _str(entry, "path", where, "")
    if not synthetic and not path:
        raise RegistryError(f"{where}: non-synthetic dataset needs a path")
    path = _check_rel(path, where, allow_empty=synthetic)
    row_filter = _str(entry, "row_filter", where, "none")
    if row_filter not in ("none", "eval_mask", "policy_negative_scoreable"):
        raise RegistryError(f"{where}: unknown row_filter {row_filter!r}")
    split_file = _check_rel(_str(entry, "split_file", where, ""), where, allow_empty=True)
    split_part = _str(entry, "split_part", where, "")
    split_column = _str(entry, "split_column", where, "")
    if split_file and not split_part:
        raise RegistryError(f"{where}: split_file needs split_part")
    cols = entry.get("columns", [])
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise RegistryError(f"{where}: columns must be a list of strings")
    return Dataset(
        id=dataset_id,
        task=task,
        enabled=enabled,
        path=path,
        sha256=_check_sha(_str(entry, "sha256", where, ""), where),
        notes=notes,
        source=_check_rel(_str(entry, "source", where, ""), where, allow_empty=True),
        synthetic=synthetic,
        split_file=split_file,
        split_sha256=_check_sha(_str(entry, "split_sha256", where, ""), where),
        split_part=split_part,
        split_column=split_column,
        split_value=_str(entry, "split_value", where, ""),
        split_source=_check_rel(_str(entry, "split_source", where, ""), where, allow_empty=True),
        row_filter=row_filter,
        columns=tuple(cols),
    )


def parse_registry(doc: dict[str, Any], source: str = "") -> Registry:
    if doc.get("schema") != REGISTRY_SCHEMA:
        raise RegistryError(f"registry schema must be {REGISTRY_SCHEMA!r}")
    reg = Registry(source=source)
    for index, raw in enumerate(doc.get("models", [])):
        model = _parse_model(raw, index)
        if model.id in reg.models:
            raise RegistryError(f"duplicate model id {model.id!r}")
        reg.models[model.id] = model
    for index, raw in enumerate(doc.get("datasets", [])):
        dataset = _parse_dataset(raw, index)
        if dataset.id in reg.datasets:
            raise RegistryError(f"duplicate dataset id {dataset.id!r}")
        reg.datasets[dataset.id] = dataset
    return reg


def load_registry(path: Path | str = DEFAULT_REGISTRY) -> Registry:
    path = Path(path)
    try:
        with path.open("rb") as handle:
            doc = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise RegistryError(f"registry not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"registry {path} is not valid TOML: {exc}") from exc
    return parse_registry(doc, source=str(path))


# -- compatibility ----------------------------------------------------------


def check_compatible(model: Model, dataset: Dataset) -> None:
    """Raise IncompatibleError with a clear message unless the pair can run."""
    if dataset.task not in COMPATIBLE[model.task]:
        allowed = ", ".join(COMPATIBLE[model.task])
        raise IncompatibleError(
            f"model {model.id} (task {model.task}) cannot run on dataset {dataset.id} "
            f"(task {dataset.task}); it needs a dataset with task: {allowed}"
        )


def compatible_datasets(reg: Registry, model: Model) -> list[str]:
    return [d.id for d in reg.datasets.values() if d.task in COMPATIBLE[model.task]]


def sync_manifest(reg: Registry) -> list[dict[str, str]]:
    """Every enabled file that has a repo source, as sync_bench.sh needs it (deduplicated by path).

    Files with no ``source`` (the Llama files, which the Pi downloads itself) are left out.
    """
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(owner: str, role: str, source: str, path: str, sha: str) -> None:
        if not source or path in seen:
            return
        seen.add(path)
        rows.append({"owner": owner, "role": role, "source": source, "path": path, "sha256": sha})

    for m in reg.models.values():
        if not m.enabled:
            continue
        for f in m.files():
            add(f"model:{m.id}", f.role, f.source, f.path, f.sha256)
    for d in reg.datasets.values():
        if not d.enabled or d.synthetic:
            continue
        add(f"dataset:{d.id}", "data", d.source, d.path, d.sha256)
        if d.split_file:
            add(f"dataset:{d.id}", "split", d.split_source, d.split_file, d.split_sha256)
    return rows


# -- artifact checks --------------------------------------------------------


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(
    root: Path,
    rel: str,
    expected: str,
    hash_it: bool = True,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """State of one artifact: present, size, actual sha256 and whether it matches the pin.

    ``ok`` is None when the file is absent or no sha is pinned, or was not hashed (hash_it
    False), True when the pin matches, False when it does not. ``incomplete`` is True when the
    file is present but its size differs from ``expected_bytes`` (a partial download).
    """
    full = root / rel
    if not rel or not full.is_file():
        return {
            "path": rel,
            "present": False,
            "bytes": None,
            "expected_bytes": expected_bytes,
            "sha256": None,
            "expected_sha256": expected or None,
            "ok": None,
            "incomplete": False,
        }
    size = full.stat().st_size
    actual = file_sha256(full) if hash_it else None
    ok = None if (not expected or actual is None) else (actual == expected)
    return {
        "path": rel,
        "present": True,
        "bytes": size,
        "expected_bytes": expected_bytes,
        "sha256": actual,
        "expected_sha256": expected or None,
        "ok": ok,
        "incomplete": expected_bytes is not None and size != expected_bytes,
    }


def model_state(root: Path, model: Model, hash_it: bool = True) -> dict[str, Any]:
    """Status labels, most severe first:

    disabled; incomplete (present, wrong size: a partial download); sha mismatch; not
    downloaded / missing; present (not verified: pinned but not hashed); present (sha not
    pinned); ready (every pinned file hashed and matching).
    """
    files = [
        check_file(root, f.path, f.sha256, hash_it, f.bytes) | {"role": f.role}
        for f in model.files()
    ]
    present_all = all(f["present"] for f in files)
    bad = [f for f in files if f["ok"] is False]
    if not model.enabled:
        status = "disabled"
    elif any(f["incomplete"] for f in files):
        status = "incomplete"
    elif bad:
        status = "sha mismatch"
    elif not present_all:
        status = "not downloaded" if model.hf else "missing"
    elif any(f["expected_sha256"] and f["ok"] is None for f in files):
        status = "present (not verified)"
    elif any(f["ok"] is None for f in files):
        status = "present (sha not pinned)"
    else:
        status = "ready"
    return {"id": model.id, "status": status, "files": files}


def dataset_state(root: Path, dataset: Dataset, hash_it: bool = True) -> dict[str, Any]:
    if dataset.synthetic:
        return {
            "id": dataset.id,
            "status": "ready (synthetic)" if dataset.enabled else "disabled",
            "files": [],
        }
    files = [check_file(root, dataset.path, dataset.sha256, hash_it) | {"role": "data"}]
    if dataset.split_file:
        files.append(
            check_file(root, dataset.split_file, dataset.split_sha256, hash_it) | {"role": "split"}
        )
    if not dataset.enabled:
        status = "disabled"
    elif any(f["ok"] is False for f in files):
        status = "sha mismatch"
    elif not all(f["present"] for f in files):
        status = "missing"
    else:
        status = "ready"
    return {"id": dataset.id, "status": status, "files": files}
