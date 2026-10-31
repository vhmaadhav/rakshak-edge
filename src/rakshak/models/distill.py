"""Teacher transfer logits for distillation (torch-free: matching, loading and pool building).

``sms_teacher --transfer-texts`` writes one ``transfer_<teacher>_s<seed>_<run>.parquet`` per run
(split file x partition), where ``<run>`` is the run name with ``:`` replaced by ``_``. Each file
holds the teacher logits of every row outside that run's test partition, with columns ``id``,
``split`` (``train`` | ``cal`` | ``unlabeled``), ``logit_0`` and ``logit_1``.

The student never reuses one file across runs. ``find_transfer_file`` picks the file whose run name
matches, ``build_pool`` checks that the file's train and cal membership equals the run's own
partitions and that no test id of the run is present, and the pool holds the labelled train rows
(with teacher logits) plus the ``unlabeled`` rows (teacher logits only, no label). A
``manifest.json`` in the dataset folder, when present, pins each file's sha256.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rakshak.eval import data as D

MANIFEST = "manifest.json"
TRANSFER_GLOB = "transfer_*.parquet"
TRANSFER_RE = re.compile(r"^transfer_(?P<teacher>.+?)_s(?P<seed>\d+)_(?P<run>.+)\.parquet$")
COLUMNS = ("id", "split", "logit_0", "logit_1")
SPLIT_TRAIN, SPLIT_CAL, SPLIT_UNLABELED = "train", "cal", "unlabeled"
SPLITS = (SPLIT_TRAIN, SPLIT_CAL, SPLIT_UNLABELED)


def safe_run_name(run_name: str) -> str:
    """The run name as the teacher wrote it into a file name (``source_holdout:x`` ->
    ``source_holdout_x``)."""
    return run_name.replace(":", "_")


def resolve_teacher_dir(explicit: str | Path) -> Path:
    """The folder holding the transfer files: ``explicit`` itself or one child folder down (the
    attached dataset may sit one level below its mount point). Like the data roots, only the given
    path is tried."""
    folder = Path(explicit)
    if folder.is_dir():
        candidates = [folder, *sorted(p for p in folder.iterdir() if p.is_dir())]
        for candidate in candidates:
            if any(candidate.glob(TRANSFER_GLOB)):
                return candidate
    raise FileNotFoundError(f"no {TRANSFER_GLOB} under {folder} or one folder down")


def find_transfer_file(folder: Path, run_name: str, teacher_seed: int = 0) -> Path:
    """The single transfer file for ``run_name`` and ``teacher_seed``. Raises when there is none,
    or more than one (for example two teacher models in one folder)."""
    want = safe_run_name(run_name)
    hits = []
    for path in sorted(folder.glob(TRANSFER_GLOB)):
        match = TRANSFER_RE.match(path.name)
        if match and match["run"] == want and int(match["seed"]) == teacher_seed:
            hits.append(path)
    if len(hits) != 1:
        names = [h.name for h in hits]
        raise FileNotFoundError(f"{len(hits)} teacher transfer files for run {run_name!r} "
                                f"(seed {teacher_seed}) under {folder}: {names}")
    return hits[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(folder: Path, run_name: str, path: Path) -> dict | None:
    """Check ``path`` against the folder's manifest when one exists. Returns the manifest entry
    (or None without a manifest). Raises on a missing entry, a wrong file name or a sha256
    mismatch."""
    manifest_path = folder / MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get("splits", {}).get(run_name)
    if entry is None:
        raise ValueError(f"{MANIFEST} has no entry for run {run_name!r}")
    if entry["file"] != path.name:
        raise ValueError(f"{MANIFEST} maps run {run_name!r} to {entry['file']}, not {path.name}")
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ValueError(f"sha256 of {path.name} is {digest}, manifest says {entry['sha256']}")
    return entry


def load_transfer(path: Path) -> pd.DataFrame:
    """Read a transfer file and check its schema: the four columns, known split values, unique
    ids and finite logits on every row."""
    df = pd.read_parquet(path, columns=list(COLUMNS))
    df["id"] = df["id"].astype(str)
    if not df["id"].is_unique:
        raise ValueError(f"{path.name} has duplicate ids")
    bad = sorted(set(df["split"]) - set(SPLITS))
    if bad:
        raise ValueError(f"{path.name} has unknown split values {bad}")
    logits = df[["logit_0", "logit_1"]].to_numpy(dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError(f"{path.name} has non-finite teacher logits")
    return df


@dataclass(frozen=True)
class Pool:
    """Training rows for one run. ``positions`` are rows of sms.parquet: the labelled train rows
    first, then the unlabeled rows (when included). ``labels`` is NaN on unlabeled rows.
    ``teacher`` is (n, 2) teacher logits [logit_0, logit_1], NaN where the teacher has no row."""

    positions: np.ndarray
    labels: np.ndarray
    teacher: np.ndarray
    n_labelled: int
    n_unlabeled: int

    def stats(self) -> dict:
        has_teacher = ~np.isnan(self.teacher).any(axis=1)
        return {
            "rows": int(len(self.positions)),
            "labelled": self.n_labelled,
            "unlabeled": self.n_unlabeled,
            "with_teacher": int(has_teacher.sum()),
            "labelled_with_teacher": int(has_teacher[: self.n_labelled].sum()),
            "unlabeled_with_teacher": int(has_teacher[self.n_labelled:].sum()),
        }


def _check_membership(name: str, split: str, file_ids: set[str], run_ids: np.ndarray) -> None:
    run_set = set(map(str, run_ids))
    if file_ids != run_set:
        only_file = len(file_ids - run_set)
        only_run = len(run_set - file_ids)
        raise ValueError(f"{name}: transfer file's {split} rows differ from the run's {split} "
                         f"partition ({only_file} only in file, {only_run} only in run)")


def build_pool(run: D.Run, sms_ids: np.ndarray, labels: np.ndarray, transfer: pd.DataFrame,
               *, include_unlabeled: bool) -> Pool:
    """The student's training pool for one run from its transfer file.

    Checks, in order: no test id of the run (``test_all``) appears in the file; the file's train
    and cal membership equals the run's train and cal partitions; every unlabeled row resolves to
    an sms row and is in neither train nor cal. ``labels`` is the 0/1 label per sms row.
    """
    if not transfer["id"].is_unique:
        raise ValueError(f"{run.name}: transfer file has duplicate ids")
    D.assert_no_test_rows(transfer["id"], sms_ids[run.test_all])
    by_split = {s: transfer[transfer["split"] == s] for s in SPLITS}
    _check_membership(run.name, SPLIT_TRAIN, set(by_split[SPLIT_TRAIN]["id"]),
                      sms_ids[run.train])
    _check_membership(run.name, SPLIT_CAL, set(by_split[SPLIT_CAL]["id"]), sms_ids[run.cal])

    logit_cols = ["logit_0", "logit_1"]
    indexed = transfer.set_index("id")
    train_logits = indexed.reindex(sms_ids[run.train])[logit_cols].to_numpy(dtype=np.float64)
    train_labels = labels[run.train].astype(np.float64)
    positions = [np.asarray(run.train, dtype=np.int64)]
    labels_out = [train_labels]
    teacher_out = [train_logits]
    n_labelled = len(run.train)
    n_unlabeled = 0
    if include_unlabeled:
        unl = by_split[SPLIT_UNLABELED]
        index = pd.Index(sms_ids)
        unl_pos = index.get_indexer(unl["id"].tolist())
        if (unl_pos < 0).any():
            raise ValueError(f"{run.name}: {int((unl_pos < 0).sum())} unlabeled ids are not in "
                             "sms.parquet")
        overlap = np.intersect1d(unl_pos, np.concatenate([run.train, run.cal]))
        if overlap.size:
            raise ValueError(f"{run.name}: {overlap.size} unlabeled rows are also train or cal")
        positions.append(unl_pos.astype(np.int64))
        labels_out.append(np.full(len(unl_pos), np.nan))
        teacher_out.append(unl[logit_cols].to_numpy(dtype=np.float64))
        n_unlabeled = len(unl_pos)
    return Pool(
        positions=np.concatenate(positions),
        labels=np.concatenate(labels_out),
        teacher=np.concatenate(teacher_out),
        n_labelled=n_labelled,
        n_unlabeled=n_unlabeled,
    )
