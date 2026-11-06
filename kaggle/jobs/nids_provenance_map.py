"""Frozen-row host mapping audit, never a new training/evaluation cohort.

Pilot UNSW first. Missing or multiple host-pair candidates make mapping
UNMEASURABLE; no nearest-neighbour matching, label tie-break or split changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import nids_baselines as historical
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _common import write_metrics

from rakshak.nids import schema as S
from rakshak.nids import splits as SP
from rakshak.nids.provenance import (
    DOWNLOAD_HOST,
    RDM_IDS,
    MetadataAuditError,
)
from rakshak.nids.provenance_match import CandidatePairs, numeric_keys

STAGE1_SHA = "7480f3a36c65985dcb8274d27bf736e93d154ff9f8a7984104547ab38ba0e36a"
HISTORICAL_ROWS = {"unsw_nb15": 17227, "cse_cic_ids2018": 53294,
                   "bot_iot": 19225, "ton_iot": 45958}
SAMPLER_SHA = "79ac87d0dd8a28b498e6dc1fb5988cb75b036a4242f07edda96cd9043de18d1e"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replay_rows(path: Path, key: str):
    """Reproduce the unchanged seed0 sampler while retaining raw numeric fields.

    Labels are read ONLY to replay the original stratified sample. They never
    enter a join key or the original-CSV reader. Compare X/y with the old sampler.
    """
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    join_columns = [c for c in columns if c not in (*S.HOST_COLUMNS, *S.LABEL_COLUMNS)]
    if len(join_columns) != 41:
        raise MetadataAuditError("expected all 41 original non-label numeric fields")
    attack_counts = Counter()
    benign_count = 0
    for batch in parquet.iter_batches(batch_size=historical.BATCH_ROWS,
                                     columns=list(S.LABEL_COLUMNS)):
        frame = batch.to_pandas()
        benign = frame["Label"].to_numpy().astype(np.int64) == 0
        benign_count += int(benign.sum())
        attack_counts.update(frame["Attack"].fillna("").astype(str)[~benign].value_counts().to_dict())
    keep_att, keep_ben = SP.plan_quotas(dict(attack_counts), benign_count, 200000)
    rng = np.random.default_rng(0)
    frames, positions = [], []
    offset = 0
    for batch in parquet.iter_batches(batch_size=historical.BATCH_ROWS):
        frame = batch.to_pandas()
        labels = frame["Label"].to_numpy().astype(np.int64)
        attacks = frame["Attack"].fillna("").astype(str).to_numpy()
        probabilities = np.ones(len(frame))
        benign = labels == 0
        probabilities[benign] = keep_ben
        attack_rows = np.flatnonzero(~benign)
        probabilities[attack_rows] = [keep_att.get(a, 1.0) for a in attacks[attack_rows]]
        keep = rng.random(len(frame)) < probabilities
        frames.append(frame.loc[keep])
        positions.append(np.flatnonzero(keep) + offset)
        offset += len(frame)
    sampled = pd.concat(frames, ignore_index=True)
    x = S.coerce_matrix(sampled[list(S.SHARED_FEATURES)].to_numpy(dtype=np.float64))
    _, first = np.unique(x, axis=0, return_index=True)
    first = np.sort(first)
    retained = sampled.iloc[first].reset_index(drop=True)
    positions = np.concatenate(positions)[first].astype("<i8")
    legacy = historical.sample_dataset(path, key, 200000, 0)
    if not (len(retained) == HISTORICAL_ROWS[key]
            and np.array_equal(x[first], legacy.X)
            and np.array_equal(retained["Label"].to_numpy(dtype=np.int64), legacy.y)):
        raise MetadataAuditError("historical sampler replay parity failed")
    numeric = retained[join_columns]
    dtypes = {name: str(numeric[name].dtype) for name in join_columns}
    return numeric, positions, dtypes, legacy


def download_archive(key: str, pinned: dict, destination: Path) -> str:
    with urllib.request.urlopen(
        f"https://api.rdm.uq.edu.au/datasets/{RDM_IDS[key]}/download", timeout=40
    ) as response:
        meta = json.loads(response.read(1024 * 1024))
    # Inspect URL origin without ever logging a signed URL or exception string.
    url = meta["url"]
    parsed = urllib.parse.urlsplit(url)
    if (meta.get("canDownload") is not True or parsed.scheme != "https"
            or parsed.hostname != DOWNLOAD_HOST):
        raise MetadataAuditError("unexpected original archive download metadata")
    request = urllib.request.Request(url, headers={"If-Match": pinned["etag"],
                                                 "Accept-Encoding": "identity"})
    digest = hashlib.sha256()
    expected, copied = pinned["archive_bytes"], 0
    with urllib.request.urlopen(request, timeout=120) as source, destination.open("wb") as target:
        if (source.status != 200 or source.headers.get("ETag") != pinned["etag"]
                or int(source.headers.get("Content-Length", "-1")) != expected):
            raise MetadataAuditError("archive identity changed since stage1")
        while copied < expected:
            block = source.read(min(8 * 1024 * 1024, expected - copied))
            if not block:
                raise MetadataAuditError("truncated original archive")
            target.write(block)
            digest.update(block)
            copied += len(block)
    return digest.hexdigest()


def audit_dataset(key: str, stage1: dict, work: Path, log) -> dict:
    paths = historical.discover(Path("/kaggle/input"), [key])
    if key not in paths:
        raise MetadataAuditError("frozen parquet missing")
    parquet = paths[key]
    pinned = stage1["datasets"][key]
    if digest_file(parquet) != pinned["frozen_input"]["sha256"]:
        raise MetadataAuditError("frozen parquet hash mismatch")
    numeric, positions, dtypes, legacy = replay_rows(parquet, key)
    keys = numeric_keys(numeric, dtypes)
    candidates = CandidatePairs(keys)
    original = pinned["official_archive"]
    members = [m for m in original["members"] if m["kind"] == "flow_header"]
    if len(members) != 1:
        raise MetadataAuditError("pilot expects exactly one original flow CSV")
    scanned = invalid = 0
    with tempfile.TemporaryDirectory(prefix="rakshak-host-audit-") as scratch:
        archive = Path(scratch) / "original.zip"
        archive_sha = download_archive(key, original, archive)
        with zipfile.ZipFile(archive) as zipped:
            member = zipped.getinfo(members[0]["name"])
            if (member.CRC != members[0]["crc32"]
                    or member.file_size != members[0]["uncompressed_bytes"]):
                raise MetadataAuditError("original member metadata mismatch")
            with zipped.open(member) as source:
                # Explicitly excludes both original CSV label columns.
                for frame in pd.read_csv(source, chunksize=100000, dtype=str,
                                         usecols=[*dtypes, *S.HOST_COLUMNS]):
                    row_keys = numeric_keys(frame, dtypes)
                    invalid += sum(k is None for k in row_keys)
                    candidates.observe(row_keys, frame[S.HOST_COLUMNS[0]].to_numpy(),
                                       frame[S.HOST_COLUMNS[1]].to_numpy())
                    scanned += len(frame)
                    if scanned % 1000000 == 0:
                        log(f"{key} original rows scanned={scanned}")
    report = {**candidates.summary(), "raw_rows_scanned": scanned,
              "raw_numeric_rows_unjoinable": invalid, "original_archive_sha256": archive_sha,
              "frozen_input_sha256": pinned["frozen_input"]["sha256"],
              "ordered_numeric_join_schema": dtypes, "join_uses_labels": False,
              "sampler_seed": 0, "sampler_parity": True,
              "frozen_position_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
              "ordered_join_key_sha256": hashlib.sha256(b"".join(keys)).hexdigest(),
              "current_input_historical_byte_identity": "UNVERIFIED original hash absent"}
    if report["complete_unambiguous"]:
        pairs = candidates.pairs(keys)
        groups, method = SP.host_groups([p[0] for p in pairs], [p[1] for p in pairs])
        splits = SP.random_split(len(positions), historical.FRACTIONS, 0, legacy.fam_idx)
        path = work / f"{key}_host_mapping.npz"
        np.savez_compressed(path, parquet_positions=positions, host_groups=groups,
                            train=splits[0], cal=splits[1], test=splits[2])
        report.update({"mapping_sha256": digest_file(path), "group_method": method,
                       "group_count": int(len(np.unique(groups))),
                       "largest_group_share": float(np.bincount(groups).max() / len(groups))})
    report["mapping_status"] = "COMPLETE" if report["complete_unambiguous"] else "UNMEASURABLE"
    return report


def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(HISTORICAL_ROWS), default="unsw_nb15")
    args = parser.parse_args(ctx.argv)
    stage_files = list(Path("/kaggle/input").rglob("nids_provenance_stage1.json"))
    if len(stage_files) != 1 or digest_file(stage_files[0]) != STAGE1_SHA:
        raise MetadataAuditError("stage1 receipt missing or hash mismatch")
    if digest_file(Path(historical.__file__)) != SAMPLER_SHA:
        raise MetadataAuditError("historical sampler source mismatch")
    report = {"dataset": args.dataset, "stage1_sha256": STAGE1_SHA,
              "models_fitted": 0, "rows_scored": 0, "all_four_mapping_gate": "NOT_EVALUATED"}
    try:
        report.update(audit_dataset(args.dataset, json.loads(stage_files[0].read_text()),
                                    Path("/kaggle/working"), ctx.log))
    except Exception as exc:
        # Arbitrary transport/CSV errors can contain URLs or row values.
        report.update({"error": str(exc) if isinstance(exc, MetadataAuditError)
                       else type(exc).__name__, "mapping_status": "UNMEASURABLE"})
    if report["mapping_status"] != "COMPLETE":
        report["all_four_mapping_gate"] = "UNMEASURABLE: pilot mapping failed"
    write_metrics("nids_provenance_mapping.json", report)
    ctx.results["summary"] = (
        f"{args.dataset} host mapping {report['mapping_status']}; no model result")
    return "ok" if "error" not in report else "partial"
