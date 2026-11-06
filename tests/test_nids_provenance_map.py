"""End-to-end provenance matching on eight invented flows, no network."""

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rakshak.nids import schema as S

JOBS = Path(__file__).resolve().parents[1] / "kaggle" / "jobs"
sys.path.insert(0, str(JOBS))

import nids_provenance_map as job  # noqa: E402


@pytest.mark.parametrize("ambiguous", [False, True])
def test_full_mapping_replays_rows_ignores_csv_labels_and_refuses_ambiguous_pairs(
    tmp_path, monkeypatch, ambiguous,
):
    extra = ["L4_SRC_PORT", "L4_DST_PORT", "MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN",
             "TCP_WIN_MAX_OUT", "DNS_QUERY_ID", "FTP_COMMAND_RET_CODE"]
    numeric = pd.DataFrame({name: np.ones(8, dtype=np.int64)
                            for name in [*S.SHARED_FEATURES, *extra]})
    numeric["IN_BYTES"] = np.arange(8, dtype=np.int64) + 1
    frame = numeric.assign(Label=[0, 1] * 4, Attack=["Benign", "DoS"] * 4)
    parquet = tmp_path / "frozen.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), parquet)
    monkeypatch.setitem(job.HISTORICAL_ROWS, "unsw_nb15", 8)
    monkeypatch.setattr(job.historical, "discover", lambda *args: {"unsw_nb15": parquet})
    replayed, positions, _, _ = job.replay_rows(parquet, "unsw_nb15")
    assert replayed.equals(numeric)
    assert positions.tolist() == list(range(8))
    original = frame.assign(IPV4_SRC_ADDR=[f"192.0.2.{i + 1}" for i in range(8)],
                            IPV4_DST_ADDR="198.51.100.1", Label=1 - frame.Label)
    if ambiguous:
        duplicate = original.iloc[[0]].copy()
        duplicate["IPV4_SRC_ADDR"] = "192.0.2.100"
        # Different label cannot be used to resolve this deliberately ambiguous match.
        duplicate["Label"] = 0
        original = pd.concat([original, duplicate], ignore_index=True)
    archive = tmp_path / "original.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("flows.csv", original.to_csv(index=False))
    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo("flows.csv")
    pinned = {"datasets": {"unsw_nb15": {
        "frozen_input": {"sha256": job.digest_file(parquet)},
        "official_archive": {"members": [{"kind": "flow_header", "name": "flows.csv",
                                             "crc32": info.CRC,
                                             "uncompressed_bytes": info.file_size}]},
    }}}

    def fake_download(key, metadata, target):
        shutil.copyfile(archive, target)
        return hashlib.sha256(archive.read_bytes()).hexdigest()

    monkeypatch.setattr(job, "download_archive", fake_download)
    result = job.audit_dataset("unsw_nb15", pinned, tmp_path, lambda message: None)
    assert result["sampler_parity"]
    assert result["unmapped_keys"] == 0
    assert result["ambiguous_keys"] == int(ambiguous)
    assert result["complete_unambiguous"] == (not ambiguous)
    assert result["join_uses_labels"] is False
    assert "192.0.2." not in str(result)
    assert (tmp_path / "unsw_nb15_host_mapping.npz").exists() == (not ambiguous)
