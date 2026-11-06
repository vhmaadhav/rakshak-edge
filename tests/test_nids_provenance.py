"""Synthetic archive/HTTP regressions; no network or real flows."""

import io
import json
import sys
import urllib.error
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rakshak.nids.provenance import (
    DOWNLOAD_HOST,
    MetadataAuditError,
    RangeArchive,
    inspect_zip_header,
)


def archive_bytes(header=b"IPV4_SRC_ADDR,IPV4_DST_ADDR,IN_BYTES\n"):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("flows.csv", header + b"SYNTHETIC_SOURCE,SYNTHETIC_DESTINATION,42\n")
    return out.getvalue()


class Response(io.BytesIO):
    status = 206

    def __init__(self, data, start, end, total):
        super().__init__(data)
        self.headers = {"ETag": '"fixture"', "Content-Range": f"bytes {start}-{end}/{total}"}


def opener_for(blob, mutate=None):
    def open_range(request, timeout):
        assert timeout == 40
        start, end = map(int, request.get_header("Range")[6:].split("-"))
        response = Response(blob[start:end + 1], start, end, len(blob))
        if mutate:
            mutate(response)
        return response
    return open_range


def test_bounded_zip_header_does_not_emit_flow_or_signed_url():
    blob = archive_bytes()
    secret_url = f"https://{DOWNLOAD_HOST}/archive.zip?secret=DO_NOT_REPORT"
    archive = RangeArchive(secret_url, opener=opener_for(blob))
    result = inspect_zip_header(archive)
    assert result["all_csvs_have_hosts"]
    assert result["members"][0]["columns"] == ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "IN_BYTES"]
    assert "SYNTHETIC_SOURCE" not in str(result)
    assert "DO_NOT_REPORT" not in str(result) + str(archive.metadata)
    assert archive.metadata["range_response_bytes"] < 8 * 1024 * 1024


@pytest.mark.parametrize("bad", ["status", "range", "etag", "short"])
def test_refuses_unreliable_range_server(bad):
    def mutate(response):
        if bad == "status":
            response.status = 200
        elif bad == "range":
            response.headers["Content-Range"] = "bytes 1-1/1000"
        elif bad == "etag":
            response.headers.pop("ETag")
        else:
            response.truncate(0)
    with pytest.raises(MetadataAuditError):
        RangeArchive(f"https://{DOWNLOAD_HOST}/file.zip", opener=opener_for(b"abc", mutate))


def test_budget_and_changed_identity_stop_without_full_download():
    blob = archive_bytes()
    archive = RangeArchive(f"https://{DOWNLOAD_HOST}/file.zip", budget=2,
                           opener=opener_for(blob))
    with pytest.raises(MetadataAuditError, match="budget"):
        archive.read()
    archive = RangeArchive(f"https://{DOWNLOAD_HOST}/file.zip", opener=opener_for(blob))
    archive._etag = '"different"'
    with pytest.raises(MetadataAuditError, match="changed"):
        archive.read(1)


def test_http_error_redacts_url_and_unexpected_origin_is_refused():
    secret_url = f"https://{DOWNLOAD_HOST}/archive.zip?secret=DO_NOT_REPORT"

    def denied(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, secret_url, None, None)

    with pytest.raises(MetadataAuditError) as caught:
        RangeArchive(secret_url, opener=denied)
    assert str(caught.value) == "archive HTTP 403"
    with pytest.raises(MetadataAuditError, match="origin"):
        RangeArchive("https://unrelated.example/flow.zip", opener=denied)


def test_stripped_hosts_and_oversized_header_are_not_silently_accepted():
    result = inspect_zip_header(io.BytesIO(archive_bytes(b"IN_BYTES,OUT_BYTES\n")))
    assert not result["all_csvs_have_hosts"]
    with pytest.raises(MetadataAuditError, match="header"):
        inspect_zip_header(io.BytesIO(archive_bytes()), max_header=8)
    with pytest.raises(MetadataAuditError, match="header"):
        inspect_zip_header(io.BytesIO(archive_bytes(b"192.0.2.1,192.0.2.2,42\n")))


def test_auxiliary_index_and_feature_dictionary_are_not_mistaken_for_missing_hosts():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("features.csv", "Feature,Description\nIN_BYTES,Byte count\n")
        zipped.writestr("flows.csv", ",IPV4_SRC_ADDR,IPV4_DST_ADDR,IN_BYTES\n0,A,B,42\n")
    result = inspect_zip_header(io.BytesIO(out.getvalue()))
    assert result["all_csvs_have_hosts"]
    assert result["members"][0]["kind"] == "not_flow_header"
    assert "columns" not in result["members"][0]
    assert result["members"][1]["columns"][0] == ""


def load_job():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle" / "jobs"))
    import nids_provenance
    return nids_provenance


def test_input_audit_hashes_bytes_but_count_match_does_not_claim_identity(tmp_path):
    job = load_job()
    folder = tmp_path / "frozen-slug"
    folder.mkdir()
    path = folder / "flows.parquet"
    pq.write_table(pa.table({"IN_BYTES": [42, 100]}), path)
    result = job.input_metadata(tmp_path, "frozen-slug", 2)
    assert result["row_count_matches_history"]
    assert result["historical_byte_identity"].startswith("UNVERIFIED")
    assert len(result["sha256"]) == 64
    assert result["columns"] == ["IN_BYTES"]
    assert not job.input_metadata(tmp_path, "frozen-slug", 3)["row_count_matches_history"]
    pq.write_table(pa.table({"IN_BYTES": [1]}), folder / "other.parquet")
    with pytest.raises(MetadataAuditError, match="exactly one"):
        job.input_metadata(tmp_path, "frozen-slug", 2)


def test_job_partial_failure_redacts_arbitrary_exception_and_retains_unmeasurable(monkeypatch):
    job = load_job()
    saved = []
    monkeypatch.setattr(job, "write_metrics",
                        lambda name, x: saved.append(json.loads(json.dumps(x))))
    monkeypatch.setattr(job, "input_metadata", lambda *args: {"row_count_matches_history": True})

    def bad_original(key):
        raise RuntimeError("https://example.invalid/?secret=DO_NOT_REPORT")

    monkeypatch.setattr(job, "original_metadata", bad_original)
    context = SimpleNamespace(results={}, log=lambda message: None)
    assert job.run(context) == "partial"
    result = saved[-1]
    assert not result["technical_metadata_complete"]
    assert not result["stage2_header_prerequisites"]
    assert result["mapping_status"].startswith("UNMEASURABLE")
    assert "DO_NOT_REPORT" not in str(saved) + str(context.results)
    assert result["models_fitted"] == result["rows_scored"] == 0
