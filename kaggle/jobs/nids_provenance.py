"""Stage-1 provenance audit only: frozen-input hashes and original archive headers.

No models, scoring, labels, flow rows or changed split files are output. Exact
frozen-row correspondence remains UNMEASURABLE until a later complete mapping.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from _common import write_metrics

from rakshak.nids.provenance import (
    RDM_IDS,
    MetadataAuditError,
    RangeArchive,
    inspect_zip_header,
)

# Row counts from the original saved nids_baselines result, not new split choices.
FROZEN_INPUTS = {
    "unsw_nb15": ("nfunswnb15v2", 1986745),
    "cse_cic_ids2018": ("nfcsecicids2018v2", 17129715),
    "bot_iot": ("nfbotiotv2", 30420086),
    "ton_iot": ("nftoniotv2", 13135881),
}
HISTORICAL_TREE = "dea7270125e726d5e70c4ddf139673062be2870309a81b8cec9b15a175a337ec"


def input_metadata(root: Path, slug: str, historical_rows: int) -> dict:
    candidates = sorted(p for p in root.rglob("*.parquet") if slug in p.parts)
    if len(candidates) != 1:
        raise MetadataAuditError("expected exactly one frozen input parquet")
    path = candidates[0]
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    parquet = pq.ParquetFile(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise MetadataAuditError("input changed during hashing")
    rows = parquet.metadata.num_rows
    return {"filename": path.name, "bytes": before.st_size, "sha256": digest.hexdigest(),
            "rows": rows, "row_groups": parquet.num_row_groups,
            "columns": parquet.schema_arrow.names, "historical_rows": historical_rows,
            "row_count_matches_history": rows == historical_rows,
            "historical_byte_identity": "UNVERIFIED: no original file hash was saved"}


def original_metadata(key: str) -> dict:
    url = f"https://api.rdm.uq.edu.au/datasets/{RDM_IDS[key]}/download"
    try:
        with urllib.request.urlopen(url, timeout=40) as response:
            metadata = json.loads(response.read(1024 * 1024))
        if metadata.get("canDownload") is not True:
            raise MetadataAuditError("official archive not downloadable")
        with RangeArchive(metadata["url"]) as archive:
            report = inspect_zip_header(archive)
            report.update(archive.metadata)
        return {"rdm_uuid": RDM_IDS[key], "access_type": metadata.get("accessType"),
                "download_url_recorded": False, **report}
    except urllib.error.HTTPError as exc:
        raise MetadataAuditError(f"metadata HTTP {exc.code}") from None
    except (OSError, ValueError, KeyError) as exc:
        raise MetadataAuditError(f"metadata {type(exc).__name__}") from None


def run(ctx):
    report = {"stage": "metadata_headers_only", "historical_tree": HISTORICAL_TREE,
              "mapping_status": "UNMEASURABLE: no frozen-row host mapping in stage 1",
              "models_fitted": 0, "rows_scored": 0, "datasets": {}}
    complete = True
    for key, (slug, rows) in FROZEN_INPUTS.items():
        item = {}
        for name, operation in (
            ("frozen_input", lambda slug=slug, rows=rows: input_metadata(
                Path("/kaggle/input"), slug, rows)),
            ("official_archive", lambda key=key: original_metadata(key)),
        ):
            try:
                item[name] = operation()
            except Exception as exc:
                # Never serialize arbitrary network/ZIP exception text: URLs can
                # contain signed credentials. Known audit errors are redacted.
                error = str(exc) if isinstance(exc, MetadataAuditError) else type(exc).__name__
                item[name] = {"error": error}
                complete = False
        report["datasets"][key] = item
        write_metrics("nids_provenance_stage1.json", report)
        ctx.log(f"metadata audit {key}: " + ("recorded" if not any(
            "error" in value for value in item.values()) else "incomplete"))
    eligible = complete and all(
        item["official_archive"]["all_csvs_have_hosts"]
        and item["frozen_input"]["row_count_matches_history"]
        for item in report["datasets"].values())
    report["stage2_header_prerequisites"] = eligible
    report["technical_metadata_complete"] = complete
    write_metrics("nids_provenance_stage1.json", report)
    ctx.results["summary"] = (
        f"stage1 metadata complete={complete}, stage2 prerequisites={eligible}; "
        "row mapping UNMEASURABLE, no model result")
    return "ok" if complete else "partial"
