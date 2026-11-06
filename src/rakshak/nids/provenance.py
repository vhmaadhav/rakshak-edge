"""Bounded metadata inspection of the original UQ NetFlow archives.

Run against real archives on Kaggle only. Signed download URLs stay in memory;
reports contain headers and archive metadata, never flow rows or credentials.
This does not establish row correspondence with a processed dataset.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from typing import Any

RDM_IDS = {
    "unsw_nb15": "8c6e2a00-ef9c-11ed-827d-e762de186848",
    "ton_iot": "a4ad7080-ef9c-11ed-a964-b70596e96ad5",
    "bot_iot": "b97fa3c0-ef9c-11ed-8007-832d3bc76ba8",
    "cse_cic_ids2018": "ce5161d0-ef9c-11ed-827d-e762de186848",
}
DOWNLOAD_HOST = "uq-rdm-datasets.s3.ap-southeast-2.amazonaws.com"
HOST_COLUMNS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"}


class MetadataAuditError(RuntimeError):
    """A deliberately URL-free error safe for saved reports."""


class RangeArchive(io.RawIOBase):
    """Seekable HTTP range reader with a hard cumulative response-byte limit.

    Refuse a server that ignores Range, a changing ETag, or an invalid byte range.
    No full-download fallback: even an unexpected large ZIP is bounded.
    """

    def __init__(
        self, url: str, *, budget: int = 8 * 1024 * 1024,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ):
        super().__init__()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname != DOWNLOAD_HOST:
            raise MetadataAuditError("unexpected archive origin")
        self._url, self._opener = url, opener
        self._position, self._spent, self._budget = 0, 0, budget
        self._size: int | None = None
        self._etag: str | None = None
        self._range(0, 1)

    @property
    def metadata(self) -> dict:
        return {"archive_bytes": self._size, "etag": self._etag,
                "range_response_bytes": self._spent}

    def _range(self, start: int, length: int) -> bytes:
        if length < 1 or self._spent + length > self._budget:
            raise MetadataAuditError("archive metadata byte budget exceeded")
        headers = {"Range": f"bytes={start}-{start + length - 1}",
                   "Accept-Encoding": "identity"}
        if self._etag is not None:
            headers["If-Match"] = self._etag
        request = urllib.request.Request(self._url, headers=headers)
        try:
            with self._opener(request, timeout=40) as response:
                if response.status != 206:
                    raise MetadataAuditError("archive server did not honour Range")
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match:
                    raise MetadataAuditError("invalid archive Content-Range")
                first, last, size = map(int, match.groups())
                if first != start or last != start + length - 1 or size <= last:
                    raise MetadataAuditError("archive range bounds mismatch")
                etag = response.headers.get("ETag")
                if not etag or (self._etag is not None and self._etag != etag):
                    raise MetadataAuditError("missing or changed archive ETag")
                if self._size is not None and self._size != size:
                    raise MetadataAuditError("archive size changed")
                # Read only the requested length; reject truncated responses.
                data = response.read(length)
                if len(data) != length:
                    raise MetadataAuditError("truncated archive range response")
        except urllib.error.HTTPError as exc:
            raise MetadataAuditError(f"archive HTTP {exc.code}") from None
        except (OSError, ValueError) as exc:
            raise MetadataAuditError(f"archive transport {type(exc).__name__}") from None
        self._size, self._etag = size, etag
        self._spent += len(data)
        return data

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        assert self._size is not None
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}
        if whence not in origin or origin[whence] + offset < 0:
            raise MetadataAuditError("invalid archive seek")
        self._position = origin[whence] + offset
        return self._position

    def read(self, size: int = -1) -> bytes:
        assert self._size is not None
        length = max(0, self._size - self._position)
        if size >= 0:
            length = min(length, size)
        if not length:
            return b""
        data = self._range(self._position, length)
        self._position += len(data)
        return data


def inspect_zip_header(archive: Any, *, max_header: int = 65536) -> dict:
    """Read CSV header(s), never emit data lines or extract archive members."""
    with zipfile.ZipFile(archive) as zipped:
        members = [m for m in zipped.infolist()
                   if not m.is_dir() and m.filename.lower().endswith(".csv")]
        if not members or len(members) > 4:
            raise MetadataAuditError("expected one to four CSV archive members")
        rows = []
        for member in members:
            if member.flag_bits & 1:
                raise MetadataAuditError("encrypted archive member")
            with zipped.open(member) as source:
                header = source.readline(max_header + 1)
            if len(header) > max_header or not header.endswith(b"\n"):
                raise MetadataAuditError("missing or oversized CSV header")
            try:
                columns = next(csv.reader([header.decode("utf-8-sig")]))
            except (UnicodeError, csv.Error):
                raise MetadataAuditError("invalid UTF-8 CSV header") from None
            # CSV exports may contain an unnamed index or auxiliary header names.
            # Recognize a flow header by its semantic anchors, rather than an
            # invented uppercase-only vocabulary. Never emit an unrecognized
            # first line (which might otherwise be a data row).
            recognized = "IN_BYTES" in columns and bool(
                {"OUT_BYTES", *HOST_COLUMNS} & set(columns))
            if not recognized:
                rows.append({"name": member.filename,
                             "header_sha256": hashlib.sha256(header).hexdigest(),
                             "column_count": len(columns), "kind": "not_flow_header"})
                continue
            # Header-only result; no raw data values or signed URLs.
            rows.append({"name": member.filename, "uncompressed_bytes": member.file_size,
                         "compressed_bytes": member.compress_size, "crc32": member.CRC,
                         "header_sha256": hashlib.sha256(header).hexdigest(),
                         "columns": columns, "kind": "flow_header",
                         "has_both_hosts": HOST_COLUMNS <= set(columns)})
    flows = [r for r in rows if r["kind"] == "flow_header"]
    if not flows:
        raise MetadataAuditError("no recognized NetFlow CSV header")
    return {"members": rows, "all_csvs_have_hosts": all(r["has_both_hosts"] for r in flows)}
