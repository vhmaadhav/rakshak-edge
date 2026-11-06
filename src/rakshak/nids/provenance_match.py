"""Label-free numeric row matching; ambiguous host pairs are never resolved."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from rakshak.nids.provenance import MetadataAuditError


def numeric_keys(frame: pd.DataFrame, dtypes: dict[str, str]) -> list[bytes | None]:
    """SHA256 of ordered numeric fields, cast to the frozen parquet's precision.

    Reject nonfinite/non-numeric rows, integer overflow/fractions and integers
    outside exact float64 range. No labels, hosts, imputation or fitted transforms.
    Casting raw CSV floats to frozen float precision is explicit in the manifest.
    """
    if {"Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"} & set(dtypes):
        raise MetadataAuditError("labels or identifiers are forbidden join fields")
    if not dtypes:
        raise MetadataAuditError("empty join schema")
    values = np.empty((len(frame), len(dtypes)), dtype="<f8")
    valid = np.ones(len(frame), dtype=bool)
    for i, (name, spec) in enumerate(dtypes.items()):
        dtype = np.dtype(spec)
        if dtype.kind not in "iuf":
            raise MetadataAuditError("join schema must be numeric")
        numbers = pd.to_numeric(frame[name], errors="coerce").to_numpy()
        okay = np.isfinite(numbers)
        if dtype.kind in "iu":
            limits = np.iinfo(dtype)
            okay &= (numbers >= limits.min) & (numbers <= limits.max)
            okay &= (numbers >= -(2**53)) & (numbers <= 2**53)
            okay &= numbers == np.floor(numbers)
        with np.errstate(over="ignore", invalid="ignore"):
            cast = np.where(okay, numbers, 0).astype(dtype).astype(np.float64)
        okay &= np.isfinite(cast)
        valid &= okay
        values[:, i] = cast
    values[values == 0] = 0.0  # canonical positive zero
    return [hashlib.sha256(row.tobytes()).digest() if okay else None
            for row, okay in zip(values, valid, strict=True)]


class CandidatePairs:
    """Accumulate all matching source rows without reading source labels."""

    def __init__(self, wanted: Sequence[bytes | None]):
        if not len(wanted) or any(key is None for key in wanted) or len(set(wanted)) != len(wanted):
            raise MetadataAuditError("frozen join keys must be valid and unique")
        self.wanted = set(wanted)
        self.first: dict[bytes, tuple[str, str]] = {}
        self.ambiguous: set[bytes] = set()
        self.matching_rows = 0

    def observe(self, keys, sources, destinations) -> None:
        for key, src, dst in zip(keys, sources, destinations, strict=True):
            if key not in self.wanted:
                continue
            if not isinstance(src, str) or not isinstance(dst, str) or not src or not dst:
                raise MetadataAuditError("matching source row lacks host identifiers")
            pair = (src, dst)
            self.matching_rows += 1
            if key in self.first and self.first[key] != pair:
                self.ambiguous.add(key)
            self.first.setdefault(key, pair)

    def summary(self) -> dict:
        return {"frozen_rows": len(self.wanted), "matched_keys": len(self.first),
                "unmapped_keys": len(self.wanted - self.first.keys()),
                "ambiguous_keys": len(self.ambiguous), "matching_raw_rows": self.matching_rows,
                "complete_unambiguous": len(self.first) == len(self.wanted)
                and not self.ambiguous}

    def pairs(self, ordered_keys) -> list[tuple[str, str]]:
        if not self.summary()["complete_unambiguous"]:
            raise MetadataAuditError("host mapping incomplete or ambiguous")
        return [self.first[key] for key in ordered_keys]
