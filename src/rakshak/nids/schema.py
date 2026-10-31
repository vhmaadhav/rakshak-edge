"""Frozen shared NetFlow-v2 feature schema, dtype coercion, and the label taxonomy.

The schema is the 33 candidate features of docs/NETWORK_IDS_PLAN.md section 4.2 (names as in the
NF-*-v2 feature files). Identifiers (IPs, ports), timestamps, TTL, TCP sequence numbers and window
sizes, and the host-window ``ct_*`` counters are not in the schema: they either leak the capture
or cannot be computed the same way on the Pi.

``SCHEMA_VERSION`` is part of every exported model and every result file. Changing the feature
list or the transforms means a new version; nothing here is tuned on a target test set.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

SCHEMA_VERSION = "netflow-v2-shared-1"

SHARED_FEATURES: tuple[str, ...] = (
    # protocol and service
    "PROTOCOL",
    "L7_PROTO",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    # volume
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    # rate
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    # packet size
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    # TCP flags (bit masks)
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    # retransmission
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    # DNS
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
)

# Identifier and grouping columns read by the Kaggle job. They are never features.
# Host columns are optional: some dhoogla NF-*-v2 copies have them removed. Without both, the
# job skips T4 and never invents hosts.
HOST_COLUMNS: tuple[str, ...] = ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")
IDENTIFIER_COLUMNS = HOST_COLUMNS
LABEL_COLUMNS: tuple[str, ...] = ("Label", "Attack")

# Categorical and bit-mask features: clipped, never log-transformed.
CATEGORICAL: frozenset[str] = frozenset(
    {
        "PROTOCOL",
        "L7_PROTO",
        "ICMP_TYPE",
        "ICMP_IPV4_TYPE",
        "TCP_FLAGS",
        "CLIENT_TCP_FLAGS",
        "SERVER_TCP_FLAGS",
        "DNS_QUERY_TYPE",
    }
)

# Fixed caps applied before the transform. They are set from the feature meaning, not from data,
# so they cannot leak information from a split. Values above a cap are clipped to it.
_CAP_BYTES = 1e12
_CAP_PKTS = 1e9
_CAP_MS = 1e10
_CAP_RATE = 1e12
_CAP_PKT_LEN = 65535.0
_CAP_BITS = 255.0
CAPS: dict[str, float] = {
    "PROTOCOL": 255.0,
    "L7_PROTO": 65535.0,
    "ICMP_TYPE": 255.0,
    "ICMP_IPV4_TYPE": 255.0,
    "IN_BYTES": _CAP_BYTES,
    "OUT_BYTES": _CAP_BYTES,
    "IN_PKTS": _CAP_PKTS,
    "OUT_PKTS": _CAP_PKTS,
    "FLOW_DURATION_MILLISECONDS": _CAP_MS,
    "DURATION_IN": _CAP_MS,
    "DURATION_OUT": _CAP_MS,
    "SRC_TO_DST_SECOND_BYTES": _CAP_RATE,
    "DST_TO_SRC_SECOND_BYTES": _CAP_RATE,
    "SRC_TO_DST_AVG_THROUGHPUT": _CAP_RATE,
    "DST_TO_SRC_AVG_THROUGHPUT": _CAP_RATE,
    "MIN_IP_PKT_LEN": _CAP_PKT_LEN,
    "MAX_IP_PKT_LEN": _CAP_PKT_LEN,
    "LONGEST_FLOW_PKT": _CAP_PKT_LEN,
    "SHORTEST_FLOW_PKT": _CAP_PKT_LEN,
    "NUM_PKTS_UP_TO_128_BYTES": _CAP_PKTS,
    "NUM_PKTS_128_TO_256_BYTES": _CAP_PKTS,
    "NUM_PKTS_256_TO_512_BYTES": _CAP_PKTS,
    "NUM_PKTS_512_TO_1024_BYTES": _CAP_PKTS,
    "NUM_PKTS_1024_TO_1514_BYTES": _CAP_PKTS,
    "TCP_FLAGS": _CAP_BITS,
    "CLIENT_TCP_FLAGS": _CAP_BITS,
    "SERVER_TCP_FLAGS": _CAP_BITS,
    "RETRANSMITTED_IN_BYTES": _CAP_BYTES,
    "RETRANSMITTED_IN_PKTS": _CAP_PKTS,
    "RETRANSMITTED_OUT_BYTES": _CAP_BYTES,
    "RETRANSMITTED_OUT_PKTS": _CAP_PKTS,
    "DNS_QUERY_TYPE": 65535.0,
    "DNS_TTL_ANSWER": float(2**31 - 1),
}

LOG_FEATURES: tuple[str, ...] = tuple(f for f in SHARED_FEATURES if f not in CATEGORICAL)

assert len(SHARED_FEATURES) == 33 and len(set(SHARED_FEATURES)) == 33
assert set(CAPS) == set(SHARED_FEATURES)
assert CATEGORICAL <= set(SHARED_FEATURES)

# Attack families. ``benign`` is index 0; the order is part of the model contract.
FAMILIES: tuple[str, ...] = (
    "benign",
    "port_scan",
    "brute_force",
    "dos_ddos",
    "botnet_c2",
    "injection_exploit",
    "exfil_theft",
    "other",
)
FAMILY_INDEX: dict[str, int] = {name: i for i, name in enumerate(FAMILIES)}
ATTACK_FAMILIES: tuple[str, ...] = FAMILIES[1:]


def _name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_columns(
    actual: Sequence[str], wanted: Sequence[str]
) -> tuple[dict[str, str], list[str]]:
    """Map each wanted name to its actual column name in a parquet file.

    An exact match wins. Otherwise a single column whose name agrees after lower-casing and
    dropping separators counts (``in_bytes`` for ``IN_BYTES``). Two candidates for one name
    raise ValueError. Returns (wanted -> actual for every match, wanted names with no match).
    """
    exact = set(actual)
    matched: dict[str, str] = {}
    missing: list[str] = []
    for name in wanted:
        if name in exact:
            matched[name] = name
            continue
        hits = [a for a in actual if _name_key(a) == _name_key(name)]
        if len(hits) > 1:
            raise ValueError(f"ambiguous columns for {name}: {hits}")
        if hits:
            matched[name] = hits[0]
        else:
            missing.append(name)
    return matched, missing


def missing_features(
    columns: Iterable[str], features: Sequence[str] = SHARED_FEATURES
) -> list[str]:
    """Features in ``features`` that are not among ``columns``, in schema order."""
    have = set(columns)
    return [f for f in features if f not in have]


def coerce_matrix(values, features: Sequence[str] = SHARED_FEATURES) -> np.ndarray:
    """Raw feature values (rows x features, columns in ``features`` order) to model input.

    NaN becomes 0. Values are clipped to ``[0, CAPS]`` (so negatives and infinities are handled
    by the caps). Non-categorical features get ``log1p``. Returns float32. The same function
    runs on Kaggle, in the Pi monitor and in tests.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(features):
        raise ValueError(f"expected (rows, {len(features)}) values, got shape {x.shape}")
    x = np.where(np.isnan(x), 0.0, x)
    caps = np.array([CAPS[f] for f in features], dtype=np.float64)
    x = np.clip(x, 0.0, caps)
    for j, name in enumerate(features):
        if name not in CATEGORICAL:
            x[:, j] = np.log1p(x[:, j])
    return x.astype(np.float32)


def coerce_frame(df, features: Sequence[str] = SHARED_FEATURES) -> np.ndarray:
    """``coerce_matrix`` on the named columns of a DataFrame. Missing columns raise KeyError."""
    missing = missing_features(df.columns, features)
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return coerce_matrix(df[list(features)].to_numpy(dtype=np.float64), features)


# ---------------------------------------------------------------------------
# Label taxonomy
# ---------------------------------------------------------------------------

_EXACT: dict[str, str] = {
    "benign": "benign",
    "normal": "benign",
    # UNSW-NB15 (NF-UNSW-NB15-v2)
    "analysis": "other",
    "backdoor": "botnet_c2",
    "dos": "dos_ddos",
    "exploits": "injection_exploit",
    "fuzzers": "other",
    "generic": "other",
    "reconnaissance": "port_scan",
    "shellcode": "injection_exploit",
    "worms": "other",
    # ToN-IoT (NF-ToN-IoT-V2)
    "ddos": "dos_ddos",
    "injection": "injection_exploit",
    "mitm": "other",
    "password": "brute_force",
    "ransomware": "other",
    "scanning": "port_scan",
    "xss": "injection_exploit",
    # BoT-IoT (NF-BoT-IoT-V2)
    "theft": "exfil_theft",
    # CSE-CIC-IDS2018 (NF-CSE-CIC-IDS2018-V2)
    "bot": "botnet_c2",
    "infilteration": "other",
    "infiltration": "other",
    "sql injection": "injection_exploit",
}


def normalise_label(name: str) -> str:
    """Lower case, every run of non-alphanumeric characters becomes one space."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _rule_family(norm: str) -> str | None:
    """Family for a label string the exact table does not know, or None.

    Checked in order: injection words first, so ``Brute Force -XSS`` is an injection. The match
    runs on the string with spaces removed, so ``DDoS attacks-LOIC-HTTP`` and ``FTP-BruteForce``
    match as well.
    """
    compact = norm.replace(" ", "")
    if "xss" in compact or "injection" in compact:
        return "injection_exploit"
    if "brute" in compact:
        return "brute_force"
    if "ddos" in compact or "dos" in compact:
        return "dos_ddos"
    if compact.startswith("bot") or "botnet" in compact:
        return "botnet_c2"
    if "backdoor" in compact:
        return "botnet_c2"
    if "scan" in compact or "recon" in compact:
        return "port_scan"
    if "theft" in compact or "exfil" in compact:
        return "exfil_theft"
    return None


def map_attack(name: str | None, label: int) -> tuple[str, str]:
    """Attack family and how it was decided, for one row.

    ``label`` is the dataset's binary label (0 benign, 1 attack). It wins for benign rows. For
    attack rows the attack string decides; a benign-looking string on an attack row becomes
    ``other`` with how ``conflict``. ``how`` is one of ``label``, ``exact``, ``rule``, ``unknown``
    or ``conflict``. Only ``unknown`` and ``conflict`` need a human look.
    """
    if int(label) == 0:
        return "benign", "label"
    norm = normalise_label(name or "")
    if not norm:
        return "other", "unknown"
    if norm in _EXACT:
        family, how = _EXACT[norm], "exact"
    else:
        ruled = _rule_family(norm)
        if ruled is None:
            return "other", "unknown"
        family, how = ruled, "rule"
    if family == "benign":
        return "other", "conflict"
    return family, how
