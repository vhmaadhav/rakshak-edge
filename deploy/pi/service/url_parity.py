"""U4 parity: score PhreshPhish URLs through the edge service's URL code path and dump float32 bits.

Run the same file on the laptop (repo root) and on the Pi (``~/rakshak-bench``) and compare the
two JSON files with ``--compare``. Each row is scored the way ``POST /analyze_url`` does it:
``canonicalize(text).urls`` then ``best_url_hit`` (``rakshak.url_model``) on a scorer whose cache is
empty, so each request is one ONNX call with that message's URLs, as in the service. The raw path
(``score_urls([url])`` on the dataset string) is kept as a second column.

    python deploy/pi/service/url_parity.py --root . --out run/url_parity_laptop.json
    python deploy/pi/service/url_parity.py --compare laptop.json pi.json

Model files are checked by sha256 against the pins in ``rakshak.fusion.url_expert`` before loading.
No network, no sockets. Reads the parquet with pyarrow; needs numpy, onnxruntime, regex, tldextract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
from pathlib import Path

DEFAULT_PARQUET = "kaggle/out/url_threshold/output/parity_300.parquet"
DEFAULT_MODEL_DIR = "kaggle/out/url_student/output"
TAU_URL = 0.6308487409530461


def f32_hex(x: float) -> str:
    return struct.pack(">f", float(x)).hex()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_rows(root: Path, parquet: Path, model_dir: Path) -> dict:
    sys.path.insert(0, str(root / "src"))
    import numpy as np
    import onnxruntime as ort
    import pyarrow.parquet as pq

    from rakshak.canonicalize import canonicalize
    from rakshak.fusion.url_expert import (
        SCALER_HASH,
        URL_HASH,
        FrozenURLExpert,
    )
    from rakshak.url_model import best_url_hit

    onnx_path = model_dir / "best_url_student.onnx"
    scaler_path = model_dir / "url_side_scaler.json"
    if _sha(onnx_path) != URL_HASH or _sha(scaler_path) != SCALER_HASH:
        raise SystemExit("model or scaler sha256 does not match the pins")

    table = pq.read_table(parquet).to_pydict()
    scorer = FrozenURLExpert(model_dir)
    rows = []
    for i, url in enumerate(table["url"]):
        scorer.cache.clear()
        raw = scorer.score_urls([url])[0]
        scorer.cache.clear()
        urls = canonicalize(url).urls
        hit = best_url_hit(urls, scorer) if urls else None
        rows.append({
            "id": table["id"][i],
            "url": url,
            "kaggle_f32_hex": table["score_f32_hex"][i],
            "kaggle_score": float(table["score"][i]),
            "label": int(table["label"][i]),
            "raw_f32_hex": f32_hex(raw),
            "raw_score": float(raw),
            "n_canon_urls": len(urls),
            "svc_url": hit.url if hit else None,
            "svc_f32_hex": f32_hex(hit.score) if hit else None,
            "svc_score": float(hit.score) if hit else None,
        })
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "onnx_sha256": URL_HASH,
        "scaler_sha256": SCALER_HASH,
        "tau_url": TAU_URL,
        "n": len(rows),
        "rows": rows,
    }


def compare(a_path: Path, b_path: Path) -> dict:
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())
    assert len(a["rows"]) == len(b["rows"]), "row counts differ"
    result = {"a": str(a_path), "b": str(b_path), "n": len(a["rows"])}
    for col in ("raw", "svc"):
        key = f"{col}_f32_hex"
        diff = [r["id"] for r, s in zip(a["rows"], b["rows"], strict=True)
                if r[key] != s[key]]
        result[f"{col}_bit_mismatches"] = len(diff)
        result[f"{col}_mismatch_ids"] = diff[:10]
    flips = sum(
        1 for r, s in zip(a["rows"], b["rows"], strict=True)
        if (r["svc_score"] is not None and s["svc_score"] is not None)
        and ((r["svc_score"] >= TAU_URL) != (s["svc_score"] >= TAU_URL))
    )
    result["svc_flips_at_tau_between_files"] = flips
    return result


def vs_kaggle(doc: dict) -> dict:
    import struct as st

    def f(h: str) -> float:
        return st.unpack(">f", bytes.fromhex(h))[0]

    def kag(h: str) -> float:  # the Kaggle column stores little-endian bytes
        return st.unpack("<f", bytes.fromhex(h))[0]

    rows = doc["rows"]
    raw_max = max(abs(f(r["raw_f32_hex"]) - kag(r["kaggle_f32_hex"])) for r in rows)
    svc_rows = [r for r in rows if r["svc_f32_hex"]]
    svc_max = max(abs(f(r["svc_f32_hex"]) - kag(r["kaggle_f32_hex"])) for r in svc_rows)
    raw_bit_eq = sum(f(r["raw_f32_hex"]) == kag(r["kaggle_f32_hex"]) for r in rows)
    svc_flips = sum(
        (f(r["svc_f32_hex"]) >= TAU_URL) != (kag(r["kaggle_f32_hex"]) >= TAU_URL)
        for r in svc_rows
    )
    return {
        "raw_bit_equal_to_kaggle": raw_bit_eq,
        "kaggle_column": "score_f32_hex (little-endian float32)",
        "raw_max_abs_diff_vs_kaggle": raw_max,
        "svc_max_abs_diff_vs_kaggle": svc_max,
        "svc_flips_at_tau_vs_kaggle": svc_flips,
        "rows_with_url": len(svc_rows),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=Path("."),
                   help="repo root or bench root (has src/)")
    p.add_argument("--parquet", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--out", type=Path)
    p.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = p.parse_args()
    if args.compare:
        print(json.dumps(compare(*args.compare), indent=2))
        return
    root = args.root.resolve()
    parquet = args.parquet or root / DEFAULT_PARQUET
    model_dir = args.model_dir or root / DEFAULT_MODEL_DIR
    doc = score_rows(root, parquet, model_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc))
    summary = {k: v for k, v in doc.items() if k != "rows"}
    summary.update(vs_kaggle(doc))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
