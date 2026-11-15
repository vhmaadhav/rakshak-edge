"""Offline checks for the U4 parity helper (deploy/pi/service/url_parity.py). No model is loaded."""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "deploy" / "pi" / "service" / "url_parity.py"
spec = importlib.util.spec_from_file_location("url_parity", HELPER)
url_parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(url_parity)


def _row(i, raw, svc, kaggle):
    return {
        "id": str(i),
        "url": f"https://x{i}.test/",
        "kaggle_f32_hex": struct.pack("<f", kaggle).hex(),
        "kaggle_score": kaggle, "label": 0, "raw_f32_hex": url_parity.f32_hex(raw),
        "raw_score": raw, "n_canon_urls": 1, "svc_url": f"https://x{i}.test/",
        "svc_f32_hex": url_parity.f32_hex(svc), "svc_score": svc,
    }


def test_compare_counts_bit_mismatch_and_flip(tmp_path):
    tau = url_parity.TAU_URL
    a = {"rows": [_row(0, 0.1, 0.1, 0.1), _row(1, tau + 0.01, tau + 0.01, tau + 0.01)]}
    b = {"rows": [_row(0, 0.1, 0.1, 0.1), _row(1, tau - 0.01, tau - 0.01, tau + 0.01)]}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    out = url_parity.compare(pa, pb)
    assert out["raw_bit_mismatches"] == 1
    assert out["svc_bit_mismatches"] == 1
    assert out["svc_flips_at_tau_between_files"] == 1


def test_vs_kaggle_reads_little_endian_column():
    doc = {"rows": [_row(0, 0.25, 0.25, 0.25), _row(1, 0.9, 0.9, 0.9)]}
    out = url_parity.vs_kaggle(doc)
    assert out["raw_bit_equal_to_kaggle"] == 2
    assert out["raw_max_abs_diff_vs_kaggle"] == 0.0
    assert out["svc_flips_at_tau_vs_kaggle"] == 0


def test_expit_matches_logistic_and_is_finite_at_extremes():
    np = pytest.importorskip("numpy")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rakshak.fusion.url_expert import _expit

    x = np.array([-800.0, -3.0, 0.0, 3.0, 800.0])
    y = _expit(x)
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert y[2] == np.float32(0.5)
    assert np.allclose(y, 1 / (1 + np.exp(-x)), atol=1e-6)
