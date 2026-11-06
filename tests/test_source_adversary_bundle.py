"""The source-adversary wrapper must carry its frozen source-control evaluation pipeline."""

import importlib.util
from pathlib import Path


def test_source_adversary_bundle_includes_shared_evaluation_job():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("source_builder", root / "kaggle/build_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "kaggle/jobs/sms_source_balance.py" in module.bundle_names("sms_source_adversary")
