"""The training job must carry the guarded shared runner into its private kernel."""

import importlib.util
from pathlib import Path


def test_hardneg_bundle_includes_shared_guarded_controls():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('hardneg_builder', root/'kaggle/build_kernel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = module.bundle_names('sms_transactional_hardneg')
    assert 'kaggle/jobs/sms_source_balance.py' in names
    assert 'src/rakshak/models/distill.py' in names
    assert 'src/rakshak/models/transactional_negatives.py' in names


def test_fused_hardneg_bundle_contains_common_fusion_job():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        'hardneg_eval_builder', root/'kaggle/build_kernel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = module.bundle_names('fusion_hardneg')
    assert 'kaggle/jobs/fusion_transactional.py' in names
    assert 'src/rakshak/fusion/hardneg_gate.py' in names
