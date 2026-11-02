"""Policy/cal split identifiers must freeze before even loading SMS rows."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rakshak.fusion.escalation_audit import FOLDS


def test_required_alert_freezes_native_cal_before_any_sms_access(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('required_alert_fixture',
                                                repo/'kaggle/jobs/evidence_required_alert.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    models = {fold: (tmp_path, {'experts': {'url': {'artifact_sha256': '1'*64}}}, '2'*64)
              for fold in FOLDS}
    monkeypatch.setattr(job, 'v1_inputs', lambda *args: models)
    monkeypatch.setattr(job.D, 'resolve_data_root', lambda: tmp_path)
    monkeypatch.setattr(job, 'frozen_url_folder', lambda _: tmp_path)
    monkeypatch.setattr(job, 'FrozenURLExpert', lambda _: object())
    native = {'split_id': 'synthetic-native-cal', 'cal_ids': ['native0', 'native1'],
              'cal_ids_sha256': '3'*64, 'threshold_selection': {'threshold': .73}}
    monkeypatch.setattr(job, 'calibrate_native_url', lambda *args: native)
    class FrozenBeforeSMS(Exception):
        pass
    def stop_at_sms(_):
        policy = json.loads((tmp_path/'frozen_required_alert_policy.json').read_text())
        assert policy['native_url_calibration'] == native
        assert set(policy['parent_fold_sha256']) == FOLDS
        assert policy['shipment_approved'] is False
        raise FrozenBeforeSMS
    monkeypatch.setattr(job.D, 'load_sms', stop_at_sms)
    ctx = SimpleNamespace(argv=['--v1-tree', '1'*64, '--v1-manifest-sha256', '2'*64],
                          src=repo/'src', work=tmp_path, input_root=tmp_path,
                          results={'git_sha': '4'*40, 'tree_sha256': '5'*64}, log=lambda _: None)
    with pytest.raises(FrozenBeforeSMS):
        job.run(ctx)


def test_required_alert_bundle_includes_its_actual_frozen_input_reader():
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('required_alert_builder',
                                                repo/'kaggle/build_kernel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert 'kaggle/jobs/evidence_escalation.py' in module.bundle_names('evidence_required_alert')
