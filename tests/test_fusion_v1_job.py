"""Portable student feed and private-publication isolation, synthetic fixtures only."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rakshak.models.features import featurize


def module_at(relative, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo/'kaggle/jobs'))
    spec = importlib.util.spec_from_file_location('v1_'+Path(relative).stem, repo/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onnx_student_feed_is_canonical_rule_side_and_batched(monkeypatch):
    job = module_at('kaggle/jobs/fusion_v1.py', monkeypatch)
    feats = featurize(['Synthetic OTP fixture: do not share.']*257)
    batches = []
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name=value) for value in ('raw', 'canon', 'side')]
        def run(self, _, feed):
            n = len(feed['raw'])
            start = sum(batches)
            assert feed['raw'].dtype == np.int64
            assert np.array_equal(feed['raw'], feats.raw_bytes[start:start+n])
            assert np.array_equal(feed['canon'], feats.canon_bytes[start:start+n])
            assert np.array_equal(feed['side'], feats.side('canon')[start:start+n])
            batches.append(n)
            return [np.zeros(n)]
    assert np.array_equal(job.onnx_scores(Session(), feats), np.full(257, .5))
    assert batches == [256, 1]


@pytest.mark.parametrize('checkpoint', [False, True])
def test_private_stage_excludes_messages_and_rejects_incomplete_job(
        tmp_path, monkeypatch, checkpoint):
    pub = module_at('kaggle/publish_fusion_v1.py', monkeypatch)
    source = tmp_path/'source'
    source.mkdir()
    manifest = {'schema': 'rakshak-fusion-v1', 'shipment_approved': False,
                'fitting': {'tree_sha256': '2'*64, 'git_sha': '1'*40, 'kernel_version': 1},
                'artifacts': {}}
    keys = pub.KEYS - {'student_onnx'} if checkpoint else pub.KEYS
    if checkpoint:
        manifest['student_format'] = 'pytorch_checkpoint'
    for key in keys:
        path = source/key
        path.write_bytes(key.encode())
        manifest['artifacts'][key] = {'file': key, 'sha256': pub.digest(path)}
    manifest['validation_references'] = {}
    for fold in pub.COMPANIONS:
        directory = source/fold.replace(':', '_')
        directory.mkdir()
        child = {**manifest, 'fold': fold, 'validation_references': {}}
        for entry in manifest['artifacts'].values():
            (directory/entry['file']).write_bytes((source/entry['file']).read_bytes())
        child_path = directory/'fusion_v1.json'
        child_path.write_text(json.dumps(child))
        manifest['validation_references'][fold] = {
            'manifest': str(child_path.relative_to(source)), 'sha256': pub.digest(child_path)}
    path = source/'fusion_v1.json'
    path.write_text(json.dumps(manifest))
    metrics = {'artifact_gate': 'go', 'manifest_sha256': pub.digest(path)}
    result = {'status': 'ok', 'tree_sha256': '2'*64, 'git_sha': '1'*40}
    (source/'metrics.json').write_text(json.dumps(metrics))
    (source/'results.json').write_text(json.dumps(result))
    (source/'external_messages.parquet').write_bytes(b'NEVER UPLOAD THIS FIXTURE')
    push = tmp_path/'push.log'
    push.write_text('Kernel version 1 successfully pushed.\n')
    out = tmp_path/'stage'
    pub.stage(source, out, push)
    assert not (out/'external_messages.parquet').exists()
    assert len(list(out.rglob('fusion_v1.json'))) == 6
    meta = json.loads((out/'dataset-metadata.json').read_text())
    assert meta['isPrivate'] is True
    assert meta['id'] == pub.DATASET
    push.write_text('Kernel version 2 successfully pushed.\n')
    with pytest.raises(ValueError, match='pushed version'):
        pub.stage(source, tmp_path/'bad', push)
    push.write_text('Kernel version 1 successfully pushed.\n')
    result['status'] = 'partial'
    (source/'results.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='complete'):
        pub.stage(source, tmp_path/'bad', push)


def test_all_fold_experts_freeze_before_first_outcome(tmp_path, monkeypatch):
    import sys

    import pandas as pd

    from rakshak.eval import data as D
    from rakshak.fusion.frozen_bundle import sha256

    job = module_at('kaggle/jobs/fusion_v1.py', monkeypatch)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setenv('RAKSHAK_WORK', str(tmp_path))
    names = {'campaign', 'lang_shift', 'temporal', 'source_holdout:holdout_scamshield',
             'source_holdout:holdout_uci_sms_spam',
             'source_holdout:holdout_mishra_soni_smsphishing'}
    sms = pd.DataFrame({'id': [f'fixture{i}' for i in range(16)],
                        'text': ['synthetic notification']*16, 'label_binary': [0, 1]*8,
                        'label_scam': [0, 1]*8, 'campaign_id': [str(i//2) for i in range(16)]})
    runs = [D.Run(name, np.arange(4), np.arange(4, 8), np.arange(8, 16),
                  np.arange(8, 16), {}) for name in names]
    monkeypatch.setattr(job.D, 'resolve_data_root', lambda: tmp_path)
    monkeypatch.setattr(job.D, 'load_sms', lambda _: sms)
    monkeypatch.setattr(job.D, 'load_runs', lambda *a, **k: runs)
    monkeypatch.setattr(job.D, 'group_values', lambda *a: {})
    frozen = []
    def freeze(ctx, args, frame, labels, ids, features, split, references):
        assert np.array_equal(split.train, np.arange(4))
        if split.name == 'campaign':
            assert set(references) == names - {'campaign'}
        manifest = {'description': 'newly fitted (not previously shipped)',
                    'fusion': {'thresholds': {'t_safe': .2, 't_scam': .8}},
                    'student_cal_1pct_threshold': .5}
        path = ctx.work/'fusion_v1.json'
        path.write_text(json.dumps(manifest))
        frozen.append(split.name)
        return {'manifest': manifest, 'sha256': sha256(path), 'cal_scores': [.2, .8]*2,
                'session': None, 'url': None, 'history': {}, 'work': ctx.work}
    monkeypatch.setattr(job, 'fit_and_freeze', freeze)
    def score(_, features):
        assert set(frozen) == names  # no test/external scoring while any state is unfrozen
        return np.full(len(features), .5)
    monkeypatch.setattr(job, 'onnx_scores', score)
    monkeypatch.setattr(job, 'matrix', lambda frame, score, url: score)
    monkeypatch.setattr(job, 'predict_frozen', lambda state, values: values)
    monkeypatch.setattr(job, 'reports', lambda *a: {'label_scam': {}})
    monkeypatch.setattr(job, 'evaluate_scores', lambda *a, **k: {})
    monkeypatch.setattr(job, 'load_manifest', lambda *a, **k: {})
    monkeypatch.setattr(job, 'baseline_gate', lambda *a: {'decision': 'kill'})
    monkeypatch.setattr(job, 'mark_exact_duplicates', lambda frame: frame)
    monkeypatch.setattr(job, 'assign_campaigns', lambda frame: (frame, {}))
    def external(_):
        assert set(frozen) == names
        return sms.assign(text_available=True)
    monkeypatch.setattr(job.pd, 'read_parquet', external)
    ctx = SimpleNamespace(argv=['--kernel', 'maadhavvh/rakshak-fixture',
                                '--kernel-version', '1'], work=tmp_path, input_root=tmp_path,
                          src=tmp_path, results={}, log=lambda _: None)
    assert job.run(ctx)
    metrics = json.loads((tmp_path/'metrics.json').read_text())
    assert set(metrics['internal']) == names
    assert metrics['safety_gate']['decision'] == 'kill'
