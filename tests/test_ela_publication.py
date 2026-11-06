"""Only exact experiment artifacts are staged, never generated/message rows."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_private_archive_records_unmeasured_runtime_and_checks_exact_weights(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('ela_pub_fixture', repo/'kaggle/publish_ela.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    audit, exported, runtime = (tmp_path/name for name in ('audit', 'export', 'runtime'))
    for folder in (audit, exported, runtime):
        folder.mkdir()
    adapter = audit/job.ADAPTER_DIR
    adapter.mkdir()
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'LICENSE', 'NOTICE',
                 'README.md', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json'):
        (adapter/name).write_bytes(name.encode())
    for name in (job.PTE_NAME, 'LICENSE', 'NOTICE', 'config.json', 'tokenizer.json',
                 'ela_8da4w_xnnpack.yaml'):
        (exported/name).write_bytes(name.encode())
    (audit/'results.json').write_text(json.dumps({'status': 'ok'}))
    (audit/'metrics.json').write_text(json.dumps({'comparison': {'decision': 'go'}}))
    (exported/'results.json').write_text(json.dumps({'status': 'error'}))
    (runtime/'results.json').write_text(json.dumps({'status': 'error',
                                                  'tree_sha256': job.RUNTIME_TREE}))
    (runtime/'pte_generations.jsonl').write_text('synthetic forbidden payload\n')
    monkeypatch.setattr(job, 'AUDIT', job.digest(audit/'metrics.json'))
    monkeypatch.setattr(job, 'ADAPTER', job.digest(adapter/'adapter_model.safetensors'))
    monkeypatch.setattr(job, 'PTE', job.digest(exported/job.PTE_NAME))
    monkeypatch.setattr(job, 'LICENSE', job.digest(exported/'LICENSE'))
    (exported/'ela_export_metrics.json').write_text(json.dumps({
        'export': {'exit_code': 0}, 'pte': {'sha256': job.PTE}}))
    target = tmp_path/'private-stage'
    manifest = job.stage(audit, exported, runtime, target)
    assert manifest['quality_retained'] is None
    assert manifest['shipment_approved'] is False
    assert manifest['hf_publish'] == 'HF publish pending'
    assert not (target/'pte_generations.jsonl').exists()
    assert json.loads((target/'dataset-metadata.json').read_text())['isPrivate'] is True
    (exported/job.PTE_NAME).write_bytes(b'wrong weights')
    with pytest.raises(ValueError, match='hashes changed'):
        job.stage(audit, exported, runtime, tmp_path/'bad')


def test_privacy_verification_uses_owner_record_and_rejects_public_or_wrong_owner():
    from types import SimpleNamespace

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        'ela_privacy_fixture', repo/'kaggle/publish_ela.py')
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    response = SimpleNamespace(ref=job.DATASET, owner_ref='maadhavvh', is_private=True,
                               current_version_number=1, total_bytes=123)
    def get(request):
        assert request.owner_slug == 'maadhavvh'
        assert request.dataset_slug == 'rakshak-ela-models-b20261101'
        return response
    service = SimpleNamespace(get_dataset=get)
    assert job.verify_private(service, SimpleNamespace())['isPrivate'] is True
    response.is_private = False
    with pytest.raises(RuntimeError, match='private owner'):
        job.verify_private(service, SimpleNamespace())
    response.is_private = True
    response.owner_ref = 'wrong-owner'
    with pytest.raises(RuntimeError, match='private owner'):
        job.verify_private(service, SimpleNamespace())
