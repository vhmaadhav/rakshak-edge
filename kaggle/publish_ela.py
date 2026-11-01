#!/usr/bin/env python3
"""Archive the exact experiment adapter/PTE privately after the final runtime receipt.

No training, inference, API labelling, deployment approval or artifact modification.
Run with the installed Kaggle CLI interpreter. No message/generation files uploaded.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DATASET = 'maadhavvh/rakshak-ela-models-b20261101'
ADAPTER = '1bdf548fd40ee245ae9a32693af04a89e13448bbe5a9224be7d0808403986e87'
PTE = 'af24472d931ee449a571368f5f1f70d61822c25398c6627b8cf26159a649e98c'
AUDIT = 'aaac0fafe216011d5d75386b7216978aaa9b80730cab6c2663df6f29a9356d8c'
LICENSE = '8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74'
RUNTIME_TREE = '23c75bd0aa885c0493e5b0d654c1603c65a56ff6c6b34392f44a6241f8ddbc67'
PTE_NAME = 'Llama-Rakshak-ELA-QLoRA-8da4w-xnnpack.pte'
ADAPTER_DIR = 'Llama-Rakshak-ELA-QLoRA'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage(audit, exported, runtime, destination):
    audit, exported, runtime, destination = map(Path, (audit, exported, runtime, destination))
    original = json.loads((audit/'results.json').read_text())
    export_receipt = json.loads((exported/'results.json').read_text())
    final = json.loads((runtime/'results.json').read_text())
    metrics = json.loads((audit/'metrics.json').read_text())
    export_metrics = json.loads((exported/'ela_export_metrics.json').read_text())
    if (original['status'] != 'ok' or metrics['comparison']['decision'] != 'go'
            or digest(audit/'metrics.json') != AUDIT
            or final['status'] not in ('ok', 'error', 'fail')
            or final['tree_sha256'] != RUNTIME_TREE
            or export_metrics['export']['exit_code'] != 0
            or export_metrics['pte']['sha256'] != PTE):
        raise ValueError('exact completed gate/export/runtime sources required')
    if (digest(audit/ADAPTER_DIR/'adapter_model.safetensors') != ADAPTER
            or digest(exported/PTE_NAME) != PTE or digest(exported/'LICENSE') != LICENSE):
        raise ValueError('selected adapter/PTE/licence hashes changed')
    optional = ['pte_quality.json', 'metrics.json', 'runtime_ops.json', 'runtime_install.json']
    files = [(exported/name, name) for name in (
        PTE_NAME, 'LICENSE', 'NOTICE', 'config.json', 'tokenizer.json', 'ela_8da4w_xnnpack.yaml')]
    files += [(audit/ADAPTER_DIR/name, ADAPTER_DIR+'/'+name) for name in (
        'adapter_model.safetensors', 'adapter_config.json', 'LICENSE', 'NOTICE',
        'README.md', 'chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json')]
    files += [(audit/'metrics.json', 'qlora_audit.json'),
              (audit/'results.json', 'qlora_audit_receipt.json'),
              (exported/'ela_export_metrics.json', 'export_metrics.json'),
              (exported/'results.json', 'export_receipt.json'),
              (runtime/'results.json', 'runtime_receipt.json')]
    files += [(runtime/name, 'runtime_'+name) for name in optional if (runtime/name).is_file()]
    quality = (json.loads((runtime/'pte_quality.json').read_text())
               if final['status'] == 'ok' else None)
    if final['status'] == 'ok' and quality is None:
        raise ValueError('successful runtime must have actual quality report')
    manifest = {'description': 'private experimental artifacts; Built with Llama',
                'adapter_sha256': ADAPTER, 'pte_sha256': PTE,
                'pte_bytes': (exported/PTE_NAME).stat().st_size,
                'sources': {'audit': original, 'export': export_receipt, 'runtime': final},
                'quality_retained': None if quality is None else quality['retained'],
                'runtime_status': final['status'], 'shipment_approved': False,
                'scope': 'rule/contract design and repeated benchmark in-sample; '
                         'not independent validation or Pi/Android performance',
                'hf_publish': 'HF publish pending', 'files': {}}
    for source, name in files:
        if source.is_symlink() or not source.is_file():
            raise ValueError('required allowlisted artifact missing or symlinked')
        manifest['files'][name] = {'sha256': digest(source), 'bytes': source.stat().st_size}
    destination.mkdir(parents=True, exist_ok=False)
    for source, name in files:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (destination/'model_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (destination/'README.md').write_text(
        '# Experimental ELA artifacts\n\nBuilt with Llama.\n\n'
        'Exact selected QLoRA adapter and adapted8da4w PTE; see model_manifest.json.\n'
        'Shipment is not approved. Quality retention may be unmeasured or failed; '
        'inspect the runtime receipt/report. No Pi5/Android claim.\n'
        'Raw contract compliance is low despite higher EN evidence-point pass. '
        'Verifier and deterministic fallback remain necessary.\n'
        'Rule/contract design and repeated benchmark are in-sample. '
        'No independent Indian validation claim. HF publish pending.\n')
    (destination/'dataset-metadata.json').write_text(json.dumps({
        'id': DATASET, 'title': 'Rakshak ELA experimental models B20261101',
        'licenses': [{'name': 'other'}], 'isPrivate': True,
        'description': 'PRIVATE mixed licence. Built with Llama. '
                       'Experimental selected adapter and adapted PTE; shipment not approved.'},
        indent=2)+'\n')
    return manifest



def verify_private(service, request):
    request.owner_slug, request.dataset_slug = DATASET.split('/')
    result = service.get_dataset(request)
    if (result.ref != DATASET or result.owner_ref != request.owner_slug
            or result.is_private is not True or result.current_version_number < 1):
        raise RuntimeError('could not verify private owner/version publication')
    return {'ref': result.ref, 'isPrivate': result.is_private,
            'version': result.current_version_number, 'totalBytes': result.total_bytes}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', required=True, type=Path)
    p.add_argument('--exported', required=True, type=Path)
    p.add_argument('--runtime', required=True, type=Path)
    p.add_argument('--stage', required=True, type=Path)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    stage(args.audit, args.exported, args.runtime, args.stage)
    if not args.publish:
        print('Staged validated experiment artifacts; no upload requested.')
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

    api = KaggleApi()
    api.authenticate()
    response = api.dataset_create_new(str(args.stage), public=False,
                                      convert_to_csv=False, dir_mode='zip')
    if getattr(response, 'error', None) or getattr(response, 'status', None) == 'error':
        raise RuntimeError('private artifact creation rejected; no overwrite/version attempted')
    with api.build_kaggle_client() as client:
        receipt = verify_private(client.datasets.dataset_api_client, ApiGetDatasetRequest())
    (args.stage/'publication_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'Private experimental model bundle verified: {DATASET}; HF publish pending')


if __name__ == '__main__':
    main()
