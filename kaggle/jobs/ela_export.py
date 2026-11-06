"""Conditional adapter-merged 8da4w XNNPACK export and frozen slot quality. Built with Llama."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import write_metrics

from rakshak.llm.llama_notice import write_model_notice

AUDIT_TREE = '1c08f453b8b906cb67faaea2979259f79fa1391ba1ec27d8c976892d6f1c797b'
AUDIT_HASH = 'aaac0fafe216011d5d75386b7216978aaa9b80730cab6c2663df6f29a9356d8c'
TUNED_TREE = 'd8120f4e72e43bb1b11ef28af26e782d3f71df0b1e8eac60df3264fb0dac3de6'
GENERATIONS_HASH = '864aea4be4f5fbafe0ea227891e29b62a426c9d8a1229f713493a807e092f6ca'
ADAPTER_HASH = '1bdf548fd40ee245ae9a32693af04a89e13448bbe5a9224be7d0808403986e87'
PTE_NAME = 'Llama-Rakshak-ELA-QLoRA-8da4w-xnnpack.pte'


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sources(root):
    matches = {}
    for path in Path(root).rglob('results.json'):
        result = json.loads(path.read_text())
        tree = result.get('tree_sha256')
        if tree in (AUDIT_TREE, TUNED_TREE):
            if result.get('status') != 'ok' or tree in matches:
                raise ValueError('one successful source per exact tree required')
            matches[tree] = path.parent
    if set(matches) != {AUDIT_TREE, TUNED_TREE}:
        raise ValueError('successful final GO audit and inference sources required')
    audit, tuned = matches[AUDIT_TREE], matches[TUNED_TREE]
    if hash_file(audit / 'metrics.json') != AUDIT_HASH:
        raise ValueError('final gate audit hash changed')
    decision = json.loads((audit / 'metrics.json').read_text())
    if decision['comparison']['decision'] != 'go':
        raise ValueError('export is forbidden without final recorded GO')
    adapter = audit / 'Llama-Rakshak-ELA-QLoRA'
    if hash_file(adapter / 'adapter_model.safetensors') != ADAPTER_HASH:
        raise ValueError('selected adapter hash changed')
    if hash_file(tuned / 'generations.jsonl') != GENERATIONS_HASH:
        raise ValueError('FP16 comparison source changed')
    return adapter, tuned


def merge(adapter, scratch):
    # This fresh process completes BEFORE torch is replaced by the export runtime.
    import importlib.metadata
    try:
        importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rakshak.llama_source import (
        check_meta_shapes,
        hf_to_meta_state_dict,
        params_from_hf_config,
        resolve_model_dir,
    )

    scratch = Path(scratch)
    base, source = resolve_model_dir(lambda: None)
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    # Transformers5 normalises rope fields; converter expects the pinned source schema.
    original_config = json.loads((base/'config.json').read_text())
    params = params_from_hf_config(original_config)
    state = hf_to_meta_state_dict(model.state_dict(), params,
                                  tie_word_embeddings=model.config.tie_word_embeddings)
    shapes = check_meta_shapes(state, params)
    torch.save(state, scratch / 'merged_meta.pth')
    (scratch / 'params.json').write_text(json.dumps(params))
    shutil.copyfile(base/'config.json', scratch/'config.json')
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(scratch / 'tokenizer')
    (scratch / 'merge.json').write_text(json.dumps({
        'source': source, 'adapter_sha256': hash_file(Path(adapter) / 'adapter_model.safetensors'),
        'safe_merge': True, 'dtype': 'float16', 'shapes': shapes,
        'torch': torch.__version__,
        'merged_checkpoint_sha256': hash_file(scratch/'merged_meta.pth')}))


def quality(pte, tokenizer, tuned_folder, output):
    from rakshak.llm.ela_export import register_runtime_ops

    registration = register_runtime_ops()
    (Path(output)/'runtime_ops.json').write_text(json.dumps(registration, indent=2))
    from executorch.extension.llm.runner import TextLLMRunner

    from rakshak.ela_eval import (
        append_payload,
        ela_prompt,
        injection_case,
        max_new_tokens_for,
        package_for_sms,
        score_advisory,
    )
    from rakshak.eval import data as D
    from rakshak.llm.ela_export import generate_pte, retention_report
    from rakshak.verifier import render_advisory

    tuned_folder, output = Path(tuned_folder), Path(output)
    source = [json.loads(line) for line in
              (tuned_folder/'generations.jsonl').read_text().splitlines()]
    fp16 = [r for r in source if r['arm'] in ('slot', 'slot_hinglish')]
    manifest = json.loads((tuned_folder/'sample_manifest.json').read_text())
    sms = D.load_sms(D.resolve_data_root()).set_index('id')
    payloads = {row['sample_id']: injection_case(i) for i, row in enumerate(
        r for r in manifest['rows'] if r['injected'])}
    runner = TextLLMRunner(pte, tokenizer)
    rows = []
    with (output/'pte_generations.jsonl').open('w') as handle:
        for ref in fp16:
            text = sms.loc[str(ref['sample_id']), 'text']
            if ref['injected']:
                family, payload = payloads[str(ref['sample_id'])]
                if family != ref['family']:
                    raise ValueError('frozen injection family changed')
                text = append_payload(text, payload)
            _, package = package_for_sms(text)
            lang = 'hinglish' if ref['arm'] == 'slot_hinglish' else 'en'
            raw, tokens, wall, stats = generate_pte(
                runner, ela_prompt(package, lang=lang), max_new_tokens_for(package))
            scored = score_advisory(raw, package, lang)
            if scored['points_n'] != ref['points_n']:
                raise ValueError('frozen evidence point denominator changed')
            row = {**scored, **{k: ref[k] for k in
                   ('arm', 'sample_id', 'campaign_id', 'side', 'injected', 'family')},
                   'output': raw, 'advisory': render_advisory(raw, package, lang),
                   'new_tokens': tokens, 'latency_s': wall, 'runner_stats': stats}
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            handle.flush()
    report = retention_report(source, rows)
    (output/'pte_quality.json').write_text(json.dumps(report, indent=2))


def child(ctx, mode, args, scratch, *, timeout):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]/'src')
    env['OMP_NUM_THREADS'], env['MKL_NUM_THREADS'] = '2', '2'
    log = ctx.work / f'{mode}.log'
    with log.open('w') as handle:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), mode, *map(str, args)],
                       cwd=scratch, env=env, stdout=handle, stderr=subprocess.STDOUT,
                       timeout=timeout, check=True)


def run(ctx):
    import llama_export as export

    argparse.ArgumentParser().parse_args(ctx.argv)
    adapter, tuned = sources(ctx.input_root)
    scratch = Path(tempfile.mkdtemp(prefix='rakshak-ela-export-'))
    resources = export._resources(ctx)
    if resources['ram_gb'] < 20 or resources['free_disk_gb'] < 12:
        raise RuntimeError(f'insufficient export resources: {resources}')
    report = {'source_audit_tree': AUDIT_TREE, 'source_inference_tree': TUNED_TREE,
              'adapter_sha256': ADAPTER_HASH, 'resources': resources, 'status': 'running',
              'recipe': 'ExecuTorch1.5.1 8da4w XNNPACK GS32 embedding4,32; merged selected adapter',
              'rank': 'unmeasured: runner lacks public token-likelihood API'}
    write_metrics('ela_export_metrics.json', report)
    child(ctx, 'merge', (adapter, scratch), scratch, timeout=3600)
    report['merge'] = json.loads((scratch/'merge.json').read_text())
    write_metrics('ela_export_metrics.json', report)
    steps = []
    try:
        export._install(ctx, steps)
    finally:
        report['install'], report['versions'] = steps, export._versions()
        write_metrics('ela_export_metrics.json', report)
    folder = scratch/'export'
    folder.mkdir()
    config = ctx.work/'ela_8da4w_xnnpack.yaml'
    config.write_text(export._yaml(scratch/'merged_meta.pth', scratch/'params.json').replace(
        export.PTE_NAME, PTE_NAME))
    rc, wall, peak = export._run_export(ctx, folder, config, ctx.work/'ela_export.log')
    report['export'] = {'exit_code': rc, 'wall_s': wall, 'peak_rss_kb': peak,
                        'delegation_lines': export._delegation_lines(ctx.work/'ela_export.log')}
    write_metrics('ela_export_metrics.json', report)
    if rc != 0:
        raise RuntimeError(f'8da4w export failed with exit {rc}; see private export log')
    files = list(folder.rglob(PTE_NAME))
    if len(files) != 1:
        raise ValueError('expected one actually exported adapted PTE')
    target = ctx.work/PTE_NAME
    shutil.copyfile(files[0], target)
    shutil.copyfile(scratch/'tokenizer/tokenizer.json', ctx.work/'tokenizer.json')
    shutil.copyfile(scratch/'config.json', ctx.work/'config.json')
    report['pte'] = {'bytes': target.stat().st_size, 'sha256': hash_file(target)}
    report['licence'] = write_model_notice(ctx.work)
    with (ctx.work/'README.md').open('a') as handle:
        handle.write('\nThis package also contains the selected adapter merged and exported as '
                     + PTE_NAME + '. See ela_export_metrics.json for quality and provenance.\n')
    write_metrics('ela_export_metrics.json', report)
    # Fresh process uses installed export Torch, never the original parent interpreter's DLLs.
    child(ctx, 'quality', (target, ctx.work/'tokenizer.json', tuned, ctx.work), scratch,
          timeout=30000)
    report['quality'] = json.loads((ctx.work/'pte_quality.json').read_text())
    report['status'] = 'complete'
    write_metrics('ela_export_metrics.json', report)
    ctx.results['summary'] = (f"adapted PTE {target.stat().st_size} bytes; "
                              f"quality retained {report['quality']['retained']}; Kaggle x86 only")
    return True


if __name__ == '__main__':
    if sys.argv[1] == 'merge':
        merge(*sys.argv[2:])
    elif sys.argv[1] == 'quality':
        quality(*sys.argv[2:])
    else:
        raise SystemExit('only merge/quality subprocess modes are supported')
