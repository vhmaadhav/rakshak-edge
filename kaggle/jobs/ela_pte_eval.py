"""Runtime-only recovery on the existing adapted PTE; no export, fitting or quantisation."""

import argparse
import json
import tempfile
from pathlib import Path

from _common import write_metrics
from ela_export import PTE_NAME, child, hash_file, sources

PTE_TREE = '20c6e21feefa82e003b07b5cfe950a8e9bac64c4db0bae9887c3343f76b9719d'
PTE_HASH = 'af24472d931ee449a571368f5f1f70d61822c25398c6627b8cf26159a649e98c'


def run(ctx):
    import llama_export as export

    argparse.ArgumentParser().parse_args(ctx.argv)
    _, tuned = sources(ctx.input_root)  # final GO/FP16/adapter hashes must still match
    matches = []
    for path in Path(ctx.input_root).rglob(PTE_NAME):
        receipt = path.with_name('results.json')
        if receipt.is_file() and json.loads(receipt.read_text()).get('tree_sha256') == PTE_TREE:
            if hash_file(path) != PTE_HASH:
                raise ValueError('original adapted PTE changed')
            matches.append(path)
    if len(matches) != 1:
        raise ValueError('one exact original PTE source required')
    pte = matches[0]
    steps = []
    try:
        export._install(ctx, steps)
    finally:
        write_metrics('runtime_install.json', {'steps': steps, 'versions': export._versions()})
    scratch = Path(tempfile.mkdtemp(prefix='rakshak-pte-runtime-'))
    child(ctx, 'quality', (pte, pte.with_name('tokenizer.json'), tuned, ctx.work), scratch,
          timeout=30000)
    report = json.loads((ctx.work/'pte_quality.json').read_text())
    write_metrics('metrics.json', {'pte_source_tree': PTE_TREE, 'pte_sha256': PTE_HASH,
                                   'pte_bytes': pte.stat().st_size, 'reexported': False,
                                   'runtime_ops': json.loads(
                                       (ctx.work/'runtime_ops.json').read_text()),
                                   'quality': report, 'versions': export._versions()})
    ctx.results['summary'] = (f"same adapted PTE; quality retained {report['retained']}; "
                              'Kaggle x86 only')
    return True
