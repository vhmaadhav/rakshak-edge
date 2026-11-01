"""Export/runtime helpers; no torch or ExecuTorch imports until the Kaggle runtime call."""

import time

from rakshak.ela_eval import json_stop_ready
from rakshak.eval.point_comparison import paired_point_pass
from rakshak.llm.ela_comparison import compare


def generate_pte(runner, prompt, max_new_tokens, *, config_factory=None):
    if config_factory is None:
        from executorch.extension.llm.runner import GenerationConfig
        config_factory = GenerationConfig
    runner.reset()  # no KV or stop-state carryover between independent messages
    pieces, stats = [], {}

    def token(piece):
        pieces.append(piece)
        if json_stop_ready(''.join(pieces)):
            runner.stop()

    def on_stats(value):
        stats.update(num_generated_tokens=int(value.num_generated_tokens),
                     num_prompt_tokens=int(value.num_prompt_tokens))

    cfg = config_factory(echo=False, max_new_tokens=max_new_tokens, temperature=0.0,
                         num_bos=0, num_eos=0)
    start = time.perf_counter()
    runner.generate(prompt, cfg, token, on_stats)
    wall = time.perf_counter() - start
    if 'num_generated_tokens' not in stats:
        raise ValueError('runner did not report token counts; refuse character-based estimate')
    return ''.join(pieces), stats['num_generated_tokens'], wall, stats


def retention_report(fp16, pte):
    # Rank/freeform cannot be measured by this slot-only runtime comparison.
    arms = ('slot', 'slot_hinglish')
    before = [r for r in fp16 if r['arm'] in arms]
    comparison = compare(before, pte)
    paired = {}
    for arm in arms:
        clean = [r for r in pte if r['arm'] == arm and not r['injected']]
        base = {str(r['sample_id']): r for r in before
                if r['arm'] == arm and not r['injected']}
        paired[arm] = paired_point_pass(
            [base[str(r['sample_id'])]['points_pass_n'] for r in clean],
            [r['points_pass_n'] for r in clean], [r['points_n'] for r in clean],
            [r['campaign_id'] for r in clean])
    margins = all(row['ci95'][0] is not None and row['ci95'][0] >= -.05
                  for row in paired.values())
    comparison['injection_vs_base'] = {'slot': comparison['injection_vs_base']['slot']}
    return {'retained': comparison['decision'] == 'go' and margins,
            'paired_point_pass': paired, 'absolute_and_safety': comparison,
            'margin': 'both paired PTE-minus-FP16 lower95% bounds >= -0.05',
            'unmeasured': ['rank token-likelihood scoring', 'freeform'],
            'scope': 'same frozen sample; in-sample for rule/contract design; Kaggle x86 CPU'}


def register_runtime_ops(*, roots=None, load_library=None, get_registered=None):
    """Load wheel-provided AOT libraries before creating the LLM runner; no model changes."""
    import hashlib
    from pathlib import Path

    patterns = ('*custom_ops_aot_lib.*', '*quantized_ops_aot_lib.*')
    if roots is None:
        import importlib.util
        roots = []
        for package, pattern in zip(('executorch.extension.llm.custom_ops',
                                     'executorch.kernels.quantized'), patterns, strict=True):
            spec = importlib.util.find_spec(package)
            paths = [] if spec is None else list(spec.submodule_search_locations or ())
            matches = [p for folder in paths for p in Path(folder).glob(pattern) if p.is_file()]
            if len(matches) != 1:
                raise ValueError(f'expected one official wheel AOT library for {package}')
            roots.append(matches[0].parent)
    if load_library is None:
        import torch
        load_library = torch.ops.load_library
    if get_registered is None:
        from executorch.extension.pybindings.portable_lib import _get_operator_names
        get_registered = _get_operator_names
    loaded = []
    for root, pattern in zip(roots, patterns, strict=True):
        matches = [p for p in Path(root).glob(pattern) if p.is_file()]
        if len(matches) != 1:
            raise ValueError(f'expected one official wheel AOT library for {pattern}')
        path = matches[0]
        load_library(str(path))
        loaded.append({'name': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    names = [str(v) for v in get_registered()]
    required = ('llama::custom_sdpa', 'llama::update_cache',
                'quantized_decomposed::embedding_4bit')
    present = {op: any(op in name for name in names) for op in required}
    if not all(present.values()):
        raise ValueError(f'required runtime kernels remain unavailable: {present}')
    return {'loaded_libraries': loaded, 'required_operators': present, 'registry_n': len(names)}
