"""Calibration-only exact-weight backend diagnostic after the failed v1 artifact gate."""

import argparse
import json
from pathlib import Path

import numpy as np
from _common import write_metrics
from scipy.special import expit

from rakshak.eval import data as D
from rakshak.fusion.frozen_bundle import sha256
from rakshak.models.features import featurize
from rakshak.models.onnx_parameters import initializer_parameters

SOURCE = '79e16fe8bdaf722ecd7b202198b9a555201ded44bd90fba13b43161b90ea13f3'
FOLDS = ('lang_shift', 'source_holdout:holdout_mishra_soni_smsphishing',
         'source_holdout:holdout_scamshield')
SOURCE_ONNX = {
    'lang_shift':
        '41d02710857968de5759c662624c0ae5e7bda772b2f6d98cf233b9fabd97c36e',
    'source_holdout:holdout_mishra_soni_smsphishing':
        'e2dede4cddd10f56895c5f538ab6f105654c3110d925b5cc2a6998992b7ba404',
    'source_holdout:holdout_scamshield':
        '70b827ed5842c8f381bbe03be1c8cc7cdc3d09487675bf6bc515258c1270823a',
}
VARIANTS = ((True, 'all'), (True, 'basic'), (True, 'disabled'), (False, 'disabled'))


def run(ctx):
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import numpy_helper

    from rakshak.models.student import ModelSpec, build_model

    argparse.ArgumentParser().parse_args(ctx.argv)
    sources = []
    for path in Path(ctx.input_root).rglob('results.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('tree_sha256') == SOURCE:
            if (receipt['status'] != 'error'
                    or receipt['error'] != 'ValueError: cal-only ONNX logit parity gate failed'):
                raise ValueError('exact parity-failed source required')
            sources.append(path.parent)
    if len(sources) != 1:
        raise ValueError('one exact failed artifact source required')
    source = sources[0]
    root = D.resolve_data_root()
    sms = D.load_sms(root)
    runs = D.load_runs(root, sms, patterns=['lang_shift', 'source_holdout'],
                       labels=sms.label_binary.to_numpy(dtype=int), mask=D.eval_mask(sms))
    runs = {split.name: split for split in runs}
    torch.set_num_threads(2)
    result = {'source_tree': SOURCE, 'scope': 'calibration only; no fitting or external rows',
              'variants': [list(pair) for pair in VARIANTS], 'folds': {}}
    for name in FOLDS:
        split = runs[name]
        path = source/name.replace(':', '_')/'student.onnx'
        digest = sha256(path)
        if digest != SOURCE_ONNX[name]:
            raise ValueError('exact exported source weights hash required')
        graph = onnx.load(str(path))
        model = build_model(ModelSpec('dual', 'base', 'dual')).cpu().eval()
        expected = {key: value.detach().numpy() for key, value in model.state_dict().items()}
        params = initializer_parameters(
            expected, [(entry.name, numpy_helper.to_array(entry))
                       for entry in graph.graph.initializer])
        model.load_state_dict({key: torch.from_numpy(value) for key, value in params.items()},
                              strict=True)
        features = featurize(sms.text.iloc[split.cal].fillna('').tolist())
        block = {'source_onnx_sha256': digest, 'parameters': len(params),
                 'cal_n': len(features), 'variants': {}}
        for mkldnn, optimization in VARIANTS:
            torch.backends.mkldnn.enabled = mkldnn
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.graph_optimization_level = {'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                'disabled': ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[optimization]
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
            logit_max, prob_max, n = 0., 0., 0
            for start in range(0, len(features), 256):
                part = features.take(np.arange(start, min(start+256, len(features))))
                feed = {view: part.bytes_view(view).astype(np.int64) for view in model.views}
                feed['side'] = np.ascontiguousarray(part.side('canon'), dtype=np.float32)
                got = session.run(None, feed)[0]
                with torch.inference_mode():
                    ref = model({view: torch.from_numpy(feed[view]) for view in model.views},
                                 torch.from_numpy(feed['side'])).numpy()
                if not np.isfinite(ref).all() or not np.isfinite(got).all():
                    raise ValueError('nonfinite calibration parity outputs')
                logit_max = max(logit_max, float(np.max(np.abs(got-ref))))
                prob_max = max(prob_max, float(np.max(np.abs(expit(got)-expit(ref)))))
                n += len(part)
            block['variants'][f'mkldnn_{mkldnn}/ort_{optimization}'] = {
                'max_abs_logit_diff': logit_max, 'max_abs_probability_diff': prob_max,
                'n': n, 'pass': n == len(split.cal) and logit_max <= 1e-5}
        if sha256(path) != digest:
            raise ValueError('source weights changed')
        result['folds'][name] = block
        write_metrics('metrics.json', result)
    viable = [key for key in next(iter(result['folds'].values()))['variants']
              if all(block['variants'][key]['pass'] for block in result['folds'].values())]
    result['decision'] = 'go' if viable else 'kill'
    result['first_common_runtime'] = viable[0] if viable else None
    write_metrics('metrics.json', result)
    ctx.results['summary'] = f'cal-only same-weight parity {result["decision"]}: {viable}'
    return True
