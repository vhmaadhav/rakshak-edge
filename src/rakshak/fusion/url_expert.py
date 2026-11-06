"""Frozen URL ONNX expert with offline features and explicit per-message max aggregation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from rakshak.canonicalize import canonicalize
from rakshak.models.encoding import apply_standardizer, encode_batch
from rakshak.url_features import url_features

URL_TREE = '5131673567fcf4d3be5eda9f43d28194632d2cce7e2d0940fc06a400eede36db'
URL_HASH = '1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7'
SCALER_HASH = 'c6b3fc968b03339a33052fe814a9bb851d3035cb302b06458955c2a5e5b6501e'


def frozen_url_folder(root):
    matches = []
    for path in Path(root).rglob('best_url_student.onnx'):
        result = path.with_name('results.json')
        if not result.is_file():
            continue
        result = json.loads(result.read_text())
        if result.get('tree_sha256') != URL_TREE:
            continue
        if result.get('status') != 'ok':
            raise ValueError('URL source kernel is incomplete')
        if hashlib.sha256(path.read_bytes()).hexdigest() != URL_HASH:
            raise ValueError('frozen URL weights hash changed')
        scaler = path.with_name('url_side_scaler.json')
        if hashlib.sha256(scaler.read_bytes()).hexdigest() != SCALER_HASH:
            raise ValueError('frozen URL feature scaler changed')
        matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError('one successful hash-pinned URL source required')
    return matches[0]


class FrozenURLExpert:
    def __init__(self, folder):
        import onnxruntime as ort

        self.scaler = json.loads((Path(folder) / 'url_side_scaler.json').read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(folder) / 'best_url_student.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        inputs = self.session.get_inputs()
        if [v.name for v in inputs] != ['raw', 'side'] or inputs[0].shape[1] != 200:
            raise ValueError('unexpected frozen URL input contract')
        self.cache = {}

    def score(self, texts):
        urls = [canonicalize(str(text)).urls for text in texts]
        missing = sorted({url for values in urls for url in values} - self.cache.keys())
        for start in range(0, len(missing), 256):
            values = missing[start:start + 256]
            features = [url_features(url) for url in values]
            matrix = np.asarray([[row[name] for name in self.scaler['columns']]
                                 for row in features], dtype=np.float32)
            side = apply_standardizer(matrix, np.asarray(self.scaler['mean']),
                                      np.asarray(self.scaler['std']))
            logits = self.session.run(None, {'raw': encode_batch(values, 200).astype(np.int64),
                                             'side': side})[0].reshape(-1)
            scores = expit(logits)
            if len(scores) != len(values) or not np.isfinite(scores).all():
                raise ValueError('invalid frozen URL output')
            self.cache.update(zip(values, scores.tolist(), strict=True))
        return (np.asarray([max(self.cache[url] for url in values) if values else np.nan
                            for values in urls]), np.asarray([bool(values) for values in urls]))
