"""Hash-pinned source-control score inputs for private Kaggle audits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_TREE = '62a788d27aa9db20b84964f00b8208d3025c3be8f5128352d5c76c53b5ae589e'
METRICS_HASH = 'db3811b2ede62d577e3dad40684276d0ec0580d231d0505cafe3d5d54fb46538'


def source_folder(root):
    matches = []
    for path in Path(root).rglob('results.json'):
        result = json.loads(path.read_text())
        if result.get('tree_sha256') == SOURCE_TREE:
            if result.get('status') != 'ok' or result.get('failures'):
                raise ValueError('completed clean source kernel required')
            digest = hashlib.sha256(path.with_name('metrics.json').read_bytes()).hexdigest()
            if digest != METRICS_HASH:
                raise ValueError('source statistics hash changed')
            matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError('one successful frozen source kernel required')
    return matches[0]


def aligned_scores(path, frame):
    scores = pd.read_parquet(path)
    scores.id = scores.id.astype(str)
    if (not scores.id.is_unique or set(scores.id) != set(frame.id.astype(str))
            or not np.isfinite(scores.score).all()):
        raise ValueError('saved scores must match exactly the eligible endpoint population')
    values = scores.set_index('id').score.reindex(frame.id.astype(str)).to_numpy()
    if ((values < 0) | (values > 1)).any():
        raise ValueError('saved scores must be probabilities')
    return values

