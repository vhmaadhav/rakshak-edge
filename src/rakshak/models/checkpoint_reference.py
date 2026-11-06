"""CPU checkpoint-reference inference; import does not load Torch or a model."""

import numpy as np
from scipy.special import expit


def reference_logits(model, features):
    import torch

    model = model.cpu().eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            batch = features.take(np.arange(start, min(start+256, len(features))))
            views = {view: torch.from_numpy(batch.bytes_view(view).astype(np.int64))
                     for view in model.views}
            side = torch.from_numpy(np.ascontiguousarray(batch.side('canon'), dtype=np.float32))
            value = model(views, side).detach().cpu().numpy()
            if value.shape != (len(batch),) or not np.isfinite(value).all():
                raise ValueError('invalid checkpoint reference logits')
            parts.append(value)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def checkpoint_scores(model, features):
    return expit(reference_logits(model, features))
