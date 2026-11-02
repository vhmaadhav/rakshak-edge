"""Fixed ByT5 encoder teacher for URLs; all model compute belongs on Kaggle."""

import numpy as np
import torch
from torch import nn

MODEL = 'google/byt5-small'
REVISION = '68377bdc18a2ffec8a0533fef03b1c513a4dd49d'
CARD_SHA = '6daea4f59e1a86e7db0a44c488651130cfd39c3b92cf63eead46db865bdabaed'
CONFIG_SHA = '7845fb21b320f3fa05392ce151143502cf08729c8c89732da3293615885d3e83'
WEIGHT_SHA = '5c5aaf56299d6f2d4eaadad550a40765198828ead4d74f0a15f91cbe0961931a'


def encode_urls(urls):
    ids = np.zeros((len(urls), 201), dtype=np.int64)
    for index, url in enumerate(urls):
        raw = str(url).encode('utf-8')[:200]
        ids[index, :len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)+3
        ids[index, len(raw)] = 1
    return ids


class URLByT5(nn.Module):
    def __init__(self, config, n_side):
        super().__init__()
        from transformers import T5EncoderModel

        self.encoder = T5EncoderModel(config)
        self.head = nn.Sequential(nn.Linear(config.d_model+n_side, 128), nn.ReLU(),
                                  nn.Dropout(.1), nn.Linear(128, 1))

    def load_encoder(self, state):
        expected = self.encoder.state_dict()
        if any(key not in state or state[key].shape != value.shape
               or state[key].dtype != value.dtype or not torch.isfinite(state[key]).all()
               for key, value in expected.items()):
            raise ValueError('every exact encoder/shared pretrained parameter required')
        extra = set(state)-set(expected)
        if any(not (key.startswith('decoder.') or key == 'lm_head.weight') for key in extra):
            raise ValueError('unexpected pretrained nondecoder parameter')
        self.encoder.load_state_dict({key: state[key] for key in expected}, strict=True)

    def forward(self, ids, side):
        valid = ids != 0
        hidden = self.encoder(input_ids=ids, attention_mask=valid.long()).last_hidden_state.float()
        pooled = (hidden*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
        return self.head(torch.cat([pooled, side.float()], dim=1)).squeeze(1)


@torch.inference_mode()
def predict(model, ids, side, rows, *, batch_size=64, device='cuda'):
    model.eval()
    scores = np.zeros(len(rows), dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        part = rows[start:start+batch_size]
        scores[start:start+len(part)] = model(
            torch.as_tensor(ids[part], device=device),
            torch.as_tensor(side[part], device=device)).double().cpu().numpy()
    if not np.isfinite(scores).all():
        raise ValueError('finite teacher logits required')
    return scores
