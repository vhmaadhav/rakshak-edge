"""Compact subword pooling beside the unchanged dual-byte encoders. Kaggle compute only."""

import time

import numpy as np
import torch
from torch import nn

from rakshak.eval.metrics import pr_auc
from rakshak.models.charcnn import BASE, ByteConv, CharCNN
from rakshak.models.encoding import N_RULES
from rakshak.models.student import _state_on_cpu, distillation_loss


class MultilingualCNN(nn.Module):
    def __init__(self, table):
        super().__init__()
        table = torch.as_tensor(table, dtype=torch.float32)
        if table.ndim != 2 or table.shape[1] != 64 or torch.count_nonzero(table[0]):
            raise ValueError('expected 64-dimensional table with zero PAD')
        self.encoders = nn.ModuleDict({view: ByteConv(BASE) for view in ('raw', 'canon')})
        self.embedding = nn.Embedding.from_pretrained(table.clone(), freeze=False, padding_idx=0)
        width = sum(enc.out_dim for enc in self.encoders.values()) + N_RULES + 128
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(width, 64), nn.ReLU(),
                                  nn.Dropout(.1), nn.Linear(64, 1))

    def pool(self, token_ids):
        valid = token_ids != 0
        embedded = self.embedding(token_ids)
        mean = (embedded*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
        maximum = embedded.masked_fill(~valid.unsqueeze(-1), float('-inf')).max(1).values
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return torch.cat([mean, maximum], dim=1)

    def forward(self, views, side, token_ids):
        parts = [self.encoders[view](views[view]) for view in ('raw', 'canon')]
        return self.head(torch.cat([*parts, side, self.pool(token_ids)], dim=1)).squeeze(1)


def forward(model, feats, tokens, indices, device):
    views = {view: torch.as_tensor(feats.bytes_view(view)[indices].astype(np.int64), device=device)
             for view in ('raw', 'canon')}
    side = torch.as_tensor(feats.side('dual')[indices], device=device)
    if isinstance(model, MultilingualCNN):
        return model(views, side, torch.as_tensor(tokens[indices], device=device))
    return model(views, side)


@torch.inference_mode()
def predict(model, feats, tokens, indices, device='cuda', batch_size=512):
    model.eval()
    result = np.zeros(len(indices), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        part = indices[start:start+batch_size]
        result[start:start+len(part)] = forward(model, feats, tokens, part, device).sigmoid().cpu()
    if not np.isfinite(result).all():
        raise ValueError('non-finite multilingual scores')
    return result


def train(feats, tokens, positions, labels, teacher, cal, cfg, table, log=lambda msg: None):
    """Identical ERM minibatches/BCE/AdamW/PR stopping across all three arms."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = (CharCNN(BASE, views=('raw', 'canon'), n_side=N_RULES) if table is None
             else MultilingualCNN(table)).cuda()
    y = np.asarray(labels[positions], dtype=np.float64)
    if set(np.unique(y)) != {0, 1} or len(set(positions) & set(cal)):
        raise ValueError('train labels or train/cal membership invalid')
    if teacher is None:
        teacher = np.full((len(y), 2), np.nan)
    elif teacher.shape != (len(y), 2) or not np.isfinite(teacher).all():
        raise ValueError('teacher must cover exact original train rows')
    weight = torch.tensor((y == 0).sum()/(y == 1).sum(), dtype=torch.float32, device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_pr, best_state, best_epoch, stale, history = -np.inf, None, -1, 0, []
    started = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(positions))
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start:start+cfg.batch_size]
            logits = forward(model, feats, tokens, positions[idx], 'cuda')
            loss = distillation_loss(
                logits, torch.as_tensor(y[idx], dtype=torch.float32, device='cuda'),
                torch.as_tensor(teacher[idx], dtype=torch.float32, device='cuda'),
                pos_weight=weight, alpha=cfg.kd_alpha, temp=cfg.kd_temp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        score = float(pr_auc(labels[cal], predict(model, feats, tokens, cal)))
        history.append({'epoch': epoch, 'cal_pr_auc': score, 'train_loss': float(np.mean(losses))})
        log(f'epoch {epoch}: cal PR-AUC {score:.6f}')
        if np.isfinite(score) and score > best_pr:
            best_pr, best_state, best_epoch, stale = score, _state_on_cpu(model), epoch, 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise ValueError('no finite calibration checkpoint')
    model.load_state_dict(best_state)
    return model.cpu(), {'history': history, 'best_epoch': best_epoch, 'best_cal_pr_auc': best_pr,
                         'seconds': time.perf_counter()-started, 'rows': len(y),
                         'teacher_rows': int(np.isfinite(teacher).all(axis=1).sum())}
