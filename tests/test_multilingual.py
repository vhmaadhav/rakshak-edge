import io

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from rakshak.models.features import featurize  # noqa: E402
from rakshak.models.multilingual import MultilingualCNN, forward  # noqa: E402
from rakshak.models.student import distillation_loss  # noqa: E402


def test_masked_pool_unknown_gradients_and_roundtrip():
    torch.manual_seed(2)
    table = np.random.default_rng(1).normal(size=(10, 64)).astype(np.float32)
    table[0] = 0
    model = MultilingualCNN(table)
    ids = torch.tensor([[2, 3, 0], [0, 0, 0], [1, 0, 0]])
    pooled = model.pool(ids)
    torch.testing.assert_close(pooled[0, :64], torch.tensor(table[[2, 3]].mean(0)))
    torch.testing.assert_close(pooled[0, 64:], torch.tensor(table[[2, 3]].max(0)))
    assert torch.isfinite(pooled).all() and not pooled[1].any()
    pooled.sum().backward()
    assert not model.embedding.weight.grad[0].any()
    assert model.embedding.weight.grad[1].abs().sum() > 0
    feats = featurize(['synthetic bank message', 'test only'], max_len=16)
    tokens = np.asarray([[2, 3, 0], [1, 0, 0]])
    model.eval()
    ref = forward(model, feats, tokens, np.arange(2), 'cpu')
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = MultilingualCNN(table).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(ref, forward(restored, feats, tokens, np.arange(2), 'cpu'),
                               rtol=0, atol=0)


def test_no_teacher_full_bce_and_kd_uses_finite_targets():
    logits = torch.tensor([.1, -.4], requires_grad=True)
    labels = torch.tensor([1., 0.])
    weight = torch.tensor(2.)
    loss = distillation_loss(logits, labels, torch.full((2, 2), float('nan')),
                             pos_weight=weight, alpha=.7, temp=2.)
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=weight))
    kd = distillation_loss(logits, labels, torch.tensor([[0., -1.], [0., 1.]]),
                           pos_weight=weight, alpha=.7, temp=2.)
    assert not torch.isclose(kd, loss)
    kd.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_synthetic_cuda_training_matches_byte_control_and_reloads_hybrid(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('synthetic CUDA unit exercised on Kaggle')
    # Exact comparison needs deterministic CUDA convolution gradients in this unit only.
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', True)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', False)
    from rakshak.models.multilingual import predict, train
    from rakshak.models.student import ModelSpec, TrainConfig, train_student

    feats = featurize(['synthetic alpha', 'synthetic beta', 'test gamma', 'test delta',
                       'cal epsilon', 'cal zeta'], max_len=16)
    positions, cal = np.arange(4), np.arange(4, 6)
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    tokens = np.asarray([[2, 3, 0]]*6)
    cfg = TrainConfig(seed=7, epochs=1)
    expected, reference = train_student(ModelSpec('dual', 'base', 'dual'), feats.take(positions),
                                        labels[positions], feats.take(cal), labels[cal], cfg,
                                        device='cuda')
    model, history = train(feats, tokens, positions, labels, None, cal, cfg, None)
    assert history['teacher_rows'] == 0
    assert history['best_cal_pr_auc'] == reference['best_cal_pr_auc']
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    table = np.random.default_rng(8).normal(size=(4, 64)).astype(np.float32)
    table[0] = 0
    teacher = np.asarray([[0., .5], [0., -.5]]*2)
    hybrid, history = train(feats, tokens, positions, labels, teacher, cal, cfg, table)
    assert history['teacher_rows'] == 4
    scores = predict(hybrid.cuda(), feats, tokens, cal)
    assert scores.shape == (2,) and np.isfinite(scores).all()
    restored = MultilingualCNN(table)
    restored.load_state_dict(hybrid.cpu().state_dict())
    np.testing.assert_array_equal(scores, predict(restored.cuda(), feats, tokens, cal))
