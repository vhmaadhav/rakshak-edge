"""Small synthetic checks; no model fitting or benchmark data."""

import numpy as np
import pytest

from rakshak.models.source_balance import SourceObjective, source_probabilities


def test_balance_gives_equal_source_mass_without_using_labels():
    _, _, prob = source_probabilities(["large"] * 9 + ["small"])
    assert prob[:9].sum() == pytest.approx(0.5)
    assert prob[9] == pytest.approx(0.5)
    pytest.importorskip("torch")
    objective = SourceObjective(["large"] * 9 + ["small"], 10, "balanced")
    assert objective.prob[:9].sum() == pytest.approx(0.5)
    assert objective.prob[9] == pytest.approx(0.5)
    assert np.array_equal(objective.order(np.random.default_rng(4)),
                          objective.order(np.random.default_rng(4)))
    assert len(objective.order(np.random.default_rng(0))) == 10
    with pytest.raises(ValueError, match="aligned"):
        SourceObjective(None, 10, "dro")


def test_dro_weights_harder_source_and_gradients_reach_losses():
    torch = pytest.importorskip("torch")
    obj = SourceObjective(["easy", "easy", "hard"], 3, "dro", eta=1.0)
    losses = torch.tensor([0.1, 0.1, 2.0], requires_grad=True)
    loss = obj.reduce(losses, np.arange(3))
    assert obj.weights()["hard"] > obj.weights()["easy"]
    loss.backward()
    assert bool((losses.grad > 0).all())
    assert obj.reduce(torch.tensor([0.5]), np.array([0])).item() == pytest.approx(0.5)


def test_per_row_kd_loss_preserves_original_mean_and_masks_unlabelled_rows():
    torch = pytest.importorskip("torch")
    from rakshak.models.student import distillation_loss
    logits = torch.tensor([0.3, -0.2, 0.7])
    labels = torch.tensor([1.0, 0.0, float("nan")])
    teacher = torch.tensor([[0.0, 0.5], [float("nan"), float("nan")], [0.0, 1.0]])
    kwargs = {"pos_weight": torch.tensor(2.0), "alpha": 0.7, "temp": 2.0}
    per_row = distillation_loss(logits, labels, teacher, reduction="none", **kwargs)
    mean = distillation_loss(logits, labels, teacher, **kwargs)
    assert per_row.mean().item() == pytest.approx(mean.item())
    obj = SourceObjective(None, 3)
    assert obj.reduce(per_row, np.arange(3)).item() == pytest.approx(mean.item())


def test_paired_mean_f1_uses_campaigns_and_all_seeds():
    from rakshak.eval.source_comparison import paired_seed_f1

    y = [1, 1, 0, 0]
    a = [[1, 1, 0, 0]] * 3
    b = [[1, 0, 0, 0]] * 3
    result = paired_seed_f1(y, a, b, ["x", "x", "z", "z"], n_boot=30)
    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(2 / 3)
    assert result["training_seeds"] == 3
    same = paired_seed_f1(y, a, a, ["x", "x", "z", "z"], n_boot=30)
    assert same["delta_ci95"] == [0, 0]


def test_inverse_source_frequencies_sum_to_equal_group_mass():
    _, _, probability = source_probabilities(["large"] * 9 + ["small"])
    assert probability[:9].sum() == pytest.approx(0.5)
    assert probability[9] == pytest.approx(0.5)


def test_mean_f1_is_mean_of_seed_scores_not_f1_of_a_voted_prediction():
    from rakshak.eval.source_comparison import paired_seed_f1

    y = [1, 1, 0, 0]
    # Seed F1 values: 1, 2/3, 4/5. Majority-vote prediction would be perfect.
    predictions = [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]]
    result = paired_seed_f1(y, predictions, predictions, ["a", "b", "c", "d"], n_boot=30)
    assert result["a"] == pytest.approx((1 + 2 / 3 + 4 / 5) / 3)
    assert result["a"] < 1
    assert result["delta_ci95"] == [0, 0]
