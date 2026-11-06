"""Train-only conditional source-adversary control; three seeds, six frozen splits, Kaggle T4."""

from sms_source_balance import run as source_controls

from rakshak.models.source_adversary import verify_torch_invariants


def run(ctx):
    verify_torch_invariants("cuda")
    ctx.log("conditional source-adversary synthetic gradient/RNG/zero-coefficient checks passed")
    return source_controls(ctx, arms=("erm", "adversarial"), method_reference={
        "gradient_reversal": "https://www.jmlr.org/beta/papers/v17/15-239.html",
        "conditioning": ("https://proceedings.neurips.cc/paper/2018/hash/"
                         "ab88b15733f543179858600245108dd8-Abstract.html"),
        "label_shift_warning": "https://proceedings.mlr.press/v97/zhao19a.html",
        "scope": "source-only regularization; derived experiment, not exact DANN/CDAN",
    })
