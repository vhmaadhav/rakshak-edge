# Conditional source-adversary experiment — 2026-11-01

This new experiment is on `ci/source-adversary-20261101`, based on C guard milestone
`56892e7` / PR #3. It does not add changes to PR #3 or PR #2. Benefit/runtime on Rakshak is
**Verified on the named frozen splits**; recorded gate is go (see SOURCE_ADVERSARY_RESULTS.md). No deployment or independent held-out claim.

## Hypothesis and frozen gates

A source-prediction adversary on learned byte-CNN features, conditioned on the true training
label, can reduce source shortcuts without forcing different class proportions to match.
It might improve ScamShield transfer where balanced sampling and DRO failed. This is a
preplanned method family; its regularization parameters are fixed before this new run.

Go: ScamShield intervention-minus-ERM mean F1 has a strictly positive paired campaign-bootstrap
95% lower bound, and no other source fold loses more than the ERM F1 CI half-width. Otherwise
kill the intervention. All three training seeds1/2/3, the three source folds and campaign,
lang_shift, temporal are reported. No choosing a winning seed. One thousand common campaign
resamples, seed13; intervals conditional on these three seeds, not a future seed population.
Use calibration-max-F1 thresholds for this primary comparison. Report absolute CIs and paired
deltas/p-values. The candidate is fixed, without test-driven hyperparameter search.

## Primary-source basis and limits

- [Ganin et al., JMLR2016](https://www.jmlr.org/beta/papers/v17/15-239.html): gradient reversal
  sends the opposite domain-loss gradient into an encoder while the domain head learns to
  classify domains. Their protocol uses unlabeled target data. We use **training sources only**,
  so this is source-invariance regularization, not a reproduction of target adaptation.
- [Long et al., NeurIPS2018](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html):
  CDAN conditions on classifier predictions, with multilinear/entropy mechanisms. This motivates
  conditioning; our separate true-training-label heads are a simpler adaptation, not exact CDAN.
- [Zhao et al., ICML2019](https://proceedings.mlr.press/v97/zhao19a.html): invariant features and
  small source error need not give low target error; differing label marginals can make
  unconditional alignment harmful. Source/label imbalance and the all-positive IMC source
  motivate caution. Class conditioning does not guarantee a solution to that problem.

Implementation API references (PyTorch2.11, matching the Kaggle image):
[custom autograd](https://docs.pytorch.org/docs/2.11/notes/extending.html),
[RNG fork](https://docs.pytorch.org/docs/2.11/random.html),
[module pre-hooks](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html).
The coefficient is a non-Tensor context value; backward returns no gradient for it.
CPU RNG is always forked, even with an empty accelerator-device list. The hook returns no
replacement input. Gradient reversal is an intentional surrogate gradient, checked against
its intended sign rather than the finite-difference derivative of identity. API documentation
is not a runtime-test pass; the pre-corpus synthetic checks still must succeed on Kaggle.

## Fixed implementation

Same dual-base byte CNN and rule side features, same train-only teacher KD and ERM sampling.
The source adversary sees the concatenated learned raw/canonical pooled features (384 dimensions
for dual-base) **before** the classification MLP/dropout. It excludes immutable rule inputs.
The inference classifier and parameter count remain unchanged; source heads are training-only.

One MLP per true training-label class: input384 → hidden64/ReLU → supported training sources.
No head dropout. Skip a class with fewer than two training sources; fail if neither class can
be regularized. Source CE balances label/source cells with inverse full-training-cell counts,
normalized within each minibatch. It sees no calibration/test/external source/label metadata.
Source-head CE has coefficient1 for its own parameters; the encoder's reversed gradient ramps
linearly from0 to0.05 over the first three epochs (epoch1=.05/3, epoch3=.05). The classifier loss
is unchanged. Auxiliary initialization preserves backbone/dropout RNG; hooks are removed at
the end, and auxiliary parameters are never registered as classifier children.

Both arms: seeds1/2/3, AdamW lr.002/wd.0001, batch128, KD alpha.7/T2, max15epochs/patience3,
selected by cal PR-AUC only. Default ERM training does not construct an adversary. The zero-weight
synthetic control checks matching class logits/gradients. No local fitting or real-data evaluation.

`sms_source_adversary` runs synthetic Torch checks on Kaggle before corpus access: gradient sign,
RNG preservation, zero-coefficient backbone gradients, live encoder/head gradients and unchanged
classifier state keys/hook cleanup. Missing local Torch is a skip, never a pass.

## Data, endpoints and claims

`build_training_pool` uses only the original frozen train IDs and matched per-run teacher
logits; raw extra rows are disabled. Split files and external rows are unchanged. Frozen
temporal's three train/test campaign overlaps and77 positives remain caveats.

All internal F1 endpoints use legacy spam-inclusive `label_binary`, not resolved fraud labels.
Campaign models also report every seed's GroundLM/SmishX FPR/recall at calibration-max-F1 and
calibration-1%-FPR thresholds for both available label endpoints. External results select no
arm, seed, epoch or threshold. PhreshPhish is a URL protocol and is not applicable here.

**In-sample for rule-feature design and repeated frozen benchmark comparisons.** ScamShield
test rows informed the rules. GroundLM is design-exposed; corrected train-only SmishX is
**independent, non-Indian-specific**. No clean independent held-out improvement claim.
Independent Indian validation remains **OPEN** under the post-freeze consented/privacy-scrubbed
real-SMS plan; never tune on those messages.

The saved-output audit uses the exact completed source tree and `--arms adversarial`; same
acceptance rule as before. Full args, kernel versions/trees, failures and outcomes go in STATUS.
