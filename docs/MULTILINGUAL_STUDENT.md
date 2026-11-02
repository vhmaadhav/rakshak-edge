# Multilingual student — predeclaration 2026-11-01

Current clean language-shift headline: no-KD byte student meanF1.405842
[.369672,.441243] versus historical teacher-b.802. The contaminated≈.81 result
remains WITHDRAWN; clean single-teacher KD.351977 is KILL. This experiment targets
missing multilingual representation; it does not assume stronger KD alone fixes it.

## Primary-source check and hypothesis (before code or compute)

[mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-small/blob/abc32620dd4f6ab06f5fbe905dc25f310618e09f/README.md)
reports MIT; public model API independently reports licenceMIT and revision
`abc32620dd4f6ab06f5fbe905dc25f310618e09f` (checked2026-11-01).
Card SHA256 `a56c26f251c188a1c0b783fcddb461e78f754b61ee948975fafbb2b77c863353`.
[Pinned config](https://huggingface.co/jhu-clsp/mmBERT-small/blob/abc32620dd4f6ab06f5fbe905dc25f310618e09f/config.json)
specifies ModernBERT, hidden384, vocab256000,22layers. Only its input embedding
rows/tokenizer will initialize the compact branch; do not deploy all22layers.
The [authors' paper](https://arxiv.org/abs/2509.06888) studies multilingual encoder
pretraining. It does **not** demonstrate this compressed embedding+byteCNN hybrid
or SMS results. Whether its static embedding prior helps Rakshak is **UNVERIFIED**.
No reproduced SOTA or inherited clean external-performance claim. Public pretrained
corpus overlap with these messages is unknown; train-only downstream fitting is
not a guarantee about upstream pretraining membership. Keep MIT attribution and
exact original licence/tokenizer source revision with private artifacts.

Hypothesis: a compact pretrained multilingual subword branch gives the byteCNN
useful cross-language structure, so train-only mmBERT task-logit distillation can
improve language-shift F1 without harming the other frozen folds.

## Fixed experiment

Three arms, each seeds1/2/3, all six unchanged frozen folds:
1. `byte_nokd`: matched dual/base byteCNN + existing canonical rule side features,
   full hard-label BCE, no teacher;
2. `multilingual_nokd`: same byteCNN + compact multilingual embedding branch,
   full BCE, no teacher (diagnostic ablation, not the candidate selected for GO);
3. `multilingual_kd`: identical hybrid with cached mmBERT teacher0 logits ONLY on
   that fold's exact original train IDs, alpha.7/temperature2.

No unlabeled transfer extras, new synthetic augmentation, source reweighting, adversarial loss,
new label policy or external/test row may enter training. Exact original frozen train IDs
include the corpus's existing synthetic training rows; they are retained equally in all arms.
This clarifies the conflicting earlier “no synthetic rows” wording before any model run;
no new synthetic rows are added and no synthetic cal/test rows are permitted. Cached teacher targets
are in-sample single-teacher training outputs, **not OOF/ensemble**. Do not combine
other outer-fold teachers that may have trained on this fold's test rows.

Per fold, tokenize only canonical original **train** text with the pinned public
tokenizer for vocabulary selection. Keep at most16382 most frequent token IDs,
deterministic count-descending/token-ID-ascending ties; reserve PAD+UNK for a
maximum16384-row table. Never use cal/test/external token frequency, vocabulary,
embedding PCA or normalization statistics. Initialize selected rows from the
pinned pretrained input embedding table, project384→64 by unwhitened centred
PCA fitted only to those selected train-token rows, deterministic component signs
(largest-absolute loading positive). PAD=zero, UNK=mean selected projected vector.
Failure to obtain64components on train is unmeasurable, no test-driven fallback.
Freeze vocabulary/row mapping/PCA source hashes before cal/test transformation.
The64dim table is trainable on train IDs. Encode up to128 canonical subword tokens;
masked mean+max pooling joins the existing dual-byte pooled vectors/rule side
features before a64hidden/dropout.1 classifier. Unseen tokens map UNK. No lexical
family, token list or encoder size changes from outcomes. Byte paths remain the
existing base widths3/4/5,32dim embeddings/64filters and byte-length limit.

Same AdamW.002,weight_decay.0001,batch128,max15epochs/patience3,cal PR-AUC stopping
and cal-only max-F1/1%-FPR thresholds. Seed model construction BEFORE initialization;
identical ordered original training IDs/batch seeds/class weighting within each
seed. NoKD gets full BCE. Save selected CPU checkpoints + vocabulary/projection/
tokenizer licence and exact source/model/teacher/fold/train-ID hashes before
scoring test/external rows. No model execution on this laptop.

## Candidate gate (frozen before outcomes)

GO only if `multilingual_kd` lang_shift mean-F1 campaign-clustered95%CI lower bound
exceeds the **fresh matched byte_nokd point mean**, AND the paired lang_shift
KD-minus-byte-control CI excludes0 positively, AND no other frozen-fold mean-F1
loss exceeds its matched byte-control CI half-width. Required capacity gate:
serialized classifier weights<=8MiB (report tokenizer/vocabulary/source footprint
separately). Missing fold/seed/required CI or artifact blocksGO; any observed
failure KILL. Do not replace the fresh control with historical.405842 or select
the ablation after seeing results. `multilingual_nokd` is reported to distinguish
representation versus soft-target effects, with no promotion from this experiment.
No learning-rate/alpha/table-size sweep or external-selected threshold.

Paired bootstrap:1,000 shared campaign multiplicities across both arms/all3fixed
seeds, report both absolute mean CIs and paired differences/null-centred p-values
for every fold. Conditional-on-fixed-seeds intervals do not cover initialization
uncertainty generally. Report all per-seed checkpoints/thresholds, every source
holdout, campaign/lang_shift/temporal, and GroundLM/SmishX both fraud/legacy labels.
These are student diagnostics; deployed full-system FPR still cites frozenv1 or a
named successor. No new PhreshPhish URL or NIDS result from this SMS experiment.

Rule design used GroundLM/ScamShield; intervention and representation selection
uses aggregate GroundLM/SmishX and repeated frozen-language results: **in-sample
for intervention selection**, no untouched held-out improvement claim. Temporal's
original overlaps/77positives unchanged. Independent Indian validation stays OPEN.
A scientific GO is not shipment approval: fused safety and Arm memory/latency/
format checks remain necessary. Llama+ExecuTorch advisory requirements unchanged.
Only private Kaggle/T4, max2kernels, one wait_event and matching fetch per job;
private Kaggle artifacts with verified isPrivate, **HF publish pending**.

Status: predeclared only; no implementation, training or model result yet. Complete
higher-priority policy/hardneg milestones before using spare capacity for this arm.

Pinned packaging check2026-11-02: revision has pytorch_model.bin, not safetensors;
weight563633326bytes, SHA256be7c9ce2af947fb5bb17a3bc919552f898c5ef4164bf0a7190cf25a0f98f85b6
from the primary HF tree API. First job failed before data access; recovery uses
this identical pinned weight with weights_only=True and exact embedding shape.
No representation or scientific gate change.

Primary [pinned repository file metadata](https://huggingface.co/api/models/jhu-clsp/mmBERT-small/tree/abc32620dd4f6ab06f5fbe905dc25f310618e09f) supplies the weight size/LFS SHA above. No local weights downloaded.
