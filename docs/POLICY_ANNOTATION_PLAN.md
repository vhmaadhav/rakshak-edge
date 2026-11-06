# Open-model train-only policy annotation — 2026-11-03

Hypothesis: separating promotions/transactional traffic from fraud in previously
null training-policy rows can address label-policy confusion. This is a hypothesis,
not evidence that relabelling fixes the deployable system's false-alarm rate.

First experiment is an inventory only, before model inference. Preserve every
original split; freeze eligible null-policy train IDs for each of the six folds,
input/split hashes, and a campaign-spread source/language/legacy-class-stratified
200-ID human-audit sample. Use existing chunk/ledger/schema tooling with an explicit
open-model JobSpec. Do not run the old cloud/Claude jobs or include calibration.

Candidate source checked via HF model card: Apache-2.0
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
revision989aa7980e4cf806f80c7fef2b1adb7bc71aa306. No local weights/inference,
paid API, HF Job, Claude/OpenAI-written target or advisory-model substitution.
Llama/ExecuTorch advisory requirements remain unchanged.

Data gate predeclared before compute:

- Every proposal must be bound to an original TRAIN ID in its named fold;
  zero raw-cal/test IDs in that fold, zero external rows, zero existing policy
  labels overwritten. Exclude exact duplicate rows and synthetic rows from this
  real-message annotation pilot. At apply time recheck the named fold's scope.
- An ID may have a different role in another fold. Sidecars are per-fold;
  never globally rewrite sms.parquet or use its model proposal as another
  fold's evaluation truth. Original evaluation labels stay unchanged.
- Accepted model outputs must pass the existing annotation JSON schema and
  verbatim evidence-span checks; reject invalid outputs, never invent a label.
  No inference recipe/prompt gate has been launched yet. Inventory has no model.
- The blinded 200-item human audit must be prepared privately before model
  labels become approved for training. Human fields start empty, never filled
  by this assistant. Training approval remains FALSE until the actual manual
  review is available and assessed; no automatic schema pass implies correctness.
- Kill/refuse on any membership or overwrite violation. Inventory completeness
  is a technical prerequisite, not a classifier performance go.

Human sample and text-containing chunk files remain private Kaggle/HF artefacts,
never git or public logs. Source labels/model suggestions are hidden from the
reviewer. The current inventory generates zero model labels and changes zero
evaluation labels. Independent Indian post-freeze validation remains OPEN.

Before inference, separately freeze the annotation prompt, generation settings,
model/tokenizer/weight hashes and bounded retry/abstention recipe. Report all
accepted/rejected/unclear counts and sample coverage. Only after genuine review
can a new matched training experiment declare its own three-seed, all-fold,
external and fused-FPR/recall gates. No retuning on external/test sets.


## Inventory result (no inference)

Private policy_inventory-d-scope-20261103 v1 completed in84.6s,
tree5e51464453b9a12c9e1aad62451999cc0acd06cb7c6a34150b34ab0c6890e474;
ONE wait/fetch and verified scope/audit hashes. Eligible distinct union74,700;
campaign51,189, lang_shift41,612, temporal34,725, source_holdout
Mishra66,070/ScamShield4,137/UCI61,643. These are missing policy categories,
not a measured count of unknown binary fraud labels. Do not conflate them.

Exactly200 blinded human rows are prepared privately, not reviewed. No Qwen
inference/approved training label or changed evaluation label exists yet.
Next: freeze prompt/generation/source hashes, then a bounded200-item annotation
pilot; no mass annotation or new classifier claim from the inventory.
