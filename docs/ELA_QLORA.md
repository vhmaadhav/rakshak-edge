# Verifier-gated Llama ELA experiment

Built with Llama. This is an experiment, not a shipped model or a semantic-entailment result.
The base result is from PR #2 (`2463245`); this work has its own branch/PR.

## Frozen decision rule (declared before generation/training)

- Go: English pooled point-pass campaign-bootstrap lower 95% bound >40%, versus base
  16/62 =25.81% [13.04%,40.82%]; Hinglish not worse than 23/37 =62.16%
  [40.61%,79.59%] beyond CI overlap; zero raw/rendered unsupported advisory claims; no new
  harmful ELA slot flips across the five injection families. Positive raw/rendered safe/no-risk
  reassurance is separately reported and also fails safety, even when no attack-induced flip
  occurs; the training gate already rejects such targets.
- Kill: EN lower bound <=25% or any new unsupported advisory claim. Other outcomes are
  inconclusive and do not authorize export. Never change these gates from the frozen test.
- Exact paired McNemar and descriptive two-sided Fisher tests compare tuned versus base
  harmful flips. Campaign-bootstrap rate differences retain pairing. Rank uses only pairs
  with candidates in both clean/injected outputs in both models. Freeform is exploratory.
- Campaign bootstrap is the primary interval. Ordinary Wilson intervals are separate IID
  references; there is no invented “clustered Wilson” method. Zero observed harms do not
  prove zero population risk.

## Data provenance and separation

Teacher: `unsloth/Llama-3.2-1B-Instruct` at
`5a8abab4a5d6f164389b1079fb721cfab8d7126c`. Its [pinned model card](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/5a8abab4a5d6f164389b1079fb721cfab8d7126c/README.md)
was read on 2026-11-01 and lists `llama3.2`, the Llama 3.2 Community License.
The mirror contains no licence file. Its card links [Meta's agreement](https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/LICENSE),
retained verbatim at that commit, SHA256
`8cc15535a8a34b41888f644b339a1a9eb428af793a4f5e24df58a3e5b1487d74`.
Private retained adapter packages carry that agreement, NOTICE and Built with Llama attribution.
The saved-output audit adds these to a hash-checked copy of the original selected adapter;
it does not alter weights or make a deployment claim.
No Apache licence claim is made. Qwen is not used; the official
[Qwen2.5-3B-Instruct card](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/README.md)
explicitly lists `qwen-research`, confirming that this size is not Apache-2.0.

`ela_data` reads no corpus. It creates abstract evidence packages from the frozen evidence-type
vocabulary, with no SMS text, spans, URLs, phone numbers or brands. Target sentences come only
from the existing deterministic candidate bank and the pinned Llama's Kaggle generations.
Summaries use the existing deterministic risk summary. No Claude/OpenAI-authored target
sentences enter the dataset. Generation instructions and schema serialization are code.

Both `check_points` and the evaluation scorer's unsupported-ID/claim check must pass. In
particular a copied E99 is rejected even when the verifier alone permits it. No rejected sentence
is rewritten into a passing target. A valid teacher JSON string may be wrapped verbatim as one
point; malformed JSON is rejected. Composed points retain the singleton parent IDs, full parent
records, teacher pin/seed, template fingerprint and raw generation/rejection audit.

Exact prompt/target copies deduplicate across sources. The sorted evidence-type multiset is
hashed into train/dev; languages, evidence order and risk variants of one composition stay
together. Empty/singleton packages train for vocabulary coverage. Dev contains unseen
compositions; it is synthetic development data, not a test or external set.

## Runtime and evaluation plan

On Kaggle T4 only: NF4 plus double quantization, frozen 1B base, LoRA r16/alpha32/dropout.05
on all linear layers, AdamW lr2e-4, batch2/accumulation8, max768 tokens without truncation.
Only completion tokens contribute loss. At most three epochs, patience1, selected solely by
verifier pass on 48 fixed synthetic-dev records (24 per language); earliest checkpoint wins ties.
The exact runtime versions and any rejected oversized rows are recorded in `training.json`.

Final inference reloads the saved adapter onto the fp16 pinned base to match the original base
inference dtype. Before inference the evaluator must exactly reproduce the frozen 300-row
manifest, 150 Hinglish subset and 150 injection pairs; it refuses changed membership. Frozen
base generation SHA256:
`8a898ef0d8edcf92598729ef8bf6e94e7bc5674f7d914d854aed02cf799e04d1`.
All five attack families and the corrected rank-token accounting are retained.
A separate saved-output `ela_paired_audit` reports paired pooled point-pass differences and
null-centred campaign-bootstrap p-values for EN/Hinglish. It performs no inference or tuning
and does not change the absolute gates declared above.

The frozen 300-row campaign sample is the predeclared endpoint, evaluated once after dev-only
selection. **In-sample for rule/contract design and repeated benchmark comparison:** base outcomes
were observed before this experiment and ScamShield test rows informed rule design (see the
design-set register in `docs/GROUNDLM_FPR.md`). A source split name does not undo that exposure.
This is not a clean independent held-out improvement claim. No evaluation message is a target. This experiment does not establish advisory generalization on all source_holdout,
lang_shift, temporal or external messages; detector results on those splits are separate.
No raw test message enters training. Export is attempted only after a go decision; NF4 QLoRA
training is not itself an ExecuTorch 8da4w export.

Primary implementation references: [QLoRA](https://arxiv.org/abs/2305.14314),
[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization),
[bitsandbytes in Transformers](https://huggingface.co/docs/transformers/quantization/bitsandbytes).
Runtime compatibility and export quality remain UNVERIFIED until their Kaggle jobs succeed.

### Conditional export check (declared before model outcomes)

Only a final QLoRA **go** permits export. Export the selected adapter merged into its pinned
base; a prebuilt or unadapted base artifact cannot stand in for it. Start with the existing
ExecuTorch 1.5.1 8da4w XNNPACK recipe, without test-driven calibration or repeated recipe changes.
On the same frozen slot/Hinglish samples and five-family injection pairs, quality is retained
only if both languages' paired point-pass difference intervals (PTE minus FP16 adapter) have
lower bounds >= -5 percentage points, the original absolute quality gates still pass, and
there are zero unsupported/reassuring claims and no new slot harmful flips. A successful load
or a non-significant difference alone does not establish retention. Report the actual differences
and intervals even if the criterion fails; no test-driven retry. Candidate ranking is unmeasured
unless the runtime actually exposes the needed token likelihoods. Record artifact bytes/hash and
runtime versions; Kaggle x86 timings are not Pi or Android measurements.

## Run register

| Job / version | Source tree | Result | Conclusion |
| --- | --- | --- | --- |
| `ela_data-b-data-20261101` / 1 | `aea5cd47a85a` | error, 303.2 s; 256/256 paraphrases rejected | Kill this incomplete dataset; no training/publication. JSON envelope mismatch. |
| `ela_data-b-data2-20261101` / 1 | `91cd95088bb2` | ok, 331.9 s; 131/256 paraphrases kept | Keep data path: 10,934 train / 3,026 dev, disjoint compositions; no model-quality claim. |

Successful targets are published as **private** Kaggle dataset
`maadhavvh/rakshak-ela-synthetic-b20261101`, version1; API privacy verified.
Manifest SHA256 `03d5b85ec0d6e7ab313cb5b0fe849a6696d003a490f5deb52a4f20f333a6c69a`.

Detailed args, checks and subsequent results belong in `docs/STATUS.md`. Artifacts remain private;
no training rows or weights are committed. No quality improvement is claimed while jobs are pending.
