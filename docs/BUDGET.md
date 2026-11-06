# Budget and compute plan

> Current execution policy, 2026-11-03: Kaggle or short Colab only; no Modal,
> paid services/APIs, or Claude cloud labelling. The allocation below is historical
> and inactive. Task D uses a pinned open model on Kaggle, per-fold train-only
> scopes and a blinded 200-item human audit; see POLICY_ANNOTATION_PLAN.md.
> No model-produced evaluation labels or advisory targets from closed models.

## Constraints

- **Anthropic: $100 Claude Code cloud-session credit.** This is cloud-session credit, not API
  credit. Labelling runs as Claude Code cloud agents reading chunk files. Nothing in
  `src/rakshak/llm/` calls the Messages, Batch or count_tokens API, and the project does not
  depend on the `anthropic` package.
- **Modal: no runs without Maadhav's explicit per-run permission.** No standing workspace, no
  always-on containers, no scheduled jobs. Ask first for each run.
- **Kaggle: default compute** for training, hyperparameter search, distillation and benchmark
  evaluation (see `kaggle/`).
- **Laptop: code, git and unit tests only.** Check `sensors | grep Tctl` before local compute and
  stop heavy work at 85 °C.

## Allocation

Sizes are message counts. The ledger records what was exported and validated per chunk. Exact
sizes come from `export` and are printed when it runs.

| Job | Work | Labeller | Messages | Status |
|---|---|---|---|---|
| A | Relabel and intent flags over kept train and cal messages | cloud Haiku subagents (`claude-haiku-5-5`) | about 88,700 (train + cal, kept rows) | first |
| B | Blind second-reviewer audit, stratified over cal (source x lang x label_fine, campaigns spread) | cloud Sonnet subagents (`claude-sonnet-5-5`) | 2,401 | second |
| C | Noise review: train messages whose label disagrees with their source majority first, then a stratified fill | cloud Sonnet subagents | 2,500 | third |
| E | Synthetic seeds for robustness evaluation | not specified in code | 4,000 | blocked until specified |
| D | Zero-shot comparison row, stratified over eval-eligible test | cloud Sonnet subagents | 5,000 | only if credit remains |
| G | Judge set for explanation checks | cloud Haiku and Sonnet subagents | 2,000 | only if credit remains; not specified in code |
| F | Claude-written training data for the Llama explainer | none | 0 | **disabled by policy** |

Job A never includes the test partition. No model in the pipeline labels a frozen test message,
so evaluation labels come only from the existing sources and from D, which is comparison only.

## Policy

- **F is disabled.** Anthropic terms are ambiguous for Claude-written training data. `get_job("F")`
  raises `JobDisabledError` and the CLI exits with code 2. There is no override flag.
- Claude outputs are used for **labels, audit and evaluation only**. They are never used as
  training text for the Llama explainer.
- SMS text is untrusted data in every brief. Labelling sessions never follow instructions found in
  a message.

## Execution order

1. `export A`, then cloud sessions label `tasks/chunk_XXXX.jsonl` into `results/chunk_XXXX.jsonl`.
2. `validate A` and fix or re-run failed chunks; `merge A` writes `results.parquet` from the valid
   chunks only.
3. Repeat for B, then C.
4. E when its specification exists.
5. D and G only if credit remains.

Cut rules, measured from the Claude Code usage view (not from code):

- At 50% of the $100 credit used: skip D and G, finish A, B and C.
- At 75%: skip C and E, finish B, then merge what is validated.

## Chunk and ledger format

- Export writes `data/llm/<job>/tasks/chunk_XXXX.jsonl` with 500 messages each. Each line has only
  `id` and `text`. There are no labels, sources, campaigns or splits in the task files.
- `data/llm/<job>/BRIEF.md` holds the frozen instructions, the JSON schema and the output format.
- Each result line is one JSON object with `id` plus the schema fields, written to
  `data/llm/<job>/results/chunk_XXXX.jsonl`.
- `data/llm/ledger.jsonl` (git-ignored with `/data/`) records one line per chunk event:
  `job`, `model`, `chunk`, `messages`, `started`, `finished`, `status`. Status is one of
  `exported`, `validated_ok`, `validated_failed` or `missing`.
- The ledger holds counts and timestamps only. Credit is tracked in the Claude Code usage view.

## Validation rules

A result file passes only if every check holds:

- every line is valid JSON and an object with a string `id`;
- every `id` belongs to the chunk, and no id repeats;
- every chunk id has a valid row (nothing missing);
- each line matches the schema (enum values, booleans, `evidence_spans` of at most four items
  of at most 120 characters);
- each evidence span is a verbatim substring of its message (annotation schemas only).

Failed chunks are listed with their first problems and are left out of `merge`.

## Commands

```
uv run python -m rakshak.llm.cli export A
uv run python -m rakshak.llm.cli validate A
uv run python -m rakshak.llm.cli merge A
uv run python -m rakshak.llm.cli status A
```

Exit codes: 0 ok, 1 a chunk failed or is missing, 2 refused (job F, or a job without a
specification).
