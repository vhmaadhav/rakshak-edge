# Evidence-Locked Advisory (ELA): research notes for the Llama 3.2 1B layer

Snapshot: 2026-10-31. Owner: research worker. Scope: constrained decoding on ExecuTorch, evidence-faithful explanations, cheap fine-tuning, prompt-injection defences, Hindi/Hinglish output. No model was run, nothing was installed, nothing was committed or posted.

Status tags used below:
- **[OPENED]** the page or PDF was fetched in this session and the claim was checked against it.
- **[SNIPPET]** the claim comes from a search-result summary only. Check the paper before quoting a number.
- **[SECONDARY]** an aggregator, mirror or blog stated it. Not a primary source.
- **[UNVERIFIED]** could not be confirmed. Do not cite as fact.

Our own estimates and calculations are labelled as such.

---

## 0. Findings that change the current plan

1. **ExecuTorch 1.5.1 has a logit-processor hook, but no grammar.** PR #19517 ("Add LogitProcessor interface for pre-sampling logit transforms") was merged on 2026-05-15 [OPENED]. The v1.5.1 tag contains `extension/llm/sampler/logit_processor.h` (an `ET_EXPERIMENTAL` abstract class with pure virtual `Error process(Tensor logits)`) and `extension/llm/runner/text_token_generator.h`, which calls `apply_logit_processors()` after `text_decoder_runner_->step(...)` and before `logits_to_token(...)` [OPENED, raw files at tag v1.5.1]. The PR text names grammar masks as a motivating use, and no grammar implementation is in the tree [OPENED]. `EXECUTORCH_PLAN.md` section 4 says there is "no logit hook in the documented API". That is still true of the documented `GenerationConfig`, but it is out of date for the source. Masking therefore does not need a runner fork. It needs a C++ subclass registered on the generator. Whether the Android `LlmModule` and the Python `executorch.runtime` path expose this is **[UNVERIFIED]**.
2. **Issue #19476 is still open with no linked PR** [OPENED, fetched 2026-10-31]. Do not wait for it.
3. **The 7.7% verifier pass rate is most likely prompt-induced copy bias, not a decoding problem.** Few-shot example IDs leak into outputs, and a single default action dominates. Both are classic demonstration biases (Zhao et al. 2021; Sclar et al. 2024). Fix the prompt before adding a grammar.
4. **Full-schema constrained decoding can hurt output quality.** Tam et al. (EMNLP Industry 2024) report that stricter format restrictions degrade reasoning, while JSON mode can improve classification accuracy [SNIPPET]. Our decision fields are classification-like (choose one action, choose evidence IDs). That is the favourable case. The summary text should be a template, not a constrained generation.
5. **The injection numbers are small and only partly independent.** For 11/50 flips versus 0/50, the Wilson 95% intervals are 12.8 to 35.2% and 0 to 7.1%. A two-sided Fisher exact test gives p ≈ 0.0005 [our calculation]. This holds only if the 50 trials are one attack family. Report per-family counts with intervals before using the figure in the report.
6. **Attacker-controlled strings can still reach the evidence package.** Domain names, brand strings and sender IDs are attacker-controlled. If they appear verbatim in the evidence JSON, the evidence-locked arm has an indirect injection channel. The design should map them to fixed vocabulary or escape them. This is a design check, not a published result.

---

## 1. Ranked recommendations

Costs are rough. Kaggle is the compute target per `CONTEXT.md`. Thresholds in the last column are **our suggestions** for pre-declaration, not sourced values.

| Rank | Method | Evidence (status) | Expected effect | Cost | How to test in Kaggle eval |
|---|---|---|---|---|---|
| 1 | **Remove demonstration leakage.** No few-shot examples that contain real evidence IDs or action codes. Use per-request randomised opaque IDs (not `E1..En`). Shuffle the order of allowed action codes. Calibrate the action prior (Zhao et al. 2021). | Zhao et al. 2021 documents majority-label, recency and common-token bias [OPENED abstract via search]. Sclar et al. 2024 shows accuracy swings of up to 76 points from format changes for LLaMA-2-13B [SNIPPET]. Ali et al. 2024 documents copy bias [OPENED abstract]. | Should remove copying of example IDs by construction. Should spread action choices. Effect on the 7.7% is **unmeasured**. | Low. Prompt change only. | Re-run the current v1 prompt and the new one on the same 500 evidence packages, three prompt formats each, and report verifier pass rate with Wilson CI. Pre-declare a go threshold (suggested: pass rate ≥ 60% and no single action above 50% of outputs). |
| 2 | **Choose slots, do not generate them.** Summary is a deterministic template. The LLM only scores K candidates per decision field (action, each evidence ID set) by log-likelihood. Verifier applied afterwards. | Robinson et al. (ICLR 2023) show option scoring and symbol-choice differ by model ability [SNIPPET]. Candidate scoring is our design (`EXECUTORCH_PLAN.md` section 4, v2). Naive masking can distort the model distribution (Park et al. 2024, [SNIPPET]). | Removes free text the verifier cannot check. Reduces attack surface. No grammar needed. | Low to medium. K prefill passes per decision. Prefix KV reuse, if the runner supports it, is **[UNVERIFIED]**. | Ablation arms: (a) free JSON, (b) JSON with v1 verifier, (c) candidate scoring. Same packages, same seed. Report pass rate, action accuracy against labels, and unsupported-claim rate. |
| 3 | **Verifier-gated fine-tuning.** Train the 1B on deterministic-template targets plus teacher paraphrases. Keep only paraphrases that pass the same verifier (exact ID and action match, entailment check on the summary). QLoRA first, then the ExecuTorch QAT+LoRA export path. | TempLM reports unfaithfulness falling from 83% to 0% out of domain for a BART model [SNIPPET, ACL 2023 Findings]. QLoRA reports 4-bit NF4 matching 16-bit fine-tuning on academic benchmarks [SNIPPET]. Meta's QAT+LoRA recipe is SFT, then QAT, then frozen-backbone LoRA, then DPO [OPENED, Meta blog]. | Likely the largest faithfulness gain, if verifier filtering works. Risk: the fine-tune re-introduces copying. | Medium. QLoRA on a single T4 is plausible; time and VRAM on T4 are **[UNVERIFIED]**. Teacher: Qwen3 0.6B to 4B are Apache 2.0 per the Qwen blog [SNIPPET]. Llama-derived checkpoints must carry the "Llama" prefix (licence, `EXECUTORCH_PLAN.md`). | Train on Kaggle. Compare base, template-only SFT, and verifier-filtered SFT on the same held-out campaigns. Suggested go threshold: fine-tuned arm beats base on unsupported-claim rate with non-overlapping Wilson intervals. |
| 4 | **Injection: architecture first, then adaptive evaluation.** Keep raw SMS out of the LLM. Any attacker-controlled string must be fixed-vocabulary or escaped. Output may contain only allowed IDs and codes, so a flip stays inside the allowed set. Report harmful flips, not any text change. Evaluate with adaptive attacks. | Spotlighting (Hines et al. 2024) reduces attack success from above 50% to below 2% on GPT-family models, with datamarking and encoding [SNIPPET]. "The Attacker Moves Second" (Nasr et al. 2025) bypasses 12 published defences with above 90% success for most [SNIPPET]. A 2026 preprint finds output filtering was the only defence that held (zero leaks across 15,000 attacks), while model-self-defences all broke [OPENED abstract, preprint, no 1B models named]. | Architecture-level control is the only defence with evidence that survives adaptive attacks. Model-side training (StruQ, SecAlign) is untested at 1B. | Low for the evaluation. Medium for any training. | Build an adaptive attack set against the full pipeline (attacker controls only SMS text). Report harmful-flip rate per family with Wilson CI. Suggested go threshold: zero harmful flips in the locked arm, with 95% upper bound stated. |
| 5 | **Faithfulness metric stack.** Exact ID precision and recall for the evidence claims. AIS-style entailment on the free-text summary, with a small grounded-factuality checker (MiniCheck-FT5) as the judge. FActScore-style atomic-fact precision as a secondary metric. Human audit of a random subset. | AIS (Rashkin et al. 2021) [SNIPPET]. MiniCheck-FT5 (770M) reported at GPT-4-level accuracy on LLM-AggreFact [SNIPPET]. AttributionBench: fine-tuned GPT-3.5 reaches about 80% macro-F1 on binary attribution, and harder on long evidence [SNIPPET]. FActScore [SNIPPET]. | Measurement, not a gain. Prevents over-claiming: automatic judges are unreliable on hard cases. | Low. Kaggle GPU for MiniCheck. | 200-item human audit with two raters. Report Cohen's kappa. Report judge agreement with humans before using the judge's numbers. |

Also worth doing, lower priority: **Hindi and Hinglish output by template.** See section 6. Hindi is an officially supported Llama 3.2 language [OPENED]. Romanised Hinglish is not listed. Test before any model is used for Hinglish prose.

---

## 2. Question 1: constrained decoding, slot-filling, candidate ranking, copy bias

### 2.1 Tools and whether they fit ExecuTorch

| Tool | Mechanism | Fit with ExecuTorch 1.5.1 | Evidence |
|---|---|---|---|
| **llama.cpp GBNF / `json_schema`** | Grammar compiled to a per-step token mask. Supports a subset of JSON Schema. Guidance notes that some schema features are unsupported or slow. | **Not usable in our runtime.** llama.cpp is not ExecuTorch, and the plan says llama.cpp is not a legal fallback for the Llama layer. Useful only as a reference spec or for host-side ablations if the team agrees. | Grammar README via mirrors [SECONDARY]. The canonical `ggml-org/llama.cpp` README was not opened. |
| **XGrammar** (arXiv 2411.15100) | Context-free grammars as byte-level pushdown automata. Splits tokens into context-independent (precomputed) and context-dependent (checked at runtime). Claims near-zero overhead on a laptop and an iPhone with 4-bit models (paper claim). | **No ExecuTorch integration found.** Published integrations are WebLLM, vLLM, TensorRT-LLM and MLC-LLM [SNIPPET]. A C++ core could in principle be wired into the new `LogitProcessor`. Not verified. | Paper [SNIPPET]. |
| **Outlines** (arXiv 2307.09702) | Regex and CFG to finite-state machine, with a precomputed index of allowed tokens per state. JSON Schema support. Python library. | No ExecuTorch integration found. Python-side masking in a custom loop would be a re-implementation. | Paper [SNIPPET, ar5iv]. |
| **lm-format-enforcer** (noamgat, MIT) | Character-level parser intersected with a tokenizer prefix tree. JSON Schema, regex, JSON mode. | No ExecuTorch integration listed. Integrations listed: Transformers, llama.cpp, vLLM, TensorRT-LLM, LlamaIndex, LangChain [SNIPPET, GitHub]. | Repo README [SNIPPET]. |
| **llguidance** (guidance-ai, Rust) | Earley parser plus regex-derivative lexer, with token masks computed per step. The project quotes about 1.5 ms per mask on a 128k-vocab tokenizer for a typical JSON schema (its own figure). | Rust library with Python bindings. Used by Guidance, llama.cpp (behind a CMake flag) and SGLang [SNIPPET]. A C ABI for linking into the ExecuTorch C++ `LogitProcessor` would be needed. **[UNVERIFIED]** whether a C ABI exists. Licence not confirmed. | GitHub repo [SNIPPET]. |
| **Guidance** | Program-level generation library on top of llguidance. | Python. No ExecuTorch integration found. | [SNIPPET]. |
| **Logit masking through ExecuTorch `LogitProcessor`** | Subclass `LogitProcessor::process(logits)`, set the disallowed logits to -inf, and advance our own state after sampling. | **Feasible in C++ on 1.5.1 source** [OPENED]. Needs a custom C++ runner or `llama_main` patch. Android and Python exposure **[UNVERIFIED]**. Accepting a token needs a post-sample callback, which the 1.5.1 interface does not show. The state update would have to be inferred from the chosen token, which the processor cannot see. **[UNVERIFIED]** whether this is a real gap. | PR #19517 and v1.5.1 source [OPENED]. |

**GitHub issue #19476** ("Wire grammar-constrained decoding end-to-end", opened 2026-05-11) proposes `GenerationConfig.grammar`, a `GrammarLogitProcessor`, and `accept_token()` after sampling, with a target mask overhead under 100 µs. It is open, with no linked PR and no comments [OPENED].

**PR #19351** (merged 2026-05-12) lets `TextDecoderRunner` reuse a persistent sampler and says it "lays groundwork" for grammar-constrained decoding [OPENED].

**Answer to the direct question.** No shipped ExecuTorch grammar or JSON-constrained decoding exists in v1.5.1 [OPENED]. Masking is possible through the v1.5.1 `LogitProcessor` in C++. A grammar engine would have to be integrated by us. Python-side masking through the `executorch.runtime` path is **[UNVERIFIED]**.

### 2.2 Slot-filling versus free generation

- Tam et al. (EMNLP 2024 Industry, arXiv 2408.02442): format restrictions degrade reasoning tasks, more so for stricter formats. JSON mode can increase accuracy on classification tasks [SNIPPET]. Our decision fields are classification-like. The summary is not.
- Park et al. (arXiv 2405.21047, NeurIPS 2024): grammar-constrained decoding skews the sampled distribution, so outputs are valid but low-likelihood under the model. Proposed ASAp converges to the model distribution [SNIPPET].
- DOMINO (arXiv 2403.06988, ICML 2024): reports that naive constraining can lower JSON-constrained GSM8K accuracy (from 41.5% to 30.8% in one setting), while minimally invasive constraining recovers it [SNIPPET, secondary summary]. Check the model and setting in the paper.
- JSONSchemaBench (arXiv 2501.10868): frameworks differ by up to 2x in how many schemas they cover. Constrained decoding can speed generation by about 50% (paper claim) [SNIPPET].

**Implication.** Keep the free text out of the constrained path. Use a deterministic template for the summary. Use slot choice (candidate scoring) for action and IDs. Use full grammar masking only if candidate scoring fails the eval.

### 2.3 Candidate selection and ranking

- Robinson, Rytting and Wingate (ICLR 2023, arXiv 2210.12353): cloze scoring (option log-likelihood) versus multiple-choice symbol prompting. Symbol prompting wins only for models with high "multiple-choice symbol binding" ability [SNIPPET]. A 1B model is unlikely to have high MCSB. This is an inference from the paper's framing, **[UNVERIFIED for 1B]**. Prefer log-likelihood option scoring and run it as an ablation.
- Zhao et al. (ICML 2021, arXiv 2102.09690): calibrate with a content-free input ("N/A") to remove majority-label, recency and common-token bias [SNIPPET, ar5iv].

### 2.4 Copy bias and few-shot leakage

- Copy bias is documented: LLMs sometimes copy answers from in-context examples. Ali, Wolf and Titov (arXiv 2410.01288, Oct 2024) identify copying neurons with Integrated Gradients and report gains from pruning them [OPENED abstract]. **The claim that smaller models copy more is not confirmed on the abstract page [UNVERIFIED].**
- A 2024 EMNLP paper finds copying grows with model capability to recognise similar examples (arXiv 2407.17011, "Unveiling In-Context Learning") [SNIPPET]. Treat as a counterpoint: copying is not limited to small models.
- Sclar et al. (arXiv 2310.11324, ICLR 2024): formatting changes alone move accuracy by up to 76 points; FormatSpread reports a range across plausible formats [SNIPPET].

**Mitigations for us, in order of cost:**
1. Remove real IDs and action codes from any example. Use format-only examples with placeholder tokens that are not in the allowed set (e.g., `<ID_A>`).
2. Randomise the opaque IDs per request. Reject any output that uses an ID from a prior request (the verifier can check this).
3. Shuffle option order and report the spread across orders.
4. Calibrate the action prior with the content-free method.
5. Neuron pruning (Ali et al.) needs weight access and a fork. Not recommended for v1.

---

## 3. Question 2: faithfulness metrics and verifier loops

### 3.1 Metrics

| Metric | What it measures | Cost | Small-model evidence |
|---|---|---|---|
| **Exact evidence precision / recall** (ours) | Fraction of cited IDs that exist in the package; fraction of required IDs cited. Deterministic. | Zero. | Not applicable. This is the strongest check available for ID claims. |
| **AIS** (Rashkin et al., arXiv 2112.12870) | Whether a statement is attributable to a source, human-labelled. Basis for later automatic judges. | Human. | Framework, not a model result [SNIPPET]. |
| **ALCE citation precision / recall** (Gao et al., EMNLP 2023, arXiv 2305.14627) | NLI-based citation support, validated against humans. | Moderate. | Reports that even the best models lack full citation support on ELI5 about half the time [SNIPPET]. Validation accuracy for citation precision is 77.6% against humans [SNIPPET]. |
| **FActScore** (Min et al., EMNLP 2023, arXiv 2305.14251) | Share of atomic facts supported by a knowledge source. Precision only. | Moderate (retrieval plus a judge). | Public code, `pip install factscore` [SNIPPET]. Designed for biographies; adapting it to a short advisory is ours. |
| **AttributionBench** (arXiv 2402.15089) | Benchmark for attribution evaluators. | Benchmark. | Fine-tuned GPT-3.5 about 80% macro-F1 on binary attribution. Harder on long evidence. Models do not close the gap by scale alone [SNIPPET]. |
| **MiniCheck** (Tang et al., arXiv 2404.10774) | Small grounded fact-checker (Flan-T5, 770M for FT5). LLM-AggreFact benchmark with 10 datasets. | Low on a Kaggle GPU. | Reported GPT-4-level accuracy at about 400x lower cost [SNIPPET]. Judge on our own data before trusting it. |
| **RAGTruth** (Niu et al., ACL 2024, arXiv 2401.00396) | About 18k RAG responses with case- and word-level hallucination labels. | Dataset. | A fine-tuned small model reported competitive with prompt-based GPT-4 detection [SNIPPET]. Not our domain; useful as a calibration set for the judge. |

### 3.2 Model size and faithfulness

- arXiv 2407.14088 ("Impact of Model Size on Fine-tuned LLM Performance in Data-to-Text Generation"): faithfulness, measured with BARTScore, often declined as parameter count grew within families, across roughly 1B to 13B [SNIPPET]. Bigger is not automatically more faithful.
- TrueBrief (arXiv 2601.04212, Dec 2025): preference optimisation with injected hallucinations, aimed at small models, summarisation focus [SNIPPET, preprint]. Results **[UNVERIFIED]**.
- No 2025 paper was found that measures 1B to 3B structured-evidence grounding specifically. This is a gap for our own eval to fill.

### 3.3 Verifier-loop designs

- **Self-RAG** (Asai et al., ICLR 2024, arXiv 2310.11511): trains the model to emit reflection tokens for retrieval need and support. Needs training at 7B and 13B [SNIPPET]. Not a fit for a 1B without training.
- **Deterministic verifier with template fallback** (our design, `CONTEXT.md` section 4): the strongest guarantee for ID claims. It does not check free text. Keep the summary template-based, or put it through the judge in rec 5.
- **Output filtering outside the model** was the only defence with zero leaks in a 2026 preprint on prompt-injection evaluation (arXiv 2604.23887, >20,000 attacks, nine configurations, adaptive attacker) [OPENED abstract]. It names no 1B models.

---

## 4. Question 3: cheap fine-tuning for grounded verbalisation

### 4.1 Published evidence

| Item | Result | Status |
|---|---|---|
| LoRA (Hu et al., arXiv 2106.09685) | Frozen weights with low-rank adapters; no added inference latency; on-par quality with much fewer trainable parameters on GPT-3 [SNIPPET]. | SNIPPET |
| QLoRA (Dettmers et al., arXiv 2305.14314) | 4-bit base with NF4 and LoRA matches 16-bit fine-tuning on academic benchmarks. A 65B model fits in under 48 GB during fine-tuning [SNIPPET]. | SNIPPET |
| Meta QAT+LoRA recipe for Llama 3.2 1B/3B | BF16 SFT, then QAT with torchao, then frozen backbone with BF16 LoRA, then DPO. Reports "competitive accuracy" with BF16. Exact benchmark numbers are in charts only. | OPENED (Meta blog). Exact numbers **[UNVERIFIED]**. |
| TorchAO QAT with LoRA (secondary) | 1.89x training speedup and 36.1% lower memory versus the baseline in the cited setting. | SECONDARY (TorchAO docs via search). Settings **[UNVERIFIED]**. |
| TempLM (ACL 2023 Findings, arXiv 2205.11055) | Distilling into template-based generators cut out-of-domain unfaithfulness in BART from 83% to 0% (reported). | SNIPPET. Predates LoRA-era work. |
| Synthetic teacher data for data-to-text (INLG 2024 GEM challenge) | Filtered teacher data for Llama-2 on WebNLG-style triples worked for English. Mixed results for other languages. | SNIPPET |
| Cross-domain data-to-text with a 1.7B student (arXiv 2608.23391) | Reported distillation from teacher data with no target-domain references. Date unclear. | **[UNVERIFIED]** |

### 4.2 ExecuTorch QAT+LoRA path

- ExecuTorch's Llama docs cover prequantised INT4 QAT+LoRA checkpoints on XNNPACK. Export uses `export_llama` with QAT and LoRA flags (`-qat -lora 16`, 8da4w preq mode) according to a secondary summary of ExecuTorch PRs [SECONDARY]. Flags **[UNVERIFIED]** against the v1.5.1 tag.
- The plan's `llama_xnnpack_qat.yaml` config should be checked against the same tag before use.
- Our own QAT run requires torchao QAT, which we did not confirm in primary docs [UNVERIFIED].

### 4.3 Practical notes

- **Licences.** Qwen3 0.6B, 1.7B and 4B dense models are Apache 2.0, and the family covers 119 languages and dialects [SNIPPET, Qwen blog]. Qwen2.5-1.5B is listed as Apache 2.0 by aggregators [SECONDARY]. Hindi coverage in Qwen is **[UNVERIFIED]**. Check the model cards on Hugging Face before choosing a teacher.
- **Llama as teacher or base.** Any Llama-derived model must carry "Llama" in its name. Whether Llama 3.2 terms restrict using Llama outputs as training data for a non-Llama model is **[UNVERIFIED]**. Read the licence before using Llama outputs as teacher data.
- **T4 limits.** Whether T4 handles BF16 well, and the memory and time for 1B QLoRA on T4, are **[UNVERIFIED]** in this session. The plan's rule is to run the training on Kaggle and keep the laptop free.
- **Data.** Train on deterministic templates and verifier-passed paraphrases only. Keep the held-out campaigns out of the teacher prompts.

---

## 5. Question 4: prompt-injection defences and how to measure them

### 5.1 Defences

| Defence | Mechanism | Evidence | Applies to a 1B on-device model? |
|---|---|---|---|
| **StruQ** (Chen et al., arXiv 2402.06363, USENIX Security 2025) | Reserved delimiter tokens separate instructions from data. Structured instruction tuning. | ASR below 2% for all but one attack (TAP), on Alpaca and Mistral; small utility cost [SNIPPET, v2]. | Needs training and reserved tokens. Untested at 1B. |
| **SecAlign** (Chen et al., arXiv 2410.05451, CCS 2025) | Preference optimisation on injected and clean responses. | Reported 0% ASR against optimisation-free attacks and under 15% against optimisation-based attacks, on 7B to 8B models [SNIPPET]. | Needs training. Untested at 1B. |
| **Meta SecAlign** (arXiv 2507.02735) | Open 8B and 70B models with built-in defence. The 70B is tied to Llama-3.3-70B; the 8B base is **[UNVERIFIED]**. | SNIPPET | Not a 1B model. |
| **Spotlighting** (Hines et al., arXiv 2403.14720) | Delimiting, datamarking (interleaved marker) and encoding of untrusted text. | ASR above 50% reduced to below 2% on GPT-family models, with little task cost. Encoding depends on the model being able to decode it [SNIPPET]. | Encoding is risky at 1B. Datamarking may be workable. Untested. |
| **Instruction hierarchy** (Wallace et al., arXiv 2404.13208) | System over user over tool-output priority, learned by fine-tuning. | Fine-tuned GPT-3.5 Turbo; robust to unseen attack types with small loss on standard tasks. Over-refusal noted [SNIPPET]. | Needs training. |
| **CaMeL** (Debenedetti et al., arXiv 2503.18813) | System layer separates control and data flow, with a custom interpreter tracking provenance. | Reported AgentDojo results differ between versions (77% vs 67% of tasks with provable security) [SNIPPET]. Check the version. | Architectural. Applies to our design in spirit: the LLM never controls flow. |
| **Architecture: no raw text, fixed vocabulary, deterministic verifier** (our design) | Attacker text never enters the model. Output limited to allowed codes. | Supported by the 2026 preprint result (section 3.3) [OPENED abstract]. Not tested on our pipeline. | Yes. This is the one available to a 1B model without training. |

### 5.2 Evaluation suites

| Suite | Size | Notes | Source |
|---|---|---|---|
| **AgentDojo** (NeurIPS 2024 D&B, arXiv 2406.13352) | 97 user tasks and 629 security test cases | Extensible environment. Designed for tool-using agents. Our pipeline is not an agent, so adapt the injection tasks, not the environment. | [SNIPPET] |
| **BIPIA** (arXiv 2312.14197, v4) | 5 scenarios, 250 attacker goals, 25 LLMs | More capable models were more vulnerable in their study. White-box embedding and adversarial training nearly eliminated ASR. Black-box defences did not [SNIPPET]. | [SNIPPET] |
| **InjecAgent** (arXiv 2403.02691) | 1,054 test cases, 17 user tools, 62 attacker tools | ReAct GPT-4 attacked 24% of the time; 47% with a hacking prompt [SNIPPET]. | [SNIPPET] |
| **Adaptive attacks** (Nasr et al., arXiv 2510.09023) | Method, not a suite | Tuned gradient, RL, random-search and human attacks bypassed 12 defences, with above 90% ASR for most. Many originally reported near-zero ASR [SNIPPET]. | [SNIPPET] |
| **Prompt Guard 2** (Meta, 22M and 86M) | Classifier | 22M is DeBERTa-xsmall based, English-focused. 86M is multilingual. 512-token limit. Gated on Hugging Face. Licence **[UNVERIFIED]**. | [SNIPPET, gated; mirror info] |

### 5.3 How to measure injection robustness credibly

1. **Define the threat model per channel.** SMS text is the only attacker-controlled input in the headline setting. Check whether URL domains, brand strings and sender IDs reach the LLM.
2. **Measure the harmful outcome, not any change.** For ELA, harmful means the verdict or the action code moves in the attacker's direction. Text changes inside the allowed set do not count as harmful unless they change the action.
3. **Build the attack set from families, not one template.** Use the AgentDojo and BIPIA scenario ideas, adapted to SMS. Hold out whole families for the final number.
4. **Include adaptive attacks** that see the defence, following Nasr et al. Report the attack budget.
5. **Use intervals.** Wilson intervals per family. Fisher or exact tests for arm comparisons. Report n for every cell.
6. **Include benign look-alikes.** Injection-looking legitimate messages (bank OTP reminders with imperative text) to measure over-refusal, consistent with the low-FPR objective in `CONTEXT.md`.
7. **Sweep prompt format.** Following Sclar et al., report the spread across formats.

---

## 6. Question 5: Hindi and Hinglish output

- Llama 3.2 model card: English, German, French, Italian, Portuguese, Hindi, Spanish and Thai are officially supported. Developers may fine-tune for other languages under the licence [OPENED, GitHub `llama-models`]. **Romanised Hinglish is not listed**, so Hinglish output should be treated as out of scope until tested.
- Tokenization disparity: Petrov et al. (NeurIPS 2023, arXiv 2305.15425) report that the same text can differ in token length by up to 15x across languages, across 17 tokenizers [SNIPPET]. Hindi-specific figures for the Llama 3 tokenizer are **[UNVERIFIED]**. Measure them on our templates.
- IndicGenBench (ACL 2024, arXiv 2404.16816): 29 Indic languages, with a large gap to English in all of them [SNIPPET]. Its tasks are summarisation, MT and cross-lingual QA, not advisory text.
- COMI-LINGUA (arXiv 2503.21670, Findings of EMNLP 2025): human-annotated Hindi-English data for training and evaluation [SNIPPET].
- CodeMixBench (arXiv 2505.05063): code-mixed prompts with Hinglish, performance drops for smaller models. It is a code benchmark, not advisory text [SNIPPET].
- arXiv 2504.19070 (Hinglish chatbot, Gemma3-4B): reports Code-Mixing Index (CMI), BERTScore, and avoids BLEU and ROUGE for code-mixed text [SNIPPET].
- HinglishEval (IISc, 2025): HumanEval translated into Hinglish. Not peer-reviewed [SNIPPET].

**Recommended evaluation for us.**
- Language-free checks first. Verifier pass rate and action accuracy do not depend on language, since IDs and codes are fixed.
- Language-specific checks second: script (Devanagari or Latin) rate, CMI, and a bilingual human rating of about 100 outputs per language.
- Hypothesis to test, not a fact: Llama 3.2 1B will drift to English or mix scripts for Romanised Hinglish prompts. **[UNVERIFIED]**.
- Practical default: human-written Hindi and Hinglish templates, with the model used only for English paraphrase, unless the eval shows otherwise.

---

## 7. Proposed additions to the Kaggle eval (suggestions, not yet pre-declared)

| Arm | Description | Primary metric |
|---|---|---|
| A0 | Current v1 prompt, free JSON, verifier | Verifier pass rate (Wilson CI) |
| A1 | No demonstration leakage, opaque per-request IDs, shuffled codes | Same |
| A2 | Template summary, candidate scoring for IDs and action | Same, plus action accuracy |
| A3 | A2 plus verifier-gated QLoRA | Same, plus unsupported-claim rate |
| I1 | Injection family suite, locked arm | Harmful-flip rate per family |
| I2 | Same suite, free-text arm (replication of the 11/50 result) | Same |
| L1 | Hindi and Hinglish templates, as in section 6 | Script rate, CMI, human rating |

Run the final numbers on the `.pte` exported through ExecuTorch, not on an HF fp16 model. Quantisation changes behaviour, which matters for the copy-bias results.

---

## 8. Open items and unverified claims

- ExecuTorch: whether the Android `LlmModule` and the Python runtime expose the 1.5.1 `LogitProcessor` **[UNVERIFIED]**. Whether the `LogitProcessor` interface needs a post-sample hook for stateful masks **[UNVERIFIED]**.
- ExecuTorch: torch pin for 1.5.1 (see `EXECUTORCH_PLAN.md`), flags for QAT+LoRA export, and the exact `llama_xnnpack_qat.yaml` contents **[UNVERIFIED]** against the tag.
- llama.cpp grammar README: canonical URL not opened. Mirror content only [SECONDARY].
- XGrammar, Outlines, lm-format-enforcer, llguidance: performance and licence claims are from papers, READMEs or aggregators, not measured here.
- DOMINO accuracy numbers: check model and setting in the paper before quoting [SNIPPET].
- CaMeL: AgentDojo figures differ between versions [SNIPPET].
- SecAlign and StruQ: numbers are for 7B to 8B models, not 1B.
- Meta SecAlign 8B base model **[UNVERIFIED]**.
- Prompt Guard 2 licence **[UNVERIFIED]**.
- Llama 3.2 licence: whether Llama outputs may be used as teacher data for a non-Llama model **[UNVERIFIED]**.
- Qwen2.5 Hindi coverage **[UNVERIFIED]**.
- Llama 3.2 1B copy-bias size dependence: the claim that smaller models copy more is **[UNVERIFIED]**.
- Fisher and Wilson numbers in section 0 are our calculations on the reported counts. They assume independence, which may not hold.

---

## 9. Bibliography

Status tags as defined at the top. Links are the URLs seen in this session.

**ExecuTorch and runtimes**
1. pytorch/executorch issue #19476, "[LLM Runner] Wire grammar-constrained decoding end-to-end". https://github.com/pytorch/executorch/issues/19476 [OPENED, state open, no linked PR, fetched 2026-10-31]
2. pytorch/executorch PR #19517, "Add LogitProcessor interface for pre-sampling logit transforms", merged 2026-05-15. https://github.com/pytorch/executorch/pull/19517 [OPENED]
3. pytorch/executorch PR #19351, "Allow TextDecoderRunner to reuse a persistent Sampler", merged 2026-05-12. https://github.com/pytorch/executorch/pull/19351 [OPENED]
4. ExecuTorch v1.5.1 source, `extension/llm/sampler/logit_processor.h` and `extension/llm/runner/text_token_generator.h`. https://github.com/pytorch/executorch/tree/v1.5.1/extension/llm/sampler and raw file at https://raw.githubusercontent.com/pytorch/executorch/v1.5.1/extension/llm/sampler/logit_processor.h [OPENED]
5. llama.cpp grammars README (mirror only). https://sources.debian.org/src/llama.cpp/7965%2Bdfsg-1/grammars/README.md [SECONDARY]
6. XGrammar, arXiv 2411.15100. https://arxiv.org/abs/2411.15100 [SNIPPET]
7. Outlines, Willard and Louf, arXiv 2307.09702. https://arxiv.org/abs/2307.09702 [SNIPPET]
8. lm-format-enforcer. https://github.com/noamgat/lm-format-enforcer [SNIPPET]
9. llguidance. https://github.com/guidance-ai/llguidance [SNIPPET]
10. JSONSchemaBench, arXiv 2501.10868. https://arxiv.org/abs/2501.10868 [SNIPPET]
11. DOMINO, Beurer-Kellner, Fischer and Vechev, arXiv 2403.06988 (ICML 2024). https://arxiv.org/abs/2403.06988 [SNIPPET]
12. Grammar-Aligned Decoding, Park et al., arXiv 2405.21047 (NeurIPS 2024). https://arxiv.org/abs/2405.21047 [SNIPPET]
13. Tam et al., "Let Me Speak Freely?", arXiv 2408.02442 (EMNLP 2024 Industry). https://arxiv.org/abs/2408.02442 [SNIPPET]

**Output bias and copy bias**
14. Zhao et al., "Calibrate Before Use", arXiv 2102.09690 (ICML 2021). https://arxiv.org/abs/2102.09690 [SNIPPET]
15. Sclar et al., "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design", arXiv 2310.11324 (ICLR 2024). https://arxiv.org/abs/2310.11324 [SNIPPET]
16. Ali, Wolf and Titov, "Mitigating Copy Bias in In-Context Learning through Neuron Pruning", arXiv 2410.01288. https://arxiv.org/abs/2410.01288 [OPENED abstract]
17. Robinson, Rytting and Wingate, "Leveraging LLMs for Multiple Choice Question Answering", arXiv 2210.12353 (ICLR 2023). https://arxiv.org/abs/2210.12353 [SNIPPET]

**Faithfulness**
18. Rashkin et al., "Measuring Attribution in NLG Models" (AIS), arXiv 2112.12870. https://arxiv.org/abs/2112.12870 [SNIPPET]
19. Min et al., "FActScore", arXiv 2305.14251 (EMNLP 2023). https://arxiv.org/abs/2305.14251 [SNIPPET]
20. Gao et al., "ALCE", arXiv 2305.14627 (EMNLP 2023). https://arxiv.org/abs/2305.14627 [SNIPPET]
21. Li et al., "AttributionBench", arXiv 2402.15089. https://arxiv.org/abs/2402.15089 [SNIPPET]
22. Tang et al., "MiniCheck", arXiv 2404.10774. https://arxiv.org/abs/2404.10774 [SNIPPET]
23. Niu et al., "RAGTruth", arXiv 2401.00396 (ACL 2024). https://arxiv.org/abs/2401.00396 [SNIPPET]
24. Asai et al., "Self-RAG", arXiv 2310.11511 (ICLR 2024). https://arxiv.org/abs/2310.11511 [SNIPPET]
25. "Impact of Model Size on Fine-tuned LLM Performance in Data-to-Text Generation", arXiv 2407.14088. https://arxiv.org/abs/2407.14088 [SNIPPET]
26. TrueBrief, arXiv 2601.04212 (preprint). https://arxiv.org/abs/2601.04212 [SNIPPET, UNVERIFIED results]
27. Deep et al., "Evaluation of Prompt Injection Defenses in Large Language Models", arXiv 2604.23887 (preprint, 2026). https://arxiv.org/abs/2604.23887 [OPENED abstract]

**Fine-tuning**
28. Hu et al., "LoRA", arXiv 2106.09685. https://arxiv.org/abs/2106.09685 [SNIPPET]
29. Dettmers et al., "QLoRA", arXiv 2305.14314. https://arxiv.org/abs/2305.14314 [SNIPPET]
30. Meta, "Introducing quantized Llama models with increased speed and a reduced memory footprint". https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/ [OPENED]
31. Llama 3.2 model card, `meta-llama/llama-models`. https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md [OPENED]
32. TempLM, ACL 2023 Findings, arXiv 2205.11055. https://arxiv.org/abs/2205.11055 [SNIPPET]
33. Qwen3 blog. https://qwenlm.github.io/blog/qwen3/ [SNIPPET]
34. Qwen2.5-1.5B-Instruct model card (licence). https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct [SECONDARY; not opened]

**Injection**
35. Chen et al., "StruQ", arXiv 2402.06363 (USENIX Security 2025). https://arxiv.org/abs/2402.06363 [SNIPPET]
36. Chen et al., "SecAlign", arXiv 2410.05451 (CCS 2025). https://arxiv.org/abs/2410.05451 [SNIPPET]
37. Chen et al., "Meta SecAlign", arXiv 2507.02735. https://arxiv.org/abs/2507.02735 [SNIPPET]
38. Hines et al., "Spotlighting", arXiv 2403.14720. https://arxiv.org/abs/2403.14720 [SNIPPET]
39. Wallace et al., "The Instruction Hierarchy", arXiv 2404.13208. https://arxiv.org/abs/2404.13208 [SNIPPET]
40. Debenedetti et al., "Defeating Prompt Injections by Design" (CaMeL), arXiv 2503.18813. https://arxiv.org/abs/2503.18813 [SNIPPET]
41. Nasr et al., "The Attacker Moves Second", arXiv 2510.09023. https://arxiv.org/abs/2510.09023 [SNIPPET]
42. Debenedetti et al., "AgentDojo", arXiv 2406.13352 (NeurIPS 2024 D&B). https://arxiv.org/abs/2406.13352 [SNIPPET]
43. Yi et al., "BIPIA", arXiv 2312.14197. https://arxiv.org/abs/2312.14197 [SNIPPET]
44. Zhan et al., "InjecAgent", arXiv 2403.02691. https://arxiv.org/abs/2403.02691 [SNIPPET]
45. Llama Prompt Guard 2 (gated). https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M [SNIPPET, mirror info]
46. Fuziwara Filho et al., "A Cross-Lingual Security Evaluation of Prompt Injection Defenses in Small Language Models", SBSeg. https://sol.sbc.org.br/index.php/sbseg_estendido/article/view/44551 [SNIPPET]
47. Chen et al., "DefensiveTokens", arXiv 2507.07974. https://arxiv.org/abs/2507.07974 [SNIPPET]

**Multilingual**
48. Petrov et al., "Language Model Tokenizers Introduce Unfairness Between Languages", arXiv 2305.15425 (NeurIPS 2023). https://arxiv.org/abs/2305.15425 [SNIPPET]
49. Singh et al., "IndicGenBench", arXiv 2404.16816 (ACL 2024). https://arxiv.org/abs/2404.16816 [SNIPPET]
50. COMI-LINGUA, arXiv 2503.21670 (Findings of EMNLP 2025). https://arxiv.org/abs/2503.21670 [SNIPPET]
51. CodeMixBench, arXiv 2505.05063. https://arxiv.org/abs/2505.05063 [SNIPPET]
52. Hinglish sample-efficient chatbot, arXiv 2504.19070. https://arxiv.org/abs/2504.19070 [SNIPPET]
53. HinglishEval (IISc, 2025). https://kiac.iisc.ac.in/wp-content/uploads/2025/06/HinglishEval.pdf [SNIPPET, not peer-reviewed]

**Secondary, not primary sources (do not cite as primary)**
54. TorchAO QAT and LoRA figures and ExecuTorch QAT+LoRA export flags, via search summaries of ExecuTorch PRs 6486, 10064 and 10077. https://github.com/pytorch/executorch/pull/6486 [SECONDARY, UNVERIFIED flags]
