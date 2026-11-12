# Llama 1B explainer on ExecuTorch: speed and faithfulness options (Pi 5 and Android Arm)

Research date: 2026-11-06. Scope: Llama-family models only, ExecuTorch runtime, offline on-device, about 2 weeks, free compute (Kaggle). Source quality is mixed: many results are vendor claims, search snippets, or single blogs. Each item is tagged where it matters. "Inference" marks my own reasoning from cited facts. "Unverified" marks claims I could not confirm.

Tool-use note: about 17 tool calls, slightly above the 15-call guideline, to cover the gaps. Two fetches failed (a dev.to page returned 404, and an Arm URL redirected), so the Gemma and Arm-blog points below rely on search snippets where noted.

## Key Question 1: ExecuTorch Llama speed levers on Cortex-A76 (Pi 5) and phones

### Takeaway
Decode on the Pi 5 is memory-bandwidth-bound, so the measured 9.8-11.6 tok/s is already close to the ceiling for a ~1.1 GB INT4 model. Thread tuning and kernel choice will help prefill more than decode. The biggest decode wins are fewer generated tokens, and possibly batched decode or shared-prefix KV reuse, but neither is confirmed for this runner.

### Cited Findings
- Pi 5 sustains about 13.8 GB/s memory bandwidth; decode tokens/s = effective bandwidth / bytes read per token; the page's own smallest-model example gives about 40 tok/s with 346 MB read per token. Pi 5 is 4x Cortex-A76 with 2 GB LPDDR4X on a 32-bit interface in that article. — [Arm Learn: LLM decode roofline](https://learn.arm.com/learning-paths/cross-platform/llm-decode-roofline/how-decode-uses-memory/)
- Earlier ExecuTorch Pi 5 work (May 2024) reported Llama3-8B INT4 at roughly 2 tok/s on Pi 5. This is old and not for the current release. — [PyTorch dev-discuss: Llama3-8b on Pi 5 with ExecuTorch](https://dev-discuss.pytorch.org/t/run-llama3-8b-on-a-raspberry-pi-5-with-executorch/2048)
- Ollama (llama.cpp backend) with llama3.2:1b Q4_K_M reported 11.22 tok/s on Pi 5 in a lab study. Snippet-level only, not fetched; treat as a rough comparison point. — [Stratosphere lab study on Pi 5 LLM performance](https://stratospherelinuxips.readthedocs.io/en/develop/immune/research_rpi_llm_performance.html)
- Arm and Meta (vendor-reported): Llama 3.2 1B quantized on Samsung S24+ via ExecuTorch with KleidiAI through XNNPACK gave prefill over 350 tok/s and decode over 40 tok/s. KleidiAI improved prefill by an average of over 20%. Peak RSS (max seq len 2048) fell from 3.1 GiB (BF16) to 1.9 GiB (SpinQuant). Benchmark: sequence 128, prompt 64, 6 of 8 CPUs, single warmup, over adb. Not stated: group size, whether i8mm was active. — [Arm Community: LLM inference with quantized Llama and ExecuTorch KleidiAI](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai)
- Arm says the KleidiAI kernel gain of up to 20% applies to Cortex-A v9 CPUs with the i8mm extension. Cortex-A76 has dotprod but not i8mm, so this gain should not be assumed on the Pi 5. — same Arm source; the "no i8mm on A76" point is from the project context and is a hardware fact I did not re-verify.
- Kernel name cited by Arm: `matmul_clamp_f32_qai8dxp_qsi4c32p` (4-bit weights, per-block scale, 8-bit activations). — same Arm source
- Independent caution (search snippet only, dev.to fetch returned 404): a Gemma 4 E2B port on Pi 5 with ExecuTorch 1.2.0 reported 0.87 tok/s versus 6.71 tok/s for llama.cpp. The author said XNNPACK rejected fused INT4 subgraphs on aarch64, and the workaround `per_op_mode=True` disables fusion. Not the same model, so it is a warning, not a direct counterexample. — [dev.to: Gemma 4 on ExecuTorch, Pi 5](https://dev.to/bamb00boy/first-gemma-4-executorch-deployment-on-raspberry-pi-5-and-why-its-77x-slower-than-llamacpp-gmb)
- Meta/ExecuTorch OnePlus 12 numbers (snippet only, from a search aggregator): 1B BF16 19.2 decode tok/s at ~3,185 MiB RSS; 1B SpinQuant 50.2 tok/s at ~1,921 MiB RSS. — [ExecuTorch Llama example / PR context](https://github.com/pytorch/executorch/pull/6486/files) (found via search; the exact figures were not checked in the PR)

### Inferences
- Decode ceiling for the current .pte: the 1.14 GB file implies about 1.1 GB read per decode token (weights dominate; KV cache is small at short context). 13.8 GB/s / ~1.1 GB ≈ 12 tok/s. Measured 9.8-11.6 tok/s is about 80-95% of that ceiling. So more threads or kernel tweaks will change decode little. Decode gains must come from fewer tokens or fewer bytes per token. This is my calculation, not a cited figure.
- Smaller weights per token are the main decode lever: quantize embedding/lm_head more aggressively or drop the output projection size if the runner allows it (unverified for this export). Llama 3.2 1B is about 1.2B parameters; the tied embedding is a large fraction of bytes.
- Prefill (compute-bound) benefits from thread count and from KleidiAI/dotprod kernels. Sweep 3, 4 threads and measure prefill separately from decode. Thermal limits (70-73 °C under 3-4 threads) mean 4 threads may throttle; test 3 vs 4 over 20+ calls.
- Batched decode (several slots decoded in one forward pass) would amortize the weight read across sequences, since decode is bandwidth-bound. This is the most promising throughput lever I can reason about, but I found no evidence that the ExecuTorch Python runner supports batch>1 decode for Llama. Unverified; needs a check of `examples/models/llama` in the ExecuTorch repo.
- Shared-prefix KV reuse (system prompt cached once, reused across calls) would cut prefill, which is about 1-2 s of the 1.7 s p50 path. I found no ExecuTorch-specific evidence that its runner supports cross-call KV reuse. General prefix-caching sources (vLLM-style) say reuse is exact-token and invalidated by any early change, so the prompt prefix must be byte-identical across calls. Unverified for ExecuTorch.
- max_seq_len: lowering it reduces KV-cache memory but does not change decode bandwidth much at short context. Its main effect is on memory and on whether the static-shape graph fits. Use the smallest value that covers prompt + 20 tokens.

### Gaps
- No ExecuTorch-published benchmark for Llama 1B on Cortex-A76 or Pi 5 in 2025-2026 was found. The Arm numbers are for S24+ (Armv9 with newer features). The Pi 5 numbers are the user's own measurements.
- Whether the ExecuTorch Python runner exposes prefix-KV reuse or batched decode: not found. Needs repo inspection.
- Whether the current XNNPACK build on aarch64 fuses INT4 subgraphs for this export: not verified; the Gemma report suggests this can fail on some exports. Check the export log for fused vs unfused kernels and compare with `per_op_mode`.

## Key Question 2: Llama-family alternatives (1B vs 3B, distilled/pruned, Indic fine-tunes) and size/speed

### Takeaway
Within Llama 3.2, staying at 1B is the only option that fits the 4 GB Pi 5 comfortably. The 3B model should roughly halve decode tok/s under bandwidth scaling, which I calculated from parameter count; I found no measured 3B-on-Pi-5 ExecuTorch figure. Indic fine-tunes exist but I found no evidence that they help Hinglish specifically.

### Cited Findings
- Llama 3.2 1B and 3B are supported in the ExecuTorch Llama example with 4-bit groupwise quantization. The example defers accuracy to the Llama 3.2 model card. — [ExecuTorch Llama example (via search)](https://android.googlesource.com/platform/external/executorch/+/HEAD/examples/models/llama)
- Llama 3.2 1B Instruct FP8 (Neural Magic): 5-shot MMLU 47.66 base vs 47.76 quantized; the card says quantized stays within 1.0% of the base on its tested benchmarks. This is FP8, not INT4, so it shows that 1B quantization can be near-lossless at higher precision, not that INT4 is. — [Neural Magic Llama 3.2 1B Instruct FP8 model card](https://huggingface.co/neuralmagic/Llama-3.2-1B-Instruct-FP8)
- IFEval for Llama 3.2 1B Instruct: 0.45 in an Embedl edge-inference benchmark CSV (unquantized). — [Embedl Edge-Inference-Benchmarks data](https://huggingface.co/spaces/embedl/Edge-Inference-Benchmarks/blob/refs%2Fpr%2F6/data/acc-Llama-3.2.csv)
- Hindi, zero-shot cross-lingual accuracy (Paramanu paper): Llama-3.2 1B about 34.5 average, Llama-3.2 3B about 40.1. Hindi only, not Hinglish. — [Paramanu (arXiv 2401.18034)](https://arxiv.org/pdf/2401.18034) (figures from search snippet, not fetched)
- Community Hindi fine-tune of Llama 3.2 1B Instruct (Ryder99): model card claims significantly improved Hindi and comparable English. No published benchmark attached. Unverified. — [featherless.ai listing](https://featherless.ai/models/Ryder99/Llama-3.2-1B-Instruct-Hindi)
- Meta describes Llama 3.2 quantized 1B/3B as QAT+LoRA starting from BF16 SFT checkpoints, plus SpinQuant; SpinQuant's repo is ExecuTorch-compatible. — [Meta Llama 3.2 model cards](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)

### Inferences
- Llama 3.2 3B INT4 would be about 2x the weight bytes of 1B; with the same bandwidth ceiling, decode would fall to about 5-6 tok/s on the Pi 5 (my scaling, not measured). RSS would also rise well above the current 1.84 GB, which is risky on a 4 GB board with no cooler. Keep 1B unless faithfulness gains are shown to need it.
- A Llama 3.2 1B Hindi fine-tune might help Devanagari but is unlikely to produce clean romanised Hinglish. It adds a second model artifact to validate. Low priority.
- Distilled or pruned Llama variants (smaller than 1B) are not in scope: I found no Llama-family sub-1B model with published quality numbers for this use.

### Gaps
- No measured ExecuTorch figure for Llama 3.2 3B on Pi 5 or A76. The 3B estimate is my scaling.
- No independent Hinglish evaluation of any Llama-family 1B model.

## Key Question 3: Faithfulness for tiny LMs (constrained decoding, few-shot, stop rules, reranking, QAT vs PTQ, entailment checkers)

### Takeaway
The clearest path to fewer verifier failures is adapter quality: the QAT recipe Meta and ExecuTorch use (BF16 LoRA, 4-bit groupwise weights, dynamic 8-bit activations) is the reference point, and the current 8da4w export likely does not match it. Constrained decoding is the right mechanism, but ExecuTorch's grammar support is not shipped. Verifier-guided reranking is cheap to build but costs decode time that the Pi cannot easily spare.

### Cited Findings
- ExecuTorch grammar-constrained decoding (issue #19476): open since 2026-05-11, no linked PRs, labelled triaged, on the project board as "Todo". Plan: create a GrammarLogitProcessor when `GenerationConfig.grammar` is set, call accept_token after sampling, and add a JSON-schema end-to-end test. Target: mask overhead under 100 µs. — [pytorch/executorch #19476](https://github.com/pytorch/executorch/issues/19476)
- LogitProcessor interface: PR #19517 adds a single virtual `process(logits)` hook that edits logits in place between forward and sampler, plus methods to add, clear and count processors on the text token generator. Only an unofficial mirror was found in search, so verify on the official repo. — [PR #19517 (mirror, via search)](https://gh.ali360.cloudns.ch/pytorch/executorch/pull/19517)
- Meta-recipe QAT+LoRA on ExecuTorch (ExecuTorch example README, via search): 4-bit groupwise weights with group size 32, 8-bit per-token dynamic activations, LoRA adaptor weights and activations kept in BF16. Accuracy deferred to the model card. — [ExecuTorch example README, via search](https://github.com/pytorch/executorch/pull/6486/files)
- TorchAO QAT on Llama-3.2-1B (MMLU-Pro): QAT recovered 36.0% of a 4.5-point loss. A partial recovery, not a full one. Unsloth reports up to ~70% recovery on larger models. — [PyTorch: QAT in torchao II](https://pytorch.org/blog/quantization-aware-training-in-torchao-ii/), [Unsloth QAT blog](https://unsloth.ai/docs/blog/quantization-aware-training-qat) (both via search, numbers not fetched)
- Recover-LoRA (AMD, arXiv 2510.08600) is a data-free repair method for degraded models. Not a QAT method; its numbers do not transfer to this setup. — [arXiv 2510.08600](https://arxiv.org/pdf/2510.08600)
- MiniCheck (arXiv 2404.10774, April 2024, rev. October 2024): fine-tuned small fact-checkers. Authors report a 770M-parameter MiniCheck-FT5 reaches GPT-4-level accuracy on LLM-AggreFact at about 400x lower cost. Author-reported; not independently checked. — [MiniCheck arXiv](https://arxiv.org/abs/2404.10774)

### Inferences
- The retention gap (FP16 pass 97%, 8da4w 84%) is most likely a train/export mismatch: an adapter learned on a quantized base or with a different quant scheme than the export. The Meta-recipe match (g32 weights, dynamic int8 activations, BF16 LoRA, QAT rather than post-hoc PTQ) is the first thing to check. Expect partial recovery only, consistent with the 36% TorchAO figure, so set the retention gate with that in mind.
- Constrained decoding is the strongest faithfulness lever for the specific defects in the notes (echoing, unterminated, quote-stop). Until a grammar processor ships, a C++ runner with a custom LogitProcessor (the 1.5.1 interface, per the project notes) is the practical route. It can ban quote tokens, enforce a sentence-end stop, and bias toward slot-allowed vocabulary. Effort: medium, needs C++ build on the Pi and Android.
- Few-shot vs zero-shot: for a 1B model the usual finding is that few-shot examples of the exact target format help more than instructions, but I found no citable 1B-specific study. Unverified; test a 2-3 example prompt on the eval set.
- Verifier-guided reranking (k candidates, keep first verifier pass, template otherwise): cost is additional decode. At ~10 tok/s and 20 tokens per sentence, k=2 adds about 2 s when the first candidate fails. Given 19-33% failure, expected cost is about 0.3-0.7 s per sentence. Reasonable only with early stop.
- MiniCheck-class checkers (AlignScore-based, 770M) are too large to co-run with the 1.84 GB LLM on a 4 GB Pi 5 with a thermal ceiling. Use them offline on Kaggle to label failures, not on device. The deterministic verifier stays the on-device gate.

### Gaps
- No direct evidence that grammar or logit-bias constrained decoding improves faithfulness for a 1B Llama on ExecuTorch. The reasoning is from the defect profile, not a measured result.
- No independent measurement of QAT+LoRA retention on the Llama 3.2 1B ExecuTorch export; the TorchAO 36% figure is a different benchmark.
- MiniCheck's on-device footprint was not measured.

## Key Question 4: Hinglish and code-mixed generation with small Llama models

### Takeaway
I found no evidence that Llama 3.2 1B produces usable romanised Hinglish, and the project's own result (8/37 pass) agrees. The practical path is templates with human-written Hinglish phrasing per evidence type, with the LLM limited to English or to copying verified slot text. Generating Devanagari and transliterating is untested here and adds a second failure point.

### Cited Findings
- Hindi is an officially supported Llama 3.2 language. Llama 3.2 is trained on a wider set. — [Meta Llama 3.2 model cards](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)
- Hindi literature uses LLM-as-judge (GPT-4o) and newer Hindi versions of IFEval/MT-Bench/GSM8K. No romanised code-mixed evaluation of a 1B Llama was found. — [Llama-3-Nanda-10B-Chat review](https://themoonlight.io/review/llama-3-nanda-10b-chat-an-open-generative-large-language-model-for-hindi) and the 2025 Hindi benchmark paper (arXiv 2508.19831, search only)
- Paramanu (arXiv 2401.18034): Llama-3.2 1B weak on Hindi zero-shot (~34.5 avg). — [Paramanu](https://arxiv.org/pdf/2401.18034)

### Inferences
- Stop-on-quote and unterminated failures in romanised Hinglish are consistent with a model that has weak code-mixed priors and a stop rule that fires on quote marks. Fixing the stop rule (sentence-end, ignore quote marks in-sentence) may recover some outputs, but expect poor fluency.
- Templates with 2-3 Hinglish variants per evidence type, plus slot fill, give deterministic verifier pass and no hallucination. This is the highest-impact, lowest-effort option for Hinglish.
- Devanagari generation then transliteration: adds an extra model call or a transliteration library; Hindi quality at 1B is weak per Paramanu. Not recommended unless a Devanagari eval shows the 1B passes.

### Gaps
- No Hinglish or romanised code-mixed benchmark for Llama 3.2 1B. The eval on this project (8/37) is the only signal I have.
- No measured comparison of template vs LLM for Hinglish.

## Key Question 5: Is the LLM worth it vs templates?

### Takeaway
2025 studies show LLM-written security explanations can help users, but the comparisons are against hand-written warnings, not against carefully written templates. For this project, the evidence supports an LLM only if it beats a well-written template on the verifier and on a small comprehension check. Templates should be the baseline and the fallback.

### Cited Findings
- Cau et al., "Can Large Language Models Improve Phishing Defense?" (arXiv 2507.07916, July 2025): between-subjects study, N = 750. Warnings generated by Claude 3.5 Sonnet and Llama 3.3 70B matched or beat hand-crafted explanations at reducing phishing susceptibility. Claude-generated warnings did best. Abstract-level, not fetched. — [arXiv 2507.07916](https://papers.cool/arxiv/2507.07916)
- PhishXplain (arXiv 2505.06836, May 2025): local LLaMA 3.2 generated contextual phishing warnings. Prolific study, 150 participants: explanations improved detection and user confidence, mostly for low-proficiency users. — [arXiv 2505.06836](https://arxiv.org/pdf/2505.06836v1) (search snippet)
- "Does Johnny Get the Message?" (arXiv 2505.22435) proposes an evaluation framework for LLM-rewritten alerts; I did not find a human comprehension result in the snippets. — [Literature review](https://www.themoonlight.io/review/does-johnny-get-the-message-evaluating-cybersecurity-notifications-for-everyday-users)
- Sajadi et al. (Drexel preprint, 2025): Claude 3, GPT-4 and Llama 3 flagged insecure code in only 12.6-40% of cases. This is a caution about relying on LLMs to judge security, not about explanation quality. — [Drexel research record](https://researchdiscovery.drexel.edu/esploro/outputs/preprint/Do-LLMs-Consider-Security-An-Empirical/991022030047404721)

### Inferences
- The explanation content (what the evidence means, what to do) is what helps users. Well-written templates can carry that content if each evidence type has a precise, specific explanation. The LLM's marginal value is variety and personalization, which is the part this project already struggles to verify.
- A cheap test: run template vs LLM sentences through the same verifier and a 10-20 person comprehension check on the Kaggle or local team. Decide the default from that, not from the literature.

### Gaps
- No study compares a carefully written template to a 1B on-device LLM for security explanations.
- The Cau and PhishXplain results use larger or hosted models or hand-written baselines; they do not show the 1B case.

## Ranked options (impact / effort)

1. **Template-first with LLM as optional** (high impact on faithfulness and Hinglish; low effort). Ship deterministic templates for all evidence types and Hinglish. Keep the LLM as a gated enhancement. Measure the verifier pass rate and comprehension for both.
2. **Fix the stop rule and retry** (medium-high impact on unterminated ~9%; low effort). Stop at sentence end, ignore quote marks, one retry on verifier failure, then template. Cost: about 0.3-0.7 s per sentence on average, mostly on failures.
3. **Align the adapter export with the Meta/ExecuTorch QAT recipe** (high impact if the adapter was trained on a different quant scheme; medium-high effort on Kaggle). Check group size 32, dynamic 8-bit activations, BF16 LoRA, and train with QAT. Expect partial recovery; re-set the retention gate.
4. **C++ runner with a custom LogitProcessor** (high impact on echo, quote-stop, and vocabulary control; medium-high effort). Needed because the Python runner does not expose the hook. Watch PR #19517 and issue #19476 for a shipped grammar path before building.
5. **Latency: thread sweep, per-op vs fused check, prompt shortening** (medium impact on prefill; low effort). Measure prefill and decode separately at 3 and 4 threads over 20+ calls with thermal logging. Check fused INT4 kernels in the export log.
6. **Shared-prefix KV reuse or batched decode** (potentially high impact on throughput; unknown effort). Only pursue after confirming runner support in the ExecuTorch source.
7. **Verifier-guided reranking (k=2 on failure)** (medium impact on pass rate; low-medium effort). Only with early stop and a fallback.
8. **Llama 3.2 3B** (unclear impact, high memory and thermal risk on the 4 GB Pi). Not recommended for the 2-week window.
9. **MiniCheck-class checker** (offline, Kaggle only). Use to label failures and tune the verifier, not as an on-device gate.

## Two-week plan suggestion (inference)
- Days 1-3: template baseline plus Hinglish templates; stop-rule fix; thread and prefill benchmark on Pi 5 with thermal logging.
- Days 4-8: inspect the adapter export against the Meta/ExecuTorch recipe on Kaggle; re-run the retention gate.
- Days 9-12: if the C++ runner is needed, prototype a LogitProcessor for quote-ban and sentence stop on Pi 5.
- Days 13-14: comprehension check (template vs LLM) and decision.

## Unverified items to check first
- Whether the ExecuTorch Python runner supports batch>1 decode or cross-call KV reuse (Key Question 1).
- Whether the current export fuses INT4 kernels on aarch64 (Gemma report, Key Question 1).
- Exact group size and activation quantization in the project's 8da4w export (Key Question 3).
- Llama 3.2 3B memory and speed on the Pi 5 (not measured anywhere I found).
