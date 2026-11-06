# Rakshak-Edge: optimisation research (2026-11-06)

Haiku research pass, read-only. Repo read from `main` via `gh api` (HANDOFF, STATUS, CONTEXT, EXECUTORCH_PLAN, ELA_RESEARCH, verifier.py, llama_export.py). Nothing edited, committed or pushed.
Tags: **[V]** checked in a fetched or searched source. **[U]** unverified. **[D]** my derivation from cited numbers.

## Repo facts that drive the ranking
- Llama is the only heavy compute. The student is under 10 MB and the NIDS model is 6.7 KB.
- Open blockers: GroundLM FPR run not done (HANDOFF task 1). No `.pte` has run on hardware. `llama_export` has produced no output. Rank-arm "0 s, 0 tokens" is flagged as suspicious.
- **Inconsistency to fix before writing:** HANDOFF reports slot-arm verifier pass 0.91, while ELA_RESEARCH cites a 7.7% verifier pass rate. These are probably different prompt versions. Reconcile them.
- **Rank arm is close to a no-op.** `rank_candidates` (verifier.py L618-640) returns 4 frames that differ only in a prefix ("This message...", "Be careful: this message..."). Log-likelihood selection among them cannot carry evidence content. The "0 tokens" is consistent with scoring-only, not generation **[U: eval job not read]**.
- Export sanity uses `max_seq_length` 2048, `GEN_TOKENS` 16, temperature 0 (llama_export.py).

## Top 5 optimisations (ranked)

1. **Measure the shipped `.pte` on real Arm hardware first.** Load `executorch-community/...SpinQuant_INT4_EO8-ET` (1.14 GB, already pinned) through `executorch.runtime` on a Pi 5 (8 GB if possible) and one Android phone. Record load time, TTFT, decode tok/s, peak RSS, and temperature. Use a 300-token prompt and 64 new tokens. *Impact:* critical, since no run exists. *Effort:* 6-10 h. *$:* 0 (own hardware). *Risk:* Pi decode may be single-digit tok/s, and a 4 GB Pi may OOM.
   - Reference: Arm reports S24+ SpinQuant prefill over 350 tok/s and decode over 40 tok/s (prompt 64, seq 128, 6 threads), PTE 1.1 GiB, RSS 1.9 GiB at max seq 2048 [V](https://developer.arm.com/community/arm-community-blogs/b/ai-and-ml-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai).
   - A Gemma 4 E2B run on Pi 5 reported 0.87 tok/s through ExecuTorch and XNNPACK [V, secondary blog, not reproduced](https://dev.to/bamb00boy/first-gemma-4-executorch-deployment-on-raspberry-pi-5-and-why-its-77x-slower-than-llamacpp-gmb). This is a warning sign for Pi performance.

2. **Make Llama do one short prefill and at most 16-24 decode tokens.** Use a template summary, a short system prompt with no real IDs in examples, and a JSON action/ID output. *Impact:* high on latency. *Effort:* 3-5 h. *$:* 0. *Risk:* candidate scoring needs K prefills per decision. Prefix/KV reuse across calls is not documented for ExecuTorch **[U]**, so the prompt must stay short.
   - Cost per token: on S25 Ultra, 256 prefill tokens take about 470 ms and decode runs about 67 tok/s [V, paper Appendix A](https://arxiv.org/html/2605.08195v1). So a 300-token prompt is about 0.5-0.6 s on that device **[D]**. The plan's 96-128 token budget is too high for Pi.
   - KV cache for Llama 3.2 1B is about 32 KB per token in fp16, which gives about 64 MB at 2048 **[D from architecture; check config.json]**. The weights dominate RSS, so `max_seq_len` is not the lever. The paper gives no RSS-vs-context numbers [V].

3. **Make Llama's contribution measurable and real.** Replace the rank arm's frame choice with a choice between evidence-bearing options, and add a no-LLM template baseline and a random-choice baseline on the same packages. Add a deterministic polarity or negation check (for example, "safe" near a HIGH risk). Use MiniCheck-FT5 (770M) only as a Kaggle-side judge, validated on 200 human-labelled items. *Impact:* high for credibility, low for speed. *Effort:* 4-6 h. *$:* 0 (Kaggle quota). *Risk:* MiniCheck is evaluated on English LLM-AggreFact, not Hinglish [V abstract](https://arxiv.org/abs/2404.10774).
   - The English slot-lexicon pass is only 28%, and the lexicon is the weakest part. Fix lexicon coverage before blaming the model.

4. **Hit FPR at or below 11 of 1,133 on GroundLM without retraining.** First run the already-planned GroundLM test with each shipped model at its cal threshold, with clustered CIs. If it fails, pick thresholds with Learn-then-Test: a finite-sample risk-control procedure that tests threshold candidates with a multiple-testing correction on a held-out negatives set [V abstract/search, arXiv 2110.01052](https://arxiv.org/abs/2110.01052). For class-conditional or per-source thresholds, Mondrian conformal is the general framework [V search, arXiv 2502.17264](https://arxiv.org/abs/2502.17264). *Impact:* high, since FPR is the primary safety objective. *Effort:* 2-4 h. *$:* 0. *Risk:* exchangeability with the cal set is likely violated across sources, so treat any bound as empirical. The 169 hash-only IIITD rows are missing.
   - INT8 for the student is low value. ExecuTorch has an XNNPACK quantizer, and its docs example is MobileNetV2 [V search](https://android.googlesource.com/platform/external/executorch/+/refs/heads/android16-release/examples/xnnpack). The student is already tiny, so the Llama path is the bottleneck. Do this only if the detector exceeds about 50 ms on a Pi.

5. **Android via the existing demo, not a new app.** Use the `executorch-android:1.5.1` AAR (LlmModule) from the stable docs, with `adb push` for the `.pte` and tokenizer. Set threads to the number of performance cores; the paper uses that rule [V paper]. Gate the Llama feature on RAM, with the threshold to be measured **[U]**. *Impact:* required for a demo. *Effort:* 12-20 h. *$:* 0. *Risk:* the LlamaDemo path differs across three sources. Use the stable docs page: [docs.pytorch.org/executorch/stable/llm/llama-demo-android.html](https://docs.pytorch.org/executorch/stable/llm/llama-demo-android.html). Note that my fetch of that page returned empty this session, so the LlmModule API is from prior research.

## Per-topic findings

**1. ExecuTorch speed, memory, pins**
- Meta README, OnePlus 12: SpinQuant decode 50.2 tok/s, TTFT 0.3 s, RSS 1,921 MiB; BF16 is 19.2 tok/s [V, cited in EXECUTORCH_PLAN, from the Meta README].
- S25 Ultra, GS32 (paper): prefill 525-529 tok/s, decode 66-67 tok/s. No power numbers. No mmap statement. [V](https://arxiv.org/html/2605.08195v1)
- KleidiAI is on by default from ExecuTorch 0.7 [V](https://huggingface.co/blog/arm/executorch-0-dot-7). Only a subset of SME micro-kernels is in XNNPACK [V snippet, Arm learning path].
- Pi 5 is Cortex-A76 (Armv8.2-A, dotprod). **Measured 2026-11-06 on our Pi: `asimddp` present, no `i8mm` in `lscpu` flags [V].** Expect SDOT kernels only.
- No published ExecuTorch Pi 5 Llama number exists. Third-party llama.cpp Q4_K_M 1B figures are about 6-12 tok/s, from secondary blogs [U].
- **Pin check:** executorch 1.5.1 released 22 Sep 2026 [V, repo doc]. torch 2.14.0 exists, GA 2 Sep 2026 [V](https://dev-discuss.pytorch.org/t/pytorch-2-14-0-general-availability/3431). Whether 1.5.1 requires exactly 2.14.0 is **[U]**; check the v1.5.1 `install_executorch.py`.
- **Loading a prebuilt .pte:** no reported incompatibility found [U]. 1.5.1 adds schema-version checks for PTE files [V, release notes snippet](https://newreleases.io/project/pypi/executorch/release/1.5.1). A `.pte` from an older exporter may be rejected, so test load first.
- Memory: constant tensors reference `.pte` data without copying [V, runtime overview](https://docs.pytorch.org/executorch/stable/runtime-overview.html). Backends may duplicate weights during prepacking, which raised peak memory in a Vulkan case [V, PR 13474 snippet](https://github.com/pytorch/executorch/pull/13474). XNNPACK behaviour is **[U]**.

**2. Prompt and latency**
- Constrained decoding: `LogitProcessor` is in the v1.5.1 source (PR #19517, merged 2026-05-15) [V per ELA_RESEARCH](https://github.com/pytorch/executorch/pull/19517). Grammar issue #19476 is open with no PR [V]. Android and Python exposure of `LogitProcessor` is **[U]**.
- Prefix or KV reuse: nothing found in ExecuTorch docs or the paper [U].
- Candidate scoring needs per-candidate logits from the Python or Android API. Exposure is **[U]**.

**3. Measurability**
- Copy bias is real: Ali et al. (arXiv 2410.01288) [V abstract]. Sclar et al. report format changes moving accuracy by up to 76 points (arXiv 2310.11324) [snippet only].
- Llama 3.2 officially supports Hindi, but not Romanised Hinglish [V, via ELA_RESEARCH from the llama-models model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md).
- TrueBrief (arXiv 2601.04212) is a Dec 2025 preprint with no named models; it is not usable evidence [V].

**4. Detector**
- Learn-then-Test (Angelopoulos et al.) is the best-fit method for picking thresholds without retraining [V abstract]. The Annals of Applied Statistics 2025 publication is from a search snippet only [U].
- Covariate-shift FPR control (Almeida, PMLR 2025) is a possible extension [snippet only].

**5. Android**
- The AAR and LlmModule API are from prior research. Demo path is inconsistent across sources.

**6. Competition-winning features**
- The Bharat AI-SoC rubric is **not found** [U]. The repo's CONTEXT.md says the mentor's focus is detection quality, on-device performance and user effectiveness.
- Typical on-device submissions show a single table with p50/p95 latency, tok/s, TTFT, peak RSS and model size, on named devices. They also show energy or thermal numbers and an airplane-mode video, plus scripts that reproduce every number **[U: general practice, no source]**.
- Rakshak currently has no on-device number, no app and no energy measurement.

## Next actions (17 days)
1. Run the GroundLM FPR test (task 1).
2. Measure the `.pte` on a Pi 5 and a phone (top-5 #1).
3. Rewrite the rank arm and add the baselines (top-5 #3).
