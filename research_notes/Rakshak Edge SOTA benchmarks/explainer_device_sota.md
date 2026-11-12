# On-device LLM security explainers and on-device LLM performance on Arm (as of 2026-11-07)

Scope note: the search budget was roughly 19 tool calls, above the ~15 guideline. Primary sources were fetched for the DeepMind blog, PhishXplain (arXiv HTML), the ExecuTorch Llama README, and the Arm energy-per-token overview. Everything else comes from search snippets and is marked UNVERIFIED where it matters. No source was found that is a "SOTA" for an SMS scam explainer on device. Rakshak's own numbers are repo-reported and were not checked.

## Key Question 1: On-device scam/phishing detection with an LLM explanation layer

### Takeaway
No published on-device SMS/message scam system with an LLM explanation layer reports a usable accuracy, latency, faithfulness, or comprehension number. The only peer-reviewed explainer with user-study and reliability numbers is PhishXplain (web phishing, Llama 3.2 3B, not on-device). Google DeepMind and GovTech Singapore describe a two-stage on-device Gemma pipeline but publish no metrics.

### Cited Findings
- Google DeepMind + GovTech Singapore (blog, 14 Oct 2026, read from primary page): two-stage design. Stage 1 is a "lightweight, low-power classifier" that screens inbound messages. Stage 2 is a "fine-tuned Gemma model" doing in-context analysis over a larger conversation snapshot, returning "Scam Alert" or "Benign Verdict"; the user gets an on-device warning. Accuracy, false-positive rate, latency, Gemma variant, and model size are all NOT reported; the page says only that it was "stress-tested" with no results. Privacy claims: on-device, training data "strictly excluded" PII, access "with the user's consent". Rollout is phased; GovTech launched an open beta of the Anti-Scam Pango app for Android at STACK 2026. [DeepMind blog](https://deepmind.google/blog/on-device-ai-scam-detection-mddi/); press coverage: [AI Weekly](https://aiweekly.co/alerts/deepmind-and-govtech-singapore-ship-on-device-gemma-scam-filter), [Fintech News SG](https://fintechnews.sg/138945/ai/google-deepmind-govtech-scam-detection-model/)
- Conflicting claim: emergent.sh says the pilot reached "89% accuracy" across WhatsApp, Telegram and SMS. No primary source supports it, and the same site makes other claims other outlets do not support. UNVERIFIED, do not cite as a benchmark. [emergent.sh](https://emergent.sh/news/google-deepmind-partners-singapore-gemma-scam)
- PhishXplain, "Explain, Don't Just Warn!" (arXiv 2505.06836, v2 read in full HTML). Real-time phishing warning for web pages, not SMS. Model: LLaMA 3.2 **3B** Q4_K_M via Ollama, CPU-only VM (Intel i5-11400, 16 GB RAM, no GPU). Numbers from the paper:
  - Table 1, 100 PhishTank sites: median 2.19 s, 2.32 GB memory.
  - Live deployment (2 Feb to 4 Mar 2025): median 5.2 s per warning, about 2.7 GB. The paper's Section 3.5 says "under 5 s", which conflicts with 5.2 s.
  - Reliability, manual review by two coders: Correct Feature Rate 0.87 ± 0.06, Feature Miss Rate 0.04 ± 0.01, Artifact Accuracy 0.93 ± 0.03, Code Snippet Accuracy 0.88 ± 0.04, composite 9.1/10. Llama 3.2 **1B** composite 7.4; Llama 3.3 70B 9.4 at 59.44 s and 75.38 GB.
  - Live: 8,752 URLs visited; GSB flagged 7,091; PXP produced warnings for 6,659 (about 94%). Manual check of 500 sites: 1,316 of 1,368 indicators correct (about 96%). No inter-rater agreement reported for this check.
  - User study, Prolific (150 recruited, 2 excluded, groups listed as n=75 each; proficiency tiers sum to 139, an internal inconsistency). Unguided accuracy on 8 sites: 6.44 vs 5.16 correct, Mann-Whitney p < 0.01. Confidence 3.83 vs 2.77 (p < 0.015). Helpfulness 4.64 vs 3.09 (p < 0.01). Feature-coding Cohen's kappa 0.72. Low-proficiency users gained most (accuracy 5.0 to 6.8).
  - No dedicated faithfulness or hallucination evaluation; the authors note small LLMs can hallucinate but do not quantify it. [arXiv HTML v2](https://arxiv.org/html/2505.06836v2)
- Cau et al. (arXiv 2507.07916): could NOT be confirmed. No search result cited that ID, and the closest candidate (a FINRA-taxonomy LLM scam-susceptibility benchmark, 37 base scenarios, 148 persona variants) is a different paper and not about on-device explanation. Gap.
- Android on-device example (single developer, UNVERIFIED): ScamRakshak uses a three-tier design: Gemma 4 via ML Kit GenAI for explanations, a LiteRT binary classifier, and regex. The LLM returns a 0-100 risk score and an explanation in Hindi and English. Gemma 4 downloads through Google AICore (about 2 GB). No metrics given. [rohitraj.tech write-up](https://rohitraj.tech/en/notes/build-on-device-ai-scam-detector-android-gemma)
- Not searched in this run: Samsung, Truecaller, and other shipped on-device scam features. Gap.

### Inferences
- The DeepMind/GovTech design (cheap classifier gate, then LLM only on flagged conversations) matches Rakshak's gate-then-explain structure. That is a design parallel, not a benchmark.
- Any SOTA claim for Rakshak's explainer on SMS would have no published comparator with metrics. The defensible comparison set is PhishXplain's reliability and user-study metrics, applied with a stated caveat that PhishXplain is 3B and web-based.
- PhishXplain's 1B vs 3B composite (7.4 vs 9.1) is directly relevant to Rakshak's 1B choice: the paper's own evidence favours 3B on explanation reliability.

### Gaps
- No primary-source metrics for any on-device SMS scam explainer (DeepMind, Samsung, Truecaller, ScamRakshak).
- Cau et al. 2507.07916 not found.
- No faithfulness or comprehension metric for an SMS explainer found.

## Key Question 2: Llama 3.2 1B (and similar) performance on Pi 5 / Cortex-A76 and mid-range Android

### Takeaway
The only primary-source ExecuTorch Llama 1B numbers are on a OnePlus 12 (Snapdragon 8 Gen 3 per general knowledge; the chip is not named in the source), not on a Pi 5. For Pi 5, no ExecuTorch numbers were found; llama.cpp/Ollama figures are blog-grade and inconsistent (about 6 to 11 tok/s for Llama 3.2 1B Q4). Rakshak's repo-reported 9.8-11.6 tok/s decode sits inside that range.

### Cited Findings
- ExecuTorch Llama README (GitHub main, fetched; no date or ExecuTorch version given). Device OnePlus 12, prompt length 64, adb binary, KleidiAI "enabled by default on the XNNPACK Backend for all ARM devices". Table (decode tok/s, TTFT, prefill tok/s, PTE size, RSS):
  - 1B BF16: 19.2, 1.0 s, 60.3, 2,358 MiB, 3,185 MiB
  - 1B SpinQuant: 50.2, 0.3 s, 260.5, 1,083 MiB, 1,921 MiB
  - 1B QAT+LoRA: 45.8, 0.3 s, 252.0, 1,127 MiB, 2,255 MiB
  - 3B BF16: 7.6, 3.0 s, 21.2, 6,129 MiB, 7,419 MiB
  - 3B SpinQuant: 19.7, 0.7 s, 89.7, 2,435 MiB, 3,726 MiB
  - Export: `python -m extension.llm.export.export_llm --config examples/models/llama/config/llama_xnnpack_spinquant.yaml` with `+base.model_class="llama3_2"`. Scheme: 4-bit groupwise weights (group 32), 8-bit per-token dynamic activations, 8-bit per-channel classifier and embeddings. Android build: `-DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`, NDK r28c. Run: `llama_main --model_path <pte> --tokenizer_path <tok> --prompt ... --seq_len 120 --warmup=1`. Only RSS reported, not PSS. Source: [ExecuTorch Llama README](https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md). The docs site (docs.pytorch.org/executorch/stable/llm/llama-demo-android.html) redirected and returned no content on fetch.
- Search snippet (secondary): an older README revision listed 1B SpinQuant at 53.41 tok/s vs 19.4 BF16 at 128 generated tokens; PR 6434 is cited for 50.2 tok/s. Different prompt lengths, so not comparable. UNVERIFIED beyond the snippet. [PR 6434](https://github.com/pytorch/executorch/pull/6434.patch)
- Arm and Meta (secondary snippet, UNVERIFIED): prefill above 350 tok/s for quantized Llama 3.2 1B on Samsung S24+ with ExecuTorch XNNPACK/KleidiAI. Source not fetched: [Arm community blog](https://community.arm.com/arm-community-blogs/b/ai-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai)
- Arm Learning Path, Android, llama.cpp with and without SME2 (secondary snippet, units ambiguous, UNVERIFIED): 1 thread 148 vs 173; 2 threads 124 vs 191. [Arm Learning Path](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/voice-assistant/7-performance/)
- Arm newsroom (vendor claim): KleidiAI cut TTFT for Llama 3 in llama.cpp by 190% on Cortex-X925, vs reference. Not Llama 3.2 1B, not A76. [Arm newsroom](https://newsroom.arm.com/?p=14033). Arm KleidiAI page claims "10 times faster" on Cortex-A76 with "3 seconds" response, unspecified model: vendor headline, UNVERIFIED. [Arm KleidiAI](https://www.arm.com/kleidi)
- Pi 5, llama.cpp via Ollama, Llama 3.2 1B Q4_K_M: 11.22 tok/s (benchmark table snippet; methodology and cooling not stated, UNVERIFIED). [SpecPicks Pi 4 vs Pi 5 review](https://specpicks.com/reviews/raspberry-pi-4-8gb-vs-pi-5-local-llm-tokens-per-second-2026). Other blog figures: about 7 tok/s and about 6 tok/s for Q4 (UNVERIFIED).
- Pi 5 throttling (snippet, UNVERIFIED): bare board, 30-minute run, first and last 60 s averages 7.8 and 5.4 tok/s (-31%), 84 °C. [Stratosphere Pi 5 LLM research](https://stratospherelinuxips.readthedocs.io/en/develop/immune/research_rpi_llm_performance.html)
- Pi 5, llama.cpp, TinyLlama-1.1B Q4_0 (different model, blog, UNVERIFIED): 14.4 tok/s generation, about 108 tok/s prompt processing. [tinyweights.dev](https://tinyweights.dev/posts/run-llms-raspberry-pi-5/)
- No Pi 5 ExecuTorch numbers found. No llama-bench pp512/tg128 Pi 5 result for a 1B Q4_0 model found. No MLC-LLM numbers found.

### Inferences
- Rakshak's 9.8-11.6 tok/s decode on Pi 5 is consistent with the 6-11 tok/s llama.cpp range on Pi 5 for Llama 3.2 1B Q4. It can be stated as in the range, not as SOTA, because no source gives an ExecuTorch Pi 5 comparator.
- Pi 5 decode is roughly 4-5x slower than the OnePlus 12 SpinQuant decode (50.2 tok/s). Phone numbers must not be used as Pi 5 targets.
- Rakshak RSS 1.84 GB is in the same range as the OnePlus 12 1B SpinQuant RSS of 1,921 MiB (RSS includes different runtimes and prompt lengths; not directly comparable).
- A Pi 5 comparison of ExecuTorch vs llama.cpp is a gap that needs a bench run; no published data exists in the sources reached.

### Gaps
- ExecuTorch on Pi 5 (any version).
- llama.cpp Q4_K_M and Q4_0 pp/tg with llama-bench on Pi 5 for 1B, from a primary source.
- Snapdragon 8 Gen 2 / 7-series / Dimensity 1B ExecuTorch numbers, and Qualcomm HTP results.
- Energy per token for any phone or Pi ExecuTorch run.
- ExecuTorch version and date for the README table.

## Key Question 3: Faithfulness/grounding metrics and prompt-injection benchmarks for small LMs

### Takeaway
MiniCheck (arXiv 2404.10774, EMNLP 2024) is the best-documented small grounding checker. Prompt-injection evidence is mainly from AgentDojo-style agent benchmarks, which are not SMS pipelines. Adaptive attacks break several defences, so a 0/150 result needs a stated confidence bound and an adaptive-attack test before it is called robust.

### Cited Findings
- MiniCheck (Tang, Laban, Durrett; arXiv 2404.10774, EMNLP 2024): MiniCheck-FT5 at 770M; LLM-AggreFact aggregates 10 datasets with human sentence-level labels; claims GPT-4-level accuracy at about 400x lower cost. Margin over AlignScore-Large reported as roughly 4-10% absolute (secondary summary). Release: code, models, leaderboard at llm-aggrefact.github.io. [arXiv abs/HTML](https://arxiv.org/html/2404.10774v2); numbers beyond the abstract are from search snippets, UNVERIFIED for exact values.
- VERITYGATE: a 2026 paper, author Sachin Gupta, GroundLM 2026 workshop. A four-gate checker for declared evidence IDs, entities, numbers and claim types, aimed at checking LLM explanations from structured evidence. Code and data released. Snippets do not confirm a small-model evaluation. [ACL Anthology 2026.groundlm-1.5](https://aclanthology.org/2026.groundlm-1.5/). Note: the author name is a match to a teammate in the user's memory notes; that match is not verified as the same person.
- Related small verifiers (snippets, UNVERIFIED numbers): ThinknCheck (arXiv 2604.01652), a 1B grounded claim verifier with a rationale before its verdict; CaLM (arXiv 2406.05365, ACL Findings 2024), a small LM verifying a larger LM's grounded output. A June 2026 coverage-aware faithfulness paper (arXiv 2606.09376) fine-tunes Qwen2.5-3B; not verified.
- AgentDojo (arXiv 2406.13352): 629 security test cases; the authors report attack success falls to 8% with a secondary attack detector (snippet, UNVERIFIED for exact figure).
- Adaptive attacks paper (arXiv 2503.00061): an LLM-based detector shows 0% ASR without adaptation and 31.25% under multi-objective GCG adaptive attack. Adaptive ASR exceeds 50% on InjecAgent. [arXiv 2503.00061](https://arxiv.org/pdf/2503.00061) (snippet-level, UNVERIFIED for exact figures).
- CommandSans (arXiv 2510.08829): BIPIA with GPT-4o, overall ASR 13.8%; also reports AgentDojo, InjecAgent, and Agent Security Bench. [arXiv 2510.08829](https://arxiv.org/pdf/2510.08829) (snippet, UNVERIFIED).
- Meta Llama Prompt Guard 2: 86M (mDeBERTa-base) and 22M (DeBERTa-xsmall) classifiers. A third-party card lists 92.4 ms (86M) and 19.3 ms (22M), with column labels unclear. An arXiv paper (Semalith v1.4, arXiv 2607.22545) reports PG2-86M false-positive rates of 96% on AgentHarm-benign and 98% on ToxicChat. That paper is by a competitor and UNVERIFIED. Meta notes the classifier is not immune to adaptive attacks. [Meta model card](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/)
- Rakshak's "0/150 harmful injection flips": 3/150 = 2.0% is the rule-of-three 95% upper bound (inference, not a published figure). Adaptive attacks are not reported.

### Inferences
- For Rakshak, the most transferable metrics are: a grounding checker (MiniCheck-style, or VERITYGATE-style structured checks on evidence IDs, numbers and entities), and the rule-of-three bound on injection flips.
- AgentDojo and BIPIA are agent or email-QA settings; they transfer as templates for an SMS-injection test set, not as numbers.
- Any injection claim needs an adaptive-attack arm (GCG or RL-based, for example PISmith arXiv 2603.13026) to be credible; the adaptive-attacks paper is the evidence for this.

### Gaps
- No small-LM (1B) faithfulness numbers on scam explanations found.
- No primary numbers for BIPIA and InjecAgent defences were extracted.
- No SMS-specific injection benchmark found.

## Key Question 4: Energy per inference reporting norms on edge

### Takeaway
There is no agreed norm. Energy must be reported with its boundary (rails, idle subtraction, sampling rate). The Arm Pi 5 method uses PMIC rails read via vcgencmd, and J/token = power x window / tokens, with and without idle subtracted.

### Cited Findings
- Arm Learning Path, "Measure llama.cpp energy per token on a Raspberry Pi 5" (last updated 28 Oct 2026, read from overview): reads each PMIC rail's current and voltage, logs timestamped CSV, aligns with token windows, computes J/token with and without idle power, and repeats runs. Hardware: Pi 5 with active cooler, 64-bit Raspberry Pi OS. No numbers on the overview page. [Arm LP overview](https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token/)
- powerstamp (PyPI, author's claim): Pi 5 PMIC effective sampling about 21.2 Hz, not the 43-44 Hz a naive read count suggests, because per-rail reads were double-counted. [powerstamp](https://pypi.org/project/powerstamp/)
- Birkenfeld "LLMs on the Edge" (snippet, UNVERIFIED): Pi 5 with a USB-C multimeter, 0.93-1.76 mWh per token. Converted with 1 mWh = 3.6 J: about 3.3-6.3 J/token (inference). Snippet text is garbled.
- Glossary note: J/token figures are comparable only when the measurement boundary is stated. [Infracost glossary](https://www.infracost.io/resources/glossary/energy-per-token)
- No primary source with J/token for Llama 3.2 1B on a phone or ExecuTorch was found.

### Inferences
- Rakshak's "about 7-12 J/request (PMIC)" must state the boundary (idle subtracted or not, rails included) and the token count per request. Converting it to J/token needs the output token count, which is not in the brief.
- Using the Birkenfeld range, a 60-token explanation at about 3-6 J/token would be about 200-400 J. The Rakshak figure implies a much lower per-token energy only if the request is short or the explanation is cached. Check the output length before comparing.

### Gaps
- No published J/request figure for any on-device explainer.
- No phone-side (PMIC or external meter) J/token for ExecuTorch Llama 1B.

## Source status summary
- Read from primary sources: DeepMind blog; PhishXplain arXiv HTML v2; ExecuTorch Llama README; Arm energy-per-token overview.
- Search-snippet only (secondary): MiniCheck, VERITYGATE, AgentDojo, adaptive attacks, CommandSans, Prompt Guard 2 comparisons, Pi 5 llama.cpp figures, Arm SME2/S24+ numbers, Birkenfeld energy, ScamRakshak.
- Not found: Cau et al. 2507.07916; Samsung/Truecaller on-device scam papers; ExecuTorch on Pi 5; MLC numbers.
- Rakshak's own figures (9.8-11.6 tok/s, 1.84 GB RSS, 7-12 J/request, 0/150 flips) are repo-reported and not checked.
