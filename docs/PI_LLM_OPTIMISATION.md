# Pi 5 Llama inference: optimisation levers (2026-11-06)

Read-only research pass. Builds on `docs/OPTIMISATION_RESEARCH.md` (not repeated). No Pi access, no compute run, nothing installed.

Tags: **[V]** checked in a source fetched or searched this session (link given). **[U]** unverified, or derived from cited numbers (derivation stated).

## Reference numbers (the ceiling we are measuring against)

- Llama 3.2 1B SpinQuant on OnePlus 12, ExecuTorch: decode 50.2 tok/s, prefill 260.5 tok/s, TTFT 0.3 s at 64-token prompt, RSS 1,921 MB, PTE 1,083 MB. [V](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET)
- Samsung S24+, Llama 3.2 1B quantised, 6 of 8 CPUs, prompt 64, seq 128: prefill over 350 tok/s, decode over 40 tok/s; RSS 3.1 GiB BF16 vs 1.9 GiB SpinQuant at max seq 2048. Arm says KleidiAI gives "up to 20%" with i8mm, which our A76 lacks. [V](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/llm-inference-llama-quantized-models-executorch-kleidiai)
- Pi 5 (4x A76 at 2.4 GHz), ExecuTorch Int4 Llama 3 8B, 4 threads, 8-token prompt: decode 2.06 tok/s, model load 47 s. The author says it is "not a rigorous benchmarking effort", and the build dates from 2024. [V](https://dev-discuss.pytorch.org/t/run-llama3-8b-on-a-raspberry-pi-5-with-executorch/2048)
- Derived 1B estimate on Pi 5: decode about 10-16 tok/s, prefill about 60-90 tok/s. [U, scaled from the 8B row by parameter count, and from the phone-to-Pi decode ratio of about 3x]
- Consequence: a 300-token prompt costs about 3-5 s of prefill, more than a 24-token decode (about 2 s). Prefill is the budget. [U, derived]
- llama.cpp, Pi 5, 1B Q4: about 7 tok/s, with throughput down about 31% after 30 min uncooled. [U, secondary](https://specpicks.com/reviews/raspberry-pi-4-8gb-vs-pi-5-local-llm-tokens-per-second-2026)
- Thermal: the firmware throttles progressively from 80 to 85 C, and at 85 C both the Arm cores and GPU are throttled. [V](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- Energy: Pi 5 idle about 1.72 W, and about 5.2-5.4 W during llama-bench generation on a 0.5B Q4_0 model (about 24 tok/s, about 0.22 J/token including idle). The same page requires `throttled=0x0` before and after a valid run. [V](https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token/energy-per-token/) J/token for 1B will be higher. [U]

## Top-10 levers (ranked for Phase 1)

| # | Lever | Expected gain | Source | Effort | Runtime-only or re-export | Risk |
|---|---|---|---|---|---|---|
| 1 | Load the `.pte` once and keep a resident process (warm daemon) | Removes the cold load. The 2024 8B load was 47 s [V]. For 1B from a USB 2.0 stick, estimate 10-30 s [U] | [V](https://dev-discuss.pytorch.org/t/run-llama3-8b-on-a-raspberry-pi-5-with-executorch/2048); `warmup()` in [V](https://docs.pytorch.org/executorch/stable/llm/run-with-c-plus-plus.html) | Low, 2 h | Runtime | A crash loses state. About 2 GB resident fits in 4 GB [U] |
| 2 | Put the `.pte` and tokenizer on fast storage, not the USB 2.0 stick | At USB 2.0 speeds (about 30-35 MB/s [U]), 1.1 GB takes about 35 s. Page-ins after eviction also pay this cost | Arithmetic [U] | Low, 1 h | Runtime (hardware) | Minimal |
| 3 | Trim the prompt to about 150 tokens and cache the fixed system prefix KV | Each 100 prompt tokens is about 1-1.5 s at 60-90 tok/s [U]. Prefix reuse could remove most of a 100-150 token system prompt | `generate_from_pos` and `TextPrefiller` [V](https://docs.pytorch.org/executorch/stable/llm/run-with-c-plus-plus.html). Python exposure [U] | Medium, 3-5 h | Runtime plus prompt | Position bugs. Must check output equality against a cold run |
| 4 | Score the candidate slots in one forward pass (single-token labels, read next-token logits) instead of decoding | Removes 16-24 decode tokens (about 1.5-2.5 s [U]). K candidates cost one prefill, not K | Constrained decoding: `LogitProcessor` PR #19517 (cited in repo docs; not re-fetched) [V per repo]. Logit access from Python or C++ [U] | Medium, 3 h | Runtime, needs logits exposed | Small models are poorly calibrated on letter labels [U] |
| 5 | Thread sweep (2, 3, 4) with pinning | Decode reads about 1.1 GB of weights per token, so it is likely memory-bound. 3 threads may match 4 at lower heat [U]. No published Pi 5 3-vs-4 data found | Protocol from [V](https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token/energy-per-token/) (`-t 4`, `-p 0`) | Low, 1 h | Runtime | Result may not hold after a long run. Re-test with a 30-min soak |
| 6 | Official Active Cooler (or a fan) to avoid progressive throttling | Uncooled loss of about 31% over 30 min [U]. Throttle begins at 80 C [V] | [V](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html); [U](https://specpicks.com/reviews/raspberry-pi-4-8gb-vs-pi-5-local-llm-tokens-per-second-2026) | Low, hardware | Hardware | Cost. Conflicts with the "no cooler" premise, so the user decides |
| 7 | Re-export with 4-bit embedding (`embedding_quantize: 4,32`) | About 130 MB less of 1.1 GB [U, from 128256 x 2048 embedding params]. Smaller RSS and faster load | Flag [V](https://docs.pytorch.org/executorch/stable/llm/export-llm.html) | Medium, Kaggle export plus eval, 4 h | Re-export | Quality cost unmeasured [U]. The embedding may be tied to the output head [U] |
| 8 | Re-export with `max_context_length` 512 instead of 2048 | KV cache at 2048 is about 64 MB [U, 32 KB/token from architecture]. Smaller activation buffers. More RSS margin on 4 GB | `max_seq_length` and `max_context_length` [V](https://docs.pytorch.org/executorch/stable/llm/export-llm.html) | Low, Kaggle export, 1 h | Re-export | Caps prompt plus output. Fine if prompt is 150 tokens |
| 9 | Pick the quality checkpoint: Meta SpinQuant vs QAT+LoRA | Quality, not speed. Meta's W4 g32 linears, 8-bit dynamic activations, 8-bit classifier and embedding are documented [V]. Our drift vs fp is unknown [U] | [V](https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET); SpinQuant [V](https://arxiv.org/pdf/2405.16406) | Medium, gating | Checkpoint choice, possible re-export | Meta repo is gated. "EO8" is undocumented in the card [V] |
| 10 | Demo telemetry: per-stage ms, RSS, temperature, `get_throttled`, PMIC J per advisory | No speed gain. Judge-facing evidence | Sampler method and `throttled=0x0` rule [V](https://learn.arm.com/learning-paths/cross-platform/rpi5-energy-per-token/energy-per-token/); `get_rss_bytes()` [V](https://docs.pytorch.org/executorch/stable/llm/run-with-c-plus-plus.html) | Low, 2-3 h | Runtime | PMIC rail names and `pmic_read_adc` not confirmed [U] |

### Notes on the levers

- **mmap vs load:** not documented in the runtime or export pages [V]. Constant tensors reference `.pte` data without copying [V, from `docs/OPTIMISATION_RESEARCH.md`]. Treat page cache as the RAM cost on 4 GB [U].
- **KleidiAI on A76:** the Arm KleidiAI page lists `qp8_f32_qb4w_gemm` but does not map kernels to ISA or to A76 [V](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/measure-kleidiai-kernel-performance-on-executorch/03-executorch-node-kai-kernel/). The "20%" gain needs i8mm, so expect XNNPACK's dotprod (SDOT) path only [U].
- **C++ runner vs Python:** the overhead difference is unmeasured [U]. Prefer whichever exposes logits and `generate_from_pos`.
- **Fused INT4 failure:** a single blog reports `xnn_status_invalid_parameter` for fused INT4 subgraphs on aarch64 in ExecuTorch 1.2.0, with `per_op_mode=True` as a workaround that disables fusion [U, single source](https://dev.to/bamb00boy/first-gemma-4-executorch-deployment-on-raspberry-pi-5-and-why-its-77x-slower-than-llamacpp-gmb). Our pin is 1.5.1, so this must be tested on the Pi. It could be a top risk.
- **Smaller Llama:** Llama 3.2 text ships 1B and 3B only [U, from memory; check the model card]. 1B is the floor for the Llama-family requirement.

## Demo ("SoC-engineered") checklist

- Per-stage table on named hardware: cold load, warm TTFT, prefill tok/s, decode tok/s, verifier ms, total p50/p95.
- RSS from the runner, plus the `.pte` size and the memory map.
- Thermal trace with `throttled=0x0` before and after each run. A run with any throttle bit set is excluded.
- Energy per advisory from PMIC rails, with idle subtracted.
- On-device KPI: verifier fallback rate (share of advisories that fell back to the template). This is the cheapest quality signal and needs no judge model.

## 3-day action plan

**Day 1 (Pi, once ExecuTorch is installed): go/no-go.**
1. Load the 1.14 GB `.pte` on executorch 1.5.1. Confirm there is no `xnn_status_invalid_parameter`. Stop and try `per_op_mode` if it appears. [U]
2. Record cold load time and warm TTFT for 64, 150 and 300-token prompts. Record decode for 16 and 24 tokens.
3. Record RSS, temperature and `get_throttled` before and after.
4. Sweep 2, 3 and 4 threads over the same five prompts, three repetitions each.
5. Move the model off the USB 2.0 stick if the cold load is over 20 s.

**Day 2: runtime and prompt.**
1. Build a resident process: one load, one warmup, then requests served from memory.
2. Trim the prompt to 150 tokens or fewer. Cache the fixed system prefix with `generate_from_pos`. Check output equality against a cold run on 20 prompts.
3. Implement single-token option scoring if logits are exposed. Otherwise keep decode, with `max_new_tokens` 24 and EOS stop.
4. Add telemetry: per-stage ms, RSS, temperature, throttle bits. Add PMIC sampling if `pmic_read_adc` works on the Pi.

**Day 3: quality and evidence.**
1. Run the verifier on at least 100 on-device advisories. Report fallback rate and per-stage latency.
2. Compare on-device greedy output with a Kaggle PyTorch reference on 50 prompts (top-1 token agreement). This is [U] as a method but is cheap.
3. Only if Day 1 shows RAM or load is a problem, do the 4-bit embedding and 512-context re-export on Kaggle. Do not start it otherwise.

Expected per-advisory budget after Days 1-2: about 1-2 s warm, versus about 30-50 s cold today [U, derived].

## Open questions

1. Does the executorch 1.5.1 Python runtime expose logits and `generate_from_pos`? If not, the C++ runner is needed for levers 3 and 4. [U]
2. Does the fused-INT4 failure reported for 1.2.0 occur on 1.5.1 on aarch64? [U]
3. Does "EO8" in the community checkpoint name mean 8-bit embedding, and is the 1B embedding tied to the output head? [U]
4. The `docs/OPTIMISATION_RESEARCH.md` inconsistency is still open: slot-arm pass 0.91 vs 7.7%. Reconcile before quoting any quality number.
5. Is the official Active Cooler acceptable under the "no cooler" premise? [user decision]
6. Which PMIC rails sum to the board total, and does `pmic_read_adc` work on this Pi? [U]
7. Does the community repo need the Llama 3.2 licence accepted, and is it gated? [V: model card cites the Llama 3.2 Community License]

Files: this report is `/home/maadhav/Documents/Development/rakshak-edge/docs/PI_LLM_OPTIMISATION.md`. No other file was edited. Nothing was committed or pushed.
