# ExecuTorch plan for the Llama explanation layer

Research snapshot: 2026-10-30. Owner: research worker. Scope: ExecuTorch release, export, Android, Raspberry Pi 5, constrained decoding, HF gating, latency. Nothing was installed or downloaded. Items marked **UNVERIFIED** could not be confirmed from an official source in this session.

## 0. Bottom line

- Pin **ExecuTorch 1.5.1** (`pip install executorch==1.5.1`, Maven `org.pytorch:executorch-android:1.5.1`).
- Export Llama 3.2 1B Instruct to `.pte` with the XNNPACK backend and `qmode: 8da4w`, group size 32. The Meta pre-quantized SpinQuant and QAT+LoRA checkpoints are the quality-preferred variants, but they are gated separately.
- ExecuTorch has **no shipped grammar or JSON-constrained decoding**. The public `GenerationConfig` has no grammar field. An open issue (#19476) proposes it. Use the design in section 4.
- **Gating blocks us today.** The `vhmaadhav` account is not on the authorised list for `meta-llama/Llama-3.2-1B-Instruct`. Maadhav must accept the Llama 3.2 licence on HF first.

## 1. Latest stable release and packages

| Item | Value | Source |
|---|---|---|
| Latest stable | **v1.5.1**, GitHub "Latest" dated 23 Sep (year not shown on that page) | https://github.com/pytorch/executorch/releases |
| PyPI | `executorch` **1.5.1**, released **Sep 22, 2026**. `pip install executorch`. Python `>=3.10, <3.15`. Supported systems: Linux x86_64, Linux aarch64, macOS aarch64. Wheels: cp310 to cp314 | https://pypi.org/project/executorch/ |
| PyPI notes | Package is labelled **beta**. Torch is **not** bundled: install a compatible `torch` separately. Torch pin for 1.5.1 is **UNVERIFIED**, so take it from the v1.5.1 tag's install script | https://pypi.org/project/executorch/ |
| Maven | `org.pytorch:executorch-android`, latest release **1.5.1**. The metadata also lists a snapshot `20261001192257`; do not use it | https://repo1.maven.org/maven2/org/pytorch/executorch-android/maven-metadata.xml |
| Python on x86-64 Linux for export | 3.10 to 3.14 have wheels. **Recommend 3.12** (our choice, not from a source) | PyPI page above |

Release history on PyPI: 1.5.0 (Sep 16 2026), 1.4.1 (Aug 14), 1.4.0 (Aug 7), 1.3.1 (May 29), 1.2.0 (Apr 1), 1.1.0 (Jan 28). Source: https://pypi.org/project/executorch/

Two caches are in play. The GitHub release page and PyPI both report 1.5.1 as latest. I could not confirm nothing newer shipped after 22 Sep 2026. Re-check on the day we pin.

## 2. Export Llama 3.2 1B Instruct to `.pte`

### 2.1 Setup (x86-64 Linux)

Docs say `pytorch_tokenizers` is required for LLM export and must be installed from the source tree. So use a source checkout at the release tag:

```bash
git clone --branch v1.5.1 https://github.com/pytorch/executorch.git   # tag name assumed from release page
cd executorch
pip install -e ./extension/llm/tokenizers/
# install torch version matching 1.5.1 (UNVERIFIED pin; see install script in repo)
```

Source: https://docs.pytorch.org/executorch/stable/llm/export-llm.html

The Llama README also says to run `./install_executorch.sh` and `examples/models/llama/install_requirements.sh`. Source: https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md

### 2.2 Inputs

The docs require `consolidated.00.pth`, `params.json` and `tokenizer.model` from the Llama site or HF. The HF file layout for the gated repo is **UNVERIFIED** (access denied, see section 5).

### 2.3 Our target: BF16 weights quantized to 8da4w on XNNPACK

Config file `llama1b_8da4w_xnnpack.yaml`. The keys come from the export-LLM docs. The `group_size: 32` and `embedding_quantize: 4,32` values match Meta's stated scheme.

```yaml
base:
  model_class: llama3_2
  checkpoint: /abs/path/to/consolidated.00.pth
  params: /abs/path/to/params.json
  metadata: '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}'
model:
  use_kv_cache: True
  use_sdpa_with_kv_cache: True
quantization:
  qmode: 8da4w
  group_size: 32
  embedding_quantize: 4,32
backend:
  xnnpack:
    enabled: True
    extended_ops: True
export:
  output_name: llama3_2_1b_8da4w_xnnpack.pte
```

Command:

```bash
python -m extension.llm.export.export_llm --config llama1b_8da4w_xnnpack.yaml
# docs also show: python -m executorch.extension.llm.export.export_llm
# module prefix differs between doc sections; UNVERIFIED which works from a pip install
```

Sources: https://docs.pytorch.org/executorch/stable/llm/export-llm.html (the `qmode: 8da4w`, `group_size`, `embedding_quantize`, `xnnpack` blocks). The `export.output_name` key is from the Llama README low-bit section: https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md

`qmode: 8da4w` is the XNNPACK path. The `torchao:8da4w` kernels are a separate experimental Arm path that needs `EXECUTORCH_BUILD_KERNELS_TORCHAO=1`. Do not mix them with XNNPACK.

### 2.4 Pre-quantized checkpoints

- **SpinQuant INT4.** Config `examples/models/llama/config/llama_xnnpack_spinquant.yaml`. Use `+base.checkpoint` and `+base.params` with `model_class="llama3_2"`. The HF repo is `meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8` (gated).
- **QAT+LoRA INT4.** Config `llama_xnnpack_qat.yaml`. HF repo `meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8` (gated).
- The Llama README says these prequantized weights are also downloadable from the Llama website, and that ExecuTorch-exported copies exist on HF. The exact HF repos for those exports are **UNVERIFIED**.

Sources: https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md ; HF details from `hub_repo_details` (section 5).

Quality note from the Llama README: 1B/3B models are sensitive to plain PTQ. Meta's scheme is 4-bit groupwise linear weights (group 32), 8-bit per-token dynamic activations, and 8-bit per-channel classifier and embedding. The SpinQuant and QAT+LoRA checkpoints were built for that.

### 2.5 Tokenizer handling

- The C++ runner loads `tokenizer.model` (tiktoken-style for Llama 3) or `tokenizer.json` (HF). Source: https://docs.pytorch.org/executorch/stable/llm/run-with-c-plus-plus.html
- BOS and EOS ids must be baked into `.pte` metadata: `get_bos_id 128000`, `get_eos_ids [128009, 128001]` (Llama 3 `<|eot_id|>` and `<|end_of_text|>`).
- **UNVERIFIED:** whether the runner applies the Llama 3 instruct chat template. The `llama_main` example passes a raw `--prompt`. Plan for **us** to build the chat string ourselves (`<|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>`) and test it.

### 2.6 Expected `.pte` size

- Vendor (Meta, OnePlus 12): 1B BF16 **2,358 MiB**, 1B SpinQuant **1,083 MiB**, 1B QAT+LoRA **1,127 MiB**. Source: https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md
- ExecuTorch paper (MLSys 2026, authors' measurements, XNNPACK, Galaxy S25 Ultra): **GS128 742 MiB**, **GS32 821 MiB**. Source: https://arxiv.org/html/2605.08195v1
- Estimate for our 8da4w GS32 build: **about 0.75 to 0.85 GB** (inference from the paper, not measured). Plan around ~0.8 GB and do not bundle it in the APK.

## 3. Running the model

### 3a. Android

- **Dependency.** `implementation("org.pytorch:executorch-android:1.5.1")`. The Maven POM bundles fbjni and soloader. The file-based AAR needs `com.facebook.soloader:soloader:0.10.5` and `com.facebook.fbjni:fbjni:0.7.0`. Sources: https://repo1.maven.org/maven2/org/pytorch/executorch-android/maven-metadata.xml ; https://docs.pytorch.org/executorch/stable/using-executorch-android.html
- **LLM API.** The LLM classes ship in the same AAR. Source: https://docs.pytorch.org/executorch/stable/llm/run-on-android.html
  - `LlmModule(modelPath, tokenizerPath, temperature)` (example `temperature 0.8f`)
  - `int status = module.load();` (non-zero means failure)
  - `module.generate(prompt, callback)` or `module.generate(prompt, genConfig, callback)`
  - `module.stop()`
  - `LlmCallback`: `onResult(String token)`, `onStats(String statsJson)`, `onError(int code, String msg)`
- **Demo app.** Docs point to the Llama Android demo in `executorch-examples`. The in-repo path `examples/demo-apps/android/LlamaDemo` returned 404. The exact current repo URL is **UNVERIFIED**.
- **Verified phones.** Meta lists iPhone 15 Pro, Galaxy S24+, S22 and OnePlus 12 (16 GB RAM). Source: the Llama README above. ET paper devices: Galaxy S25 Ultra and Pixel 9 Pro XL.
- **Minimum RAM.** Not stated by any official source. **UNVERIFIED.** Working estimate: a phone with at least 6 GB total RAM, because the vendor RSS is about 1.9 GB (see section 6). We must measure on a real device.
- **Model delivery.** Sideload with `adb push` (as in the docs) or download on first run. Do not ship a 0.8 GB asset in the APK.

### 3b. Raspberry Pi 5 (aarch64)

- **pip path.** Linux aarch64 wheels exist for cp310 to cp314 (`manylinux_2_28_aarch64`). `pip install executorch==1.5.1` should work on 64-bit Pi OS. The wheel includes the `executorch.runtime` module, which runs `.pte` files with XNNPACK. Restrictions: core ATen ops and XNNPACK only. Source: https://pypi.org/project/executorch/
- **C++ runner path.** Build `llama_main` natively on the Pi with the Llama README commands: `cmake --workflow llm-release`, then the `llama-release` preset, then `llama_main --model_path=... --tokenizer_path=... --prompt=...`. The README says it was tested on macOS and Linux, not on Pi, so **UNVERIFIED** on Pi. Source: https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md
- **Cross-compile.** Not documented for Pi. The Android toolchain flow is the only documented cross-build. Any aarch64 Linux toolchain would be **UNVERIFIED**.
- **Official Pi guide.** None found. My search returned nothing Pi-specific. Treat the Pi 5 path as UNVERIFIED until we run it.
- **Pi performance and memory.** No official Pi 5 Llama numbers were found. We must measure. Our expectation is not sourced.

## 4. Constrained and structured output

**Answer: no shipped grammar or JSON-constrained decoding in the ExecuTorch LLM runner, as far as I could confirm.**

- Public `GenerationConfig` fields are `echo`, `max_new_tokens`, `warming`, `seq_len`, `temperature`, `num_bos`, `num_eos`. None is a grammar or logit-bias field. Source: https://docs.pytorch.org/executorch/stable/llm/run-with-c-plus-plus.html
- The runner has `TextTokenGenerator` with temperature sampling and EOS detection, and no logit hook in the documented API. Same source.
- Open issue **#19476**, "[LLM Runner] Wire grammar-constrained decoding end-to-end", opened 11 May 2026. It proposes `GenerationConfig.grammar`, a `GrammarLogitProcessor` injected into `TextTokenGenerator`, and `accept_token()` after sampling. Target mask overhead under 100 µs. Status at last fetch: open, no linked PR, project status "Todo". Source: https://github.com/pytorch/executorch/issues/19476
- **UNVERIFIED:** whether this landed on `main` after the cached copy. Check before relying on it.

Our design does not need token masking for v1:

1. **v1 (ship this).** Keep the output tiny. Ask for `{summary, reasons:[E*], action}` with `max_new_tokens` around 96 to 128. Parse the JSON leniently. Run the deterministic verifier: every `E*` must exist in the evidence package, and `action` must be in the allowed codes. On failure, use the deterministic template. This matches CONTEXT.md section 4.
2. **v2 (if v1 fails our eval).** Candidate scoring instead of decoding. For each enum field, score each allowed string with a prefill-only log-likelihood and take the argmax. No decode loop and no logit hook. This is our design, not from a source.
3. **v3 (only if needed).** Fork `TextTokenGenerator` in C++ and mask logits to a schema-valid token set, or wait for #19476. Costs C++ work on the Android side.

Per CONTEXT.md, the eval must include the free-form vs constrained ablation. Keep v1 and v2 as the two arms.

## 5. HF gating and access

- `meta-llama/Llama-3.2-1B-Instruct` is **Gated** and carries licence `llama3.2`. The two quantized repos `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8` and `Llama-3.2-1B-Instruct-QLORA_INT4_EO8` are also **Gated**. Source: `hub_repo_details` on the HF connector.
- **Access check.** `hf_whoami`: authenticated as **vhmaadhav**, OAuth scopes include `read-repos`. A direct read of the gated repo returned: `HF_FS_ACCESS_DENIED ... you are not in the authorized list`. **So vhmaadhav does not currently have access.** Note: the token's `expires_at` is 2026-10-30 09:16:59 UTC, so reconnect the HF connector if reads start failing.
- **What Maadhav must do.** While logged in as vhmaadhav on huggingface.co, open each repo page, accept the Llama 3.2 Community License and submit the access form if one appears. Approval may be manual and can take time. Repeat for each gated repo we use. I did not read the licence text, so its attribution and use-policy terms are **UNVERIFIED** here.

## 6. Published performance and memory (vendor numbers, cited)

**Vendor, Meta Llama README.** Llama 3.2 1B on OnePlus 12 (16 GB), prompt length 64, adb binary, KleidiAI on.

| Model | Decode tok/s | TTFT s | Prefill tok/s | PTE MiB | RSS MiB |
|---|---|---|---|---|---|
| 1B BF16 | 19.2 | 1.0 | 60.3 | 2,358 | 3,185 |
| 1B SpinQuant | 50.2 | 0.3 | 260.5 | 1,083 | 1,921 |
| 1B QAT+LoRA | 45.8 | 0.3 | 252.0 | 1,127 | 2,255 |

Source: https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md

**Paper, ExecuTorch MLSys 2026, authors' measurements.** Llama 3.2 1B, 256 prompt and 256 generated tokens, min to max over 3 runs.

| Device | Backend | GS | Prefill tok/s | Decode tok/s | Size MiB |
|---|---|---|---|---|---|
| Galaxy S25 Ultra | XNNPACK CPU | 128 | 649.8 to 658.1 | 71.7 to 72.0 | 742 |
| Galaxy S25 Ultra | XNNPACK CPU | 32 | 524.6 to 528.9 | 65.9 to 67.1 | 821 |
| Pixel 9 Pro XL | XNNPACK CPU | 128 | 270.6 to 297.0 | 34.2 to 35.0 | 742 |
| Pixel 9 Pro XL | XNNPACK CPU | 32 | 235.5 to 245.9 | 31.3 to 33.0 | 821 |

Source: https://arxiv.org/html/2605.08195v1 (paper appeared at MLSys 2026, Industry track)

These are vendor-reported numbers, not independent. The paper's authors are Meta and PyTorch staff. No Raspberry Pi numbers were found.

**Implication.** A decode of about 30 to 70 tok/s on a flagship phone means a 96 to 128 token explanation finishes in a few seconds. Pi 5 will be slower. **UNVERIFIED** estimate: single-digit to low-teens tok/s. We must measure on the Pi.

## 7. Plan, risks and fallbacks

### Steps

1. **Gating (Maadhav).** Accept the licence on the three HF repos. Gate on approval before any download. Owner: Maadhav.
2. **Pin the toolchain.** Check out `v1.5.1` on the x86-64 Linux box. Create a Python 3.12 venv. Install the torch version named in the v1.5.1 install script (**UNVERIFIED pin**). Run `pip install -e ./extension/llm/tokenizers/`. Compute rule: check `sensors | grep Tctl` first. If the export is heavy on the laptop, run it on Kaggle CPU, following the user's rule.
3. **Export v0 (BF16 to 8da4w XNNPACK).** Use the config in section 2.3. Confirm the delegation table shows linear ops delegated. Record size and RSS. Save the `.pte` and the export log under `docs/`.
4. **Host smoke test.** `llama_main` on x86-64 (native build from the Llama README) with a fixed prompt. Confirm the token stream is sane with the Llama 3 chat template. **Smoke only** on the laptop. Full runs go to Kaggle.
5. **Export v1 (SpinQuant) when the weights arrive.** Compare v0 and v1 on our explanation eval (faithfulness, unsupported-claim rate). Pick the winner.
6. **Android.** Create a thin test app on `org.pytorch:executorch-android:1.5.1` with `LlmModule`. Push the `.pte` and `tokenizer.model` with adb. Measure on a real phone: load time, TTFT, decode tok/s, RSS, thermals, airplane mode. Pick a low-RAM phone as a worst case. Note: if the phone has 6 GB or less, RSS of about 1.9 GB is a risk.
7. **Pi 5.** Install `executorch==1.5.1` from pip on 64-bit Pi OS. Run the `.pte` via `executorch.runtime` first (no C++ build). Measure with an 8 GB Pi if possible. Build `llama_main` natively only if the Python path is too slow or too fiddly. Record peak RSS, p50, p95, and thermals.
8. **Structured output.** Implement v1 (JSON, verifier, template fallback). Implement v2 (candidate scoring) for the ablation. Keep #19476 as a watch item. Do not write a C++ mask fork unless the eval demands it.

### Risks

- **Gating delay.** Maadhav's approval is outside our control. Block Llama work until it lands. The detector and template path can proceed in parallel.
- **Explanation quality at 1B.** Plain PTQ hurts 1B models. Use SpinQuant or QAT+LoRA if available. Measure, do not assume.
- **Low-RAM phones.** About 1.9 GB RSS plus the app. Gate the Llama feature on device RAM and fall back to the template below a threshold. Threshold **UNVERIFIED**.
- **Pi 5 speed and memory.** Unknown. A 4 GB Pi is tight. Plan for 8 GB or a timeout.
- **Chat template is ours.** The runner may not apply it. Test explicitly.
- **Version churn.** ExecuTorch ships minor releases every few weeks. Pin 1.5.1 and do not follow `main`.
- **Licence obligations.** Llama 3.2 licence terms, including attribution and the use policy, must be handled in the write-up. **UNVERIFIED** details.
- **llama.cpp is not a legal fallback here.** CONTEXT.md says ExecuTorch is mandatory for the explanation layer. If ExecuTorch fails, the fallback is the template path, and the write-up must say the Llama layer is absent.

### Fallbacks

- **If export fails.** Try the SpinQuant or QAT+LoRA checkpoint with its own config (section 2.4). If both fail, report the blocker and keep the template path.
- **If Android integration fails.** Demo on the Pi, which runs the same `.pte` through `executorch.runtime`.
- **If the Pi is too slow.** Reduce `max_new_tokens`, or use candidate scoring (v2) so no decode loop is needed.
- **If gating is denied.** Do not swap to an unlicensed mirror. Ask Maadhav how to proceed.

## 8. Unverified list

- Torch version pin for executorch 1.5.1.
- Exact HF file layout of the gated repos (`original/consolidated.00.pth` or similar).
- Whether the runner applies the Llama 3 chat template.
- Current repo URL for the Llama Android demo (in-repo path 404).
- Minimum RAM on Android. No official figure.
- Pi 5 build and run. No official guide found. Pi 5 performance, memory and thermals.
- Whether #19476 (grammar decoding) has merged since May 2026.
- The Llama 3.2 licence text and its obligations.
- Exact export module path (`extension.llm.export` vs `executorch.extension.llm.export`) for a pip-only install.

## Sources

- https://github.com/pytorch/executorch/releases
- https://pypi.org/project/executorch/
- https://repo1.maven.org/maven2/org/pytorch/executorch-android/maven-metadata.xml
- https://docs.pytorch.org/executorch/stable/llm/export-llm.html
- https://docs.pytorch.org/executorch/stable/llm/run-with-c-plus-plus.html
- https://docs.pytorch.org/executorch/stable/llm/run-on-android.html
- https://docs.pytorch.org/executorch/stable/using-executorch-android.html
- https://github.com/pytorch/executorch/blob/main/examples/models/llama/README.md
- https://github.com/pytorch/executorch/issues/19476
- https://arxiv.org/html/2605.08195v1
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct (via HF connector `hub_repo_details`; gated)

## Kaggle weights (checked 2026-10-30 with kaggle CLI 2.2.4)

Source: `kaggle models get`, `kaggle models instances get`, `kaggle models instances versions list` and `... versions files`. The CLI has no `--format json` for `models get`, so the output was parsed as text. Nothing was downloaded.

**Model.** `metaresearch/llama-3.2` (Meta, Llama 3.2 Community License, Kaggle model id 121027). Its instances are published under the PyTorch, Transformers and Keras frameworks (also 1B and 3B, base and instruct).

| Instance ref (`--models`) | Format | Size (bytes) | Use |
|---|---|---|---|
| `metaresearch/llama-3.2/transformers/1b-instruct` | HF Transformers, fp16 safetensors | 2,480,831,539 | ELA eval (`llama_ela_eval`) |
| `metaresearch/llama-3.2/pytorch/1b-instruct` | Meta native (`consolidated.00.pth`, `params.json`, `tokenizer.model` expected) | 2,473,861,448 | ExecuTorch export (`llama_export`) |
| `metaresearch/llama-3.2/pytorch/1b-instruct-spinquant_int4_eo8` | Meta SpinQuant INT4 (EO8), version 2 | 1,623,520,362 | Export variant v1 (section 7 step 5), not yet exported |
| `metaresearch/llama-3.2/pytorch/1b-instruct-qlora_int4_eo8` | Meta QAT+LoRA INT4 (EO8), version 2 | 1,646,138,277 | Export variant, not yet exported |

Ref form is `OWNER/MODEL/FRAMEWORK/VARIATION`. The CLI accepts lowercase framework names (`pytorch`, `transformers`), and the output shows `MODEL_FRAMEWORK_PY_TORCH` and `MODEL_FRAMEWORK_TRANSFORMERS`. Whether `kernels push` wants `pytorch` or `pyTorch` in `model_sources` is **UNVERIFIED**. The kernel mount path is `/kaggle/input/<model>/...`, also **UNVERIFIED**. Both jobs search `/kaggle/input` for the files they need, so the exact path does not matter.

**Licence consent (the gate).** `kaggle models instances versions files` returns **403 Forbidden** for `.../transformers/1b-instruct/1`, `.../pytorch/1b-instruct/1` and the two quantised 1B variants. Metadata calls work, so the gate is on the file level. We read this as the Llama 3.2 licence not yet being accepted on the Kaggle account in `~/.kaggle/kaggle.json`. The static Kaggle page rendered without the consent UI, so the exact button label is **UNVERIFIED**.

**What Maadhav must click.**
1. Log into kaggle.com as the account that owns `~/.kaggle/kaggle.json`.
2. Open https://www.kaggle.com/models/metaresearch/llama-3.2 and open the 1B Instruct variations (Transformers and PyTorch, plus the SpinQuant and QLoRA ones if we use them).
3. Accept the Llama 3.2 Community License (the "Request access" / accept prompt, if one appears). Do not sign in with any other account.
4. Check the gate is open: `kaggle models instances versions files metaresearch/llama-3.2/transformers/1b-instruct/1` should list files instead of 403. Do not run a download to test it.

**Quantised checkpoints.** SpinQuant and QLoRA exports are on Kaggle under `metaresearch` (the names match the HF repos `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8` and `-QLORA_INT4_EO8`). They are Meta checkpoints for the `llama_xnnpack_spinquant.yaml` and `llama_xnnpack_qat.yaml` configs, not ready-made `.pte` files. Whether their file layout matches what `export_llm` expects is **UNVERIFIED** until the file listing opens.

**Launch commands (not run).**
```bash
kaggle/launch.sh llama_ela_eval --gpu t4 --internet \
  --datasets maadhavvh/rakshak-data \
  --models metaresearch/llama-3.2/transformers/1b-instruct
kaggle/launch.sh llama_export --internet \
  --models metaresearch/llama-3.2/pytorch/1b-instruct
```

**Export pins used by `kaggle/jobs/llama_export.py`** (from the v1.5.1 tag):
- `executorch==1.5.1` on PyPI. Its requirements are `torchao>=0.18,<0.19` and `pytorch-tokenizers>=1.5.0`. The torch pin is not in its requirements.
- `torch==2.14.0` from `https://download.pytorch.org/whl/cpu`. This is the torch version `install_executorch.py` names at v1.5.1, and the CPU wheel exists on that index.
- `git clone --depth 1 --branch v1.5.1` for `extension/llm/tokenizers` and the `export_llm` module (`python -m extension.llm.export.export_llm`, run from the clone). The tag resolves to 3b60683.
- The v1.5.1 `llama_xnnpack_spinquant.yaml` example confirms the `export.max_seq_length` and `max_context_length` keys, which the export YAML includes.

**Still UNVERIFIED after this check:** the file layout inside the Kaggle instances; whether `pytorch/1b-instruct` contains `consolidated.00.pth` at its root; the Kaggle consent button label; the peak RSS and wall time of the export on Kaggle CPU (measured by the job, not yet run).
