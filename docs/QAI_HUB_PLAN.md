# Qualcomm AI Hub plan for on-device benchmarks

Research snapshot: 2026-10-30. Owner: research worker. Scope: what Qualcomm AI Hub (Workbench, `qai-hub` client) can and cannot do for Rakshak-Edge. Nothing was installed, no account was created, nothing was submitted. Items marked **UNVERIFIED** could not be confirmed from an official source in this session.

Canonical docs host: `workbench.aihub.qualcomm.com/docs` (the `app.aihub.qualcomm.com/docs` links 301-redirect there).

## 0. Bottom line

- AI Hub is a **cloud device farm for compiled models**. It accepts ONNX, TFLite and QNN DLC/context-binary models. It does **not** document ExecuTorch `.pte` as an input, so it cannot benchmark our Llama/ExecuTorch path.
- Use it for the **char-CNN and distilled encoder** (ONNX or TFLite). Llama/ExecuTorch numbers must come from our own phone (Android) and Pi 5.
- Workbench is described as **free to use** (FAQ). No quotas or job limits are published. Device queue times vary.
- Maadhav must create a Qualcomm ID, copy the API token from Workbench Settings, and run `qai-hub configure --api_token`. I did not and must not do this.
- Compute-unit control is only partly documented. `--compute_unit gpu` is confirmed. CPU and NPU/Hexagon selectors are **UNVERIFIED**.

## 1. Account, token, cost, data, licence

**Sign-up and token.**
- Create a Qualcomm MyAccount (Qualcomm ID), log in to Workbench, open the Settings page to copy the API token. Source: https://workbench.aihub.qualcomm.com/docs/hub/faq.html
- Install and configure: `pip install qai-hub` then `qai-hub configure --api_token API_TOKEN`; verify with `qai-hub list-devices`. Sources: https://pypi.org/pypi/qai-hub/json (package 0.57.0, released 2026-10-27, Python >=3.10); https://workbench.aihub.qualcomm.com/docs/hub/cli.html
- Accounts are per user ("Each user has their own organization"). Source: FAQ above.
- Token handling: keep it out of git. Use an env var locally or a Kaggle secret.

**Cost and quotas.**
- FAQ: "Qualcomm AI Hub Workbench, our platform, is currently completely free to use." Source: FAQ above. "Currently" means it can change.
- Some third-party pre-optimised models are paid, under the model maker's licence. Our own models are not affected. Source: FAQ above.
- No quota, rate limit or job cap is published. Jobs run "subject to availability"; "device popularity may impact queue times"; jobs "may occasionally fail due to device instability". Models over 2 GB may fail. Source: https://workbench.aihub.qualcomm.com/docs/hub/jobs.html ; FAQ above.

**What is uploaded.**
- Models and datasets uploaded to Workbench are private by default and visible only to users you share with. Job outputs (metrics) are covered by the same rule. Source: FAQ above.
- Temporary device storage is wiped when a job ends. Source: FAQ above.
- No retention period is stated for models or results kept in the account. **UNVERIFIED** (not stated; ask or read the Terms).
- Inference jobs upload **sample inputs**. Use only synthetic or public SMS text, never private messages or any user data. Source: https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html
- So the answer is: uploads include the model and any sample inputs we choose. No device or user data is sent unless we send it.

**Licence and publishing benchmark numbers.**
- Our own models: deployed assets "typically have the same distribution license as your model". Source: FAQ above.
- Qualcomm Terms of Use are the governing text. I could not read the clauses (the page returned only its title). Whether results may be published in a competition report is **UNVERIFIED**. Read https://www.qualcomm.com/site/terms-of-use before publishing numbers.
- The challenge rules on publishing third-party benchmark results are **UNVERIFIED**.
- Pre-optimised Llama assets are licensed under Meta's Llama 3.2 licence (`License: llama3` on the HF card) and Qualcomm's "Generative AI usage and limitations". Any Llama numbers in the report need that attribution. Sources: https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct ; https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct

## 2. Input formats, runtimes, compute units

**Inputs and targets (official).**
- Workbench supports "TensorFlow Lite, ONNX Runtime, and Qualcomm AI Engine Direct" as runtimes. AI Engine Direct formats are context binaries and DLC. Source: FAQ above.
- Inference jobs accept QNN DLC, QNN context binary, ONNX and TFLite. PyTorch (`.pt`/`.pt2`) must be compiled first with `submit_compile_job`. Source: https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html
- `--target_runtime qnn_dlc` appears in a compile example. The values `tflite` and `onnx` are used in examples but not listed as enum values. **UNVERIFIED** as an exhaustive list. Source: https://workbench.aihub.qualcomm.com/docs/hub/cli.html
- ExecuTorch `.pte`: **not mentioned** in the AI Hub FAQ, frameworks page, inference page or Python package description. Treat as **not supported**. Sources: https://workbench.aihub.qualcomm.com/docs/hub/frameworks.html ; https://pypi.org/pypi/qai-hub-models/json
- `qai-hub-models` (BSD-3-Clause, v0.65.0) lists export targets Qualcomm AI Engine Direct, LiteRT and ONNX. ExecuTorch is absent. Source: https://pypi.org/pypi/qai-hub-models/json ; https://github.com/quic/ai-hub-models

**ExecuTorch's own Qualcomm path (separate from AI Hub).**
- ExecuTorch has a Qualcomm QNN (AI Engine Direct) backend that delegates to Hexagon (HTP) and Adreno and writes `.pte`. Source: https://docs.pytorch.org/executorch/stable/backends-qualcomm.html (page fetched 2026-10-30).
- It needs the QNN SDK. The page recommends **QNN 2.37.0**. Verified SoCs on the page: **SM8550, SM8450** (Android/Linux) and SC8380XP (Windows on ARM). SM8650 appears only as a placeholder; SM8750 is not mentioned. Treat other SoCs as **UNVERIFIED**.
- Compile entry point (from the page): `python -m examples.qualcomm.scripts.deeplab_v3 --build_folder build-android --soc_model SM8550 --compile_only --download`. Our custom model would use `to_edge_transform_and_lower_to_qnn`. Source: same page.
- Measuring that `.pte` on a phone is our own job (adb or the Android app). AI Hub does not run it. So this is an NPU bonus path, separate from AI Hub.

**Compute unit (CPU / GPU / NPU) per profile job.**
- Confirmed: `qai-hub submit-profile-job ... --profile_options " --compute_unit gpu"`. Source: https://workbench.aihub.qualcomm.com/docs/hub/cli.html
- Confirmed: TFLite options go inside `--profile_options`, e.g. `"--tflite_options number_of_threads=4;qnn_htp_precision=kHtpQuantized"`. Same source.
- ONNX on HTP: by default, unsupported ops fall back to **CPU**. Adding `--onnx_execution_providers=qnn` runs quantised models on the NPU. Source: https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html
- Profiling reports which compute unit ran each layer. Source: same page.
- **CPU-only (Arm cores) and NPU-only selectors: UNVERIFIED.** Only `gpu` is documented as a `compute_unit` value. Run `qai-hub submit-profile-job --help` to list the values before relying on them.
- Our use: the Arm-CPU number we care about (ExecuTorch XNNPACK) is not in AI Hub. For the char-CNN, CPU vs NPU can be attempted via ONNX providers or TFLite options, but CPU-only is **UNVERIFIED**.

## 3. Device list

Official pages list these (not exhaustive):
- Phones: Samsung Galaxy S25, S25 Ultra, S25+, **S26**; Samsung S24 family; Galaxy S23 Ultra. Sources: https://workbench.aihub.qualcomm.com/docs/hub/devices.html ; https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct
- Chipsets (cached model page): Snapdragon 8 Elite, 8 Elite Gen 5 (for Galaxy), X Elite, X2 Elite, X Plus 8-core, QCS5121, Dragonwing IQ-8275/IQ-9075/IQ-X7181/Q-8750, SA8255P/8295P/8650P/8775P. Source: https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct
- An older cached models page also lists Snapdragon 8 Gen 1/2/3, Snapdragon 888, Pixel 3 to 5a, Galaxy S21 to S24 and Galaxy Tab S8. Source: search snippet of https://aihub.qualcomm.com/models (cache; **UNVERIFIED** as current).
- **Snapdragon 7-series, 6-series, 4-series, Xiaomi and OnePlus: not confirmed in any source found.** Check `qai-hub list-devices` (the official list), or the device filter `--device-attr chipset:...`. Source: https://workbench.aihub.qualcomm.com/docs/hub/cli.html
- **Raspberry Pi and generic Arm: not offered.** The closest Arm Linux devices are the Dragonwing IQ and SA-series boards and the Arduino VENTUNO Q. Source: https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct . Pi 5 is **not** available.

**Practical consequence.** The only confirmed phones are Samsung flagships. A mid-range Indian-market phone in the device farm is **UNVERIFIED**. Our mid-range Android measurement may have to be done on our own phone.

## 4. What a profile or inference job returns

**Profile job** (per https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html):
- Latency is measured over **100 iterations**. One extra iteration collects per-layer metrics. `--max_profiler_iterations` changes the count.
- Memory is reported in **bytes**, times in **microseconds**.
- Per-layer data includes the compute unit used, per-layer runtime, optrace and HTP runtime metrics.
- Artifacts (op-by-op timing summaries, QHAS, logs) depend on the job type. Format "may change without notice". Source: https://workbench.aihub.qualcomm.com/docs/hub/jobs.html
- **Min/median latency split, peak memory and model load time as named fields: UNVERIFIED.** The docs say latency and per-layer data, not these exact fields. Check the JSON from a real job.

**Inference job:**
- Inputs are a dict of feature names to tensors (e.g. `inputs=dict(x=[sample])`). Source: inference page above.
- `download_output_data()` returns `dict[str, list[np.ndarray]]` from the device. This is what we compare against our PyTorch/ONNX reference for numerical accuracy.
- Expect small numerical differences from fp16/int8 on device. Source: inference page above.

## 5. Llama 3.2 1B on AI Hub (relevance)

- Llama-v3.2-1B-Instruct is listed (`llama_v3_2_1b_instruct`). Quantisation: **w4** and **w4a16**. Runtimes in the table: **GENIE** and **GENIEX_QAIRT**, not ExecuTorch. Source: https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct
- Qualcomm-reported decode tok/s (context 4096), GENIE w4a16: **Snapdragon 8 Elite Gen 5 For Galaxy 65.2**, **8 Elite For Galaxy 67.2**, **X2 Elite 83.6**, X Elite 34.2, Dragonwing IQ-9075 41.6. TTFT ranges 0.03 to 1.1 s on the 8 Elite class. Same source.
- Caveats on the same card: "Due to licensing restrictions, we cannot distribute pre-exported model assets for this model." And "Genie support will be deprecated soon." No memory or size figures. Same source.
- Relevance: a **vendor reference point for NPU LLM throughput**, not our path. The challenge mandates ExecuTorch on Arm CPU. Genie numbers may be cited as context only and labelled as a different runtime, with Llama licence attribution.
- The Llama page lists Galaxy S25/S25 Ultra/S25+/S26 as supported devices, but its Genie performance table is keyed by chipset (8 Elite, 8 Elite Gen 5, X Elite, X2 Elite, Dragonwing, SA-series), not by phone model. The numbers are **not** an ExecuTorch comparison.

## 6. Concrete workflow for us

**Scope.** Only the char-CNN (message expert) and the distilled encoder go through AI Hub. Llama/ExecuTorch and the URL model are measured on our own hardware.

**Steps.**
1. Maadhav: create the Qualcomm ID, copy the token from Workbench Settings, pin `qai-hub` (0.57.0 on 2026-10-30; re-check on the day).
2. Export the student models to **ONNX** (char-CNN, encoder). Optionally export **TFLite** as a second format. Do the export on Kaggle, per the project rule (laptop = code only). Model export is light, but the rule holds. Export signatures and input shapes are our code; not checked against AI Hub. **UNVERIFIED** that AI Hub accepts our exact op set.
3. Confirm the device list: `qai-hub list-devices`. Pick about 6 devices: at least two Galaxy S25/S26 class, one Snapdragon 8 Gen 2 or 8 Gen 3 if listed, one X Elite if we want a laptop-class point, and **one mid-range phone if listed** (otherwise skip and note it).
4. Submit compile and profile. Python API: `submit_compile_and_profile_job` (name taken from the task brief; signature **UNVERIFIED** here, check `qai_hub` API docs). CLI equivalents: `qai-hub submit-compile-job` and `qai-hub submit-profile-job`, with `--device`, `--profile_options`, and `--onnx_execution_providers` / `--tflite_options` for the NPU vs CPU variants.
5. Run the inference job with 100 to 1000 synthetic or public SMS-sized inputs. Compare `download_output_data()` with the reference outputs. Record the max abs diff and label-flip rate.
6. Save JSON results under `benchmarks/qaihub/` (one file per model, device, compute setting, date). Include job IDs, the `qai-hub` version and the device name.
7. Completion notification: use the Kaggle ntfy event from the project's job runner, not a poll loop. Do not write a wait loop.

**Where it runs.**
- Submission is API calls only, so a laptop is fine for the submission step. No local compute.
- Kaggle is the project's compute rule. A Kaggle kernel can run steps 2 to 6 with the token as a Kaggle secret. Whether Kaggle kernels have outbound internet for `api.aihub` is **UNVERIFIED**; test with a single `qai-hub list-devices` before the run.

## 7. Caveats

- **NPU vs CPU framing.** A judge may read NPU numbers as the headline and miss that the mandated path is ExecuTorch on Arm CPU. Label every AI Hub number with runtime, compute unit, device and date. Put the CPU/ExecuTorch numbers first. Do not use NPU numbers to claim the Llama path is fast.
- **Vendor and cloud.** AI Hub devices are Qualcomm-hosted, not our hardware. Queue times and device state vary. Results are not a measure of our build on a user's phone.
- **Submission numbers come from our own phone.** The final numbers in the technical report must be measured on our physical device (Android for ExecuTorch, Pi 5 for the reproducibility target). AI Hub is a supplement for model-level comparison across chips, not the headline.
- **Model-level, not app-level.** AI Hub profiles one model graph. It does not include our canonicaliser, fusion, verifier or I/O. End-to-end latency still needs an on-device app test.
- **Licence of published numbers.** Check Qualcomm Terms of Use and the challenge rules before publishing (section 1).

## 8. UNVERIFIED list

- Qualcomm Terms of Use clauses on publishing benchmark results (page returned only its title).
- Whether the challenge rules allow third-party benchmark numbers in the report.
- Full `compute_unit` enum (only `gpu` confirmed). CPU-only and NPU-only selectors.
- Full `target_runtime` enum (only `qnn_dlc` confirmed in the docs).
- Mid-range Snapdragon (7/6/4 series), Xiaomi and OnePlus support in AI Hub.
- Exact profile result fields (min/median latency, peak memory, load time).
- Retention period for uploaded models and results.
- Whether Kaggle kernels can reach AI Hub over the network.
- Signature of `submit_compile_and_profile_job` in the current `qai-hub` release (0.57.0).
- ExecuTorch QNN support for SoCs beyond SM8550 and SM8450 (SM8650, SM8750).

## Sources

- https://workbench.aihub.qualcomm.com/docs/ (redirected from app.aihub.qualcomm.com/docs)
- https://workbench.aihub.qualcomm.com/docs/hub/faq.html
- https://workbench.aihub.qualcomm.com/docs/hub/cli.html
- https://workbench.aihub.qualcomm.com/docs/hub/frameworks.html
- https://workbench.aihub.qualcomm.com/docs/hub/devices.html
- https://workbench.aihub.qualcomm.com/docs/hub/jobs.html
- https://workbench.aihub.qualcomm.com/docs/hub/inference_examples.html
- https://workbench.aihub.qualcomm.com/docs/hub/index.html
- https://pypi.org/pypi/qai-hub/json
- https://pypi.org/pypi/qai-hub-models/json
- https://github.com/quic/ai-hub-models
- https://aihub.qualcomm.com/models/llama_v3_2_1b_instruct
- https://aihub.qualcomm.com/ (sign-up and device categories)
- https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct
- https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
- https://www.qualcomm.com/site/terms-of-use (footer link; content not retrieved)
- Companion file: `docs/EXECUTORCH_PLAN.md` (this repo)
