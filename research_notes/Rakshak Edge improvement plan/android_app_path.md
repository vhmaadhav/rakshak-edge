# Android app path: ExecuTorch Llama 3.2 1B + ONNX classifier (Rakshak Edge)

Research date: 2026-11-06. Scope: fastest low-risk path to an offline Android demo APK. Coverage is partial. Sources were mostly search snippets and doc pages; the LlamaDemo source tree and LlmModule Java signatures were NOT verified. Items marked UNVERIFIED need a check before the team relies on them.

## ExecuTorch Android: AAR and LlmModule

### Takeaway
Use `org.pytorch:executorch-android` from Maven and start from the official LlamaDemo app in the ExecuTorch repo. The exact LLM API (LlmModule) and current LlamaDemo build steps were not confirmed from primary source in this pass. Pin the version from the docs page you actually build against.

### Cited Findings
- The official docs say to add `org.pytorch:executorch-android:${executorch_version}`; the example pins `1.0.0`. Each release's AAR is published to Maven and S3. ABIs: arm64-v8a and x86_64. — [using-executorch-android (stable)](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
- A standalone AAR carries no dependency metadata, so the direct-AAR route needs `com.facebook.fbjni:fbjni:0.7.0` and `com.facebook.soloader:soloader:0.10.5` declared explicitly. The Maven route does not need them. — [using-executorch-android (stable)](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
- Docs for versions 1.1 through 1.5 exist under the same path pattern (`/executorch/1.5/`, `/1.3/`, etc.), so 1.5.x docs exist. I did not confirm the 1.5.x AAR coordinate or its Maven listing. — [1.5 docs index, via search](https://docs.pytorch.org/executorch/1.5/using-executorch-android.html)
- The LlamaDemo app lives at `examples/demo-apps/android/LlamaDemo` in the ExecuTorch repo. Docs say to open it in Android Studio and that instrumented tests run with `./gradlew connectedAndroidTest` after pushing a `.pte` and `tokenizer.bin` to `/data/local/tmp/llama`. — [LlamaDemo docs, via search snippet](https://docs.pytorch.org/executorch/stable/llm/llama-demo-android.html) (page body not returned by fetch; treat as snippet-level)
- Search snippet describes the LLM Java API as: construct `LlmModule` with model category, model path, tokenizer path, and temperature; call `load`; call `generate(prompt, sequence_length, listener)`. — same LlamaDemo docs, snippet-level
- Stats/memory: the ExecuTorch LLM runner docs show `Stats` can report resident set size (via `get_rss_bytes()`). `reset()` clears prefill state and KV-cache position, so call it between independent prompts. A warmup run is recommended before timing. — [ExecuTorch main, run-with-c-plus-plus](https://docs.pytorch.org/executorch/main/llm/run-with-c-plus-plus.html)
- Build flags for native CMake builds include XNNPACK, data loader, module, runner util, and tensor extensions. — [PR 5456 diff, via search](https://github.com/pytorch/executorch/pull/5456/files) (snippet-level)

### Inferences
- Start from the LlamaDemo app, swap the model and tokenizer for our exported `.pte`, and wire the Kotlin UI to our pipeline. Avoid writing a custom JNI runner.
- Export with the same ExecuTorch version as the AAR. Mismatched export and runtime versions are the most likely cause of load failures (this is general practice, not a sourced finding).

### Gaps
- LlmModule exact Java/Kotlin signatures, callback interface name, and `stop()` API: NOT verified from source. Read `extension/android` in the repo before coding.
- Minimum Android API level, minimum RAM, and NDK/AGP versions for the current LlamaDemo: not found.
- Current LlamaDemo README fetch returned 404 at the tried path. Use the repo directly.

## Measured speed and memory for Llama 3.2 1B on phones

### Takeaway
Only one phone has a number in the sources I reached: about 50 tok/s decode for SpinQuant INT4 on a OnePlus 12 (Snapdragon 8 Gen 3). The ExecuTorch benchmark table gives 50.2 tok/s decode and 1,921 MiB RSS for SpinQuant 1B, but that table is not tied to a named phone in the snippet. No mid-range or MediaTek/Exynos numbers were found.

### Cited Findings
- Llama 3.2 1B SpinQuant 4-bit: about 50 tok/s decode on a OnePlus 12, about 2.6x the bf16 model. — [ExecuTorch PR 6434, via search](https://github.com/pytorch/executorch/pull/6434/files) (snippet-level, vendor-reported)
- ExecuTorch LLM benchmark table (prompt length 64): bf16 1B = 19.2 tok/s decode, 60.3 tok/s prefill, 3,185 MiB RSS; SpinQuant 1B = 50.2 tok/s decode, 260.5 tok/s prefill, 1,921 MiB RSS. Measured with an adb binary, KleidiAI not enabled by default. — [ExecuTorch PR 6434 / docs, via search](https://github.com/pytorch/executorch/pull/6434.diff) (snippet-level)
- React Native ExecuTorch docs list peak memory for Llama 3.2 1B on Android (XNNPACK): 3.2 GB base, 1.9 GB SpinQuant, 2.2 GB QLoRA; longer context raises it. Same project suggests quantized 1B for 6 GB RAM devices. — [RN ExecuTorch useLLM](https://docs.swmansion.com/react-native-executorch/docs/llms/useLLM) and [memory management](https://www.mintlify.com/software-mansion/react-native-executorch/advanced/memory-management) (third-party wrapper, not ExecuTorch core)
- Arm learning path: Llama 3.2 1B Vulkan export (8-bit dynamic activations, 4-bit weights) yields about 1.8 GB file; changing quantization, group size, or context changes memory. — [Arm learning path](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/executorch-vulkan-learning-path/how-to-3-export-the-vulkan-pte/) (snippet-level)
- 3B SpinQuant decodes at 19.7 tok/s in the same ExecuTorch table (vs 7.6 bf16). — [ExecuTorch PR 6434 / docs, via search](https://github.com/pytorch/executorch/pull/6434.diff)

### Inferences
- Plan for roughly 1.9 GB peak RSS and a model file in the 1.1 to 1.8 GB range. Phones with 6 GB RAM are the realistic floor; 8 GB or more is safer for a demo run with the app open.
- Expect decode speed well below 50 tok/s on mid-range phones. This is an inference from the single data point, not measured. A 1-2 sentence report with 30 to 60 generated tokens is likely fast enough for a demo.
- Our pipeline does not need the LLM for every item. Keep the template fallback path so the demo still completes if generation is slow or the phone is memory-constrained.

### Gaps
- No mid-range Snapdragon 7-series, Dimensity, or Exynos measurements found.
- No independent (non-vendor) benchmark found.
- Thermal throttling behaviour during sustained generation: not found.

## ONNX Runtime Android vs exporting the classifier to ExecuTorch

### Takeaway
ONNX Runtime Android is the lower-risk route for the ~10 MB classifier because it avoids a second export pipeline. The Android artifact name was NOT confirmed from the docs I reached.

### Cited Findings
- ONNX Runtime Java docs list Maven artifacts `com.microsoft.onnxruntime:onnxruntime` (CPU, desktop) and `onnxruntime_gpu`. They do not list an Android artifact. Java API classes: `OrtEnvironment`, `OrtSession`, `OrtSession.SessionOptions`, `OnnxTensor`, `Result`. — [ONNX Runtime Java get-started](https://onnxruntime.ai/docs/get-started/with-java.html)
- A search result says the Android build is published as an AAR, with the Android build guide at [onnxruntime.ai/docs/build/android](https://onnxruntime.ai/docs/build/android.html). The guide says AARs are generated under `java/build/android/outputs/aar` when built with the Java option.

### Inferences
- The likely artifact is `com.microsoft.onnxruntime:onnxruntime-android` (UNVERIFIED, from memory; check Maven Central before use).
- Exporting the classifier to ExecuTorch is possible but adds a second runtime and a second export path. Not worth it for a 10 MB model unless the team already has an ExecuTorch export script.

### Gaps
- Exact current ONNX Runtime Android version and AAR size: not verified.
- Text tokenisation on device: not researched. Our canonicaliser and tokenizer need a Kotlin or Java port, or we pre-tokenise on the Pi. Recommend porting the tokenizer vocabulary as a flat file.

## Porting Python rules and verifier: Chaquopy vs Kotlin

### Takeaway
Rewrite the rules and verifier in Kotlin. Chaquopy adds a Python interpreter and bundled packages to the APK, and no APK size figure for Chaquopy was found. Keep the Python code in the repo as the reference and test oracle, and check the Kotlin port against it.

### Cited Findings
- Chaquopy is native code, so ABI filters are the main size control. Chaquopy 17.0.0 supports AGP 7.3.x to 9.1.x and Python 3.13 or newer for the 16 KB page-size requirement. — [Chaquopy wrapper summary, via search](https://awesome.ecosyste.ms/projects/github.com%2Fchaquo%2Fchaquopy) (third-party summary; the official chaquo.com docs were not fetched)
- No published Chaquopy APK size numbers were found.

### Inferences
- Rules are regex and string logic; the verifier is deterministic. Both port to Kotlin with modest effort and no interpreter.
- Chaquopy would keep the Python code unchanged but adds build complexity and APK size. This is a judgment, not sourced.

### Gaps
- Chaquopy size and cold-start cost: not measured. Official Chaquopy docs not read.

## Thin client to Pi vs fully on-phone

### Takeaway
Not researched in this pass. The hackathon brief asks for offline on-device operation, so the on-phone path is the safer reading. A Pi-backed mode would be a fallback only.

### Gaps
- What judges accept for on-device challenges: not researched. Check the PS3 problem statement wording directly.

## SMS UX within Play policy

### Takeaway
Play restricts READ_SMS and the other SMS and call-log permissions to apps that are the registered default SMS, Phone, or Assistant handler, or otherwise qualify under a declaration. For our app, share intent and paste are the policy-safe entry points. Sideloaded demo APKs can use other methods, but this is not a Play path.

### Cited Findings
- Restricted groups include READ_SMS, RECEIVE_SMS, SEND_SMS, WRITE_SMS, READ_CALL_LOG and others. Apps that do not qualify must remove them from the manifest. — [Google Play SMS and Call Log policy](https://support.google.com/googleplay/android-developer/answer/9047303) (fetched)
- Permitted uses include default SMS, Phone, and Assistant handling. Apps must be actively registered as the default handler before requesting the permissions, and must stop using them when they are no longer the default. — same page (fetched)
- Listed alternatives: Share Intent, SMS Retriever API, SMS User Consent API, Incoming Call Retriever API, Digital Credentials API, and the SMS, Share, and Dial intents. A user-initiated picker is not named on the page. — same page (fetched)
- Apps that lack a declaration or fail the criteria "may be removed from Google Play." — same page (fetched)
- Effective 2027-01-27, account verification via phone call is no longer a permitted use for READ_CALL_LOG. — same page (fetched)
- Google's 2019 policy notice confirms the declaration form requirement. — [Android Developers blog, 2019](https://android-developers.googleblog.com/2019/01/reminder-smscall-log-policy-changes.html) (snippet-level)

### Inferences
- Share intent (`ACTION_SEND` with `text/plain`) from the SMS app into ours is the cleanest demo flow. Paste is the fallback. This is an inference from the listed alternatives.
- Notification listener access for SMS is not in the listed alternatives and is likely to be policy-sensitive. Avoid it for Play; treat as sideload-only at most. Not verified.

### Gaps
- Sideload-only rules (for a non-Play demo APK) are not sourced. The APK can be sideloaded technically; whether the judges accept a sideloaded demo is not verified.

## Thermal, memory, and low-memory-killer practice

### Takeaway
Not researched in this pass. Use the ExecuTorch Stats RSS reading during the demo to watch memory, and keep the LLM off the main thread.

### Cited Findings
- ExecuTorch Stats exposes RSS; `reset()` between prompts; warmup before timing. — [ExecuTorch main, run-with-c-plus-plus](https://docs.pytorch.org/executorch/main/llm/run-with-c-plus-plus.html)
- Peak RSS of about 1.9 GB for SpinQuant 1B. — see above

### Gaps
- mmap behaviour of the `.pte` on Android: no ExecuTorch-specific source found.
- Foreground service, background kill, and thermal throttling guidance: not researched.

## Recommended next steps (inferred, for the team to confirm)
1. Clone `pytorch/executorch`, open `examples/demo-apps/android/LlamaDemo`, and build it unchanged on one phone with an 8 GB or larger RAM budget.
2. Read `extension/android` to confirm LlmModule signatures and the callback name before writing app code.
3. Export Llama 3.2 1B SpinQuant with the same ExecuTorch version as the AAR.
4. Port the rules and verifier to Kotlin, checked against the Python outputs on a fixed test set.
5. Add ONNX Runtime Android after confirming the artifact on Maven Central.
6. Build the share-intent input flow; keep paste as fallback.
