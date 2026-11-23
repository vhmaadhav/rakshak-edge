# Phase 1 submission checklist (audit of 2026-11-23)

Audit of `main` at `6d9b60b` (after PRs #26 and #27). Each line below was checked against code, build files,
the organisers' page, or a run on a fresh clone. Repo documents were not taken as evidence; where a document
disagreed with the code, the code won and the document was corrected in the same change as this file.

## 1. Deadline and submission route (organisers' page)

Source: <https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html>, read 2026-11-23.

| Item | Value | Status |
|---|---|---|
| Phase 1 (Open Qualifying) deadline | **23 November 2026** | Confirmed on the page. **No time of day and no time zone are given.** No extension found on any organiser source |
| Shortlist results | mid-December 2026 | Confirmed |
| Phase 2 (PoC) deadline | 5 February 2027 | Confirmed |
| Finals | 15–16 February 2027 | Confirmed |
| How to submit | "hosted on GitHub and submitted using the challenge submission form" | Confirmed. **The page gives no form link** |
| Repo must be public? | Not stated | Unknown |
| Report / video / app format, length, template | Not stated | None given |
| Scoring rubric | Not stated | None given |
| Questions | support@armbharatchallenge.com; technical: tech-queries@armbharatchallenge.com (PS number in the subject) | From the page |

Ask the organisers today for: the cut-off time and time zone, the form link (if the team leader does not
already have it), and whether a private repo with judge access is acceptable.

## 2. Problem Statement 3 requirements against the code

| Requirement (organisers' wording, shortened) | Evidence in code | Verdict |
|---|---|---|
| Runs on Arm edge hardware, no cloud | Pi service binds loopback only and refuses other hosts (`deploy/pi/service/rakshak_edge_service.py` `serve`); `start_service.sh` runs it under `unshare -rn`. Android manifest removes `INTERNET` and `ACCESS_NETWORK_STATE` (`android/app/src/main/AndroidManifest.xml` lines 10–11) | Met |
| Llama-family model for the advisory part, on-device, ExecuTorch as runtime | Pi: `deploy/pi/bench/rbench/runners/pte_direct.py` (`executorch.extension.pybindings.portable_lib`), default `--llm-runner direct`; `.pte` pinned in `deploy/pi/bench/registry.toml`. No other LLM runtime anywhere in the tree. Android: `org.pytorch:executorch-android:1.5.1`, called from `android/app/src/main/java/org/rakshak/edge/llm/LlmExplainer.kt` | Met on the Pi. Android is wired but Llama has not been run on a phone |
| Detect scam SMS | SMS student + rules, `POST /analyze` | Met |
| Detect phishing URLs | URL student + domain rules, `POST /analyze_url` | Met |
| Monitor and classify network traffic | `POST /analyze_flows`, `src/rakshak/nids/host_rules.py`. Re-run of the 26-file holdout (rules only, regenerated with `make_holdout_pcaps.py`): 11 / 11 attacks, benign alerts 1 / 6 (default) and 0 / 6 (`--nids-dns-min-distinct 3`) | Met, on **synthetic** captures only |
| Explain threats in plain language | Template + Llama sentences filtered by the verifier (`src/rakshak/verifier.py`) | Met |
| Recommend remediation | `src/rakshak/remediation.py`, returned with every verdict | Met |
| Hailo pipeline documented if used | Not used | n/a |

## 3. The four deliverables

| Deliverable | State on 2026-11-23 | Blocking? |
|---|---|---|
| Source code (GitHub) | On GitHub, **repository is private**, no licence file | **Yes**, until visibility or judge access is settled |
| Application | Pi service and Android app exist in source. **No APK or GitHub release exists**; the `release` build type has no signing config. This Mac has no Android SDK or JDK 17, so it was not built here | **Yes**: attach an installable APK (e.g. `app-benchmark.apk`, debug-signed) or say how judges get the app |
| Demonstration video | Not in the repo, no link anywhere (`docs/START_HERE.md` placeholder) | **Yes** |
| Technical report | Not in the repo, no link anywhere. `docs/SYSTEM_V1.md` (frozen 2026-11-12) is its stated source | **Yes** |

## 4. What a judge can run from a fresh clone

| Step | Result on a fresh clone (macOS, 2026-11-23) |
|---|---|
| `uv sync --locked` | Works |
| `uv run pytest -q -n auto` | 2,255 passed, 43 skipped, 4 failed. The 4 need Linux (`/proc`, short AF_UNIX paths, process stop state) and fail the same way on `main` |
| `uv run ruff check src tests` | Passes |
| `python3 deploy/pi/service/ui_mock.py` | Serves the page with fake data (it ignores `--help` and starts the server) |
| Real service, `rakshak_edge_service.py --root .` | **Exits at start**: `model sms_student is not ready … 'present': False` |
| `scripts/release_manifest.py` | Writes the manifest, exits 1: 15 of 18 pinned files missing |

Files the service and the app need, none of them in git:

| File | sha256 (from `registry.toml`) | Public source |
|---|---|---|
| `models/sms_student/best_student.onnx` (also the Android asset `sms_student.onnx`) | `5341e14c…2466` | **None** |
| `models/url_student/best_url_student.onnx` + `url_side_scaler.json` | `1bbd2c57…3fc7`, `c6b3fc96…01e` | **None** |
| `models/nids/student_cse_cic_ids2018_int8.onnx` (service refuses to start without it) | `d3dfdcb1…8d14` | **None** |
| Llama 3.2 1B SpinQuant `.pte` + `tokenizer.model` | `8715cdba…da3b`, `82e9d319…1b55` | Hugging Face `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` (revision in `src/rakshak/llama_source.py`), Llama 3.2 licence |

The three ONNX models are small (388 KB, 2.7 MB, 6.7 KB). They are not on this Mac; they live in
`kaggle/out/` on the machine that ran the Kaggle jobs and on the Pi. Without them a judge can run the tests
and the mock page, but not the detector.

## 5. Numbers and their backing

Backed by committed raw data: `docs/results/{explainer_speed,nids_xdata,recall_v2}/metrics.json`; the NIDS
holdout (regenerated from the committed generator); the paired-transaction and Indian-strata counts in
`docs/VALIDATION_REPORT.md` (arithmetic re-checked).

Reported in prose only (raw logs stay on the devices): all Pi latency, throughput and power numbers; the
OnePlus 13R Android benchmark; SMS campaign F1 0.9813 (the stored run log says 0.9749; unresolved);
robustness recall 0.9803; URL OOD figures. Say so if a judge asks, or commit the raw files.

## 6. Before submitting

- [ ] Confirm the cut-off time, time zone and form link with the organisers.
- [ ] Decide repo visibility (public, or private with judge access) and add a `LICENSE` (README says one is needed before a public release).
- [ ] Publish the three ONNX models (GitHub release assets, with the sha256 above) so the service and the app run from a clone.
- [ ] Build and attach an installable APK to the same release.
- [ ] Record the demo video and link it in `docs/START_HERE.md` "Release placeholders".
- [ ] Write the technical report (from `docs/SYSTEM_V1.md` plus `docs/VALIDATION_REPORT.md` and `docs/benchmarks.md`) and link it.
- [ ] Fill the team roles and affiliations in `README.md` "Team" (the page requires a Team Leader, and a Faculty Mentor for student teams).
- [ ] Fill the frozen commit SHA and the release manifest hash in `docs/START_HERE.md`.
- [ ] Close or update issue #1 (the 2026-11-01 handoff), which is out of date.
- [ ] Submit the form.

## 7. Known stale text left as is

`docs/SYSTEM_V1.md` is frozen by its own rule ("a later change is a new version"). Its sections 0 and 5 still
say the URL path is uncommitted, HEAD is `90809f2`, and the Android app is not built. All of that is out of
date: the URL path was committed in `2b3c1bd`, `main` is at `6d9b60b`, and the app builds. Its line references
into `edge_decision.py` and `rakshak_edge_service.py` have also moved.

## 8. How this was checked

Six parallel checks on clean worktrees of `main`, plus spot checks of each key finding: the organisers'
page and searches for other organiser sources; the Android manifest, Gradle files and the Llama call path; the
Pi runner, routes, bind address and outbound connections (the only non-loopback connect is the startup
egress probe to `1.1.1.1:443`, which is meant to fail inside `unshare -rn`); a rules-only re-run of the NIDS
holdout; a fresh-clone run of every README step; and a claim-by-claim check of the judge-facing documents
against code and committed results.
