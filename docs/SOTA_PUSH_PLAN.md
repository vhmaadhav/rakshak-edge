# SOTA push before Phase 1 (2026-11-12 to 2026-11-23)

Maadhav, 2026-11-12: "this is not enough, we have to make this as SOTA as possible". Approved by Claude under his
delegation of plan decisions (2026-11-11); he has not reviewed it line by line. Each workstream gets its own
pre-registration (gates and threshold rule written before any test scoring), as before. Kaggle only, no Modal.

## Where we are against the published targets

| Task | Published reference | Ours now | Gap |
|---|---|---|---|
| Indian legit SMS FPR (GroundLM 1 294 negatives) | XLM-R 270M 0.39 % **in-distribution** (random split); zero-shot APIs 12.1-38.3 %; its own English-only control 21.9 % | student 41.6 % (cross-source); system quiet 0.18 % visible (but SmishX visible recall only 23.5 %) | detector is far behind |
| Same, cross-source | none published | **mmBERT-base teacher fold (no GroundLM/SmishX in training): 1-6 / 1 133 at 88.8-91.5 % SmishX label-1 kept** (`docs/TEACHER_XSOURCE_RESULTS.md`) | not deployed, no proper threshold, recall on GroundLM positives not scored |
| URL-only phishing (PhreshPhish) | no URL-only row; URL+HTML rows only | AP 0.484-0.941 across base rates; **69 % FPR out of domain** | generalisation |
| NIDS cross-dataset | XGBoost UNSW→CIC F1 0.81 (arXiv 2609.13511) | model 0.24; rules-first 11/11 on synthetic holdout only | no real-traffic evaluation |
| On-device scam explainer | none publish metrics | Pi: p50 0.96 s, 3.1 J, verifier-gated | first of its kind; keep |

The single biggest lever is the evidence we already have: **a pretrained multilingual encoder, trained without any
Indian legit data, almost separates GroundLM's legit Indian SMS from real scams** (AUC 0.995). A deployable version
of that, with an honest threshold, would be the first cross-source on-device result at about 1 % FPR, against
zero-shot GPT-4o's 14.6 % on the same 1 294 negatives.

## Workstreams, ranked

**W1. Encoder detector on the device (SMS). Highest priority.**
- W1a (scoring only): score the existing teacher fold on the full suite: GroundLM 1 133 negatives and its policy
  positives (paired with GroundLM XLM-R's released predictions), SmishX smishing 226 and ham 606, campaign test
  (clean rows), lang_shift. Threshold from the fold's own calibration split only. Arms: teacher alone, student AND
  teacher, student OR teacher. Row-level output.
- W1b (train): the same fold recipe on `jhu-clsp/mmBERT-small` (140 M), 3 seeds, so a Pi-sized encoder exists.
  Also re-run mmBERT-base seeds 1-2 for the variance the single checkpoint lacks.
- W1c (export): ONNX with a quantisation that keeps parity (per-channel weight-only INT8 on the encoder
  MatMuls, embeddings and head kept in fp32/fp16; the dynamic INT8 export failed at argmax 0.906). Gate:
  argmax agreement ≥ 0.995 and max |Δp| ≤ 0.02 against PyTorch on 2 000 rows.
- W1d (Pi): memory and latency with `deploy/pi/bench/encoder_bench.py` under `thermal_run.py --kill 65`, beside
  the Llama explainer in the same 4 GB. Gate: RSS fits with the explainer, p95 ≤ 400 ms per SMS.
- Ships if W1a passes its pre-registered gates and W1c/W1d pass. Otherwise the report carries W1a as a measured
  cross-source result and v0 stays.

**W2. URL generalisation.** Retrain the URL student on PhreshPhish train plus the `url_domain` train split
(grouped by registered domain), and score PhreshPhish test and `url_domain` test at a threshold from a held-out
cal split drawn from both. Gate: `url_domain` test FPR ≤ 5 % at PhreshPhish-test FPR ≤ 1.5 %. If it passes,
the U2 check (GroundLM visible warnings) is re-run for quiet mode under a new pre-registration.

**W3. NIDS on real labelled traffic.** Run the shipped rules-first pipeline on real labelled pcaps (CIC-IDS2017
or similar public captures with attack timestamps, e.g. PortScan, SSH-Patator, FTP-Patator, DoS days), on
Kaggle CPU. Report per-family detection and benign alert rates per host-window. The rules are not retuned;
any miss is reported.

**W4 (only if time remains).** KD from the W1 encoder into the byte student, with the GroundLM-free training data,
judged by the same gates.

## Order and compute

W1a and W3 research start now. W1b/W1c start in parallel on Kaggle T4. W2 follows after W1b frees a GPU slot
(Kaggle allows two GPU sessions). Every result goes to its own `docs/*_RESULTS.md` and to `docs/SYSTEM_V1.md`
only if it ships.

## W1d procedure (Pi: encoder latency and memory beside the Llama explainer)

Prepared 2026-11-12, nothing run yet. Pi state at preparation (read only): onnxruntime 1.31.1 and
tokenizers 0.23.3 and numpy 2.5.4 import in `~/rakshak-bench/.venv`; transformers is absent and not
needed; the ORT build lists the `MatMulNBits` contrib op (schema only; its ARM kernel is confirmed
only by a first run). 3.8 GB RAM available of 4.0 GB, 20 GB disk free, SoC 44.7 C, throttled 0x0,
`rakshak-edge` inactive, no rbench or Llama process, `current_job.json` state `done`.

Rows are pre-tokenised on the laptop, so the Pi needs no tokenizer install. Only `encoder_bench.py`
measures; the Llama explainer's peak (1 839.3 MB, `docs/BASELINE_PI.md`) is ADDED as an estimate,
labelled as such in the JSON (`combined`), not measured together.

Steps (`PI=hermoine@192.168.29.39`, `SSH` and `SCP` below use the key and host alias of the Pi):

1. Laptop: `sensors | grep Tctl` (stop at 85 C). Build the rows, 500 seeded cal rows:
   `.venv/bin/python deploy/pi/bench/build_encoder_rows.py --tokenizer <TOKENIZER_DIR> --out <ROWS_NPZ> --rows 500 --seed 0`
   (needs tokenizers and the rakshak package on the laptop; `tokenizer.json` from the W1c export).
2. Laptop: copy the code (no model files, no delete):
   `PI_HOST=$PI PI_HOST_KEY_ALIAS=pi5.local deploy/pi/bench/sync_bench.sh --code-only`
3. Laptop: copy the model and rows, then check hashes:
   `SSH="ssh -i ~/.ssh/pi5 -o IdentitiesOnly=yes -o HostKeyAlias=pi5.local"; SCP="scp -i ~/.ssh/pi5 -o IdentitiesOnly=yes -o HostKeyAlias=pi5.local"`
   `$SSH $PI 'mkdir -p ~/rakshak-bench/models/encoder/<NAME>'`
   `$SCP <ENCODER_ONNX> <ROWS_NPZ> $PI:rakshak-bench/models/encoder/<NAME>/`
   `sha256sum <ENCODER_ONNX> <ROWS_NPZ>; $SSH $PI 'sha256sum ~/rakshak-bench/models/encoder/<NAME>/*'`
   (the two lists must match; the ONNX sha256 is also recorded in the bench JSON.)
4. Pi, read only, before starting: SoC temperature must be below 55 C, i.e. `cat /sys/class/thermal/thermal_zone0/temp`
   gives less than 55000; `vcgencmd get_throttled` gives `0x0`; `systemctl is-active rakshak-edge` says
   `inactive`; `ls ~/rakshak-bench/results/.rbench.lock` finds nothing. Otherwise do not start.
5. Pi, run (the guard pauses at 63 C, resumes at 58 C, kills at 65 C; the bench stops itself at 64 C):
   `cd ~/rakshak-bench && .venv/bin/python deploy/pi/thermal_run.py --name encoder-bench --pause 63 --resume 58 --kill 65 --run-dir ~/rakshak-bench/run -- .venv/bin/python deploy/pi/bench/encoder_bench.py --model models/encoder/<NAME>/<ENCODER_ONNX_NAME> --rows models/encoder/<NAME>/<ROWS_NPZ_NAME> --threads 3 --seq-lens 32 64 128 --runs 200 --warmup 10 --max-temp 64 --companion-peak-mb 1839.3 --out run/encoder-<NAME>.json`
   Run it in the foreground over ssh or under `nohup ... &`; expect about 20 minutes. Read the
   `run/encoder-<NAME>.json` file when it finishes; no polling loop.
6. Pass rule, read from the JSON: `realistic.latency_ms.p95` at most 400 ms (the per-SMS gate);
   `combined.estimated_sum_mb` at most 3 796 MB (4 049 MB total minus the 253 MB idle `used` read on
   the Pi; a derived rule, to confirm before it is used as a gate). A real co-run with the Llama
   explainer is the follow-up if this passes. Discard the run and repeat it if the CSV in
   `run/encoder-bench-*.csv` shows a pause (temperature above 63 C), since a pause distorts latency.
7. Laptop, after the run: copy `run/encoder-<NAME>.json` into `bench_results/encoder/` (gitignored).
   Exit codes: 0 done, 2 refused at start (too hot), 3 stopped early at 64 C; the JSON records which.
