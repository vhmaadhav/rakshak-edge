# URL phishing detector in the edge service (pre-registration)

Written 2026-11-12, before any threshold is computed for this model. Approved by Claude under Maadhav's
delegation of plan decisions (2026-11-11); Maadhav asked for this step ("put the URL detector into the service").
Frozen once the threshold job is launched.

## Model

`best_url_student.onnx`, sha256 `1bbd2c57cb541a7cbc287887ef51976d54240a353655827a511c63ec56213fc7` (charcnn-wide,
PhreshPhish train, seed 0), scaler `url_side_scaler.json` (sha `c6b3fc96…`). Inputs: raw int64[200] and side
float32[29] from `rakshak.url_features.url_features(url)`, URL string only (no HTML, WHOIS or network). It is already
pinned in `deploy/pi/bench/registry.toml` but not loaded by the service. No threshold is registered for it.

## Threshold rule (one number, fixed before any test scoring)

The model's own calibration split is the 5 % grouped carve of PhreshPhish `official.train` (18 087 rows,
`rakshak.eval.phreshphish`, the same seed as `url_student.py`). It is rebuilt exactly and checked against the logged
size. **τ_url = the smallest threshold with FPR ≤ 1 % on that carve's benign rows.** τ_url is written to the job
output before any other set is scored. A message's URL score is the max over the URLs that
`canonicalize(text).urls` extracts, as in `rakshak.fusion.url_expert.FrozenURLExpert`.

## Service behaviour

- `POST /analyze` (SMS): if any URL scores ≥ τ_url, a new evidence item `phishing_url_model` (the URL's
  registered domain, the score) is added. Like rule evidence it is **visible in quiet mode**: at least
  UNCERTAIN, with the remediation "do not open the link; check the sender through the official app or number".
  With the SMS student also over its threshold, HIGH. The explainer gets an English and Hinglish phrase for the
  new type. The verifier treats it like the other URL evidence types.
- `POST /analyze_url` (new): one URL in, URL score plus the domain rules (`brand_domain_mismatch`,
  `shortened_url`, `suspicious_url`), verdict and remediation out. The page gets a "Check a link" box.
- The SMS student, its threshold (0.35128) and the rules are unchanged.

## Gates (all must hold to ship the URL path on by default)

| # | Gate | Current system | Pass |
|---|---|---|---|
| U1 | PhreshPhish `official.test` FPR at τ_url (does the threshold transfer) | – | ≤ 1.5 % (recall reported) |
| U2 | GroundLM scoreable negatives, quiet-mode visible warnings (`eval_groundlm.py`, 1 133; 22 contain a URL) | 2 (0.18 %) | ≤ 4 |
| U3 | SmishX smishing, quiet-mode visible recall (226; 210 contain a URL) | 53 (23.5 %) | ≥ 53 |
| U4 | Pi: service-path URL scores vs laptop on 300 PhreshPhish test URLs | – | bit-identical, 0 flips at τ_url |
| U5 | Pi: added latency of URL scoring per `/analyze` | – | p95 ≤ 50 ms |

Reported, not gated: recall at τ_url on PhreshPhish test and on `url_domain` test (out-of-domain); the PhreshPhish
base-rate APs (already in `docs/BASELINE.md`); SmishX legit quiet FPR; feature skew between the dataset `url` column
and `canonicalize`-extracted URLs on a PhreshPhish test sample; GroundLM URL-model hits per bucket.

If U2 fails, the URL evidence is shown only in verify mode, and the measured result is reported. If U1 fails,
the threshold is reported as not transferring and the path stays off.

## Compute

Threshold and offline metrics: Kaggle CPU job `url_threshold` (data `maadhavvh/rakshak-data`, ONNX and scaler from
kernel `maadhavvh/rakshak-url-student`). U2/U3: `deploy/pi/service/eval_groundlm.py --rescore` on the laptop
(about 25 s, light; temp rule applies). U4/U5: on the Pi under the thermal rules (start < 55 °C, never ≥ 65 °C).

## Amendment 2026-11-12 (night): corroborated URL evidence in quiet mode (pre-registered before computing)

U2 failed by one message (5 vs ≤ 4). New arm, written before it is computed: in quiet mode, `phishing_url_model`
evidence becomes visible only when it is **corroborated**, i.e. the SMS student is over its threshold (0.35128) or any
rule evidence fired on the message. Verify mode is unchanged. Same τ_url (0.6308), same eval
(`eval_groundlm.py run --mode quiet` with URL scores), same gates: U2 GroundLM quiet visible ≤ 4 / 1 133 and U3 SmishX
smishing quiet visible ≥ 53 / 226. Also reported: SmishX ham quiet visible FPR, and campaign rows (unchanged, because
the URL job did not score campaign test). If U2 and U3 pass, quiet mode ships with corroborated URL evidence.
