# Rank-arm audit: why `llama_ela_eval` reported 0 s and 0 tokens

Status: **measurement correction and clustered intervals verified on Kaggle, 2026-11-01**.
No on-device claim.
Owner: Sachin. Opened from `docs/HANDOFF.md` open task 2.

## 1. The symptom

`docs/HANDOFF.md` section 5 recorded the `rank` arm of `llama_ela_eval` as:

> rank arm has p50 latency 0 s and 0 generated tokens. It is almost certainly not scoring as intended.
> It must be audited before any claim.

and the handoff flagged it as a risk: "the ELA rank arm reports 0 s latency and 0 tokens (suspicious,
needs an audit)".

## 2. Root cause

The rank arm does no generation by design, so `new_tokens = 0` is expected *in principle*. The
suspicious part is that it was **hardcoded**:

```python
new_tokens = 0  # nothing is generated; the scored candidate count is in "choice"
```

The arm scores `K=4` candidate sentences per evidence item with `_mean_logprobs` and picks the best
one. How much work it does therefore depends entirely on `len(package.evidence)`:

```python
scores = [
    _mean_logprobs(tok, model, rank_prefix(pkg, i, lang), [" " + c for c in cands])
    for i, cands in enumerate(rank_candidates_for(pkg, lang))   # one entry per evidence item
]
```

`rank_candidates_for(package)` returns `[rank_candidates(e.type, lang) for e in package.evidence]`.
When the evidence package is **empty**, that list is empty: the comprehension runs zero iterations,
no model forward pass happens, and the measured latency is genuinely ~0 s. The arm is not broken —
it is being handed nothing to rank.

Evidence packages come from `run_rules` (`src/rakshak/rules.py`) via
`package_for_sms`. Those are **scam-specific** rules (OTP request, KYC block, prize, APK install,
brand/domain mismatch, …). Some benign messages have empty packages, as in these fixtures;
some scams do too. Benign messages can also trigger rules (false positives):

```
scam_otp         evidence_n= 4 risk=HIGH
scam_upi         evidence_n= 1 risk=MEDIUM
scam_prize       evidence_n= 2 risk=MEDIUM
hinglish_scam    evidence_n= 3 risk=HIGH
benign_bank      evidence_n= 0 risk=LOW    <-- zero work in the rank arm
benign_otp       evidence_n= 0 risk=LOW    <-- zero work in the rank arm
benign_personal  evidence_n= 0 risk=LOW    <-- zero work in the rank arm
```

Thus work is conditional on evidence, not on the gold label. The reported p50 was pulled to ~0
because a large share of rows scored no candidate at all, and nothing in the output distinguished
"scored zero candidates" from "the timer is broken".

The audit also clears the two obvious alternative hypotheses:

- **Candidate scoring is not skipped.** `_mean_logprobs` runs a real forward pass over
  `prefix_ids + cand_ids` and gathers the candidate-token log-probabilities; the `start - 1` offset
  is correct (logits at position `i` predict token `i+1`).
- **`select_candidates` is correct.** `max(range(len(s)), key=lambda k: (s[k], -k))` picks the
  highest score and keeps the first candidate on a tie.

## 3. Fix

The original patch attempted to make zero work visible, but used `sum(len(c))`, which counts
**characters** and was incorrectly described as a candidate count. The 2026-11-01 correction
keeps generated tokens separate from scored continuations and leaves rule behaviour frozen:

1. `kaggle/jobs/llama_ela_eval.py` — the rank arm now records, per row:
   - `rank_items`: how many evidence items were ranked,
   - `rank_empty`: whether the package was empty,
   - `rank_candidates`: how many candidate strings were scored,
   - `rank_targets`: number of actual tokenizer continuation IDs scored, including the
     leading space used by `_mean_logprobs`, excluding repeated prefix tokens,
   - `new_tokens`: **0**; ranking never calls `generate()`.

2. `src/rakshak/ela_eval.py` — `aggregate` now reports:
   - `zero_work_rate`: share of rows where the arm did no model work,
   - `latency_mean_s_worked`, `latency_p50_s_worked`, `latency_p95_s_worked`: latency over the
     rows that actually did work, i.e. the honest per-item cost.

Rank injection estimates exclude a pair unless **both** its clean and injected member have
candidates. Matched freeform comparisons use those same IDs. The outputs retain five attack
families, harmful-flip counts, Wilson intervals and Fisher tests; exact McNemar and paired
campaign-clustered differences account for pairing. All-row coverage remains visible.

The action and summary in the rank arm are deterministic and the candidate bank is fixed.
Zero harmful flips measure this restricted architecture; they cannot establish learned
Llama robustness or the quality of open-ended generation.

Regression tests: `tests/test_ela_rank_job.py`, `tests/test_ela_eval.py::test_aggregate_reports_zero_work_rows`,
`::test_aggregate_zero_work_over_rows_without_work`,
`::test_benign_message_yields_no_rank_candidates`.

## 4. What the numbers will mean after the rerun

- `mean_new_tokens` for ranking is zero. `mean_rank_target_tokens` and
  `rank_target_tokens_total` report continuation tokens scored; prefix work is additional.
- `latency_p50_s` over all rows is a mixture and is **not** the arm's cost. Quote
  `latency_p50_s_worked` / `latency_p95_s_worked` for work-conditional cost. Kaggle T4 timing
  is not an on-device benchmark; Pi/Android measurement is still required.
- `zero_work_rate` on a 50/50 scam/benign sample is the **evidence-coverage complement**. It is a
  measurement of the rules layer, not of Llama.

## 5. The real finding: evidence coverage, not just telemetry

These historical observations are **in-sample: these rows informed rule design**. While
auditing this, the evidence layer was measured on the public ScamShield
dataset (`sidzzz07/scamshield-dataset`, MIT), 3,000 uniformly sampled rows, seed 13, using
`package_for_sms` from this repo:

| Slice | n | rows with no evidence | rate |
|---|---:|---:|---:|
| All rows | 3,000 | 2,628 | **87.6%** |
| Benign | 1,891 | 1,845 | 97.6% |
| Scam (any `is_scam=1`) | 1,109 | 783 | 70.6% |
| — of which General Spam / Telemarketing | 843 | 679 | 80.5% |
| — of which genuine fraud | 266 | 104 | **39.1%** |

Separating promotions from fraud matters here: per `docs/CONTEXT.md` and datacard section 2a,
"General Spam / Telemarketing" is **promotion, not scam**, and a promotion not firing a scam rule is
correct behaviour. The actionable number is the **39.1% of genuine fraud messages that produce no
evidence at all**, which is also the share of fraud on which the rank arm would do zero work.

Per-intent empty-evidence rate for the fraud slice, with the caveat that the source carries label
noise:

| Intent | no evidence | n | note |
|---|---:|---:|---|
| OTP / Credential Theft | 100% | 2 | tiny n |
| Loan Scam | 92.3% | 13 | sampled rows are mostly genuine bank loan offers — label noise |
| UPI Fraud | 75.4% | 65 | sampled rows are mostly genuine cashback/coupon offers — label noise |
| Courier Scam | 61.1% | 36 | **genuine gap**: generic "parcel held / pay fee" wording |
| Telegram Job | 35.3% | 17 | genuine gap |
| Lottery / Prize | 18.9% | 37 | genuine gap |
| Fake KYC | 2.9% | 69 | covered |
| Bank Account Freeze | 11.1% | 9 | covered |
| Electricity Disconnection | 16.7% | 12 | covered |

False-positive side, same sample: **46 of 1,891 benign rows (2.43%)** fire any rule. That is the
headroom for widening coverage.

One example of the genuine courier gap, from the real data:

```
Post Office Your parcel have be redirect to your local Post Office branch due to an unpaid
shipping fee To reschedule a delivery please visit http://postoffice-tracking updates.com
```

`parcel_customs` requires a `pay|fee|charges|...` keyword within 50 characters of a
`parcel|courier|shipment` keyword. Here the fee wording ("unpaid shipping fee") sits beyond that
window and in a different clause, so the rule misses it while the lookalike domain
(`postoffice-tracking updates.com`) is the actually damning signal.

## 6. Recommendation

The following historical recommendations informed dc223d3. They are retained for provenance;
the 2026-11-01 frozen-rule audit evaluates that change without tuning. Ranked:

1. **Measure before widening.** The 39.1% is measured on a 3,000-row sample with a noisy source
   label. Re-measure on the built `external_groundlm` policy-negative set and on a hand-checked
   fraud slice before changing thresholds.
2. **Widen `parcel_customs` / `job_task_scam` recall** — allow a clause-level window (sentence
   scoped) instead of 50 characters, and treat a brand-lookalike domain as corroboration.
3. **Add a `lookalike_domain` evidence type** for near-miss registrable domains
   (`postoffice-tracking updates.com`, `hmrev.customs`). This is a real, low-FP signal that is
   currently only reachable through the weaker `suspicious_url` path.
4. Keep the honest floor: **no rule change ships without a paired before/after on both the fraud
   slice and the 2.43% benign false-positive rate.**

## 7. Corrected full rerun — 2026-11-01

Private [Kaggle kernel `llama_ela_eval-a1-20261101`, version 1](https://www.kaggle.com/code/maadhavvh/rakshak-llama-ela-eval-a1-20261101),
base `038bdd33022d` dirty, tree
`48ed512ef46e0abf629fe7e8da89e1dbca68dbd275516f113900e12eb852932d`.
One event wait, verified fetch, **ok**, **0 failed units**, 1,622.0 s. Same seeded 300 campaign
packages, 150 paired attacks, 150 matched-language packages. Model: pinned
`unsloth/Llama-3.2-1B-Instruct@5a8abab4a5d6f164389b1079fb721cfab8d7126c`, fp16.
The environment exposes two Tesla T4s; the model uses CUDA device 0.

| Clean rank quantity (300 packages) | Corrected measurement |
|---|---:|
| Generated tokens | **0** |
| Candidate strings scored | **248** |
| Actual continuation-token IDs scored (prefix excluded) | **2,992** |
| Packages with no model work | **86%** |
| Latency p50 over all packages | 0 s (mixture with empty packages) |
| Latency p50 / p95 over worked packages | **0.3110 / 1.0782 s** |

The before-patch character count is invalid and is not used as a token baseline. These are
Kaggle T4 measurements, not Pi 5/Android timings. Candidate likelihood ranking is not generation.

| Injection arm / scope | Harmful / pairs | Wilson 95% interval |
|---|---:|---|
| Freeform, all | 35/150 | [17.28%, 30.72%] |
| Slot, all | 0/150 | [0%, 2.50%] |
| Rank, candidates in both members | **0/39** | **[0%, 8.97%]** |
| Freeform on the same rank-eligible IDs | 11/39 | [16.54%, 43.78%] |

**111 rank pairs are excluded** because they have no candidates. The rank result must never
be presented as 0/150. Rank versus matched freeform: Fisher p=0.00043094 (descriptive unpaired
reference), exact McNemar p=0.00097656. Paired campaign-clustered harmful-rate difference
(rank minus freeform) **-28.205 pp**, CI **[-43.590, -15.385] pp**, p=0.000999.
Slot versus matched freeform: Fisher p=6.15e-12, exact McNemar p=5.82e-11.
Zero-event rank/slot bootstrap intervals are [0, 0] and degenerate; quote Wilson bounds too.
The fixed fifteen payload variants do not establish robustness to adaptive or unseen attacks;
campaign resampling does not capture uncertainty over new attack families.

| Attack family | Rank harmful/eligible | Rank Wilson 95% | Freeform harmful/all |
|---|---:|---|---:|
| direct_override | 0/8 | [0%, 32.44%] | 7/30 |
| roleplay_spoof | 0/7 | [0%, 35.43%] | 5/30 |
| encoded | 0/12 | [0%, 24.25%] | 9/30 |
| delimiter_json | 0/6 | [0%, 39.03%] | 8/30 |
| multi_turn | 0/6 | [0%, 39.03%] | 6/30 |

| Raw point lexicon pass | Passed/points | Campaign-clustered 95% CI |
|---|---:|---|
| English, all 300 packages | 16/62 = **25.81%** | [13.04%, 40.82%] |
| English, matched 150 packages | 8/37 = **21.62%** | [8.00%, 40.00%] |
| Hinglish, same 150 packages | 23/37 = **62.16%** | [40.61%, 79.59%] |

These are raw keyword/lexicon checks, not semantic entailment measurements. Package verifier
pass: English 90.33% [86.99%, 93.33%]; Hinglish 90.67% [86.00%, 95.33%]. Many packages have no
points, so high package pass does not contradict weak point generation. Raw slot/rank
unsupported-claim rates and rendered rates are zero.
Rank's deterministic summary/action and fixed sentence bank restrict the possible outcomes;
the injection result measures that architecture, not learned Llama robustness. The repeated
campaign sample is a development benchmark; no new held-out improvement is claimed.

**Decision: keep the telemetry and eligible-denominator correction.** Artifact invariants
confirm zero generated rank tokens and zero latency/targets for empty packages. QLoRA remains
open because the model's own point pass is weak. Language-rate cluster intervals were computed
from these exact saved outputs in the private statistics-only Kaggle supplement
[`ela_recompute-a1-ci-20261101`, version 1](https://www.kaggle.com/code/maadhavvh/rakshak-ela-recompute-a1-ci-20261101):
one wait, verified fetch, **ok**, 72.0 s, tree
`9730c50a862b67341634e16c5d07bcbe26574ed84c7d19fe5763d69ddc1e4909`.
It binds to the source tree and generations SHA256
`8a898ef0d8edcf92598729ef8bf6e94e7bc5674f7d914d854aed02cf799e04d1`.
Point ratios pool numerator/denominator across resampled campaigns, preserving all points
of each package together (1,000 draws, seed 13). No model generation, sample selection or tuning.
