# Rank-arm audit: why `llama_ela_eval` reported 0 s and 0 tokens

Status: **closed** (code-level root cause found and fixed; the GPU rerun is still needed for the numbers).
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
brand/domain mismatch, …). A benign message hits none of them, so its package is empty:

```
scam_otp         evidence_n= 4 risk=HIGH
scam_upi         evidence_n= 1 risk=MEDIUM
scam_prize       evidence_n= 2 risk=MEDIUM
hinglish_scam    evidence_n= 3 risk=HIGH
benign_bank      evidence_n= 0 risk=LOW    <-- zero work in the rank arm
benign_otp       evidence_n= 0 risk=LOW    <-- zero work in the rank arm
benign_personal  evidence_n= 0 risk=LOW    <-- zero work in the rank arm
```

So the rank arm did work only on the scam half of the sample. The reported p50 was pulled to ~0
because a large share of rows scored no candidate at all, and nothing in the output distinguished
"scored zero candidates" from "the timer is broken".

The audit also clears the two obvious alternative hypotheses:

- **Candidate scoring is not skipped.** `_mean_logprobs` runs a real forward pass over
  `prefix_ids + cand_ids` and gathers the candidate-token log-probabilities; the `start - 1` offset
  is correct (logits at position `i` predict token `i+1`).
- **`select_candidates` is correct.** `max(range(len(s)), key=lambda k: (s[k], -k))` picks the
  highest score and keeps the first candidate on a tie.

## 3. Fix

Two changes, both about **making zero work visible** rather than changing detection behaviour:

1. `kaggle/jobs/llama_ela_eval.py` — the rank arm now records, per row:
   - `rank_items`: how many evidence items were ranked,
   - `rank_empty`: whether the package was empty,
   - `rank_targets`: how many candidate strings were scored,
   and `new_tokens` is the real scored-candidate count instead of the constant `0`.

2. `src/rakshak/ela_eval.py` — `aggregate` now reports:
   - `zero_work_rate`: share of rows where the arm did no model work,
   - `latency_mean_s_worked`, `latency_p50_s_worked`, `latency_p95_s_worked`: latency over the
     rows that actually did work, i.e. the honest per-item cost.

Regression tests: `tests/test_ela_eval.py::test_aggregate_reports_zero_work_rows`,
`::test_aggregate_zero_work_over_rows_without_work`,
`::test_benign_message_yields_no_rank_candidates`.

## 4. What the numbers will mean after the rerun

- `mean_new_tokens` for the rank arm is now a **scored-candidate** count, not a generation count.
  Do not compare it to the slot or freeform arms' generated-token counts.
- `latency_p50_s` over all rows is a mixture and is **not** the arm's cost. Quote
  `latency_p50_s_worked` / `latency_p95_s_worked` for on-device claims.
- `zero_work_rate` on a 50/50 scam/benign sample is the **evidence-coverage complement**. It is a
  measurement of the rules layer, not of Llama.

## 5. The real finding: evidence coverage, not just telemetry

While auditing this, the evidence layer was measured on **real** data — the public ScamShield
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

Telemetry is fixed; the coverage work is a separate, larger change and is **not** done here. Ranked:

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
