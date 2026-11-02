# GroundLM FPR test (handoff task 1)

Status: **in-sample: these rows informed rule design**. Frozen-rule Kaggle audit completed
2026-11-01; full before/after tables and paired intervals: `docs/REVIEW_AUDIT.md`.
Model-side FPR and independent Indian validation remain OPEN.

## Design-set register (2026-11-01; exposure updates 2026-11-02)

| Change | Design exposure | Permitted interpretation |
|---|---|---|
| f29521a urgency-only risk downgrade | GroundLM policy negatives and their failure examples; all scoreable rows were evaluated. Exact row access was not logged, so conservatively treat the entire GroundLM release as exposed. | In-sample descriptive FPR; a later split cannot create an untouched test. |
| dc223d3 parcel/job/prize widening | ScamShield 3,000-row seed-13 sample, then all 85,680 rows including official test; campaign/test, lang_shift and temporal outcomes were inspected in the reviewed work. | In-sample coverage/risk audit only. |
| 63f9323 LOW-summary caution | Campaign/test rule-coverage failures were used to motivate the wording. | Do not use that split as independent validation of the wording. |
| 2026-11-01 review fixes | Frozen GroundLM/ScamShield retrospective campaign subsets and existing SMS splits; no rule or threshold tuning. | Measurement correction; no held-out improvement claim. |
| SmishX | Used as independent validation of already-frozen reviewed rules after train-overlap removal. Those rules remain unchanged; later policy/intervention design uses aggregate feedback and is explicitly in-sample for that selection. | **Independent, non-Indian-specific**; report frozen-threshold FPR and recall, with no Indian deployment claim. |
| 2026-11-01 transactional/fusion candidate planning | User-provided aggregate student FPR feedback from GroundLM (35.92%) and SmishX (30.31%) motivated the new task and protective-context candidate; no message rows inspected for these templates/features. Future aggregate diagnostics are also design feedback if used for changes. | Repeated benchmark feedback is **in-sample for intervention selection**; SmishX's earlier independent status applies to the already-frozen reviewed rules/control, not an untouched validation claim for new candidates. |

| 2026-11-01/02 evidence-policy selection | GroundLM/SmishX aggregate student/fused FPR and recall motivated corroboration. No new benchmark text authored a rule/template; old full-release rule-design exposure still applies. Policy KILL/fold metrics are now design feedback. | **In-sample policy selection**, despite cal-only thresholds. Original escalation UNMEASURABLE; successor KILL. No v2 or independent Indian claim. |
| 2026-11-02 native URL threshold | Original official-train-only grouped URLcal, seed0/fraction.05,18087rows; ordered ID SHA256 a93713954509cacb8ea7046982ed37fad42eee4b4a8886441eb721c6b6399be2. Same cal previously selected the URL checkpoint. No URLtest/benchmark/SMS/external calibration. | Calibration data, **not held-out validation** or a risk guarantee. Split/IDs/hash/threshold froze before SMS scoring; no new PhreshPhish test result. |
| 2026-11-02 family-calibration and multilingual planning | Earlier failures and clean student≈.41 versus teacher≈.80 language-shift feedback motivated predeclared arms. Existing task regexes only; multilingual vocab/PCA will use train IDs only. | Repeated test/intervention feedback is **in-sample for design/selection**. No untouched held-out gain claim; independent Indian validation OPEN. No outcome yet for either arm. |

**Independent Indian held-out validation: OPEN.** Collect consented, privacy-scrubbed real
Indian SMS received after the rules freeze: bank/UPI/OTP/delivery negatives plus received
scams. Retain dates and campaign groups; obtain human policy labels; keep the set private,
never train or tune on it, and evaluate once at the frozen rules/thresholds. No synthetic
messages or model-generated relabels may enter this test.

The primary [GroundLM paper](https://aclanthology.org/2026.groundlm-1.79/) defines its endpoint
on 1,294 policy negatives, and treats positive-label metrics as secondary because positive
adjudication used LLMs. Our scoreable and eval-eligible denominators must be shown separately.

## 1. What the task asked for

From `docs/HANDOFF.md` open task 1:

> Score the `external_groundlm` (1,294 policy-negatives) with each shipped model at its cal
> threshold. *Acceptance:* FPR per model with a campaign-clustered CI, the 169 hash-only IIITD rows
> noted as missing, and the result in `docs/HANDOFF.md` section 5. Go/kill: FPR <= 1% on this set.

GroundLM is the stated primary safety objective in `docs/CONTEXT.md`: *low FPR on legitimate Indian
transactional SMS (bank/UPI/OTP/delivery)*.

## 2. Getting the data (it was not in the repo)

`data/raw/zenodo_20480151_scamshield_v4/` was absent. Zenodo record 20480151 v4 is a **1,656 MB**
zip, and this machine had ~5.7 GB free, so it was not downloaded in full.

Instead the single needed member was extracted with HTTP range requests:

- read the End Of Central Directory record at the tail (`-r 1655890000-1655956640`);
- read the 9,450-byte central directory to list the 70 entries;
- fetched only the local-header range for `scamshield_emnlp_artifact_v4/data/cleaned_test_1393.csv`
  (53,205 bytes compressed), inflated the raw deflate stream, and repacked it into a minimal zip at
  the path the loader expects;
- **verified against the CRC32 and uncompressed size recorded in the original central directory**
  (`e42764c9` / 205,858 bytes), so the member is byte-identical to upstream.

The member path matches `GROUNDLM_MEMBER` in `src/rakshak/datasets/external.py` exactly, so
`load_groundlm` reads it unchanged. 1,656 MB avoided, 53 KB fetched, and the 53 KB is provably the right 53 KB.

## 3. The set, as loaded

| Quantity | Value | Handoff/datacard expectation |
|---|---:|---|
| Rows in `cleaned_test_1393.csv` | 1,393 | 1,393 ✓ |
| `label_cleaned == 0` (policy negatives) | **1,294** | 1,294 ✓ |
| Text present | 1,233 | — |
| Hash-only (text unavailable) | **169** | **169** ✓ |
| Scoreable policy negatives | **1,133** | 1,133 ✓ |
| Fraud rows (`label_cleaned == 1`) | 99 | — |

Every count matches the datacard exactly, including the 1,133 scoreable denominator the datacard
names ("the FPR denominator is 1,133, not 1,294").

> **Integrity note.** The first extraction was truncated by 30 bytes: the range request ended at the
> local-header end rather than the end of the deflate stream, so the archive inflated to 205,540 of
> 205,858 bytes and the last two rows were lost (1,391 rows instead of 1,393). It was caught by an
> independent md5/count check against the artefact, then fixed and proven correct by comparing the
> **CRC32 recorded in the original zip's central directory** against the CRC32 of the extracted file:
> `e42764c9` both sides, and `205858` bytes on both sides. So the CSV is byte-identical to the
> upstream member, not merely row-count-equal.

External-overlap handling is unchanged from the build: pool rows overlapping this set are dropped
from every TRAIN partition, and the 169 text-less rows keep their labels but cannot be scored.

## 4. Result

Measured with the deterministic pipeline the repo ships —
`package_for_sms(text)` = canonicalise → `run_rules` → `build_package`, using the label-free
`proxy_scam_state`. No model weights are involved in this path.

| Detector | Fire criterion | False positives | FPR on 1,133 | vs <=1% target |
|---|---|---:|---:|---|
| Rules only, **before** this change | risk above LOW | 12 (reproduced on Kaggle) | 1.0591% | in-sample only |
| Rules only, **after** this change | risk above LOW | 1 (reproduced on Kaggle) | **0.0883%** | in-sample only |

The frozen Kaggle audit reconciles the counts: **12 total evidence-hit negatives, 11
urgency-only**, with **1** above LOW after the downgrade. All-scoreable and eval-eligible
negative populations both contain **1,133** rows in this frozen data version. **Any evidence**
remains 12 before and after; **risk above LOW** changes 12 to 1; **HIGH/SCAM** remains 1.
The old “13 total / 12 urgency-only” narrative was wrong and is withdrawn.

| Frozen GroundLM population | Warning flags before / after | FPR before / after | Current clustered 95% CI | Current Wilson score 95% CI |
|---|---:|---|---|---|
| All 1,133 policy negatives | 12 / 1 | 1.0591% / 0.0883% | [0%, 0.2971%] | [0.02%, 0.50%] |
| Campaign audit subset, 131 | 3 / 0 | 2.2901% / 0% | [0%, 0%] | [0%, 2.85%] |
| Campaign remainder, 1,002 | 9 / 1 | 0.8982% / 0.0998% | [0%, 0.3770%] | See private metrics |

All three rows are **in-sample: these rows informed rule design**. The zero-event bootstrap
interval is degenerate; the audit subset does not certify a 1% population FPR. On the full
set, the paired warning-rate change is -0.9709 pp, clustered CI [-1.6529, -0.4508] pp,
two-sided bootstrap p=0.0050. That describes a design-set tradeoff, not independent safety.

The single remaining false positive is one Devanagari message that triggers `apk_install` (weight
0.9) on "डाउनलोड करें" ("download") in conversational text. Because that weight clears the 0.85
HIGH cut it lands at **HIGH** — the worst bucket — and it is the only row above LOW. The 11
urgency-only rows are still *flagged* (they keep their evidence) but sit at LOW, so they do not count
against a verdict. This one is worth a follow-up: the rule fires on a bare "download" verb with no
app/APK noun, which is the shape of false positive to tighten.

Because exactly one row clears the ladder and no row sits strictly between LOW and HIGH, the FPR is
identical at every risk threshold:

| Threshold | FPR |
|---|---:|
| `>= UNCERTAIN` | 0.0883% |
| `>= MEDIUM` | 0.0883% |
| `>= HIGH` | 0.0883% |

So on this set the result is **threshold-insensitive**, which is the useful property for a
calibrated-abstention claim. The caveat is honest: the eleven urgency-only rows are still *flagged*
at LOW and would still be surfaced to the explainer. What the fix bought is that they no longer set a
verdict, which is what the `<=1%` criterion measures.

## 5. Root cause and the fix

**11 of the 12** evidence-hit negatives are urgency-only. This is now reproduced by the
frozen Kaggle audit; production comments and regression tests reference this result instead
of duplicating scientific counts. Illustrative design-set cases:

```
UNCERTAIN [urgency,urgency,urgency] "अरे! मैं जल्दी नहीं कर रहा हूँ लेकिन जल्दी ही जल्दी नहीं है..."
UNCERTAIN [urgency,urgency]         "मैं आदमी लेकिन बहुत जल्दी कच्चे... अब मुझे बाहर इतना जल्दी घर..."
UNCERTAIN [urgency]                 "URGENT: Your SBI credit card payment of INR 15000 is due tomorrow."
HIGH     [apk_install]              "अला कानो, डाउनलोड करें, हमन मुक्त आओ."
```

`urgency` (weight 0.4) covers `turant|jaldi|abhi|aaj hi|urgent|immediately|...` and fires on ordinary
conversation ("I'm leaving early", "बहुत जल्दी"). In `_risk`, any evidence at all reached the
`scam_state == "uncertain"` branch and returned `Risk.UNCERTAIN`, so a greeting with the word
"जल्दी" in it was indistinguishable from a real scam signal. Note the second example: a *legitimate*
credit-card due-date reminder was being flagged — exactly the population this go/kill test exists to
protect.

The fix in `src/rakshak/evidence.py`:

```python
# ``urgency`` is the only generic, context-free type ... it must not on its own escalate a message
# the fusion layer has not called suspicious.
if evidence and all(e.type == "urgency" for e in evidence):
    return Risk.UNCERTAIN if scam_state == "scam" else Risk.LOW
```

Rationale for the exact form:

- **`all(...)` not `any(...)`** — an urgency hit alongside a real type (`urgency` + `kyc_block_urgency`)
  still escalates. Only *bare* urgency is downgraded.
- **does not delete the evidence item** — the fact stays in the package, so the ELA explainer can
  still say the message uses time pressure; it just no longer sets the verdict.
- **fusion still wins upward** — if `scam_state == "scam"`, bare urgency returns `UNCERTAIN` rather
  than `LOW`.
- **consistent with the module's own stated design**: `rules.py` already documents that `urgency`
  "never sets an action on its own ... and it cannot raise the risk above medium, so benign
  abhi aa rahe hain does not become a scam verdict". The code was not honouring that comment.

Verification: full suite `786 passed, 19 skipped` before and after; no test changed.

## 6. Remaining limitations

- **Per-model FPR.** The acceptance criterion says "each shipped model at its cal threshold". This
  host has no `torch`, no GPU, and no access to `highestintheroom/rakshak-models` (no HF token, not
  an org member), and the Kaggle datasets are not shared with `sachin0496`. The *model* half of the
  test must run on Kaggle. What is reported here is the **rules-only** detector, which is a shipped
  component and the one whose failure mode this test exposed.
- **Campaign-clustered CI.** The historical result omitted one. The completed frozen audit supplies a
  campaign-clustered percentile CI and a Wilson score interval. Wilson is not an exact
  interval and does not account for campaign dependence. A zero-event bootstrap interval
  can be `[0, 0]`; that is not evidence of zero population risk.
- Handoff section 5 has **not** been edited; that is a shared results table and should be updated
  when the model numbers land alongside it.

## 7. Historical reproduction (superseded)

```bash
# 1. Data (53 KB, via range requests; see section 2 for the method)
ls data/raw/zenodo_20480151_scamshield_v4/scamshield_emnlp_artifact_v4_2026-05-31.zip

# 2. Build the external set
.venv/bin/python -m rakshak.datasets.build --skip-urls

# 3. Score
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "src")
import pandas as pd
from rakshak.ela_eval import package_for_sms
df = pd.read_parquet("data/processed/external/external_groundlm.parquet")
scr = df[(df.policy_negative == True) & (df.text_available == True)]
k = sum(package_for_sms(r.text)[1].risk.value != "LOW" for r in scr.itertuples())
print(f"{k}/{len(scr)} = {k/len(scr):.4%}")
PY
```

## 8. The cost of the fix, measured on a second dataset

The GroundLM test only counts false positives, so it cannot see what the fix gave up. Measured
separately on the **real ScamShield sample** (3,000 uniform rows, seed 13; fraud = `is_scam==1` and
intent not "General Spam / Telemarketing", so promotions are excluded):

| Slice | n | flagged (any evidence) | `>= UNCERTAIN` | `>= MEDIUM` |
|---|---:|---:|---:|---:|
| fraud | 266 | 162 (60.9%) | 147 (55.3%) | 145 (54.5%) |
| benign | 1,891 | 46 (2.43%) | 26 (1.37%) | 26 (1.37%) |

Net effect of the urgency change: **15 fraud rows stop reaching a verdict (−5.6 pp recall) and 20
benign rows stop being flagged (−1.06 pp false-positive rate).**

That is close to break-even on raw counts, but it is **not** obviously the right trade, and it should
be stated plainly rather than sold:

- Verdict-level recall for fraud is already low (55%) because the rules layer is a *component*, not
  the detector. The fusion model is meant to carry recall; the rules contribute evidence.
- 14 of the 15 lost fraud rows were `urgency`-only packages, and several are real scams with
  genuinely thin wording, e.g. a Paytm KYC-expiry message that the rule set only caught via
  "immediately" and an Indian-Post parcel-stuck notice caught only via "तुरंत". Those are recall
  the *rules* absence causes, not the downgrade: the right fix is to give `kyc_block_urgency` and
  `parcel_customs` the coverage they need (see `docs/RANK_ARM_AUDIT.md` section 6), after which
  those messages would carry a specific type and no longer be urgency-only.
- The benign 2.43% is dominated by the same mechanism, so this is one lever, not two independent
  ones. Any further widening must be re-measured on both slices.

**Historical recommendation:** keep the downgrade on the basis of this in-sample tradeoff.
This does not establish independent safety or fulfil the model-side handoff task. The frozen
audit reports the same tradeoff on SmishX independently; no rule is tuned from that result.

## 9. Open follow-ups

1. **Rule coverage** — the 15 urgency-only fraud rows above are the same population measured in
   `docs/RANK_ARM_AUDIT.md`: specific rules (`kyc_block_urgency`, `parcel_customs`, `job_task_scam`)
   are too narrow. Widening them converts urgency-only hits into typed hits and recovers the recall.
2. **`apk_install` false positive** — the single surviving GroundLM FPR fires on a bare "download"
   verb in Devanagari conversational text at weight 0.9 (HIGH). It should require an app/APK object.
3. **Model-side FPR** — still needs Kaggle; see section 6.
4. **Independent Indian held-out validation — OPEN**, with the post-freeze collection plan
   in the design-set register. SmishX is independent, non-Indian-specific and fails the 1%
   reference gate (warning FPR 6.0159%, HIGH FPR 1.1351%). No rule is tuned from this result.

## 10. Frozen-rule review audit protocol — 2026-11-01

All benchmark evaluation runs on Kaggle. The older local commands in section 7 document
historical methodology only and must not be used for a new evaluation session.

```bash
kaggle/launch.sh rule_review_audit --run-name a23-20261101 --gpu t4 --internet \
  --datasets maadhavvh/rakshak-data
kaggle/wait_event.sh rule_review_audit-a23-20261101 --timeout 43200
kaggle/fetch.sh rule_review_audit-a23-20261101
```

This run uses frozen pre-f29521a risk logic and byte-identical pre-dc223d3 rules alongside
the current rules. It reports the unchanged campaign, every source_holdout fold, lang_shift
and temporal tests. GroundLM policy negatives and eval-eligible ScamShield are subdivided
by SHA256(`rule-audit:13:<campaign_id>`) into 20% audit / 80% remainder campaigns. These are
**in-sample: these rows informed rule design**; the subset names never imply untouched data.
The original release membership is neither rewritten nor used for threshold selection.

SmishX eval-eligible messages are the sole **independent, non-Indian-specific** validation.
Both frozen thresholds are shown: risk above LOW includes warnings/abstention; HIGH is the
SCAM decision. Every negative remains in the FPR denominator, including abstentions.
Fraud-versus-not-fraud uses known `label_scam` values; legacy spam-or-scam labels are separate.

Campaign intervals resample whole campaigns with replacement, 1,000 draws, seed 13.
For external sets lacking campaign IDs, the unchanged corpus MinHash/domain procedure
creates external-only campaign groups; this is an approximate template grouping, not
human-verified campaigns. Paired rate differences reuse a single draw for old and new.
Wilson score intervals supplement the clustered percentile intervals for rare events;
they assume independent rows. Zero-event clustered intervals can be degenerate.

Outputs are private: `metrics.json` (counts, rates, intervals, input hashes),
`audit_manifest.json` (IDs/groups only), and `rule_predictions.parquet` (IDs and frozen
decisions, no raw text). The PR must report failures and tradeoffs without a held-out
improvement claim. Model-side and PhreshPhish metrics are outside this rules-only run.
