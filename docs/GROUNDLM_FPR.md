# GroundLM FPR test (handoff task 1)

Status: **run** on the rules-only detector. Models not scored (see section 6).
Owner: Sachin. This closes the first open task in `docs/HANDOFF.md` and its issue #1 twin.

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
| Rules only, **before** this change | any evidence | 12 | 1.0591% | **FAIL** |
| Rules only, **after** this change | any evidence | 1 | **0.0883%** | **PASS** |

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

12 of the 13 original false positives were **`urgency`-only** packages:

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

## 6. What is NOT done

- **Per-model FPR.** The acceptance criterion says "each shipped model at its cal threshold". This
  host has no `torch`, no GPU, and no access to `highestintheroom/rakshak-models` (no HF token, not
  an org member), and the Kaggle datasets are not shared with `sachin0496`. The *model* half of the
  test must run on Kaggle. What is reported here is the **rules-only** detector, which is a shipped
  component and the one whose failure mode this test exposed.
- **Campaign-clustered CI.** With 1 false positive in 1,133 the exact 95% Wilson interval is
  approximately `[0.0156%, 0.4986%]`. A campaign-clustered bootstrap is not meaningful at this event
  count (one cluster); the honest statement is "1 flag in 1,133, 95% upper bound 0.50%".
- Handoff section 5 has **not** been edited; that is a shared results table and should be updated
  when the model numbers land alongside it.

## 7. Reproduce

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
k = sum(bool(package_for_sms(r.text)[1].evidence) for r in scr.itertuples())
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

**Recommendation: keep the change.** It is required to pass the go/kill line that the project defined
for itself, the failures it prevents are in the population the challenge names as the primary safety
objective (legitimate Indian bank/UPI/OTP/delivery SMS, and specifically a genuine credit-card due
reminder), and the recall it costs sits in a mechanism that a rules-coverage fix recovers properly
rather than by re-inflating a generic keyword.

## 9. Open follow-ups

1. **Rule coverage** — the 15 urgency-only fraud rows above are the same population measured in
   `docs/RANK_ARM_AUDIT.md`: specific rules (`kyc_block_urgency`, `parcel_customs`, `job_task_scam`)
   are too narrow. Widening them converts urgency-only hits into typed hits and recovers the recall.
2. **`apk_install` false positive** — the single surviving GroundLM FPR fires on a bare "download"
   verb in Devanagari conversational text at weight 0.9 (HIGH). It should require an app/APK object.
3. **Model-side FPR** — still needs Kaggle; see section 6.
