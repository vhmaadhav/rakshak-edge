# Recall V2: results (2026-11-20, InstaBox VM, CPU)

Plan and gates: `docs/RECALL_V2_PLAN.md` (written and committed before any arm was scored). Raw numbers:
`docs/results/recall_v2/metrics.json`. Per-row visibility flags (no text) are on the VM at
`/data/workspace/rakshak/jobs/recall_eval/out/row_flags.parquet`. Script: `deploy/pi/service/eval_recall_v2.py`.

**Verdict: R3 (reputation-guarded URL evidence) fails G1, G3 and G5. Quiet mode stays as V1. No default
changes in this branch.** `URL_QUIET_POLICY` stays `"off"`. The guard code and the `guarded` policy are
kept behind that flag for the next run.

Caveat on every number: GroundLM and SmishX informed earlier rule and URL-gate design, so these results
confirm or refute a design on the sets that motivated it. They are not an untouched test (plan §5).

## 1. Setup (pinned)

| Item | Value |
|---|---|
| SMS student | retrained on InstaBox (`sms_student` campaign dual-base seed 1), sha256 `25885eed…a1b6`, threshold 0.08318 |
| URL student | retrained on InstaBox (char-CNN wide, PhreshPhish train, seed 0), sha256 `0df79bab…772f` |
| τ_url | 0.84208, set by the shipped rule on the grouped cal carve: cal benign FPR 0.996 % (113 / 11,347), cal recall 89.2 %. Written to disk before any other set was scored. |
| Tranco | V354N top-100,000, sha256 `4b000826…4e3c` (matches the pre-scoring record), 100,000 domains loaded |
| τ_NP (R4) | 0.999213 = order statistic k = 114,100 of n = 115,196 Indian-legit negatives (finee_synth 100,572, cloveai ham 13,672, ridham115 783, skyaara 169). α = 1 %, δ = 5 %, violation probability 0.0494. Empirical FPR on that set 0.95 %. |

The retrained URL model gets 89.1 % PhreshPhish `official.test` phishing recall at 0.97 % benign FPR, without
the guard. The shipped V1 model got 92.0 %. Same recipe; different hardware, library versions and training
budget.

## 2. Arms (quiet mode; counts with Wilson 95 % intervals)

| Arm | GroundLM negatives visible (1,133) | GroundLM fraud visible (91) | SmishX smishing visible (226) | SmishX legit visible (606) | Campaign recall (11,193) | Campaign FPR (10,952) |
|---|---|---|---|---|---|---|
| R0 off (V1) | **2** (0.18 %, 0.05–0.64) | 11 (12.1 %) | **53** (23.5 %, 18.4–29.4) | 23 (3.80 %, 2.54–5.63) | 16.2 % | 0.64 % |
| R1 corroborated | 5 (0.44 %, 0.19–1.03) | 35 (38.5 %) | 158 (69.9 %, 63.6–75.5) | 51 (8.42 %) | 20.2 % | 0.88 % |
| R2 on | 5 | 35 | 165 (73.0 %, 66.9–78.4) | 53 (8.75 %) | 20.3 % | 0.93 % |
| **R3 guarded** | **5** | 35 | **159** (70.4 %, 64.1–75.9) | **39** (6.44 %, 4.74–8.68) | 20.1 % | **0.92 %** |
| R4 R3 + NP tier (exploratory) | 5 | 35 | 162 (71.7 %, 65.5–77.2) | 39 (6.44 %) | **57.4 %** (56.5–58.3) | 0.92 % (0.76–1.12) |

PhreshPhish `official.test` at τ_url:

| | Phishing recall (76,800) | Benign FPR (91,260) | Share on an "established" domain |
|---|---|---|---|
| No guard | 89.1 % (89.0–89.4) | 0.97 % | phishing 16.8 %, benign 59.9 % |
| Guard | **73.4 %** (73.1–73.7) | 0.63 % | |

`url_domain` test (reported, not gated, circularity caveat): benign FPR 30.0 % → 27.7 % with the guard,
phishing recall 93.7 % → 81.4 %. Only 8.5 % of its benign URLs are on established domains, so the guard
does little there.

## 3. Gate verdicts for R3 (pre-registered limits, unchanged)

| # | Measure | Limit | R3 | Verdict |
|---|---|---|---|---|
| G1 | GroundLM negatives visible | ≤ 4 | 5 | **fail** |
| G2 | SmishX smishing visible | ≥ 136 | 159 | pass |
| G3 | SmishX legit visible | ≤ 34 | 39 | **fail** |
| G4 | Campaign visible FPR | ≤ 1.0 % | 0.92 % | pass |
| G5 | PhreshPhish phishing recall with the guard | ≥ 80 % | 73.4 % | **fail** |

## 4. R0 reproduction check

The plan requires R0 to reproduce V1 exactly: 2 / 1,133 and 53 / 226. **Both match.**

SmishX legit came out 23 / 606, against V1's 22. This number was not one of the two named in the plan, but
it was checked anyway. The plan's statement that "quiet-mode visibility does not depend on the student" is
not quite true. An urgency-only message, whose only evidence is weight-0.4 `urgency`, becomes UNCERTAIN, and
so visible, only when the student also says scam (`src/rakshak/evidence.py`, `_risk`). In this run, 4 of
the 23 visible SmishX legit rows and 1 of the 2 visible GroundLM rows are urgency-only with the retrained
student over its threshold. 7 SmishX legit and 10 GroundLM urgency-only negatives stay hidden because the
student is under it. A different student moving one such row explains the 22 → 23 difference. Nothing else
in R0 depends on the model.

## 5. What the failure says (measured)

1. **The guard misses Indian regional senders.** All 3 GroundLM negatives added by the URL item carry a
   `bsesdelhi.com` link. BSES Delhi is a power distribution utility, and its domain is not in the Tranco
   top-100k. A global popularity list does not cover legitimate Indian regional institutions. The other 2
   GroundLM false positives are R0's rule-only rows, with no URL.
2. **Phishing on popular platforms is common in PhreshPhish.** 16.8 % of phishing URLs sit on a registrable
   domain in the top-100k, even with PSL private splitting and the user-content list. Suppressing them costs
   15.7 pp of recall, more than G5 allows. The guard does cut benign false positives by 35 % (886 → 572).
3. **The guard helps SmishX legit, but not enough.** URL-induced false positives drop from 30 (R2 53 vs R0
   23) to 16 (R3 39). G3 allows 11.
4. **Recall gains stand without the guard.** Every URL arm takes SmishX visible recall from 23 % to 70–73 %
   and GroundLM fraud recall from 12 % to 38 %. The price is 3 extra GroundLM false positives, one over the
   U2 limit.

## 6. Exploratory: the Neyman-Pearson student tier (R4)

R4 adds a VERIFY notice when the student score exceeds τ_NP. Compared with R3, it raises campaign visible
recall from **20.1 % to 57.4 %** (+4,173 messages). It adds **zero** false positives on campaign negatives
(101 → 101), GroundLM negatives (5 → 5) and SmishX legit (39 → 39). SmishX smishing goes from 159 to 162.

This is the largest effect in the run, but it is **not a result to ship on**:

- R4 was exploratory and has no gates.
- The campaign test split is in-distribution for the student, which was trained on the campaign split's
  train partition.
- The NP calibration set is 87 % synthetic (finee_synth). The 1 % FPR guarantee holds for that population,
  not for real Indian inboxes.
- Its GroundLM and SmishX rows show the NP tier adds no false positives there, but also almost no recall
  (+3 SmishX).

It earns a pre-registered run of its own on data that has not touched the student or this design.

## 7. Next candidates (not run, not pre-registered here)

- An India-specific established-sender list for the URL guard, e.g. utilities, banks and government
  domains from an official registry, frozen before scoring. It would replace or extend Tranco for `.in`
  and Indian-brand registrable domains. Finding 1 above motivates it, so its evaluation must use data
  outside GroundLM.
- The NP tier (R4) as a VERIFY-only notice, pre-registered with an untouched real-SMS negative set and a
  recall measure outside the campaign split.

## 8. Run facts

Scoring time 2,272 s on 4 shared vCPUs (student and URL ONNX, per-URL scoring with the best URL kept). The
added decision cost of the guard is one tldextract call and one set lookup per URL, which is negligible
next to the URL model (not separately timed, so no number is claimed).
