# End-to-end detection: what the deterministic layer actually does

Status: **measured** on the real campaign-disjoint test split.
Owner: Sachin. Produced for handoff "ML model for gathering info" work and to bound what the
ELA explainer can possibly say.

Reproduce:

```bash
python -m rakshak.datasets.build --skip-urls     # ~3.5 min
.venv/bin/python scripts/e2e_detection.py --split campaign --part test
.venv/bin/python scripts/e2e_detection.py --json        # stable blob
```

## 1. The result

`campaign/test`, 22,145 rows, 11,193 positives (50.5%). Data is the rebuilt public corpus
(137,153 raw SMS → 110,827 kept → 88,401 campaigns), so this is the real datacard population, not a
toy split. Decision read from `package.risk` on the label-free `proxy_scam_state`, i.e. **the rules
layer alone**, no classifier and no fusion.

```
gold -> decision
  0 -> benign        10,888
  0 -> scam              18
  0 -> uncertain         46
  1 -> benign         9,704     <-- the problem
  1 -> scam             225
  1 -> uncertain      1,264
```

| Metric | Value |
|---|---:|
| Precision (scam) | **0.9259** |
| Recall (scam) | **0.0201** |
| F1 | 0.0393 |
| FPR (benign called scam) | 0.17% |
| Abstention (UNCERTAIN/MEDIUM) | 5.9% |
| Coverage | 94.1% |
| Accuracy on decided rows | 0.5334 |
| Fraud with **no evidence at all** | **83.8%** |
| Benign with no evidence at all | 98.5% |

Top firing types: `urgency` 668, `account_suspension` 426, `prize_lottery` 423, `shortened_url` 327,
`parcel_customs` 145, `suspicious_url` 122, `brand_domain_mismatch` 99, `apk_install` 97.

## 2. What this means (read this before quoting any number)

**The rules layer is not a detector and must not be presented as one.** It is a high-precision
*evidence extractor*: when it fires, it is right 92.6% of the time, and it rarely fires on a
legitimate message (0.17%). But it stays silent on 83.8% of real fraud. Its F1 of 0.039 is not a
regression; it is what the component is designed to do, and every published headline in
`docs/HANDOFF.md` section 5 (campaign F1 0.987 for the teacher, 0.977 for the KD student) clearly
comes from a *learned* model, not from this path.

Three consequences that matter:

1. **`accuracy_when_decided` of 0.533 is not a bug, it is arithmetic.** Coverage is 94.1% because
   only 5.9% of rows abstain, and most of the "decided" benign calls are just "no rule fired".
   Reporting 53% accuracy as a detector result would be the single most misleading number available
   in this repo, so the metric is printed for transparency and flagged here.

2. **The ELA explainer can only explain 16.2% of fraud.** 83.8% of scam rows produce an empty
   evidence package, and `package.evidence` is the *only* thing the Llama layer may verbalise
   (`docs/CONTEXT.md`: "Llama never sees raw attacker text"). So for four out of five real scams the
   advisory falls back to `template` mode with nothing to cite. This is the binding constraint on
   the whole "proof-carrying explanation" claim, and it is a *detector coverage* problem, not an
   LLM problem. Any ELA work that improves explanation quality without widening evidence coverage
   is optimising the small slice.

3. **The abstention machinery is barely exercised.** At a 5.9% abstention rate the calibrated
   three-state design is effectively collapsed to two states. The intended behaviour (abstain when
   unsure) cannot appear until the underlying scores separate the classes, so the risk–coverage
   curve is degenerate in this configuration.

## 3. Why the label target is weak, and the honest caveat

`label_binary` is legacy `spam or scam = 1`, which counts **promotions as positives**. The project's
own decision (`docs/CONTEXT.md`, datacard §2a) is that promotions are *not* fraud and metrics should
be reported on scam-vs-not. So a fraction of the 9,704 "missed" positives are genuine promotional
SMS that the rules correctly ignore — the same label noise measured in `docs/RANK_ARM_AUDIT.md`,
where the "Loan Scam" and "UPI Fraud" buckets turned out to be mostly real bank and cashback offers.

That caveat softens the recall number but does **not** rescue it: the hand-checked fraud slice in
`docs/RANK_ARM_AUDIT.md` still showed 39.1% of genuine fraud with no evidence. Even under the most
charitable reading, the rules path explains well under half of real fraud.

**Done, and it does not change the conclusion.** Scoring the same path against `label_scam`
(scam-vs-not, restrictions: rows where `label_scam` is populated and `eval_ok`) on the campaign test
split — n = 16,912, positives 5,926:

| Target | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| `label_binary` (spam-or-scam) | 0.9259 | 0.0201 | 0.0393 | 0.17% |
| `label_scam` (scam-vs-not) | 0.8304 | **0.0157** | 0.0308 | 0.17% |

Recall does not improve — it gets slightly *worse* — so the "promotions inflate the miss count"
caveat is real but immaterial. Precision drops because 34 test-split rows are labelled `promotion`
and a few of those do fire scam rules. Either way the deterministic layer explains under 2% of
scam-labelled rows, and section 2's conclusions stand unchanged.

## 4. Where the recall has to come from

The rules cannot carry it and should not be asked to. Ranked by expected value:

1. **The student classifier** (`src/rakshak/models/`, KD from the mmBERT teacher). That is the
   component designed for recall; the rules exist to supply *explainable* evidence on top.
2. **Fusion** (`src/rakshak/fusion/`) combining `message_prob` + `url_prob` + rule flags. Fusion is
   what should set `scam_state`; today nothing here feeds it, so `proxy_scam_state` is used, which
   is why only rule-hit rows can reach a verdict at all.
3. **Rule coverage widening** — converts some urgency-only and no-evidence rows into typed
   evidence. This raises the *explainable* fraction, which is the ELA constraint, but on its own it
   moves recall by a few percentage points at most.

This is why the two halves of the project are not separable: the ML detector supplies recall, the
rules supply the grist for the explanation, and the numbers above show the rules half is
under-supplying.

## 5. The other frozen splits (same harness)

All three frozen partitions, same path, same decision rule:

| Split | n | pos. share | Precision | Recall | F1 | FPR | Abstain | fraud w/ no evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `campaign/test` | 22,145 | 50.5% | 0.9259 | 0.0201 | 0.0393 | 0.17% | 5.9% | 83.8% |
| `lang_shift/test` | 1,677 | 21.3% | 0.7857 | **0.0616** | 0.1143 | 0.46% | 2.6% | 70.3% |
| `temporal/test` | 11,029 | 0.7% | 0.0000 | 0.0000 | 0.0000 | 0.17% | 0.5% | 88.3% |

Three things worth noting:

- **`lang_shift` recall is 3x the campaign recall (6.2% vs 2.0%)**, not worse. That is the opposite
  of the Hinglish collapse seen for the *learned* student (`docs/HANDOFF.md`: lang_shift F1 0.36
  without KD), and it makes sense: the rules are hand-written bilingual keyword lists, so they fire
  on romanised Hindi ("OTP bhejo", "account band") just as readily as English. The learned model's
  Hinglish weakness is a different failure mode and this measurement neither confirms nor excuses
  it. It does mean "the rules fail on Hinglish" should not be asserted.
- **`temporal` is degenerate for this component** — 0 true positives at any threshold, with 68 of
  its 77 positives producing no evidence. This is the 77-dated-row caveat in
  `docs/DATA_FETCH_STATUS.md` showing up as a hard number: quote nothing from this row except as
  evidence that the split is too thin to measure a rules path.
- FPR is stable at 0.17-0.46% across all three, so the false-positive behaviour is the one property
  that does transfer. That is consistent with the GroundLM result (0.0883%).

## 6. Effect of the widened rules (`src/rakshak/rules.py`)

The sentence-scoped `parcel_customs` / `job_task_scam` and the `prizes?` / "win ... prize" additions
were merged after the measurements above. Re-running the same harness on `campaign/test` and
comparing against the pre-change tree (`git stash`, same command):

| Quantity | before | after |
|---|---:|---:|
| scam rows with no evidence | 84.60% | **83.78%** |
| `prize_lottery` hits | 370 | **423** |
| tp / fp / fn (scam decision) | 225 / 18 / 10968 | 225 / 18 / 10968 |

So roughly **92 more fraud rows in the test split now produce evidence** (0.82 pp of 11,193), almost
all from the lottery plural fix, and the scam verdict counts are untouched. That is consistent with
the independent measurement in `docs/RULE_COVERAGE.md` (sample fraud no-evidence 39.10% → 36.09%,
benign `>=UNCERTAIN` unchanged at 1.38%, whole-corpus benign 1.23% → 1.25%).

Read that effect for what it is: **this widens what the explainer can talk about, not what the
detector catches.** Recall against `label_binary` did not move at all. Verdict-level recall is a
model/fusion problem (section 4); rule coverage buys the *narrative* on ~1 pp more fraud.

## 7. Reproduce / extend

`scripts/e2e_detection.py` takes `--split` and `--part`, so the same measurement is available for
`lang_shift` (the Hinglish collapse), `temporal` (77 dated positives — do not quote as robust) and
`source_holdout`. Suggested next runs:

```bash
.venv/bin/python scripts/e2e_detection.py --split lang_shift --part test
.venv/bin/python scripts/e2e_detection.py --split source_holdout --part test
```

Once a student checkpoint is available, the honest comparison is *fused* vs *rules-only* on this
same split, with the rules-only row kept as the floor. Keep both rows in any report;
`docs/HANDOFF.md` claims must be read against the fused row, never this one.
