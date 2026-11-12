# SMS student retrain on consented inbox data (pre-registration draft)

Drafted 2026-11-07 after `docs/SMS_RETRAIN_RESULTS.md` killed all three synthetic-data arms. **APPROVED by Maadhav 2026-11-07. Frozen.** The gates, arms, donor roles, threshold rule and privacy rules are
fixed; any later change is a dated amendment made before the results it affects.

Decisions taken by Maadhav on 2026-11-07:
- Consented inbox data may be used for training, under a donor split (one donor held out entirely).
- Donors: Maadhav and Sachin.
- Storage: a separate private Kaggle dataset.

## 1. Why

- Synthetic licensed negatives (FinEE, minimal pairs) moved GroundLM FPR only from 41.6 % to about 24-33 % at a
  v0-like threshold. Real, template-held-out Indian negatives (`sms_groupneg` cross-fit) reached 0.97 %. The
  missing ingredient is real Indian legit traffic, and no licensed public corpus of it exists (OpenResearch audit,
  2026-11-07).
- The frozen threshold rule failed because its validation set was synthetic and harder than real traffic, which
  pushed τ to about 0.99. A real legit validation set is the direct fix.
- The inbox export keeps alphanumeric sender IDs (`tools/inbox_export/scrub.py`, `sender_style`, `scrub_sender`).
  That makes the first measured DLT-sender ablation possible; no published one exists.

## 2. Rule change (needs approval)

`tools/inbox_export/README.md` and `docs/INBOX_EXPORT.md` currently say reviewed inbox data must never be used for
training. On approval, both are amended to say:

> Reviewed inbox data may be used for training only under a pre-registered donor split (`docs/INBOX_RETRAIN_PLAN.md`):
> at least one donor's data is a frozen test set, never used for training, threshold choice or tuning, and scored
> once per pre-registered arm.

## 3. Donors, consent and privacy

| Role | Donor | Use |
|---|---|---|
| Train donor | Maadhav | train + threshold validation (sender-group held out) |
| Test donor | Sachin | **frozen test only**: scored once per arm, never used to pick τ, arms or epochs |

Sachin is the test donor so that the person who builds and tunes the experiments never sees the test data.

Consent and collection:
1. Each donor signs `docs/INBOX_CONSENT.md` (to be drafted on approval). It covers: purpose (training and testing an
   on-device scam detector for the Bharat AI-SoC Challenge), storage (private Kaggle dataset, owner Maadhav), the
   trained-model release, withdrawal at any time (data deleted, models retrained on request), and deletion after
   the challenge (final event 2027-02-16) unless re-consented.
2. Each donor exports **received SMS only** with SMS Backup & Restore (XML) and runs
   `python -m tools.inbox_export.export` **on their own machine** with their own names file.
3. **Review policy:** approve only messages from businesses or services (alphanumeric sender IDs, short codes)
   and received scams. **Skip every person-to-person message** (`benign_personal`), so no third party's message
   is exported. Do a privacy check on each row; the tool already enforces human approval.
4. **Review load cap:** at most 40 messages per sender ID, newest first. This is enough template variety and
   keeps review to about 1-2 hours per donor.
5. Sachin sends only his reviewed, scrubbed parquet to Maadhav over a private channel he chooses. Maadhav uploads
   both files to the private Kaggle dataset `maadhavvh/rakshak-inbox`. That dataset is never merged into
   `rakshak-data`, never committed to git, and never placed in the judge repo.

Handling rules (binding on Claude, subagents and job code):
- Claude and subagents never open, print or quote inbox row text. Only aggregate counts and metrics are read.
- Job logs and outputs for inbox sets carry random ids, labels and scores only, never text or sender strings.
- Raw backups never leave the donor's machine.

## 4. Data and splits

- Labels from the tool: `benign_transactional` and `promotion` → 0 (GroundLM's policy also treats marketing as safe);
  `scam` → 1; `benign_personal` and `unclear` → dropped.
- Train donor: group by sender ID and by near-dup template (two-level clustering as in `docs/SMS_RETRAIN_PLAN.md`
  §2.2). Hold out 20 % of **sender IDs** (seeded) as the **real legit validation set**. No sender ID appears in both
  train and validation.
- Test donor: whole file is test-only. A **template-novel subset** is also reported: test rows with no near-dup
  (Jaccard ≥ 0.8) in the train donor's training rows. It is computed in the job and only counts are output.
- Test-only dedup (as before): every train or validation row that near-dups campaign test, lang_shift, temporal,
  GroundLM, SmishX or any test-donor row is removed, whole cluster.
- The template-flooding cap of 5 rows per cap group and the total cap (min(12 000, 0.25 × campaign-train
  negatives)) are unchanged.
- If, after review and dedup, the train donor has fewer than 500 legit rows or the validation set has fewer than
  200 legit rows, **stop before training** and report. The thresholds below cannot be met reliably below that.

## 5. Arms (3 arms × 3 seeds 1, 2, 3; same dual-base recipe as v0)

- **R0: control.** The v0 recipe re-run with no extra rows, for paired comparison (the shipped v0 is a single seed).
- **R1: + inbox.** Campaign train + train-donor inbox rows (legit and any scams).
- **R2: R1 + sender feature.** Five extra side features: sender is alphanumeric, plus DLT suffix -P / -S / -T / -G.
  Rows without a sender (all existing corpora, GroundLM, SmishX) get all-zeros. The ONNX side input grows from 16 to
  21. If R2 wins, the service must pass the sender header, which is recorded as a deployment change. This is the
  ablation; its effect is R2 − R1.

## 6. Threshold rule (frozen before any test scoring)

τ = the smallest threshold with FPR ≤ 1 % on the **real legit validation set** (held-out train-donor sender IDs) and
FPR ≤ 2 % on campaign cal. If none exists, use the campaign-cal max-F1 threshold and flag the run. τ is written to
`thresholds.json` before any test set is scored (same enforcement as `sms_retrain.py`).

## 7. Gates (an arm passes if each gate holds on the seed median and on at least 2 of 3 seeds)

| # | Gate | v0 | Pass |
|---|---|---|---|
| G1 | GroundLM FP / 1 133 (cross-source: no GroundLM source in training) | 471 | ≤ 11 |
| G2 | SmishX smishing TP / 226 | 157 | ≥ 157 |
| G3 | Campaign-test F1 | 0.9813 | ≥ 0.9713 |
| G4 | ONNX size | 388 KB | ≤ 10 MB |
| G5 | **Test-donor legit FPR** (Sachin, full) | measured for v0 in the same job | ≤ 1 % point estimate; Wilson 95 % upper bound reported |
| G6 | Pi latency, winner only | p50 1.39 s / p95 2.93 s | within +10 % |

Reported, not gated: test-donor template-novel legit FPR; test-donor scam recall (likely few scams; Wilson CI);
lang_shift and temporal F1; per-bucket GroundLM FPR; row-by-row versus GroundLM XLM-R (McNemar); v0's numbers on
the same donor sets.

**Winner:** the passing arm with the lowest G5 FPR (seed median). On a tie, the simpler arm wins (R1 over R2).
R2 counts as the sender result only if it beats R1 on G5 with a paired CI excluding zero. If no arm passes,
it is a kill and is reported as one.

## 8. Compute

- New job `kaggle/jobs/sms_inbox_retrain.py`, reusing the `sms_retrain.py` structure (and its fixed report
  render), with datasets `maadhavvh/rakshak-data` + `maadhavvh/rakshak-inbox`. One kernel per arm, two at a time
  (Kaggle GPU limit). One blocking `wait_event.sh` per launch.
- Unit tests on synthetic fixtures only: the sender-ID split is disjoint; test-donor rows never reach train or
  validation; no text appears in any output; the 500/200 stop rule; sender-feature encoding; threshold ordering.

## 9. Limits

- Two donors means a single held-out person. G5 is one person's inbox: banks, telcos and apps in the test donor's
  life. It is not a population estimate.
- Donor inboxes share common bank templates. The template-novel subset is reported for that reason.
- A model trained on private messages is released as ONNX. A 92 k-parameter char-CNN has little capacity to
  memorise, but this is recorded rather than measured.

## Amendment 2026-11-07: Codex labelling (approved by Maadhav, before any inbox data was trained on)

Maadhav chose to have his inbox labelled by Codex instead of by hand. Rows are first scrubbed with
`tools.inbox_export.export.prepare` (names file, contacts, numbers, amounts, OTPs, URLs), capped at 40
per sender, with person-to-person senders skipped. Only that scrubbed text is given to Codex (OpenAI),
which labels each row and marks `privacy_ok`. `tools/inbox_export/merge_codex.py` keeps rows with
`privacy_ok=true` and a non-personal label, writes `human_reviewed=False, label_source="codex"`, and
`sms_inbox_retrain.label_inbox` accepts those rows. Consequences: (1) labels are model labels, not
human labels, so the results report says so; (2) scrubbed text leaves the laptop to OpenAI, which
Maadhav accepts for his own inbox. The consent form must say this before any other donor uses it.
Gates, arms, splits and the threshold rule are unchanged.

Addendum 2026-11-07: Maadhav's train-donor data comes from three of his own phones (backups exported
separately; each capped at 40 per sender and scrubbed with the union of all three names files), labelled
by Codex in two batches and concatenated into one `train_donor.parquet`. The cap stays at 40.

Addendum 2 (2026-11-07, Maadhav): supersedes the cap in the addendum above. With Codex labelling there is
no review load, so the 40-per-sender export cap is removed (the job's own template caps still apply), and
messages from phone numbers that are not saved contacts are included, since many scams come from plain
mobile numbers. Messages from saved contacts stay excluded (third parties who did not consent; the job
drops benign_personal anyway). All three phones are relabelled in one Codex batch (8,804 rows).

Addendum 3 (2026-11-07, Maadhav): Sachin's inbox is not available, so the test donor is Maadhav's phone 1
(2,992 Codex-labelled rows: 2,976 legit, 14 scam, 2 unclear) and the train donor is phones 2 and 3 (5,048 rows,
160 scam). Phone 2 was considered first but holds 148 of the 174 scams, which would starve training.
Limits: test and train come from the same person, so banks, templates and 32+ business sender IDs are
shared; G5 (test-donor legit FPR) is an optimistic, same-person estimate, not a cross-person one. The
results write-up must state this and report the template-novel subset. Gates, arms and the threshold
rule are unchanged; test-donor rows are still scored once per arm and never used for tuning.
