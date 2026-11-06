# Training-only transactional hard-negative arm(a) — 2026-11-01

This new arm is on `ci/transactional-hardneg-20261101`, based on PR#6. PR#6 and
PRs2/3/4/5 stay untouched and unmerged. Candidate(b) failed; no per-family
calibration is allowed until this separate(a) also fails. Training and fused audit completed; KILL. See TRANSACTIONAL_HARDNEG_FUSED_RESULTS.md.

## Hypothesis and experiment (before implementation/compute)

Independently authored deterministic benign notifications can teach the student
currency/account/UPI/OTP/delivery/utility context without treating it as fraud.
This is a hypothesis, not evidence of real Indian message authenticity or a causal
data gap. No raw external/test message is a template, target, paraphrase or example.
GroundLM/SmishX aggregate feedback already influenced intervention selection;
subsequent outcomes are in-sample for that design feedback, not untouched validation.

Two fresh arms, ERM/no augmentation and hard-negative augmentation. Same dual-base
student, original frozen training IDs, matched per-fold teacher logits only on those
IDs, no unlabeled extras. Synthetic rows appended only to the training pool with
label_binary=label_scam=0, label_policy=benign_transactional, synthetic=True and an
explicit generator/template/family/language/seed/ID. They receive full supervised
BCE and no teacher logits; no synthetic advisory targets or model-written labels.
Original supervised/KD loss stays alpha.7/T2. Same AdamW.002/wd.0001,batch128,
max15epochs/patience3 by original cal PR-AUC, seeds1/2/3. ERM sampling in both arms.
Class positive weight uses each arm's actual training counts, as the existing trainer
normally does; record it rather than claiming identical effective class weighting.

For each outer fold, append min(12000, floor(.25 * original binary-negative train
count)) synthetic rows, balanced across six families and EN/Hinglish; generator
seed20261101 fixed across model seeds. Same frozen generated prefix for all three
student seeds. Families: bank debit, bank credit, completed UPI transfer, OTP with
do-not-share, parcel status, telecom/utility bill. Variants/amounts/digits/account
suffixes/sender IDs are invented, not scraped or sampled from real messages. Sender
metadata is stored for provenance but not added to the existing body-only student
inputs. Synthetic headers imitate public structural shape, not verified registrations.
Generator reads no corpus; a later exact-normalised-key collision with any original
corpus/external row aborts the data arm rather than adapting a template from that row.
Deduplicate generated rows; assert train-only tags/IDs before fitting. No split/data
release or test/external label is changed. Persist generated rows privately on Kaggle.

All six frozen folds, all three seeds, student endpoints and **full new cal-only fused**
endpoints reported. Fusion uses the unchanged baseline schema (candidate(b)'s optional
context schema is not adopted), fixed hash-pinned URL expert/max aggregation, and
four fixed campaign cal stages for LR/Platt/thresholds/CRC diagnostics. Each arm's
thresholds fit cal only. Freeze/hash every fitted state before external decisions.
Underlying cal checkpoint selection remains a caveat; CRC is not a certified claim.
The owner confirmed no previous fusion configuration was frozen. The immutable v1 measurement reference is newly fitted, not previously shipped; this is not a shipped-state audit.

## Frozen go/kill

Every seed must satisfy GroundLM SCAM FPR<=1%, >=UNCERTAIN FPR<=10% over all fraud
negatives, and augmented fused SmishX SCAM recall>=paired no-augmentation student's
recall at that control's frozen cal-1%-FPR threshold. In addition, neither GroundLM
SCAM nor >=UNCERTAIN FPR may increase against the paired no-augmentation fused
baseline, and no frozen-fold SCAM recall loss may exceed that baseline's campaign
CI half-width. Otherwise no-go/kill; no externally selected seed or threshold.

Declare endpoint applicability now: legacy label_binary guards apply on **every**
fold, matching the original frozen detector protocol; additional fraud-label guards
apply wherever fraud positives exist. UCI's fraud recall remains explicitly NA
(zero mapped fraud positives), with its legacy spam guard reported separately.
No relabelling/substitution or claim that UCI is scam-free. Other undefined required
endpoints block go. Temporal retains77 positives/three original train/test campaign
overlaps. External fraud/legacy metrics stay separate. Report per-seed paired campaign
bootstrap differences/CIs and all absolute rates/coverage;1000 draws/seed13.

## Template-source register

The literal notification templates will be original deterministic code, detector-only;
never Llama advisory targets. The family list is the user's task brief, not a corpus.
No commercial SMS body will be copied. Public guidance supports security semantics
and header structure, not authenticity of generated wording:

- RBI digital-banking safeguards:
  https://rbikehtahai.rbi.org.in/safeguards-for-digital-banking.html
  Official indexed excerpt states credential/OTP sharing cautions. Full-page retrieval
  encountered a bot challenge; **UNVERIFIED full-page content**, no wording copied.
- NPCI UPI FAQ: https://www.npci.org.in/what-we-do/upi/faqs
  Indexed official excerpt describes a bank SMS after a completed transaction;
  current page is JS-only in retrieval. **UNVERIFIED full-page content**, no wording copied.
- TRAI advice to senders: https://www.trai.gov.in/advice-to-senders
  Official page retrieved2026-11-01 describes registered alphanumeric headers and fixed/
  variable message template structure. Generated headers are synthetic, not authenticated.
- Courier/telecom/utility notification wording is independently authored generic status
  semantics from the task brief; no genuine-provider template reproduction is claimed.

Keep the synthetic source private with the mixed-licence project. Independent Indian
held-out validation remains OPEN; generated messages cannot fill that validation set.

## Frozen reference clarification before arm(a) outcomes

Maadhav confirms no prior fusion config was shipped/frozen. A separate campaign
seed1 original-recipe cal-only bundle will become **newly fitted (not previously
shipped) v1**, with actual student/URL/rule-source artifacts and immutable JSON.
Post-training comparison must validate that private v1 manifest, expert hashes and
reporting receipt and include its unchanged external reference results. No v1
refitting during the arm(a) audit. Experimental fresh-control/hardneg cal states
retain their own role/seed/state hashes and are not described as the shipped v1.
This adds provenance/reference reporting only; it does not replace the original
fresh no-augmentation paired control or change any predeclared arm(a) gate,
threshold rule, feature, generator or model choice. V1 freeze precedes post-eval.
