# NIDS provenance audit — 2026-11-03

No IDS models were trained or scored. This is a frozen-data provenance audit,
not an improvement in LODO detection performance. Historical IID intervals remain
IID. No source/target split or evaluation membership was changed.

## Declared question and gate

Can the original host pair be recovered for EVERY frozen sampled row using all41
remaining non-label numeric fields, without label-assisted matching? Go only on
complete, unambiguous recovery; a missing or multiple-pair match makes the audit
UNMEASURABLE. UNSW is the smallest-archive pilot; its failure stops the remaining
three mapping runs and any new IDS model experiment. Do not discard difficult
frozen rows, invent campaigns or choose a first candidate to make the gate pass.

## Verified stage1 result

Private CPU nids_provenance-f-meta2-20261103 v1 completed ok94.9s;
tree519f56771a4e9b9a2c98eaf99b7d7c4bb89b2dd04e1d2d703eae45d1574216c0,
receipt7480f3a36c65985dcb8274d27bf736e93d154ff9f8a7984104547ab38ba0e36a.
ONE event wait and ONE successful fetch, provenance checked.

All four official archives contain both IPV4_SRC_ADDR and IPV4_DST_ADDR.
Actual bounded HTTP response totals were5840/5852/5904/5836bytes for
UNSW/CSE/BoT/ToN. No original archive/flow data was downloaded on the laptop.
Companion feature-description CSVs are metadata, not missing-host flow files.

| Input | Current parquet rows | Current parquet SHA256 |
|---|---:|---|
| UNSW | 1,986,745 | 7500a64cc12f25542772d9c43d6438234333c2f8c4b45471425fcde18899e5c4 |
| CSE | 17,129,715 | f299c8a2cb1223912438aae68f3818f898c97ce1824b2deef618f56f266562cb |
| BoT | 30,420,086 | 1923dd9d93cad5baf9aafb8fe74a59eded38604b544f3691636870c574bf2cc2 |
| ToN | 13,135,881 | 4e8c9ac5cfda1ba1b3f1dbe450feed0dde48b809425bf65eabcd23086afb6624 |

Counts match the historical result; this alone does not prove byte identity.
Provider version histories returnv1/v2 only, with allv2 updatesJanuary2023,
before the historicalOctober2026 launch. Latest-version resolution therefore
supports an inferredv2, but no original file hash or direct historical attachment
receipt was recovered. Current file hashes are now available for future checks.

The first stage1 run f-meta-20261103/tree5b1ec9875e54b40d7c53d73795088254d0f49debecf1c0d8cf28f04ff78a3ffc
completed partial116.4s. Its uppercase-only header-name guard rejected the
companion feature dictionary. It is retained, not counted as successful host
recovery. The corrected parser recognizes flow-header anchors, reports other
CSV members as metadata, and never emits an unrecognized first line.

## Strict UNSW pilot: event result, output receipt pending

Private CPU nids_provenance_map-f-unsw-20261103 v1:
tree6831963fe8a0e56489c981cf571c3095cb5a3528d6c0667d1cb8b7d62731ab13,
args --dataset unsw_nb15, original UNSW input and hash-pinned stage1 attached.

Its ONE completion event reports ok / host mapping UNMEASURABLE. The ONE allowed
fetch returned no files: CLI exit0 with only its version warning, followed by
fetch.sh's missing-results rejection. Exact match/ambiguity counts, runtime and
result-file provenance are PENDING; none is invented from the event. A request
for one recovery-fetch exception is pending. No retry, restart, replacement event
subscription or status poll has been performed.

The unchanged original sampler is reused as a parity control. The replay retains
all41 numeric fields and original parquet positions, reproduces seed0/cap200000/
stratified sampling/33-feature first-dedupe, then checks X/y against the original
sampler and expected17,227retained rows. Source CSV labels are explicitly excluded
from the join reader. Numeric casts use the frozen parquet precision; nonfinite
or lossy integer conversions are refused. The entire source CSV must be scanned,
including duplicates with different host pairs; first candidate is not a tie-break.

Decision: no further mapping or IDS training under this declared gate. This is
not a negative model result and does not measure a classifier F1/FPR/recall.
SMS campaign/source_holdout/lang_shift/temporal, GroundLM, SmishX and PhreshPhish
endpoints are not applicable to this metadata-only IDS audit.

## Verification and next lead

Focused synthetic checks21passed across metadata, matching, end-to-end mapping
and existing builder tests; prescribed nice/affinity/thread limits, Tctl below39C.
Ruff clean. Full milestone validation result is recorded below once fetched.

The original publisher's
[cleaning notebook](https://www.kaggle.com/code/dhoogla/nf-unsw-nb15-v2-00-cleaning)
v1 was inspected as code only (no table output). It documents removal of the two
host columns, dtype shrinking, removal of inf/NaN rows, first-duplicate retention,
index reset and parquet output. Its duplicate key includes label columns.
An exact reconstruction of the documented conversion with complete ordered-row/
dtype equality is a concrete research lead, not yet implemented or verified.
It is different from the failed label-free join; no retrospective gate rescue.

Primary dataset terms and source links: NIDS_NEXT_RESEARCH.md. Academic reuse
requires citation; commercial use requires agreement. HF_RESOURCE_RESEARCH.md
records the model/dataset shortlist and exclusions. No competitor reproduction,
SOTA, Arm runtime, shipment or independent Indian validation claim.

Full milestone unit_validation-f-full2-20261103 v1 completed ok206.4s: **857passed,0skipped,0failures/errors**,156warnings,
source1be81d36300b5415e2696a4c8b8936af3673193307562c1fa884ab6554033e7c.
One event/fetch, all139 archived files byte-identical current. First f-full
attempt rejected the incorrectly supplied selection hash before collection or
pytest; retained error66.8s, no test invocation or pass claim for that attempt.
