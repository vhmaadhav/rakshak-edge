# F continuation: protocol audit before another model — 2026-11-02

Research/audit only. No new IDS training, target inference, dataset download or
change to any frozen membership. Own branch ci/nids-source-coral-20261102 at
2463245, separate from URL and multilingual work; no deployment claim.

## Authoritative local evidence (aggregate saved JSON only)

Historical private nids_baselines, base9d1ddeca66de/dirty source
`dea7270125e726d5e70c4ddf139673062be2870309a81b8cec9b15a175a337ec`,
completed377units/0failed/4682.4s. Its results.json, not an assumption from prose,
reports no host columns for ALL FOUR inputs. T5/K5000/raw/lightgbm confidence
interval method is **iid_stratified** in every dataset. Therefore these are NOT
host/campaign-clustered intervals, and T4 host-disjoint evaluation was unavailable.
The job's generic docstring about host-clustered intervals does not describe this
particular run. No host clusters can be reconstructed from row positions alone.

| Dataset | Sample after dedupe | Benign | Target K5000 attack fraction | T5 CI method |
|---|---:|---:|---:|---|
| unsw_nb15 | 17,227 | 13,097 | .2374 | iid_stratified |
| cse_cic_ids2018 | 53,294 | 43,511 | .1886 | iid_stratified |
| bot_iot | 19,225 | 391 | .9820 | iid_stratified |
| ton_iot | 45,958 | 9,850 | .7798 | iid_stratified |

These are post-sampling/deduped reference populations, not natural deployment
base rates. Home-baseline fitting remains label-free, but the actual slices are
not generally mostly benign. Their labels were read only for the original
post-fit report; never use them to select a cleaner target calibration slice.
No new statistics or model computations were run locally for this audit.

## Primary-source research

- [Sun and Saenko, Deep CORAL, arXiv1607.01719v1](https://arxiv.org/html/1607.01719v1),
  sections3.1–3.2: differentiable covariance-distance penalty with normalization
  1/(4d^2), trained jointly with classification. Their experiment is image-domain
  adaptation, not NIDS. Source-only benign-conditioned alignment would be OUR
  derived variant, not their reproduced method or a claimed IDS result.
- [Cantone et al., arXiv2402.10974v1](https://arxiv.org/abs/2402.10974): primary
  abstract supports the cross-dataset generalization concern. Its arXiv journal
  reference says IEEE Access12:144489–144508, DOI10.1109/ACCESS.2024.3472907;
  direct DOI retrieval failed here. The inherited HTML venue discrepancy is not
  resolved by assuming the header is correct. No numerical competitor comparison
  is made under our different NetFlow protocol.
- [University of Queensland dataset project](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets)
  confirms common NetFlow-v2 datasets and links CSV downloads. Current official
  file schemas/host coverage, identity with the frozen stripped Kaggle copies,
  and an unambiguous row-preserving join are **UNVERIFIED**. Do not replace frozen
  inputs/splits or treat another copy as the same cohort on matching counts alone.

## Next decision, not yet a training predeclaration

Hypothesis to investigate: aligning source benign representations across three
training datasets could reduce site-specific shortcuts without fitting a target
attack-label mixture. A compact19-live-feature MLP with a source-only covariance
penalty is one candidate. Expected benefit **UNVERIFIED**; no hyperparameter or
scientific go gate is declared yet, no model job authorized by this document.

First establish whether original host/campaign provenance can be recovered for
EVERY frozen test row without changing membership or resolving ambiguous matches
using labels. Freeze and hash any provenance mapping before model outcomes.
Absent that, record clustered comparisons UNMEASURABLE and any IID results as
exploratory; never rename feature buckets or independent rows as real campaigns.
Retain exact existing seed0 sampling/split algorithms and frozen source-only train/
cal roles. Any future target adaptation may use ONLY its original unlabeled K
slice, removed from test, never target labels or remaining test features for fit.
A new candidate's full recipe, matched controls, paired gate and failure behavior
must be predeclared before Kaggle training. No training while this provenance
question remains unresolved. URL and multilingual remain higher-priority active
work; independent Indian SMS validation stays OPEN.

Official linked NetFlow-v2 records, checked2026-11-02 (research only):
UNSW https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848 ;
ToN https://rdm.uq.edu.au/files/a4ad7080-ef9c-11ed-a964-b70596e96ad5 ;
BoT https://rdm.uq.edu.au/files/b97fa3c0-ef9c-11ed-8007-832d3bc76ba8 ;
CSE https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848 .
Three returned empty rendered content, ToN retrieval timed out. This does not
verify file schemas, downloads, checksums or row correspondence. Original UQ
project states academic research is permitted with citation; commercial permission
requires agreement. No contact/message sent to authors and no paid service used.

## Stage-1 original archive audit, predeclared 2026-11-03

Official public `GET /datasets/<uuid>/file-details` and `/download` at
`https://api.rdm.uq.edu.au` returned all four records OPEN, canDownload=true.
The general `/datasets/<uuid>` endpoint returned401; no restricted endpoint was
bypassed. Public file-details confirms academic research with citation and a
commercial-use restriction. Signed download URLs are not saved or printed.
A HEAD request to the signed UNSW archive returned403; this does not establish
whether a signed GET/Range will work. No archive bytes were read on the laptop.

Hypothesis: original host columns can support a reproducible mapping for unchanged
frozen rows. Full audit go only if EVERY frozen row receives unambiguous host
provenance without label-assisted matching or changed membership. Otherwise the
clustered comparison is UNMEASURABLE. No model go claim from this metadata stage.

Stage1 job `nids_provenance` records current attached input SHA256, schema and row
count; bounded range reads inspect original ZIP directories and CSV headers only.
At most8MiB archive-response bytes per dataset, one to four CSV members, bounded
header size; reject ignored ranges, changed ETag/size, unexpected origins or auth
failures. No full-download fallback, flow values, fitting, scoring or split writes.
It does not recover host mappings yet. Technical completion and prerequisites for
stage2 are separate from the full mapping gate. Historical file hashes are absent:
matching historical input counts is necessary, never sufficient for byte identity.

Original aggregate counts expected: UNSW1,986,745; CSE17,129,715;
BoT30,420,086; ToN13,135,881. Four original dataset slugs confirmed from saved
kernel metadata. The static remote original-version metadata request returned404;
input version numbers are still UNVERIFIED, not inferred from current slugs.

Focused synthetic tests:10passed0.60s under nice19/taskset0-3/OMP2/MKL2;
Tctl38.2C. No full suite until milestone. Cards/provenance research is also in
HF_RESOURCE_RESEARCH.md. Next is the single private Kaggle metadata job, then
exact-cohort replay / conservative ambiguity checks only if the evidence permits.

Original historical run.py bundle was independently decoded as source text only:
its sampler job, schema module and split module are byte-identical to this
worktree. SHA256 respectively79ac87d0dd8a28b498e6dc1fb5988cb75b036a4242f07edda96cd9043de18d1e,
9c5a5a0e1ae840c6eb121fae175287e162b4dbf7ae067c936782cdec5fd8ca29,
9533ac5ab8a0e79275eb38275f904439d0c8cd4a73399e1a33a78a58d78f1f62.
This validates the replay algorithms, not historical input identity or host joins.

Read-only Kaggle dataset-version metadata on2026-11-03 returned all4 current
versions as2, with complete returned version histories1/2 dated January2023:
UNSWv2 Jan13 20:36:09.737; CSEv2 Jan14 11:24:57.123;
BoTv2 Jan14 10:34:20.047; ToNv2 Jan14 11:12:48.687 (server timestamps).
No version postdates the historical2026-10-30 launch. Thus latest-version
resolution would have usedv2; this is an inference from provider history plus
saved unversioned input refs, not a directly retrieved historical attachment
receipt. Current SHA256 pins are now available from f-meta. Preserve both the
supporting history and the missing original byte-hash limitation.

## Post-pilot research lead (not a rescue of the failed gate)

The original publisher's dataset description links the primary
[UNSW cleaning notebook](https://www.kaggle.com/code/dhoogla/nf-unsw-nb15-v2-00-cleaning).
Public Kaggle GetKernel returned version1; code cells only (no table outputs)
SHA2565134ed334b08477f70181bac569fb604c9730fda65e8e61e0a2b151893fbb396
were read for provenance research. It removes the two IP columns, applies
fastai df_shrink with obj2cat=false/int2uint=false, removes inf/NaN rows,
keeps first duplicates across remaining columns, resets row indices, then
writes parquet. The duplicate key includes the publication's label columns.

This gives a concrete next research lead: reconstruct the documented publication
conversion and prove complete ordered-row/dtype equality against the pinned
parquet before attaching retained original row positions. No such replay has
been implemented or verified. Reproducing that label-containing publication
conversion is DIFFERENT from the declared label-free41-field join. Its outcome
must not retrospectively change the strict pilot's UNMEASURABLE gate or pretend
that a label-assisted candidate tie-break passed. Resolve this protocol distinction
and predeclare any new audit separately; no further mapping/model run now.
