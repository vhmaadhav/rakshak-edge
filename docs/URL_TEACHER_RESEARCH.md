# URL teacher preparation — primary checks, 2026-11-02

Research only while higher-priority SMS/calibration jobs finish. No URL model job,
new result, fine-tuning or distillation has run. Historical transformer-L4 AP at
0.05% is .245 versus charCNN .484; that teacher does not justify useful KD.
Three-seed charCNN reproduction and a better teacher remain OPEN. Existing frozen
PhreshPhish subset protocol and original URLcal domain carve must remain unchanged.
No new model/threshold selection from PhreshPhish benchmark/test outcomes.

Candidate hypothesis, **UNVERIFIED**: pretrained byte representations may offer a
better URL teacher than the existing transformer trained from scratch. This is
not established by general NLP performance. A future fixed experiment must name
its exact training recipe and teacher cal-only eligibility gate before launch.
KD eligibility must be decided before reading that experiment's test outcomes.

[ByT5 paper, TACL2022](https://aclanthology.org/2022.tacl-1.17/),
[authors' official repository](https://github.com/google-research/byt5) and
[pinned model card](https://huggingface.co/google/byt5-small/blob/68377bdc18a2ffec8a0533fef03b1c513a4dd49d/README.md)
confirm byte-level pretrained T5, mC4 pretraining and Apache-2.0. The paper's
noise-related NLP findings do not establish URL detection or PhreshPhish gains.
Correct paper arXiv ID is2105.13626; an initially recalled2103.17239 was checked,
found to be an unrelated image-transformer paper, and is NOT a ByT5 citation.

Public primary API/card/config checked without downloading any local weights:
- google/byt5-small revision68377bdc18a2ffec8a0533fef03b1c513a4dd49d;
- README SHA2566daea4f59e1a86e7db0a44c488651130cfd39c3b92cf63eead46db865bdabaed;
- config SHA2567845fb21b320f3fa05392ce151143502cf08729c8c89732da3293615885d3e83;
- d_model1472,d_ff3584,12encoder/4decoderlayers,vocab384;
- pytorch_model.bin1198627927bytes, LFS/SHA256
  5c5aaf56299d6f2d4eaadad550a40765198828ead4d74f0a15f91cbe0961931a.
  [Pinned file metadata](https://huggingface.co/api/models/google/byt5-small/tree/68377bdc18a2ffec8a0533fef03b1c513a4dd49d).

This is a prospective training-only teacher, not an Arm student. T4 memory/time
feasibility remains UNVERIFIED. Encoder-only fine-tuning could avoid the decoder,
but no architecture, batch size, learning rate or scientific go/kill is declared
by this research note. Do not silently run an exploratory sweep from it.

Future requirements: all three student seeds/checkpoints frozen before outcomes;
train-only teacher targets, cal-only early-stop/eligibility, source/model/input
hashes, every fixed URL split, all975PhreshPhish subsets and allbase rates,
paired domain-clustered comparisons at0.05%/0.1%, size/CPU export quality and exact
URL-only qualifier. GTE's URL+HTML row is a different input protocol, no SOTA claim.
Historical benchmark feedback already informed intervention selection; disclose
that design exposure. No external/test URLs become teacher targets/calibration.
Private Kaggle only, max2kernels, oneevent/fetch; HF publish pending.
