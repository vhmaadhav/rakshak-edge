# Hugging Face resource review — 2026-11-03

Metadata/card research only: no weights, SMS rows or network-flow rows downloaded;
no local model evaluation, training, paid APIs or HF Jobs. No frozen split changed.
No performance claim follows from catalogue popularity or paper abstracts.

## Connection verification

The requested Hugging Face MCP authenticates as vhmaadhav and reports membership
in highestintheroom. Public hub_repo_details works. dataset_search still returns
`UNAVAILABLE: MCP tool dataset_search was not returned by tools/list` after the
user's access update. The private highestintheroom/rakshak-data lookup returns not
found; membership does not prove private-repository access. Read scopes are
present, write-repos is absent. **HF publish pending**; continue private Kaggle
publication with explicit owner/isPrivate verification. No credentials recorded.

Discovery fallback: free public HF catalogue metadata, followed by revision-pinned
README reads. MCP include_readme returned descriptions/metadata, so full cards
were read separately rather than inferred from tags. Search terms included Indian
SMS, transactional, Hinglish, UNSW, NetFlow, smishing and ScamShield. This is a
focused search, not proof that no other suitable dataset exists.

## Candidate decisions

| Resource | Possible use | Decision / limits |
|---|---|---|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Kaggle-only open-model label-policy annotation / template paraphrases | Apache-2.0 confirmed in pinned card. Candidate, no new annotation run. Llama remains mandatory advisory model; train IDs only, verifier/unsupported-ID gates still apply. |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Smaller annotation/paraphrase comparator | Apache-2.0 confirmed; card supports non-thinking mode. Annotation quality/throughput on our data UNVERIFIED. No inference here. |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | Semantic representation / teacher comparator to existing mmBERT-derived student | MIT confirmed. Retrieval pretraining is not evidence of scam accuracy or lower transactional FPR. Teacher use is a research hypothesis; no Arm or size claim. |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Considered paraphrase source | Card says other / qwen-research: the earlier licence uncertainty is resolved. Do not treat as Apache-2.0; prefer the verified 1.5B source. |
| [TabPFN-v2-clf](https://huggingface.co/Prior-Labs/TabPFN-v2-clf) | Possible offline tabular teacher | Card names priorlabs-1-1, describes Apache-2.0 plus extra attribution, and lists 16GB+ RAM. Actual licence terms not reviewed here. Defer; not a 4GB Pi deployment candidate based on this card. |
| [Indian scam SMS synthetic/audited](https://huggingface.co/datasets/Ridham115/indian-scam-sms-synthetic-audited) | Considered hard-negative supplement | 1,580 synthetic messages; card names Gemini and OpenAI generators and Claude-assisted labels. Exclude from advisory targets under open-model-only provenance rule. Not independent Indian validation. Do not adopt for SMS training without separate provenance/terms review; no rows read or used. |
| [Nora9029 NF-ToN-IoT-v2](https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2) | Potential host-provenance recovery lead | Card only has metadata; repository contains pre-split train/test CSVs. Original-row correspondence and host columns UNVERIFIED. A reuploader's Apache tag does not resolve original UQ terms. Do not substitute these splits for our frozen cohort. |

The SMS card explicitly documents model/domain shortcuts and limited human review.
Its claimed generators/audit process are author-reported, not independently verified.
Copies of the same synthetic set and ScamShield reuploads are not independent
validation. The bank-SMS catalogue lead did not yield a verified readable card;
no licence/consent/provenance conclusion is possible. Independent Indian validation
remains OPEN via consented, privacy-scrubbed, post-freeze inbox collection.

## Revision register

Revisions resolved from public HF API and full README fetched at that revision.
Licence column is card metadata, subject to the qualifications above.

| Repository | Revision | Card licence |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | apache-2.0 |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | other |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | apache-2.0 |
| `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` | mit |
| `Prior-Labs/TabPFN-v2-clf` | `275f19d2add9c2c0de38f9ebaded3856599fa2e7` | other |
| `Nora9029/NF-ToN-IoT-v2` | `4e951f835487ee57ee5da5dae2eaeb704908ab38` | apache-2.0 |
| `Ridham115/indian-scam-sms-synthetic-audited` | `45e92c38ba20e973ef3ff5d1893a0a710f622018` | cc-by-4.0 |

## Papers worth an experiment, not claimed IDS results

- [SCARF, Bahri et al., ICLR 2022](https://arxiv.org/abs/2106.15147):
  random feature corruption builds contrastive views for tabular pretraining.
  Could pretrain a compact source-only flow encoder without attack labels.
  Primary paper is general tabular learning; Rakshak LODO improvement UNVERIFIED.
- [VIME, Yoon et al., NeurIPS 2020](https://papers.nips.cc/paper_files/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html):
  masked-value reconstruction plus corruption-mask estimation provides another
  source-only pretext task. Must not fit on held-out test features or labels.
  Improvement over our supervised baseline UNVERIFIED.
- [SwitchTab, Wu et al., AAAI 2024](https://huggingface.co/papers/2401.02013),
  [primary arXiv](https://arxiv.org/abs/2401.02013): switched autoencoders separate
  shared and sample-specific representations. More complex alternative, deferred
  behind provenance recovery and simpler matched controls. No NIDS claim.
- [mmBERT, Marone et al.](https://huggingface.co/papers/2509.06888): supports
  multilingual-encoder research already pursued in PR #13. Our compressed
  embedding student is a derived experiment, not reproduction of the full encoder.

## Next project action

Keep SCARF/VIME as source-only pretraining candidates; do not launch models yet.
First inspect original NetFlow-v2 metadata/headers and exact frozen input versions,
then determine whether host provenance can be attached to unchanged rows without
label-assisted ambiguous matching. Existing historical IID CIs stay labelled IID.
Only after that audit can a paired, cluster-aware model gate be predeclared.
No experiment go/kill outcome or new headline number is asserted by this review.
