# Free compute options for Rakshak-Edge

Research snapshot: 2026-10-30. Scope: free or credit-based cloud GPU/compute that can be driven headlessly from a terminal, for ~300M-param encoder batch training and occasional 8B QLoRA. Nothing was signed up for, no accounts were created, nothing was posted.

Labels: **official** = from the provider's own page fetched this session. **secondary** = third-party aggregator or blog. **UNVERIFIED** = could not be confirmed. Where official and secondary sources disagree, both are shown.

Project rules that constrain the ranking (see `docs/BUDGET.md`, `docs/CONTEXT.md`): training runs on Kaggle by default; Modal needs Maadhav's explicit permission per run.

## Top 5 (free, headless, usable for our workloads)

| Rank | Option | GPU / VRAM | Free quota | Session cap | Headless CLI/API | Catches |
|---|---|---|---|---|---|---|
| 1 | **Kaggle Notebooks** (project default) | P100 16 GB or T4 16 GB (secondary) | About 30 GPU h/week, floats (secondary; official quota **UNVERIFIED**) | About 9 h per session (blogs) vs 12 h (Sep 2026 review); **UNVERIFIED** | Yes: `kaggle kernels push/status/output`, already wired in `kaggle/` (`docs/CONTEXT.md` line 53) | Internet off by default; quota floats; 8B QLoRA only in short runs on 16 GB |
| 2 | **Lightning AI Studios (Free)** | T4, L4, L40S listed; A100/H100/H200 listed but 4 h session cap; max 1 GPU per Studio, 2 concurrent | **Official:** up to 30 credits, one-time grant (5 on signup, +25 after adding a card), expire after 12 months; "~80 GPU hours on interruptible machines". Older sources say 15 credits/month (superseded by the official page) | Official: free Studio restarts every 4 h; T4/L4/L40S listed as unlimited | Yes: `lightning job run --command --machine --interruptible --name` (official flags). GPU machine names **UNVERIFIED**; job access on Free plan **UNVERIFIED** | Phone number required; card for +25 credits; no reserved machines on Free |
| 3 | **Modal (Starter)** | T4 $0.000164/s (about $0.59/h), L4 $0.000222/s, A10 $0.000306/s, A100-80GB $0.000694/s (about $2.50/h), H100 $0.001097/s (about $3.95/h) | **Official:** $30/month recurring credit; 10 GPU concurrency | UNVERIFIED | CLI/Python SDK (not checked this session) | **Needs Maadhav's per-run permission** (`docs/BUDGET.md`). Card requirement on Starter **UNVERIFIED**. Recurring monthly credit is its main advantage |
| 4 | **Saturn Cloud (free)** | Single T4-class GPU, 16 GB (secondary); 64 GB system RAM | **Official marketing page:** "150 hours a month of compute for free", no card. A July 2026 third-party comparison says Saturn no longer publishes the figure (conflict) | UNVERIFIED | UNVERIFIED | Quota figure conflicts; CLI unknown; treat as a trial until checked |
| 5 | **Azure for Students** | GPU VMs **not listed** on the page | **Official:** $100 credit, 12 months, renewable yearly while a student, no card | n/a | `az` CLI (standard; not checked) | GPU VM quota for student subscriptions **UNVERIFIED** and may need a quota request. Could be CPU-only in practice |

Sources: Kaggle [16], [docs/CONTEXT.md]; Lightning [1][2][3]; Modal [18]; Saturn [10]; Azure [12].

Why this order: Kaggle is free, recurring and already wired for headless runs. Lightning is the only other option with a verified GPU CLI and a usable free T4. Modal gives the largest recurring headless budget but is gated by the permission rule. Saturn and Azure are the least verified, so they sit at 4 and 5 pending checks.

## Checked and excluded (or not useful for headless training)

| Option | Status | Sources |
|---|---|---|
| Hugging Face ZeroGPU | Free daily quota (5 min/day free account, 40 min/day PRO, official). Gradio-SDK Spaces only, `@spaces.GPU` decorator. Not a batch-training target | [4] |
| Hugging Face Jobs | Headless (`hf jobs run`, `hf jobs hardware`, HTTP API, official). **Paid**: needs a positive credit balance. T4-small $0.40/h, A10G-small $1.00/h, L4 $0.80/h, A100-80GB $2.50/h. PRO ($9/mo) includes only $2 of inference credit per month; Jobs credit not stated. Best paid fallback | [5][6][7] |
| Google Cloud trial | $300 / 90 days (official), card required, **GPUs excluded** during trial, quota increases not allowed. Not useful for GPUs | [14] |
| AWS Free Tier | Up to $200 credits, 6 months (official). GPU instances **not** listed. Educate GPU credits **UNVERIFIED**; AWS docs describe Educate as mainly a learning platform | [15] |
| Paperspace Gradient (DigitalOcean) | Free M4000 (8 GB) notebook machine on Free plan (official list); P4000/RTX4000/A4000 need Pro, A5000/A6000/A100 need Growth. Notebooks are public. Headless path: `gradient run` deprecated per PyPI 0.8.0a2 (secondary) | [8][30] |
| SageMaker Studio Lab | **Closed to new customers** per AWS docs. Existing users only. EU docs say closure effective 7/30/26; US page does not state the date (UNVERIFIED) | [24] |
| Google Colab | Free T4 typical (not named by Google). Max 12 h, idle timeout, no fixed limits published. FAQ prioritises interactive use; scripted/remote-control use may be terminated. Headless via the official Colab CLI (`colab new/upload/exec/download/stop`), wired up in `colab/`; note the FAQ caveat on scripted use | [17] |
| Deepnote | Free plan is CPU basic (5 GB, 2 vCPU). Student Team plan has no GPU (GitHub Student Pack) | [13] |
| Intel Tiber AI Cloud | Standard-Free tier exists (secondary overview). GPU access may need review; availability inconsistent. CLI **UNVERIFIED** | [26] |
| Oracle Cloud | No free GPU verified (official page returned 403). Always Free Arm A1 CPU (up to 4 OCPU / 24 GB) per secondary sources; $300 / 30-day trial **UNVERIFIED** | secondary |
| RunPod | Startup program up to $10K and research credits up to $25K per search snippets (official `runpod.io/startup-program` not fetched). The fetched `runpod.io/research-credits` page did **not** confirm research credits. Eligibility **UNVERIFIED** | [21] |
| Vast.ai | $2,500 GPU credits for early-stage startups (official article, snippet only). No student programme found | [22] |
| TPU Research Cloud (TRC) | Free Cloud TPUs for accepted **researchers**; "Express interest" form; grant length and TPU types **UNVERIFIED**. Students not clearly eligible | [19] |
| NVIDIA LaunchPad | Time-limited free labs, up to 2 weeks after approval (secondary). No student offer found. Academia page offers no free compute | [23] |
| IndiaAI Compute Portal | **Subsidised, not free.** Students in AI/ML projects eligible subject to verification; users pay costs beyond the subsidy. Login via DigiLocker / e-Pramaan / Jan Parichay. ₹67/h headline is from launch coverage (**UNVERIFIED** for any specific GPU). No API/CLI mentioned | [20] |
| Bharat AI-SoC challenge compute | **Not found.** No challenge-provided compute located. Check the challenge portal before relying on it. **UNVERIFIED** | none |
| Qualcomm AI Hub | Free to use per FAQ (official, quoted in `docs/QAI_HUB_PLAN.md`). Device profiling only, not training | [25] |
| Arm Developer Program | Free to join; no current free Arm GPU instance found. Old Works-on-Arm offers have expired | secondary |
| GitHub Student Developer Pack | Azure $100 (see rank 5); Camber student plan 5 GPU hours (CLI **UNVERIFIED**) | [13] |

## Suggested checks before committing (none done)

1. Lightning: confirm free-plan job submission and GPU machine names with `lightning job run --help` after `pip install lightning-sdk` (no account needed to read help, but login is needed to run).
2. Kaggle: read the current weekly GPU quota and session cap in the account settings (official docs page returned only a title).
3. Saturn: confirm the 150 h figure and whether a CLI exists.
4. Azure: check whether GPU VM quota is available on a student subscription.
5. Modal: confirm the card requirement on Starter; any run needs Maadhav's explicit permission first.

## Sources

[1] https://lightning.ai/pricing
[2] https://lightning.ai/docs/platform/developers/cli/batch
[3] https://lightning.ai/docs/platform/developers/cli
[4] https://huggingface.co/docs/hub/spaces-zerogpu
[5] https://huggingface.co/docs/hub/jobs
[6] https://huggingface.co/docs/hub/jobs-pricing
[7] https://huggingface.co/pricing
[8] https://docs.digitalocean.com/products/paperspace/pricing
[10] https://saturncloud.io/try
[12] https://azure.microsoft.com/en-us/free/students
[13] https://education.github.com/pack
[14] https://docs.cloud.google.com/free/docs/free-cloud-features
[15] https://aws.amazon.com/free/
[16] https://www.kaggle.com/docs/notebooks (page returned only its title; Kaggle quota and cap figures are secondary)
[17] https://research.google.com/colaboratory/faq.html
[18] https://modal.com/pricing
[19] https://sites.research.google/trc/about/
[20] https://compute.indiaai.gov.in/
[21] https://runpod.io/research-credits (did not confirm research credits)
[22] https://vast.ai/article/vast-ai-startup-program (search snippet only)
[23] https://developer.nvidia.com/academia
[24] https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-availability-change.html (search snippet; studiolab.sagemaker.aws returned 403)
[25] https://workbench.aihub.qualcomm.com/docs/hub/faq.html (as quoted in docs/QAI_HUB_PLAN.md)
[26] https://console.cloud.intel.com/docs/about/ai_cloud_overview.html (search snippet)
[30] https://pypi.org/project/gradient/0.8.0a2 (search snippet)

Not fetched or 403/404: studiolab.sagemaker.aws (403), Oracle free tier page (403), Intel get-access page (redirected to a 404).
