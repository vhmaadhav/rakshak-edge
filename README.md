# Rakshak-Edge

Offline, adversarially robust, proof-carrying scam detection for SMS and embedded URLs on Arm devices.
Bharat AI-SoC Challenge 2026–27, Problem Statement 3.

A small multi-expert detector (message, URL, rules) decides scam / benign / uncertain with calibrated abstention.
Llama 3.2 1B, run through ExecuTorch, only puts detector-produced evidence into plain language. A deterministic
verifier rejects any explanation that cites evidence or actions the detector did not produce.

## Layout

| Path | What |
|---|---|
| `src/rakshak/` | canonicaliser, perturbation attacks, URL features, rules, evidence package, verifier, prompts |
| `src/rakshak/datasets/` | dataset loaders, dedup, campaign clustering, splits (`python -m rakshak.datasets.build`) |
| `src/rakshak/eval/` | metrics, cluster-bootstrap CIs, robustness harness, reports |
| `kaggle/` | Kaggle job launcher; all training and evaluation runs there (see `kaggle/README.md`) |
| `docs/` | context, data card, SOTA targets, ExecuTorch and Qualcomm AI Hub plans |
| `tests/` | unit tests |

## Setup

```bash
uv sync
uv run pytest -q
```

Raw and processed data live in `data/` and are not committed; see `docs/DATACARD.md` for provenance and licences.
