# W2: URL student on PhreshPhish + url_domain, results (2026-11-12)

Plan: `docs/URL_GENERAL_PLAN.md` with its pre-launch amendment (PhreshPhish official split kept; only rows whose
registered domain is in the url_domain test set were dropped from training). Training: kernel
`maadhavvh/rakshak-url-general` v1 (3 seeds; crashed in the final McNemar report: `2.0**n` overflow, fixed in log
space). Scoring: `maadhavvh/rakshak-url-general-score` (score-only mode, saved thresholds read unchanged), 1 894 s, ok.
Sets: train 568 924, cal 33 302 (combined), PhreshPhish test 168 060, url_domain test 30 432, domain-novel PhreshPhish
test 76 644.

## Verdict: KILLED (0 of 3 seeds pass G1-G3)

| Seed | τ (combined cal, ≤ 1 % FPR) | PP test FPR / recall (G1: ≤ 1.5 %, ≥ 0.85) | UD test FPR (G2: ≤ 5 %) | PP AP at 1 % base rate (G3: ≥ 0.8447) |
|---|---|---|---|---|
| 0 | 0.9174 | 1.17 % / 0.901 ✔ | 0.16 % ✔ | 0.788 ✘ |
| 1 | 0.6463 | 1.16 % / 0.915 ✔ | 0.04 % ✔ | 0.810 ✘ |
| 2 | 0.7410 | 0.93 % / 0.895 ✔ | 0.03 % ✔ | 0.785 ✘ |
| current model (τ_url 0.6308) | – | 0.92 % / 0.920 | 69.4 % | 0.8647 |

## What it means

- The fall in url_domain FPR (69 % → ≤ 0.16 %) is **in-distribution** for the new model, which trained on url_domain's
  train split. It shows that url_domain benign URLs are learnable, not that the model generalises.
- The fair generalisation number is the **domain-novel PhreshPhish test** (registered domains absent from training),
  paired against the current model: AP 0.981 vs **0.988**, recall 0.836-0.870 vs **0.887**, FPR 1.71-2.18 % vs
  **1.65 %** (McNemar p ≤ 1.8e-17 every seed). On unseen domains the new model is slightly worse.
- G3 fails because adding url_domain costs PhreshPhish ranking at low base rates (AP at 1 %: 0.785-0.810 vs 0.865).

The current URL model and τ_url stay. Low out-of-domain FPR on links inside SMS still needs a different approach. In
the service, URL evidence stays verify-only in quiet mode (`docs/URL_SERVICE_RESULTS.md`).
