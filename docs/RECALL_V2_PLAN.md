# Recall V2: reputation-guarded URL evidence in quiet mode (pre-registered plan, 2026-11-20)

Status: **plan, written before any arm below is scored.** Nothing here changes the shipped defaults unless every
gate passes. Owner of this run: Sachin (InstaBox VM, CPU). Branch `sachin/recall-v2`.

## 1. Problem

In the shipped quiet mode (`docs/SYSTEM_V1.md` §2b) a warning is visible only when rule evidence (or, in verify mode,
URL-model evidence) exists. The student score never makes a message visible on its own. Result: GroundLM visible
FPR 2 / 1,133 (0.18 %), but SmishX visible smishing recall **53 / 226 (23.5 %)** and campaign visible recall 16.2 %.

The URL model already fixes most of the recall (`docs/URL_SERVICE_RESULTS.md`: SmishX 53 → 181 / 226 when its
evidence is visible), but it was kept out of quiet mode because:

- U2 failed by one message: GroundLM visible 2 → 5 / 1,133 (limit 4); the 3 new ones are EMI/loan messages with links.
- SmishX legit visible FPR went 22 → 52 / 606; 43 of the 57 legit SmishX URLs cross τ_url.
- `url_domain` test (non-PhreshPhish benign URLs) FPR is 69.4 % at τ_url.

The common cause is **benign URLs on well-established domains that the char-CNN does not recognise**. Phishing URLs
in SMS overwhelmingly sit on fresh registrations, shorteners, free hosting or look-alike domains.

## 2. Intervention (fixed now, before scoring)

**Reputation guard for URL evidence.** A URL hit from the URL model becomes evidence in quiet mode only if the URL is
*not* "established". A URL is established when all hold:

1. Its registrable domain, computed with `tldextract` **including the Public Suffix List private section**
   (so `x.github.io`, `x.web.app`, `x.blogspot.com`, `x.netlify.app` are their own registrable domains), is in the
   **Tranco top-100,000** list `V354N` (created 2026-11-01, providers crux, farsight, majestic, radar, umbrella,
   30-day Dowdall, PLD filter; https://tranco-list.eu/list/V354N). File sha256 recorded at download.
2. The host is not a URL shortener (the shortener list already in `src/rakshak/url_features.py` / `rules.py`,
   unchanged).
3. The host is not a known user-content host on a popular parent that the PSL does not split. Fixed list, written now:
   `sites.google.com, docs.google.com, forms.gle, drive.google.com, storage.googleapis.com, firebasestorage.googleapis.com,
   dropbox.com, dl.dropboxusercontent.com, 1drv.ms, onedrive.live.com, sharepoint.com, ipfs.io, cloudflare-ipfs.com,
   dweb.link, wa.me, t.me, telegram.me, linktr.ee, bit.ly, tinyurl.com, rebrand.ly, cutt.ly, is.gd`
   (match = host equals the entry or ends with "." + entry).
4. The host is not a raw IP address and the URL has no userinfo (`@`) part.

No list is tuned afterwards. The guard is an offline file (about 1.5 MB for 100k domains, under 1 MB gzipped) and a
set lookup, so it adds no measurable latency.

Literature basis: popularity lists as benign priors, with their manipulation risks (Le Pochat et al., "Tranco: A
Research-Oriented Top Sites Ranking Hardened Against Manipulation", NDSS 2019); Public Suffix List private domains as
the hosting boundary (publicsuffix.org). The guard's recall cost on phishing hosted on popular domains is measured
directly by G5 rather than assumed.

## 3. Arms (all quiet mode, all with the same retrained models)

Models are **retrained on InstaBox** from the v0 recipes (`sms_student` campaign dual-base seed 1; `url_student`
char-CNN wide, PhreshPhish train, seed 0), because the shipped ONNX files are not readable from this account.
τ_url is recomputed with the shipped rule (smallest cal score with benign FPR ≤ 1 % on the 5 % grouped carve,
`kaggle/jobs/url_threshold.py` logic), written to disk before any other set is scored.

| Arm | URL evidence in quiet mode | Student-only tier |
|---|---|---|
| R0 | off (shipped V1) | none |
| R1 | corroborated (V1 amendment policy: student over τ or a rule fired) | none |
| R2 | on (URL ≥ τ_url) | none |
| **R3 (candidate)** | on, **reputation-guarded** (§2) | none |
| R4 (exploratory) | as R3 | Neyman-Pearson tier: student ≥ τ_NP adds a VERIFY notice; τ_NP = k-th order statistic of the student score on the pre-named Indian-legit negatives (`external/indian_legit/`: finee_synth, skyaara, ridham115 legit rows, cloveai ham), α = 1 %, δ = 5 % (smallest k with P(Bin(n, 1−α) ≥ k) ≤ δ). GroundLM is never used to set it. |

R0 must reproduce the V1 numbers exactly (2 / 1,133; 53 / 226), because quiet-mode visibility does not depend on the
student. Any mismatch stops the run until explained.

## 4. Gates for R3 (all must pass to propose it as the quiet-mode default)

| # | Measure | Limit |
|---|---|---|
| G1 | GroundLM policy negatives, quiet visible warnings (1,133 scoreable) | ≤ 4 (the U2 limit) |
| G2 | SmishX smishing, quiet visible recall (226 eval_ok) | ≥ 136 (60 %) |
| G3 | SmishX legitimate, quiet visible FPR (606 eval_ok) | ≤ 34 (baseline 22 + 2 pp) |
| G4 | Campaign test negatives, quiet visible FPR (10,952) | ≤ 1.0 % (baseline 0.61 %) |
| G5 | PhreshPhish `official.test` phishing, recall after the guard at τ_url | ≥ 80 % (V1 92.0 % without guard) |

Reported, not gated: campaign visible recall; GroundLM positives visible recall (91); PhreshPhish benign FPR with the
guard; `url_domain` test FPR with the guard (**circularity caveat**: its benign side may come from popularity lists,
so a drop there is expected and is not evidence); per-bucket GroundLM FPs; Wilson 95 % intervals everywhere; added
decision latency on x86 (Pi parity is a later step).

## 5. Contamination and honesty notes

- GroundLM and SmishX informed earlier rule design (`docs/RULE_COVERAGE.md`) and the URL-gate design. The guard in §2
  was chosen from the U2/U3 aggregate failure description (EMI/loan links, legit SmishX URLs), not from row-level
  inspection, but it is still **feedback-informed**: the R3 result is a confirmation on sets that motivated it, not
  an untouched test. Say so in every quote.
- No row text is printed in the results; only counts, buckets and domains of false positives (domains are not PII).
- If any gate fails, R3 is reported as failed and quiet mode stays as V1. No gate is changed after scoring.

## Record before scoring (2026-11-20, 04:5x UTC)

- Tranco `V354N` top-100,000 CSV (`https://tranco-list.eu/download/V354N/100000`, 100,000 rows, first row
  `1,google.com`): sha256 `4b00082641e977a0c613efbcec72f11ae300b681899de3aa5f2edcffcf364e3c`. Not committed (third-party
  list; re-download by ID). The Pi/phone install step fetches it once and checks this hash.
- Retrained SMS student (InstaBox, `sms_student` campaign dual-base seed 1, CPU): campaign test F1 0.9757, cal PR-AUC
  0.9968, ONNX 388,127 bytes, ONNX/torch parity 3.0e-6. Shipped v0 is F1 0.9813; same recipe, different hardware and
  library versions.
