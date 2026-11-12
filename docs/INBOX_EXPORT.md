# Independent Indian validation collection tools — 2026-11-01

**Validation remains OPEN. This milestone adds code only. No real inbox data was read,
collected, labelled, transmitted or added to git.** Maadhav/Sachin run it locally on their
own consented exports; any future transfer/validation set requires an explicit choice.

Hypothesis: a local scrub-and-review workflow can preserve useful bank/UPI/OTP/delivery
structure while removing identifiers before any research sharing. Experiment: synthetic
XML/JSON import, adversarial identifier fixtures, review-confirmation/output guards and
local Parquet round-trip. Milestone acceptance (recorded after focused implementation checks, before the full suite): all supported synthetic fixtures scrub their identifiers,
no unreviewed rows export, Git output and overwrite attempts are refused. Kill/withhold:
any fixture leaks its identifier or bypasses review. This is a software gate, not proof
of anonymisation on real SMS; arbitrary names/subdomains still require manual review.

Implementation/use/protocol: `tools/inbox_export/README.md`. Incoming SMS only, no
MMS/RCS attachments/call logs or sent messages. Five human labels; no model labels.
Owner-only output outside Git; no network clients, no original-message keys/timestamps
or reversible mapping exported. The script does not implement an upload command.
JSON is a supported conversion schema; SyncTech itself documents XML.

Official format source checked on2026-11-01:
https://www.synctech.com.au/sms-backup-restore/fields-in-xml-backup-files/

Future independent Indian validation must be collected after rules/thresholds freeze,
consented/privacy-scrubbed, human labelled and campaign/near-duplicate checked; never
use it for tuning. Bank/UPI/OTP/delivery/telecom and received scams are the planned mix.
No benchmark results or performance headline changed by these tools.

**Amendment 2026-11-07 (approved by Maadhav).** Reviewed inbox data may be used for training only under a
pre-registered donor split (`docs/INBOX_RETRAIN_PLAN.md`): at least one donor's data is a frozen test set, never used
for training, threshold choice or tuning, and scored once per pre-registered arm. Consent per `docs/INBOX_CONSENT.md`.
