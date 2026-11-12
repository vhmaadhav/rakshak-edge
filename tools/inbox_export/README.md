# Local inbox review

Code only; no real messages or backups belong in this repository. Run offline:

```
python -m tools.inbox_export.export /private/local/backup.xml \
  --names-file /private/local/personal-names.txt --output /private/local/reviewed.parquet
```

Optional flags:

- `--max-per-sender N` (default 40): per raw sender address, only the newest N received
  messages (by the XML `date`, undated messages after dated ones, in file order) reach review.
  The date is used for ranking only and is never written. Counts are printed on screen only.
- `--business-only` / `--no-business-only` (default ON): skips person-to-person messages, i.e.
  senders that are phone numbers. The number skipped is printed on screen. Use
  `--no-business-only` only if a person-to-person message is needed; it then needs the same review.

Alphanumeric DLT sender IDs keep their structure through scrubbing, including the optional
`-P`/`-S`/`-T`/`-G` suffix (for example `AX-HDFCBK-S`), so the suffix reaches the exported `sender` column.

Use the project's environment (pandas/pyarrow). Nothing uploads, loads a model, or
labels automatically. Output must be outside every git checkout, must not exist,
and is created with owner-only permissions. Choose a local folder that is not cloud
synced. The original backup is never modified. Only manually approved rows are written;
skipping/finishing excludes unapproved rows. No raw text, phone/contact metadata,
original IDs, exact timestamps, paths, hashes or reversible mapping are exported.

Supported XML: uncompressed UTF8 `<smses><sms body="..." address="..." type="1" .../>`.
Only received SMS (`type=1`); MMS/RCS parts, attachments, calls and sent messages are
excluded. [SyncTech's official format](https://www.synctech.com.au/sms-backup-restore/fields-in-xml-backup-files/)
was checked on 2026-11-01. SyncTech documents XML, not JSON. JSON compatibility accepts
an array, or an `sms`/`messages` array of objects with the same body/address/type/contact_name
fields; this is a conversion format, not a claim about native app JSON export.
DTD/entities and unsupported encodings are rejected. Backups over256MiB must be split locally.

Typed placeholders remove known/contact names, greeting names, phones, account/card
digits, amounts, contextual OTPs, UPI IDs, email and remaining digits/references/dates.
URL credentials are removed and paths/query/fragment replaced; domain and message
structure remain. Brand terms and alphanumeric sender structure remain where recognised.
Supply all personal names through the local names file. **Arbitrary names and identifying
subdomains/free text cannot be reliably detected by regular expressions**: each screen
requires a human privacy check, with text replacement available before confirmation.
Scrubbing is a review aid, not guaranteed anonymisation. Numeric redaction can remove
semantic detail; review whether the remaining structure is useful before retaining it.

Labels: benign_transactional, benign_personal, promotion, scam, unclear. Nothing here
turns collected SMS into independent evidence automatically: consent, freeze date,
near-duplicate/campaign checks and a never-tune validation protocol still need separate
approval/design. Never tune rules on a validation collection.

**Training use (amended 2026-11-07, approved by Maadhav):** reviewed data may be used for training only under the
pre-registered donor split in `docs/INBOX_RETRAIN_PLAN.md`. At least one donor's data is a frozen test set, never used
for training, threshold choice or tuning. Every donor signs `docs/INBOX_CONSENT.md` first. Approve business/service
messages and received scams only, and skip person-to-person messages.
