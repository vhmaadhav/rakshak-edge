# Consent to donate SMS messages: Rakshak-Edge

Plan: `docs/INBOX_RETRAIN_PLAN.md`. One copy per donor. Sign it before exporting anything. Keep the signed copy
outside git (for example a signed PDF in your own private storage). Only the fact that consent was given (name, date,
role) is recorded in the repo.

**Project.** Rakshak-Edge, an on-device scam-SMS detector for the Bharat AI-SoC Challenge 2026-27 (PS3).
Data owner and contact: Maadhav (vhmaadhav@hotmail.com).

**What I am donating.** Received SMS messages from my phone that **I personally review and approve** with the
project's local review tool (`tools/inbox_export`): messages from businesses and services (bank, UPI, OTP,
telecom, delivery, government, promotions) and scam messages I received. I will **not** approve
person-to-person messages, and so no message written by another private individual is shared.

**What leaves my phone or laptop.** Only the reviewed file the tool writes. In it, names, phone numbers, account and card
digits, amounts, OTPs, UPI IDs, emails, other numbers, references, dates and URL paths are replaced by
placeholders. Business sender IDs (for example `AX-HDFCBK-S`) are kept. No timestamps, phone contacts or original
message ids are included. My raw backup never leaves my device.

**How it is used.**
- My role: ☐ training donor (train and threshold validation) ☐ test donor (frozen test only, never used for tuning).
- Stored in a private Kaggle dataset (`maadhavvh/rakshak-inbox`) owned by Maadhav, used only by private training
  jobs. It is never published, never committed to any git repository, and never shared with judges or third parties.
- Project code and AI coding assistants handle only aggregate counts and scores. Message text is not printed or
  quoted in logs, reports or chats.
- A small model trained partly on this data (an ONNX file of about 400 KB) may be published as part of the challenge
  submission. It does not contain the messages. Small models like this are unlikely to memorise text, but this
  cannot be fully ruled out.

**My rights.** I can withdraw at any time by telling Maadhav. My file is then deleted from the dataset within 7
days, and any model trained on it is not used further and is retrained without it on request. All donated files are
deleted after the challenge's final event (16 Feb 2027) unless I give consent again in writing.

**I confirm** that the messages I approve were received on my own phone number, and that I have reviewed every
approved message for personal information.

Name: ____________________  Role: ____________________  Date: ______________  Signature: ____________________

---

Consent register (in the repo; no signatures or contact details):

| Donor | Role | Consent date | Withdrawn |
|---|---|---|---|
| Maadhav | training donor | | |
| Sachin | test donor | | |
