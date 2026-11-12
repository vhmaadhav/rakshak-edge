# Bharat AI-SoC Challenge 2026-27 (PS3 on-device cyber safety): what it requires, what it rewards, and what winners show

Research date: 2026-11-06. Days to Phase 1 deadline: 17.

Primary official source: the Arm education page for the 2026-27 edition, https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html (fetched via an extraction model; quoted text should be re-checked against the live page before any rule is relied on).

## Key question 1: Official eligibility, PS3 text, deliverables, judging, platform rules, Phase 2 hardware

### Takeaway
The 2026-27 PS3 is broader than Rakshak's current scope: it asks for network-traffic intrusion classification and remediation advice, not only SMS and URL scams. Judging weights are not published on the 2026-27 page. Platform rules permit Pi 5 (CPU-only) and require a Llama-family model plus ExecuTorch, which Rakshak already uses.

### Cited Findings
- Edition, dates: launch 19 Sep 2026; registration closed 19 Oct 2026 (per the page, so registration is closed and Rakshak's entry is already in); Phase 1 ("Open Qualifying") deadline 23 Nov 2026; Phase 1 shortlist results mid-Dec 2026; Phase 2 (PoC development) deadline 5 Feb 2027; finals and winners 15-16 Feb 2027 — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Eligibility: participants must be Indian nationals; student teams are 1-3 members from one college, nominated by that college, with a Team Leader and a Faculty Mentor; professional teams come from one organisation; a team may change composition or problem statement once within 15 days of registration; a team may enter one problem statement only — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- PS3 text (on the page): "On-device threat, phishing and scam detection (Cybersecurity & Digital Safety). Teams build a system that classifies network traffic for intrusions, detects phishing URLs and scam SMS, explains threats in plain language, and recommends remediation, all without cloud dependence." — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Technology rule: "A model from the Llama family must be used for the advisory or reasoning component." ExecuTorch is required across problem statements — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Permitted platforms for PS3: Arm-based Android smartphones, Raspberry Pi 5 (CPU-only), Raspberry Pi 5 with AI HAT+ or AI HAT+ 2, and Arm-based Linux devices. If a Hailo accelerator is used, "These are separate pipelines and should be clearly documented." — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Deliverables: source code; application (or interface); demonstration video; technical report. No page limits were specified. Projects must be hosted on GitHub and submitted through the challenge submission form — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Judging: "Judging criteria and weightings are not specified" on the 2026-27 page. Each problem statement has one winner, chosen after "the final live virtual demo and jury Q&A", at "the sole discretion of the challenge organisers" — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Prizes: a cash award from IIT Delhi per problem statement (one winner each); certificates for all members of teams that submit completed projects; finalists receive programme mementos — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html). Cash amount for 2026-27 not stated (UNVERIFIED).
- Support: support@armbharatchallenge.com (general); tech-queries@armbharatchallenge.com for technical questions, with the problem statement number in the subject line — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Hardware: the 2026-27 page does not say whether a board is supplied. In the first edition FAQ, organisers "do not supply hardware"; participants work with their college and mentors — [Arm first-edition page](https://arm-education.github.io/Arm-Developer-Labs/Challenge_Page.html) (the first-edition page was fetched and returned 403; the first-edition details below come from the 2026 FAQ extraction from [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html) and the separate first-edition extraction described under Key question 3).
- Mentoring: virtual mentoring by industry and academic experts; each problem statement has a mentoring session video; no schedule given on the 2026-27 page — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)

### Inferences
- The PS3 text asks for four things Rakshak's current write-up may not show: network-traffic intrusion classification, plain-language threat explanation, remediation recommendations, and operation without cloud. The first and last are the gaps to check. Rakshak covers SMS and URL, so either (a) add a network-traffic path, or (b) state explicitly in the report which PS3 components are in scope and why. Option (b) is the minimum; the judging rubric is unknown, so the safer choice is to name every PS3 component in the report, marking any not built as out of scope.
- The Llama requirement is met by Llama 3.2 1B as the explanation layer. The ExecuTorch requirement is met. The Pi 5 CPU-only path is permitted, so the current hardware choice is compliant.
- Because judging is at the organisers' discretion and weights are unpublished, the generic criteria in Key question 3 are the best available proxy.

### Gaps
- The exact PS3 wording must be confirmed on the live page or the official PS document. The page was read through an extraction model. The "network traffic" text may be a summary artefact (UNVERIFIED as exact wording).
- No official judging weights for 2026-27 (UNVERIFIED).
- No page limits for the technical report (UNVERIFIED).
- Phase 2 hardware provision for 2026-27: not found (UNVERIFIED).
- Whether the 2026-27 hardware list is the same as the first edition: not verified.

## Key question 2: Prior editions and sibling challenges (winners, benchmarks, demos)

### Takeaway
The first edition (2026) published winners and projects, but the winning write-ups do not report latency, accuracy, or energy numbers. Public evidence does not show what judges reward beyond the FAQ list. The closest on-device scam comparator (Google DeepMind with Singapore GovTech) publishes no metrics either, so measured numbers are a differentiator rather than a baseline.

### Cited Findings
- First edition: organised by Arm, MeitY C2S and IIT Delhi; reports say over 3,000 students in 1,238 teams, and 340 project submissions (the two figures are from different sources and conflict) — [Scribd brochure summary via search](https://www.scribd.com/document/980049641/Arm-Ind-hied-Event-pun-Brochure-Bharat-ai-soc-sc-004); [Deccan Herald](https://www.deccanherald.com/india/karnataka/mangaluru/nitk-students-secure-first-place-in-bharat-ai-soc-student-challenge-2-4006815)
- First-edition winners listed on the Arm page: PS1 Spark (Heritage Institute of Technology, Kolkata); PS2 Velocity Gate (NIT Surathkal); PS3 VigiaSENSE (NIT Rourkela); PS4 BhashaBridge (Chennai Institute of Technology); PS5 Ad Astra (BV Raju Institute of Technology, Narsapur). These PS3 entries were a road-anomaly problem, not the 2026-27 PS3 — [Arm first-edition extraction](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html) (see the Gaps note on which page shows the first edition)
- NITK winning project: "Touchless HCI for Media Control Using Hand Gestures on NVIDIA Jetson Nano"; uses MediaPipe hand landmarks; "Latency, accuracy, and FPS: Not reported"; judging criteria and test methodology not described — [Upayuktha article](https://english.upayuktha.com/2026/05/nitk-surathkal-students-win-national.html)
- First-edition performance targets per problem statement (from the first-edition extraction): PS1 offline Hindi voice "Sub-2-second response time per command"; PS2 ">90% gesture recognition accuracy", "<200 ms" latency, "≥15 FPS"; PS3 road anomaly "≥5 FPS near-real-time inference", "high precision"; PS4 "No cloud inference permitted" with near real-time latency; PS5 "Minimum 2× speedup" over CPU — [Arm first-edition extraction](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- First-edition FAQ judging aspects that judges "may look into" (no weights): understanding of the problem and alignment to programme objective; technical correctness and implementation quality; novelty; use of Arm tools and their criticality; performance and optimisation; documentation quality and completeness; demo and presentation — [Arm first-edition extraction](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- First-edition prizes: top-5 teams receive INR 1 lakh each (FAQ) — same source
- Comparator, on-device scam detection: Google DeepMind with Singapore GovTech (Anti-Scam Pango, Android beta). Two-stage design: a lightweight classifier screens messages, then a fine-tuned Gemma model analyses a conversation snapshot to confirm or discard. Runs on-device; training used anonymised scam reports. "Tested against synthetic dialogues and local corpora"; no metrics reported; battery efficiency called "a critical area of concern" with no measurements — [DeepMind blog](https://deepmind.google/blog/on-device-ai-scam-detection-mddi/) (page date from search metadata, UNVERIFIED)
- Silicon Labs Edge Intelligence Challenge (India, 2025): judging criteria not found; shortlisted teams get INR 10,000 and an edge-ML kit; application deadline 30 Sep 2025 — [Tribune](https://www.tribuneindia.com/news/business/silicon-labs-launches-edge-intelligence-challenge-to-power-indias-tech-innovation-journey) and [Silicon Labs challenge page](https://www.silabs.com/applications/artificial-intelligence-machine-learning/edge-intelligence-challenge); no winner write-ups found.

### Inferences
- No public winner write-up in these sources reports p50/p95 latency, energy per inference, RSS, or robustness tests. This does not prove judges ignore them, but it does mean the bar is not known. Measured numbers are still the cheapest way to score on "technical correctness and implementation quality" and "performance and optimisation".
- The first-edition winners' projects were demos with hardware and a model, not benchmark reports. A working, reproducible, offline demo with clear numbers is likely to rank well against them.
- The DeepMind design resembles Rakshak's two-stage shape (classifier gate, then LLM explainer). The report should present that similarity honestly and state the metrics Rakshak has that DeepMind did not publish.

### Gaps
- Judging weights and winning-report templates for any edge-AI challenge were not found.
- The first-edition winners' reports (beyond the NITK article) were not accessible; the Deccan Herald fetch returned 403.
- No winner benchmark tables were found for any Arm, Qualcomm, Samsung, MediaTek, or tinyML challenge in this pass.
- Smart India Hackathon, Google/Meta Llama impact challenges, and tinyML/MLPerf reporting rules were not searched in this pass.

## Key question 3: Technical report structure and how to present limitations

### Takeaway
No official report template exists for this challenge in the sources found. The first-edition FAQ lists the criteria judges may use, so the report should cover each one explicitly. The usual edge-AI report elements (latency percentiles, energy, memory, false-positive rates with denominators, in-sample vs independent evaluation) are standard practice, but no official source confirmed them for this challenge.

### Cited Findings
- Judging aspects listed (no weights): problem understanding and alignment; technical correctness and implementation quality; novelty; use of Arm tools; performance and optimisation; documentation quality and completeness; demo and presentation — [Arm first-edition extraction](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- The organisers' Phase 1 deliverables include a "technical report" with no page limit given — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)

### Inferences
- Map the report to the seven criteria above, one section each, so a judge can tick them off. Use Arm tool usage explicitly (ExecuTorch, KleidiAI or XNNPACK if used, the Arm-based Pi 5 target).
- For limitations: state the Rakshak weak point (external false-positive rate on legit Indian bank/UPI SMS) with its denominator, and label which numbers are in-sample (the offline campaign F1 0.977) versus independent. Unlabelled in-sample numbers read as overclaims to a technical judge.
- Include a measured table on the Pi 5: p50 and p95 end-to-end latency, time to first token if the LLM is run, peak RSS, and power/thermal telemetry. Mark any value not measured as unmeasured.
- Include a PS3 scope table: each PS3 component named in the official text, with status (built, partial, not built, out of scope).
- Include the reproduction path (scripts, seed, model hashes, dataset split) since the challenge emphasises GitHub-hosted source.

### Gaps
- No official template, page limit, or required metric list found.
- Specific edge-AI reporting norms (MLPerf Tiny/EEMBC energy methodology, p95 reporting conventions) were not verified in this pass.

## Key question 4: Demo video best practice

### Takeaway
No official demo-video guidance was found for the challenge. The first-edition criteria name "demo and presentation" and the finals use a live virtual demo with jury Q&A, so the video should be built to show the same things as the live demo.

### Cited Findings
- Finals use "the final live virtual demo and jury Q&A" — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Demo video is a required deliverable in Phase 1 — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- No length limit was stated for the video (UNVERIFIED as absent from official text; the page did not state one).

### Inferences (not cited; general practice)
- Keep it short, enough to cover the PS3 flow end to end, and show offline operation on camera: airplane mode, Wi-Fi off, and the Pi's network interface down or the radio disabled.
- Show a telemetry overlay (latency, power, temperature) so the numbers are visibly from the device, not a slide.
- Show one legit bank or UPI SMS and one scam, including a false-positive case, and show the explanation text the Llama layer produces, so the judge sees the decision and the verifier output.

### Gaps
- No length or format rule found. Length and overlay guidance above are general inference, not sourced.

## Key question 5: Indian-context requirements (languages, CERT-In/TRAI/RBI, Sanchar Saathi/Chakshu)

### Takeaway
No language requirement appears in the 2026-27 PS3 text. Chakshu is a real DoT reporting channel that Rakshak could point users to. Alignment with CERT-In, TRAI, and RBI guidance was not checked in this pass and is unverified.

### Cited Findings
- Chakshu is a DoT reporting feature on the Sanchar Saathi portal, launched in March 2024, for suspected fraud communication on calls, SMS, and WhatsApp. Reports trigger re-verification of the numbers, and failing numbers are disconnected. It is not for financial fraud or cybercrime; for money already lost, victims use the cybercrime portal or helpline 1930 — [Business Standard](https://business-standard.com/industry/news/dot-s-new-chaksu-new-portal-to-allow-citizens-to-report-fraudulent-calls-124030401000_1.html); [Outlook Money](https://business.outlookindia.com/ampstories/personal-finance/retirement/7-things-to-know-about-chakshu-portal); [Factly](https://factly.in/explainer-government-launches-chakshu-to-report-sms-call-or-whatsapp-messages-attempting-to-commit-fraud/amp)
- The Centre asked smartphone makers to preinstall a state-developed app for reporting fraud calls and messages — [Tech2 / search summary](https://www.tech2.com/cybersecurity/lost-your-phone-use-chakshu-portal-to-report-online-text-ws-l-10090463.html) (single secondary source, treat as UNVERIFIED for the detail)
- No language requirement found in the 2026-27 PS3 text. The first-edition PS1 required Hindi voice (offline Hindi commands), which suggests Indian-language support is valued in this challenge family — [Arm first-edition extraction](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)

### Inferences
- A "Report to Chakshu" action in the app (a deep link or user-guided step, not an API) is cheap and reads as Indian-context alignment. No public API for Chakshu was found; any integration claim must be checked against DoT sources first.
- Hindi and one regional language in the explainer would be a differentiator, but the 2026-27 PS3 text does not require it (UNVERIFIED that any language rule exists).

### Gaps
- Alignment with CERT-In, RBI, and TRAI (DLT) fraud guidance: not searched in this pass.
- Chakshu API or data-sharing: not found; the claim that Sanchar Saathi reads call logs and SMS with consent came from one aggregator and is UNVERIFIED.
- Hindi or regional-language requirement for 2026-27: UNVERIFIED.

## Key question 6: Where the evidence is thin (remaining 17 days)

### Takeaway
The official 2026-27 page settles the dates, eligibility, deliverables, platform rules, and PS3 text (with the caveats above). Judging weights, page limits, and Phase 2 hardware remain unknown. The highest-value remaining actions are checking the live PS3 text and emailing the organisers, and making the report and demo cover every criterion.

### Cited Findings
- Organiser contact: tech-queries@armbharatchallenge.com (subject: PS number) — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html)
- Team composition or problem statement change is allowed once within 15 days of registration, with approval at organisers' discretion — [Arm 2026-27 page](https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html). The 15-day window from registration (closed 19 Oct) has passed; any change now needs organiser approval.

### Inferences
- Email the organisers now with two questions: whether the PS3 scope includes network-traffic classification, and whether judging weights exist. This is a support query, not a public post, so it does not conflict with the no-posting rule. It should be sent only with Maadhav's approval, since it is an external message.

### Gaps
- Judging weights, report page limit, Phase 2 hardware, PS3 exact wording, Indian-regulator alignment, and edge-AI benchmark reporting norms are unverified.
- Sources conflict on the first-edition participant and submission counts (1,238 teams vs 340 submissions).
- The first-edition page is labelled "2026-27" in one extraction and dated January-March 2026 in another, so the edition labels are ambiguous. Confirm which page is which before quoting it in the report.
