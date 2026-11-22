package org.rakshak.edge.analysis

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Port of src/rakshak/rules.py: the evidence rules over a canonicalised message, in the same
 * emission order, so the ids E1..En match the Python ids for the same text.
 *
 * Patterns are the Python patterns, unchanged, compiled through [rxOf] (see PyRegex.kt).
 * Spans are UTF-16 offsets; the parity test compares ids, types, weights and matched text.
 */

/** The 16 side-feature types, in the order of rules.EVIDENCE_TYPES. */
val EVIDENCE_TYPES: List<String> = listOf(
    "credential_request",
    "upi_collect_or_pay",
    "kyc_block_urgency",
    "account_suspension",
    "prize_lottery",
    "apk_install",
    "remote_access_app",
    "electricity_disconnect",
    "parcel_customs",
    "job_task_scam",
    "investment_returns",
    "digital_arrest",
    "brand_domain_mismatch",
    "shortened_url",
    "suspicious_url",
    "urgency",
)

data class Evidence(
    val id: String,
    val type: String,
    val start: Int,
    val end: Int,
    val text: String,
    val weight: Double,
)

internal data class RawHit(val type: String, val start: Int, val end: Int, val text: String, val weight: Double)

internal data class Span(val start: Int, val end: Int)

private fun overlaps(a: Span, b: Span): Boolean = a.start < b.end && b.start < a.end

private const val SENTENCE_MAX = 400
private val SENTENCE_BREAKS = charArrayOf('.', '!', '?', '\n', '|', '।')
private val SENTENCE_END = rxOf("""(?<![0-9])[.!?;|](?=\s|$)|\n|\r|।""")

// ---------------------------------------------------------------------------
// Credential request: needs negation handling so "Do not share your OTP" does not fire.
// ---------------------------------------------------------------------------

private const val CRED_TERMS =
    "(?:otp|o\\.t\\.p|one[ -]time password|mpin|upi pin|atm pin|pin|cvv|cvc|password|" +
        "passcode|pass code|verification code|" +
        "पासवर्ड|ओटीपी|पिन|सीवीवी|वेरिफिकेशन कोड)"
private const val CRED_VERBS =
    "(?:share|send|give|tell|provide|submit|reply|forward|enter|bata\\w*|btao|bhej\\w*|" +
        "dij\\w*|dein|bolo|share kar\\w*|" +
        "बताएं|बताइए|बताइये|बताओ|बताएँ|भेजें|भेजो|भेजिए|भेजिये|दें|दीजिए|दीजिये|शेयर|" +
        "शेयर करें|शेयर करो|भेजना|बताना)"

private val CRED_PAIR_RX = rxOf(
    "(?:\\b$CRED_VERBS\\b.{0,30}\\b$CRED_TERMS\\b)|(?:\\b$CRED_TERMS\\b.{0,30}\\b$CRED_VERBS\\b)",
    ci = true,
)
private val CRED_VERB_ONLY_RX = rxOf("\\b$CRED_VERBS\\b", ci = true)
private val NEGATION_RX = rxOf(
    "\\b(?:not|never|don'?t|dont|do not|nahi|nahin|nhi|mat|kabhi|kisi ko na|na)\\b|" +
        "(?<![\\p{L}\\p{M}])(?:नहीं|न|मत|कभी)(?![\\p{L}\\p{M}])",
    ci = true,
)

private fun credentialRequest(canon: Canon): List<RawHit> {
    val text = canon.normalized
    val out = ArrayList<RawHit>()
    for (m in CRED_PAIR_RX.findAll(text)) {
        val matched = m.value
        val mStart = m.range.first
        // Every verb in the match is a candidate. A verb negated in its own sentence does not count.
        for (verb in CRED_VERB_ONLY_RX.findAll(matched)) {
            val verbPos = mStart + verb.range.first
            var lastBreak = -1
            for (c in SENTENCE_BREAKS) lastBreak = max(lastBreak, text.lastIndexOf(c, verbPos - 1))
            val sentenceStart = lastBreak + 1
            val lead = text.substring(sentenceStart, verbPos).takeLast(30)
            if (NEGATION_RX.containsMatchIn(lead)) continue
            out.add(RawHit("credential_request", mStart, m.range.last + 1, matched, 0.95))
            break
        }
    }
    return out
}

// ---------------------------------------------------------------------------
// Sentence-scoped pairing (subject and trigger in one sentence, at most `gap` apart).
// ---------------------------------------------------------------------------

private fun sentenceSpans(text: String): List<Span> {
    val out = ArrayList<Span>()
    var start = 0
    for (m in SENTENCE_END.findAll(text)) {
        val ms = m.range.first
        if (ms > start) out.add(Span(start, ms))
        start = m.range.last + 1
    }
    if (start < text.length) out.add(Span(start, text.length))
    return out
}

private fun sentencePairs(
    text: String,
    subject: Regex,
    trigger: Regex,
    guard: Regex?,
    gap: Int,
): List<Pair<Span, String>> {
    val out = ArrayList<Pair<Span, String>>()
    for (sp in sentenceSpans(text)) {
        if (sp.end - sp.start > SENTENCE_MAX) continue
        val sentence = text.substring(sp.start, sp.end)
        if (guard != null && guard.containsMatchIn(sentence)) continue
        val subs = subject.findAll(sentence).toList()
        val trigs = trigger.findAll(sentence).toList()
        if (subs.isEmpty() || trigs.isEmpty()) continue
        var done = false
        for (sub in subs) {
            for (trig in trigs) {
                if (abs(trig.range.first - sub.range.first) > gap) continue
                val lo = min(sub.range.first, trig.range.first)
                val hi = max(sub.range.last + 1, trig.range.last + 1)
                out.add(Span(sp.start + lo, sp.start + hi) to sentence.substring(lo, hi))
                done = true
                break
            }
            if (done) break
        }
    }
    return out
}

private fun patternRule(type: String, pattern: String, weight: Double): (Canon) -> List<RawHit> {
    val rx = rxOf(pattern, ci = true)
    return { canon ->
        rx.findAll(canon.normalized).map {
            RawHit(type, it.range.first, it.range.last + 1, it.value, weight)
        }.toList()
    }
}

private fun pairedRule(
    type: String,
    pattern: String,
    weight: Double,
    subject: String,
    trigger: String,
    guard: String? = null,
    gap: Int = 120,
): (Canon) -> List<RawHit> {
    val rx = rxOf(pattern, ci = true)
    val subj = rxOf(subject, ci = true)
    val trig = rxOf(trigger, ci = true)
    val grd = guard?.let { rxOf(it, ci = true) }
    return { canon ->
        val text = canon.normalized
        val flat = rx.findAll(text).map { Span(it.range.first, it.range.last + 1) }.toList()
        val out = ArrayList<RawHit>()
        for (s in flat) out.add(RawHit(type, s.start, s.end, text.substring(s.start, s.end), weight))
        for ((span, matched) in sentencePairs(text, subj, trig, grd, gap)) {
            if (flat.any { overlaps(span, it) }) continue
            out.add(RawHit(type, span.start, span.end, matched, weight))
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Pattern rules (Python _RULES, in the same order).
// ---------------------------------------------------------------------------

private val UPI_PATTERN = """(?:\bupi (?:collect|collection|request)\b|\bcollect request\b|""" +
    """\b(?:approve|accept|allow|authori[sz]e) (?:the |this |a )?""" +
    """(?:payment|request|collect\w*|mandate)\b|""" +
    """\b(?:payment|request|collect|mandate) (?:ko |ki |ka )?(?:approve|accept|allow)\w*\b|""" +
    """\b(?:approve|accept|allow)\w* (?:karo|kar do|kijiye|kariye|karein|karen|kare)\b|""" +
    """\bscan (?:the |this |any )?(?:qr|code)\b.{0,30}\b(?:receive|get|refund|cashback|money)\b|""" +
    """\b(?:receive|get) (?:money|refund|cashback|payment)\b.{0,40}\b(?:upi|pin|scan)\b|""" +
    """\bupi ?pin\b.{0,40}\b(?:daal\w*|dal\w*|enter|dale|dalen|dalein|dalo)\b|""" +
    """\bpaise (?:lene|paane|pane|receive|prapt|milne)\b.{0,40}\b(?:upi|pin|scan|qr|collect)\b|""" +
    """\b(?:upi|pin|scan|qr)\b.{0,40}\bpaise (?:lene|paane|pane|receive|milenge|milega)\b|""" +
    """\bpay\s*(?:rs\.?|₹)?\s*\d+\b.{0,30}\b(?:verify|activate|unlock|confirm)\b|""" +
    """पेमेंट.{0,20}अप्रूव|यूपीआई.{0,20}(?:कलेक्ट|रिक्वेस्ट)|""" +
    """यूपीआई\s*पिन.{0,30}(?:डालें|डालिए|डालो|दर्ज)|""" +
    """पैसे\s+(?:लेने|प्राप्त|पाने).{0,30}(?:यूपीआई|पिन|स्कैन|क्यूआर))"""

private val KYC_PATTERN = """(?:\bkyc\b.{0,80}\b(?:block\w*|suspen[sd]\w*|expir\w*|deactivat\w*|pending|""" +
    """urgent\w*|immediately|today|within \d+ ?(?:hours?|days?)|last date|freez\w*|hold)\b|""" +
    """\b(?:block\w*|suspen[sd]\w*|expir\w*|deactivat\w*|pending|urgent\w*|immediately|""" +
    """within \d+ ?(?:hours?|days?)|last date|freez\w*)\b.{0,80}\bkyc\b|""" +
    """केवाईसी.{0,60}(?:बंद|ब्लॉक|निलंबित|समाप्त|आज|तुरंत|जल्द|जल्दी|वरना|नहीं तो)|""" +
    """(?:बंद|ब्लॉक|निलंबित|समाप्त|तुरंत|जल्द|जल्दी).{0,60}केवाईसी|""" +
    """\bkyc\b.{0,40}\b(?:band|block|jald|jaldi|turant|aaj|bandh)\w*|""" +
    """\bkyc\b.{0,60}\b(?:warna|varna|vrna|nahi to|nahi toh|jayega|jaayega|jaega|hoga|hogi|""" +
    """ho jayega|ho jaayega)\b|""" +
    """\b(?:warna|varna|nahi to|nahi toh)\b.{0,60}\bkyc\b)"""

private val ACCOUNT_PATTERN = """(?:\b(?:a/?c|acc|account|khata|debit card|credit card|card|sim|wallet|upi id)\b""" +
    """.{0,40}\b(?:suspen[sd]\w*|block\w*|deactivat\w*|frozen|freez\w*|locked|on hold|""" +
    """will be closed|band ho\w*|block ho\w*|band kar\w*|band hoga|band hogi|""" +
    """ruk ja\w*|bandh\w*|lock ho\w*|freeze ho\w*)\b|""" +
    """\b(?:suspen[sd]\w*|block\w*|deactivat\w*|frozen|freez\w*|locked|band ho\w*|""" +
    """block ho\w*|band hoga|band hogi|lock ho\w*)\b.{0,40}""" +
    """\b(?:a/?c|acc|account|khata|card|sim|wallet)\b|""" +
    """खाता.{0,30}(?:बंद|ब्लॉक|निलंबित|फ्रीज)|(?:बंद|ब्लॉक|निलंबित|फ्रीज).{0,30}खाता)"""

private val PRIZE_PATTERN = """(?:\blottery\b|\blucky draw\b|\bjackpot\b|\byou (?:have )?won\b|\bwinner\b|""" +
    """\bcongratulations?\b.{0,40}\b(?:prize|won|reward|gift|cash)\b|\bprizes?\b|""" +
    """\b(?:kbc|kaun banega)\b|\binam\b|\binaam\b|\bjeet\w*\b|\bjeeta\b|""" +
    """\bwin\w*\b[^.!?\n]{0,25}\b(?:priz\w*|gold|vouchers?|jackpot)\b|""" +
    """लॉटरी|इनाम|जीते|जीता|जीती|जीतें|जीतिए|जीतेंगे|पुरस्कार|बधाई.{0,20}जीत)"""

private val APK_PATTERN = """(?:\.apk\b|\bapk\b|\binstall (?:the |this |our )?(?:app|application|apk|file)\b|""" +
    """\bdownload (?:the |this )?(?:app|application|apk)\b|""" +
    """\bapp (?:download|install)\w* (?:karo|kare\w*|karein)\b|""" +
    """\bइंस्टॉल\b|\bडाउनलोड\b)"""

private val REMOTE_PATTERN = """(?:\banydesk\b|\bteam ?viewer\b|\bquick ?support\b|\brustdesk\b|""" +
    """\bultraviewer\b|\bscreen ?share\b|\bremote (?:access|support|control)\b|\bany ?desk\b)"""

private val ELECTRICITY_PATTERN = """(?:\b(?:electricity|power|bijli|bijlee|meter|connection)\b.{0,60}""" +
    """\b(?:disconnect\w*|cut|cut off|band|kat\w*|kaat\w*|suspen[sd]\w*|block\w*|unpaid|overdue)\b|""" +
    """\b(?:disconnect\w*|cut|band|kat\w*|kaat\w*|suspen[sd]\w*)\b.{0,40}""" +
    """\b(?:electricity|power|bijli|bijlee|meter|connection)\b|""" +
    """बिजली.{0,40}(?:काट|बंद|कट|डिस्कनेक्ट)|""" +
    """कनेक्शन.{0,40}(?:काट|कट|बंद|डिस्कनेक्ट))"""

private val PARCEL_PATTERN = """(?:\b(?:parcel|package|courier|shipment|consignment|customs|duty|saman)\b.{0,50}""" +
    """\b(?:pay\w*|fee|charges?|customs|held|hold|pending|clear\w*|release|re-?deliver\w*|""" +
    """rs\.?|₹|\d+|shulk\w*|kastam|kasam|phas\w*|ruka|ruka|rok\w*|jama)\b|""" +
    """\b(?:pay\w*|fee|charges?|customs|held|clear\w*|release|shulk\w*|kastam|jama)\b.{0,50}""" +
    """\b(?:parcel|package|courier|shipment|consignment|saman)\b|""" +
    """\b(?:shulk\w*|kastam)\b.{0,50}\b(?:jama\w*|pay\w*|bharo|bhejo|karo|karna|de\w*)\b|""" +
    """पार्सल.{0,40}(?:शुल्क|पेमेंट|रोक|रुका|रुक|कस्टम)|""" +
    """(?:शुल्क|कस्टम).{0,40}(?:जमा|पेमेंट|भुगतान))"""

private val JOB_PATTERN = """(?:\bpart[- ]?time\b.{0,30}\b(?:job|work|income|earn\w*)\b|""" +
    """\bwork from home\b.{0,40}\b(?:earn\w*|income|salary|per day|daily)\b|""" +
    """\b(?:like|subscribe)\b.{0,30}\b(?:youtube|video|channel|task|review)\b|""" +
    """\b(?:hotel|product)\s+rating\b|""" +
    """\bearn\s*(?:rs\.?|₹)?\s*\d+\b.{0,30}\b(?:per|a|each|daily|/)\s*(?:task|day|hour)\b|""" +
    """\btask\b.{0,30}\b(?:commission|earn\w*|payment|reward)\b|""" +
    """\bjob offer\b|\bghar baithe\b.{0,30}\b(?:kamao|kamayi|income|earn\w*)\b|""" +
    """\b(?:roz|daily|rozana|ghar baithe|ghar se)\b.{0,30}\b(?:kamao|kamayein|kamaye|""" +
    """kamaiye|kamai|kamayi|earn\w*|income)\b|""" +
    """\b(?:kamao|kamayein|kamaye|kamaiye|kamayi|kamai)\b.{0,40}""" +
    """\b(?:roz|daily|rozana|ghante|ghanta|paise|paisa|rupaye|rs|₹)\b|""" +
    """\bonline (?:kaam|job|task)\b|""" +
    """घर बैठे.{0,30}(?:कमाए|कमाई|कमाएं)|पार्ट.?टाइम.{0,20}(?:नौकरी|काम))"""

private val INVESTMENT_PATTERN = """(?:\b(?:guaranteed|assured|fixed)\b.{0,20}\b(?:return|profit|income|returns)\b|""" +
    """\bdouble your\b|\b\d+\s?%\s*(?:daily|per day|monthly|per month|weekly|return)\b|""" +
    """\b(?:crypto|bitcoin|usdt|forex)\b.{0,40}\b(?:profit|return|invest\w*|trading|tips?|group)\b|""" +
    """\btrading (?:group|tips?|signals?|course)\b|\bstock tips?\b|""" +
    """\binvest\w*\b.{0,30}\b(?:return|profit|%)\b|""" +
    """\b(?:invest\w*|nivesh\w*|share market|trading|crypto|stock)\b.{0,50}""" +
    """\b(?:munafa|munaafa|laabh|profit|dugna|double|guarantee\w*)\b|""" +
    """\b(?:munafa|munaafa|laabh|profit|dugna)\b.{0,40}""" +
    """\b(?:invest\w*|nivesh\w*|share|trading|group|guarantee\w*|daily|roz)\b|""" +
    """\b(?:paisa|paise|rupaye|money)\s+double\b|\bdouble\s+(?:paisa|paise|rupaye|money)\b|""" +
    """निवेश.{0,30}(?:गारंटी|मुनाफा|रिटर्न|दोगुना|डबल)|""" +
    """(?:दोगुना|डबल).{0,30}(?:पैसा|पैसे|रुपये))"""

private val DIGITAL_ARREST_PATTERN = """(?:\bdigital arrest\b|""" +
    """\b(?:cbi|police|ed officer|enforcement directorate|narcotics|cyber ?crime|cyber ?cell|""" +
    """crime branch|trai officer|customs officer|income tax (?:department|officer))\b.{0,60}""" +
    """\b(?:arrest\w*|warrant|case|fir|jail|court|seiz\w*|freez\w*|video call|verify|""" +
    """money laundering|illegal|complaint|notice|summon\w*|karwai|karvai|khilaf|darj\w*|kanoon\w*)\b|""" +
    """\b(?:arrest\w*|warrant|video call|money laundering|illegal sim)\b.{0,60}""" +
    """\b(?:cbi|police|ed|narcotics|cyber|trai|customs|crime branch)\b|""" +
    """\b(?:aapke|aapka|aapko|aap)\b.{0,40}\b(?:khilaf|case darj|warrant|giraftar\w*|gunah)\b|""" +
    """गिरफ्तार|डिजिटल अरेस्ट|(?:क्राइम ब्रांच|साइबर क्राइम).{0,60}""" +
    """(?:वारंट|केस|कार्रवाई|वीडियो कॉल|वेरिफाई|गिरफ्तार)|""" +
    """(?:खिलाफ|वारंट).{0,40}(?:केस|शिकायत|दर्ज|गिरफ्तार))"""

private val URGENCY_PATTERN = """(?:\b(?:turant|turnt|tatkal|jaldi|jaldee|abhi|aaj hi|aaj hee|urgent(?:ly)?|""" +
    """immediately|immediate action|right now|last chance|today only|expires? today|last date)\b|""" +
    """\b\d{1,2} ?(?:hours?|hrs?|ghante|ghanta|ghantey)\b|""" +
    """(?<![\p{L}\p{M}])(?:तुरंत|तुरन्त|जल्दी|जल्द|आज ही|\d{1,2} घंटे)(?![\p{L}\p{M}])|""" +
    """अभी\s+(?:करें|करो|कीजिए|जुड़ें|भेजें|क्लिक|रजिस्टर|कॉल|संपर्क|अपडेट|वेरिफ\w*|कराएं|करवाएं))"""

private val RULES: List<(Canon) -> List<RawHit>> = listOf(
    ::credentialRequest,
    patternRule("upi_collect_or_pay", UPI_PATTERN, 0.85),
    patternRule("kyc_block_urgency", KYC_PATTERN, 0.8),
    patternRule("account_suspension", ACCOUNT_PATTERN, 0.75),
    patternRule("prize_lottery", PRIZE_PATTERN, 0.8),
    patternRule("apk_install", APK_PATTERN, 0.9),
    patternRule("remote_access_app", REMOTE_PATTERN, 0.9),
    patternRule("electricity_disconnect", ELECTRICITY_PATTERN, 0.8),
    pairedRule(
        type = "parcel_customs",
        pattern = PARCEL_PATTERN,
        weight = 0.7,
        subject = """\b(?:parcel|package|courier|shipment|consignment|redelivery|post office|customs|""" +
            """dhl|fedex|bluedart|blue dart|aramex|shadowfax|usps|royal ?mail|an ?post|deutsche ?post)\b""",
        trigger = """\b(?:unpaid|outstanding|overdue|on hold|held|fee|fees|charges?|duty|duties|""" +
            """shipping (?:cost|fee|charges?)|clearance|redeliver\w*|reschedul\w*)\b""",
        guard = """\b(?:amount payable|amount due|total payable|cash on delivery|cod|""" +
            """payment (?:successful|received|confirmed))\b""",
    ),
    pairedRule(
        type = "job_task_scam",
        pattern = JOB_PATTERN,
        weight = 0.8,
        subject = """\b(?:job|jobs|vacanc\w*|hiring|hire|recruit\w*|part[- ]?time|work from home|""" +
            """wfh|internship|fresher)\b""",
        trigger = """\b(?:earn|earns|earning|salary|income|kamai|kamayi|kamao|kamayein)\b""",
        gap = 80,
    ),
    patternRule("investment_returns", INVESTMENT_PATTERN, 0.8),
    patternRule("digital_arrest", DIGITAL_ARREST_PATTERN, 0.9),
    // Bare authority mentions are weak on their own; news and benign reports mention police.
    patternRule("digital_arrest", """(?:\bcbi\b|\bpolice\b|\bcrime branch\b|सीबीआई|पुलिस)""", 0.5),
    // Generic urgency: weak on its own and never mapped to an action.
    patternRule("urgency", URGENCY_PATTERN, 0.4),
)

// ---------------------------------------------------------------------------
// Domain rules over the extracted URLs.
// ---------------------------------------------------------------------------

private fun locate(text: String, needle: String): Span {
    val idx = text.lowercase().indexOf(needle.lowercase())
    return if (idx < 0) Span(0, 0) else Span(idx, idx + needle.length)
}

private fun domainRules(canon: Canon): List<RawHit> {
    val text = canon.normalized
    val out = ArrayList<RawHit>()
    for (url in canon.urls) {
        val span = locate(text, url)
        if (brandDomainMismatch(text, url)) {
            out.add(RawHit("brand_domain_mismatch", span.start, span.end, url, 0.9))
        }
        if (registeredDomain(url) in SHORTENERS) {
            out.add(RawHit("shortened_url", span.start, span.end, url, 0.6))
        }
        val f = urlFeatures(url)
        if (f.tldRisky || f.punycode || f.ipLiteral || f.hasAtSign || f.brandHostMismatch ||
            f.suspiciousTokenCount >= 2 || f.nSubdomains >= 3
        ) {
            val weight = if (f.brandHostMismatch || f.punycode || f.ipLiteral) 0.9 else 0.7
            out.add(RawHit("suspicious_url", span.start, span.end, url, weight))
        }
    }
    return out
}

/** run_rules: every rule in order, then the domain rules. Ids are E1..En in emission order. */
fun runRules(canon: Canon): List<Evidence> {
    val raw = ArrayList<RawHit>()
    for (rule in RULES) raw.addAll(rule(canon))
    raw.addAll(domainRules(canon))
    return raw.mapIndexed { i, h -> Evidence("E${i + 1}", h.type, h.start, h.end, h.text, h.weight) }
}

/** rule_flags: 1.0 for every evidence type that fires, in EVIDENCE_TYPES order. */
fun ruleFlags(evidence: List<Evidence>): FloatArray {
    val fired = evidence.mapTo(HashSet()) { it.type }
    return FloatArray(EVIDENCE_TYPES.size) { i -> if (EVIDENCE_TYPES[i] in fired) 1f else 0f }
}
