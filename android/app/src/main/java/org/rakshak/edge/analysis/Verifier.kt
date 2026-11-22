package org.rakshak.edge.analysis

import java.util.regex.Pattern
import org.rakshak.edge.data.RuleData

/**
 * Port of the points-contract checks in src/rakshak/verifier.py, for one point at a time:
 * grounding (URLs, domains, phone numbers, amounts), the safe-claim check (HIGH and MEDIUM),
 * positive-instruction and injection checks, the lexicon keyword for the evidence type, and the
 * attribution guard G2. Also the deterministic templates that replace a rejected point.
 *
 * Not ported: the JSON parsing of model output (the app assembles points itself), the summary
 * check (the summary is always the template), the rank arm, and the ELA scoring.
 */

const val MAX_POINT = 160
const val MAX_SUMMARY = 280

val TYPE_PHRASE_EN: Map<String, String> = mapOf(
    "credential_request" to "asks for an OTP, PIN or password",
    "upi_collect_or_pay" to "uses a UPI collect or payment request",
    "kyc_block_urgency" to "uses KYC urgency or a block threat",
    "account_suspension" to "claims an account or card is suspended",
    "prize_lottery" to "claims a prize or lottery",
    "apk_install" to "asks for an app or APK installation",
    "remote_access_app" to "asks for a remote-access app",
    "electricity_disconnect" to "threatens electricity disconnection",
    "parcel_customs" to "asks for a parcel or customs fee",
    "job_task_scam" to "offers a task or job with easy earnings",
    "investment_returns" to "promises guaranteed investment returns",
    "digital_arrest" to "claims to be a police or government agency",
    "brand_domain_mismatch" to "links to a domain that does not match the brand named",
    "shortened_url" to "contains a shortened link",
    "suspicious_url" to "contains a suspicious link",
    "urgency" to "uses time pressure (turant, abhi, 24 hours)",
)

val TYPE_PHRASE_HINGLISH: Map<String, String> = mapOf(
    "credential_request" to "OTP, PIN ya password maangta hai",
    "upi_collect_or_pay" to "UPI collect ya payment request bhejta hai",
    "kyc_block_urgency" to "KYC ke naam par block ka dar dikhata hai",
    "account_suspension" to "account ya card band hone ka dawa karta hai",
    "prize_lottery" to "inaam ya lottery ka dawa karta hai",
    "apk_install" to "app ya APK installation maangta hai",
    "remote_access_app" to "remote access app maangta hai",
    "electricity_disconnect" to "bijli katne ki dhamki deta hai",
    "parcel_customs" to "parcel ya customs fee maangta hai",
    "job_task_scam" to "asaan kamai ka task ya job offer deta hai",
    "investment_returns" to "guaranteed investment returns ka vaada karta hai",
    "digital_arrest" to "police ya sarkari agency hone ka dawa karta hai",
    "brand_domain_mismatch" to "brand ka naam le kar galat domain ka link deta hai",
    "shortened_url" to "chhota (shortened) link deta hai",
    "suspicious_url" to "shaq wala link deta hai",
    "urgency" to "jaldi karne ka dabav dalta hai (turant, abhi, 24 ghante)",
)

private val SAFE_TEXT_EN: Map<String, String> = mapOf(
    "high" to "Do not share OTP or PIN, and do not click links. " +
        "Verify through the official app or the number on your card.",
    "medium" to "Verify this independently before acting. Do not share OTP or PIN.",
    "uncertain" to "Could not confirm this. Verify through the official app or the number on your card.",
    // LOW means the rules found nothing, NOT that the message is safe. Keep this wording.
    "low" to "These checks found no known scam pattern, which does not make the message safe. " +
        "Never share an OTP, PIN or password, and verify unexpected requests yourself.",
)

private val SAFE_TEXT_HINGLISH: Map<String, String> = mapOf(
    "high" to "OTP ya PIN share na karein, aur link par click na karein. " +
        "Official app ya card ke number se verify karein.",
    "medium" to "Action lene se pehle ise khud verify karein. OTP ya PIN share na karein.",
    "uncertain" to "Ise confirm nahi kar paye. Official app ya card ke number se verify karein.",
    "low" to "In jaanch mein koi jaana-pehchana scam pattern nahi mila, lekin isse message safe " +
        "nahi hota. OTP, PIN ya password kabhi share na karein, aur anjaan requests khud verify karein.",
)

private val URL_RE = rxOf("""\b(?:https?|hxxps?)://\S+""", ci = true)
private val DOMAIN_RE = rxOf("""\b(?:[a-z0-9](?:[a-z0-9\-]*[a-z0-9])?\.)+[a-z]{2,24}\b""", ci = true)
private val PHONE_RE = rxOf("""\+?\d[\d \-]{8,}\d""")
private val SAFE_RE = rxOf("""\b(safe|legitimate|genuine|authentic|trusted|harmless|no risk)\b""", ci = true)
private val ACTION_VERB_RE = rxOf(
    """\b(click|tap|call|dial|pay|send|share|reply|download|install|transfer|""" +
        """open|visit|forward|enter|provide|approve|accept)\b""",
    ci = true,
)
private val INJECTION_RE = rxOf(
    """(ignore (all |the |any )?(previous|above|prior|earlier) (instructions|rules|prompt)|""" +
        """disregard (all |the )?(previous|above|prior)|system prompt|you are now|""" +
        """new instructions|as an ai|developer mode|jailbreak)""",
    ci = true,
)
private val NEGATION_BEFORE = rxOf("""\b(not|never|don'?t|dont|do not|nahi|nahin|mat|kabhi|na|no)\b""", ci = true)
private val NEGATION_AFTER = rxOf("""^\W*(?:nahi|nahin|mat|na|not|never)\b""", ci = true)
private val HINGLISH_POSITIVE_RE = rxOf(
    """\b(bhejo|bhejein|bhejiye|bhejen|batao|bataiye|bataye|dijiye|dijie|dein|daalo|dalo|""" +
        """kholo|kholein|dabao|dabayein|click karo|click karein|call karo|call karein|""" +
        """share karo|share karein|pay karo|pay karein|install karo|install karein|""" +
        """download karo|download karein)\b""",
    ci = true,
)
private val NUMBER_RE = rxOf("""\d+(?:[.,]\d+)*""")
private val CURRENCY_BEFORE_RE = rxOf("""(?:rs\.?|inr|₹|rupees?)\s*$""", ci = true)

private fun isNegated(text: String, start: Int, end: Int): Boolean {
    val before = text.substring(maxOf(0, start - 14), start)
    val after = text.substring(end, minOf(text.length, end + 14))
    return NEGATION_BEFORE.containsMatchIn(before) || NEGATION_AFTER.containsMatchIn(after)
}

private fun positiveMatches(rx: Regex, text: String): List<String> =
    rx.findAll(text).filter { !isNegated(text, it.range.first, it.range.last + 1) }
        .map { it.value }.toList()

private fun groundedDomain(candidate: String, pkg: EvidencePackage): Boolean {
    val cand = candidate.lowercase().trimEnd('.')
    if (cand in pkg.allowedDomainNames()) return true
    return pkg.urls.any { cand in it.lowercase() }
}

private fun blank(text: String, spans: List<IntRange>): String {
    val chars = text.toCharArray()
    for (r in spans) for (k in r) chars[k] = ' '
    return String(chars)
}

private fun slotNumbers(pkg: EvidencePackage): Set<String> {
    val brand = pkg.brandDomains.values.flatten()
    val vocab = (pkg.evidence.map { it.text } + pkg.urls + pkg.domains + brand + pkg.phones)
        .joinToString(" ")
    val nums = NUMBER_RE.findAll(vocab).map { m -> m.value.filter { it.isDigit() } }.toHashSet()
    nums.addAll(pkg.phones)
    return nums
}

/** Grounding: URLs, domains, phone numbers, and numbers of 3+ digits or after a currency word. */
internal fun ungroundedClaims(text: String, pkg: EvidencePackage): List<String> {
    val problems = ArrayList<String>()
    val packageUrls = pkg.urls.joinToString(" ").lowercase()
    val urlSpans = ArrayList<IntRange>()
    for (m in URL_RE.findAll(text)) {
        urlSpans.add(m.range)
        if (m.value.lowercase() !in packageUrls) problems.add("URL not in package: '${m.value}'")
    }
    val spans = ArrayList(urlSpans)
    for (m in DOMAIN_RE.findAll(text)) {
        if (urlSpans.any { m.range.first in it }) continue
        spans.add(m.range)
        if (!groundedDomain(m.value, pkg)) problems.add("domain not in package: '${m.value}'")
    }
    val bare = blank(text, spans)
    val phones = pkg.phones.toHashSet()
    val phoneSpans = ArrayList<IntRange>()
    for (m in PHONE_RE.findAll(bare)) {
        phoneSpans.add(m.range)
        if (m.value.filter { it.isDigit() } !in phones) {
            problems.add("phone number not in package: '${m.value}'")
        }
    }
    val bare2 = blank(bare, phoneSpans)
    val slots = slotNumbers(pkg)
    for (m in NUMBER_RE.findAll(bare2)) {
        val digits = m.value.filter { it.isDigit() }
        val before = bare2.substring(maxOf(0, m.range.first - 10), m.range.first)
        val currency = CURRENCY_BEFORE_RE.containsMatchIn(before)
        if ((digits.length >= 3 || currency) && digits !in slots) {
            problems.add("number not in package: '${m.value}'")
        }
    }
    return problems
}

// G2: attribution of scam intent (LOW rejects any; MEDIUM and UNCERTAIN allow a hedge; HIGH unchecked).

private val ATTRIBUTION_WORDS = listOf(
    "scam", "scams", "scammer", "scammers", "scammed", "scamming", "scammy",
    "fraud", "frauds", "fraudster", "fraudsters", "fraudulent", "defraud", "defrauds",
    "defrauding", "phish", "phishing", "phisher", "phishers", "spoof", "spoofing",
    "conman", "conmen", "con artist", "con artists", "con man", "swindle", "swindler",
    "swindlers",
    "trick", "tricks", "tricked", "tricking", "trickery", "cheat", "cheats", "cheated",
    "cheating", "steal", "steals", "stealing", "stole", "stolen", "theft", "thief", "thieves",
    "lure", "lures", "lured", "luring", "bait", "impersonate", "impersonates",
    "impersonating", "impersonation", "pretend", "pretends", "pretending", "hack", "hacker",
    "hackers", "hacking", "malicious", "bogus", "fake", "exploit", "exploits", "exploiting",
    "tactic", "tactics", "victim", "victims",
    "dhoka", "dhokha", "dhokhaa", "dhokebaaz", "dhokebaz", "dhokebaazi", "thag", "thagi",
    "thagna", "thagne", "thagon", "thagega", "jaalsaaz", "jalsaaz", "chor", "chori",
    "lootera", "luteraa", "lutere", "loot", "nakli", "farzi",
)
private val ATTRIBUTION_PHRASES = listOf(
    """trying to (?:rush|scare|pressure|trick|fool|cheat|steal|extract|dupe|deceive|defraud|con|scam|hack|rob)""",
    """(?:rush|rushes|rushing|pressure|pressuring|scare|scaring|fool|fooling|dupe|duping|deceive|deceiving|rob|robbing) you""",
    """take advantage""",
    """designed to (?:steal|trick|scam|cheat|defraud|rob|extract)""",
    """intends? to (?:steal|trick|scam|cheat|defraud|rob|extract)""",
)

private val ATTRIBUTION_RE: Regex = Pattern.compile(
    "(?<![a-z0-9])(?:" + (
        ATTRIBUTION_WORDS.map { w -> w.split(" ").joinToString("\\s+") { Pattern.quote(it) } } +
            ATTRIBUTION_PHRASES
        ).joinToString("|") + ")(?![a-z0-9])",
    Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE,
).toRegex()

private val HEDGE_RE = Pattern.compile(
    "(?<![a-z0-9])(?:(?:may|could|might)\\s+be\\s+(?:an?\\s+)?(?:scam|fraud|phishing|fake)" +
        "|(?:scam|fraud|dhoka|dhokha|thagi)\\s+ho\\s+(?:sakta|sakti|sakte)\\s+hai)(?![a-z0-9])",
    Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE,
).toRegex()
private val LIKELY_HEDGE_RE = Pattern.compile(
    "(?<![a-z0-9])likely\\s+(?:an?\\s+)?(?:scam|fraud|phishing)(?![a-z0-9])",
    Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE,
).toRegex()

/** G2 problems for one point at [risk]. Empty when the guard is off (null) or the risk is HIGH. */
fun attributionProblems(text: String, risk: Risk?): List<String> {
    if (risk == null || risk == Risk.HIGH) return emptyList()
    val hedged = risk == Risk.MEDIUM || risk == Risk.UNCERTAIN
    val hedges = HEDGE_RE.findAll(text).map { it.range }.toMutableList()
    if (hedged) hedges.addAll(LIKELY_HEDGE_RE.findAll(text).map { it.range })
    val hits = ArrayList<String>()
    for (m in ATTRIBUTION_RE.findAll(text)) {
        val inside = hedged && hedges.any { h -> m.range.first >= h.first && m.range.last <= h.last }
        if (!inside) hits.add(m.value.lowercase())
    }
    if (hits.isEmpty()) return emptyList()
    return listOf("attribution of scam intent at ${risk.name}: ${hits.toSortedSet().joinToString(", ")}")
}

/** Checks for one point. Returns the list of problems (empty means the point passes). */
fun pointProblems(
    text: String,
    etype: String,
    pkg: EvidencePackage,
    lang: String,
    risk: Risk?,
): List<String> {
    val body = text.trim()
    val problems = ArrayList<String>()
    if (body.isEmpty()) problems.add("point is empty")
    if (body.length > MAX_POINT) problems.add("point is ${body.length} chars, limit $MAX_POINT")
    problems.addAll(ungroundedClaims(body, pkg))
    if (pkg.risk == Risk.HIGH || pkg.risk == Risk.MEDIUM) {
        for (word in positiveMatches(SAFE_RE, body)) {
            problems.add("point calls the message ${word.lowercase()} while risk is ${pkg.risk.name}")
        }
    }
    for (verb in positiveMatches(ACTION_VERB_RE, body) + positiveMatches(HINGLISH_POSITIVE_RE, body)) {
        problems.add("point gives a positive instruction: '$verb'")
    }
    if (INJECTION_RE.containsMatchIn(body)) problems.add("point has prompt-injection phrasing")
    if (!hasLexiconKeyword(body, lang, etype)) problems.add("no $lang lexicon keyword for $etype")
    problems.addAll(attributionProblems(body, risk))
    return problems
}

private val LEXICON_PATTERNS: Map<String, Map<String, Regex>> by lazy {
    RuleData.pointLexicon.mapValues { (_, table) ->
        table.mapValues { (_, words) ->
            val alts = words.joinToString("|") { w ->
                w.split(" ").filter { it.isNotEmpty() }.joinToString("\\s+") { Pattern.quote(it) }
            }
            Pattern.compile("(?<![a-z0-9])(?:$alts)",
                Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE).toRegex()
        }
    }
}

/** True when the text has a lexicon keyword for the evidence type in the language. */
fun hasLexiconKeyword(text: String, lang: String, etype: String): Boolean =
    LEXICON_PATTERNS[lang]?.get(etype)?.containsMatchIn(text) == true

/** The deterministic point for one evidence type ("This message ..." in English). */
fun templatePoint(etype: String, lang: String): String =
    if (lang == "hinglish") "Yeh message ${TYPE_PHRASE_HINGLISH.getValue(etype)}."
    else "This message ${TYPE_PHRASE_EN.getValue(etype)}."

/** Python's _template_summary: the summary sentence(s) for a package, at most 280 characters. */
fun templateSummary(pkg: EvidencePackage, lang: String): String {
    val phrases = if (lang == "hinglish") TYPE_PHRASE_HINGLISH else TYPE_PHRASE_EN
    val safe = if (lang == "hinglish") SAFE_TEXT_HINGLISH else SAFE_TEXT_EN
    val base = safe.getValue(pkg.risk.name.lowercase())
    val seen = pkg.evidence.map { it.type }.distinct()
    val summary = if (seen.isNotEmpty()) {
        val described = seen.take(3).joinToString("; ") { phrases.getValue(it) }
        val prefix = if (lang == "en") "This message $described." else "Is message mein: $described."
        "$prefix $base"
    } else {
        base
    }
    return summary.take(MAX_SUMMARY)
}

/** The collapsed package of Fix D: one evidence item per type (the first of each). */
fun collapseByType(pkg: EvidencePackage): EvidencePackage {
    val first = LinkedHashMap<String, Evidence>()
    for (e in pkg.evidence) first.putIfAbsent(e.type, e)
    return pkg.copy(evidence = first.values.toList())
}
