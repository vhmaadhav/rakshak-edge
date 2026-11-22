package org.rakshak.edge.analysis

import java.net.IDN
import java.util.regex.Pattern
import org.rakshak.edge.data.RuleData

/**
 * Port of src/rakshak/url_features.py, limited to what the rules and the prompt facts read.
 *
 * Public suffixes come from the ICANN section of tldextract's bundled snapshot
 * (public_suffix_icann.txt), with the same lookup order as the Python reference in
 * android/tools/export_assets.py (suffix_index). urlsplit is approximated: the host is the
 * netloc without userinfo and port, lower-cased, with a best-effort fallback for strings that
 * urlsplit rejects. Both are checked by the parity fixture.
 */

val SHORTENERS: Set<String> = setOf(
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly", "shorturl.at",
    "tiny.cc", "rebrand.ly", "bit.do", "t.ly", "v.gd", "s.id", "lnkd.in", "buff.ly", "tiny.one",
    "clck.ru", "urlz.fr",
)

private val SUSPICIOUS_TOKENS = listOf(
    "login", "signin", "verify", "verification", "kyc", "update", "secure", "security", "account",
    "bank", "otp", "refund", "reward", "lottery", "claim", "unlock", "suspend", "blocked",
    "billing", "payment", "wallet", "invoice", "support", "helpdesk", "activate", "confirm",
    "validate", "reactivate", "upi", "pay",
)

private val RISKY_TLDS: Set<String> = setOf(
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "cc", "click", "link", "icu", "buzz", "cyou",
    "rest", "sbs", "cfd", "shop", "online", "site", "live", "work", "fun", "vip", "support",
    "monster", "zip", "mov", "quest", "su", "pw", "ws", "bond", "cam",
)

private val TOKEN_RE = rxOf("[a-z]+")
private val IPV4_RE = rxOf("""\d{1,3}(\.\d{1,3}){3}""")
private val SCHEME_RE = Regex("[A-Za-z][A-Za-z0-9+.\\-]*")

/** The public-suffix split of one host (tldextract's fields). */
data class HostSplit(val subdomain: String, val domain: String, val suffix: String, val top: String)

/** Python's _extract: labels, the suffix index, and the registered domain. */
internal fun splitHost(host: String): HostSplit {
    val h = host.lowercase().trimEnd('.')
    if (h.isEmpty()) return HostSplit("", "", "", "")
    val ascii = if (h.all { it.code < 128 }) h else try {
        IDN.toASCII(h)
    } catch (e: IllegalArgumentException) {
        return HostSplit("", "", "", "")
    }
    if (IPV4_RE.matches(ascii)) return HostSplit("", ascii, "", "")
    val labels = ascii.split('.')
    val idx = suffixIndex(labels, RuleData.suffixRules)
    val suffix = labels.subList(idx, labels.size).joinToString(".")
    val domain = if (idx >= 1) labels[idx - 1] else ""
    val sub = if (idx >= 2) labels.subList(0, idx - 1).joinToString(".") else ""
    val top = if (domain.isNotEmpty() && suffix.isNotEmpty()) "$domain.$suffix" else ""
    return HostSplit(sub, domain, suffix, top)
}

/** Reference lookup, identical to suffix_index() in android/tools/export_assets.py. */
internal fun suffixIndex(labels: List<String>, rules: Set<String>): Int {
    val n = labels.size
    for (i in 0 until n) {
        val cand = labels.subList(i, n).joinToString(".")
        if ("!$cand" in rules) return i + 1
        if (i + 1 < n && "*." + labels.subList(i + 1, n).joinToString(".") in rules) return i
        if (cand in rules) return i
    }
    return n
}

private const val AUTHORITY_END = "/?#"

/** Host of the URL the way the Python _split helper reads it. Never throws. */
internal fun hostOf(url: String): String {
    val candidate = if ("://" in url) url else "http://$url"
    val fromUrlsplit = urlsplitHost(candidate)
    if (fromUrlsplit != null && (fromUrlsplit.isNotEmpty() || url.isBlank())) {
        return fromUrlsplit.trimEnd('.')
    }
    return bestEffortHost(url)
}

/** urlsplit(candidate).hostname, lower-cased. Null where Python raises (unbalanced brackets). */
private fun urlsplitHost(candidate: String): String? {
    val idx = candidate.indexOf("://")
    if (idx < 0) return ""
    val scheme = candidate.substring(0, idx)
    if (!SCHEME_RE.matches(scheme)) return ""
    val rest = candidate.substring(idx + 3)
    val end = rest.indexOfFirst { it in AUTHORITY_END }.let { if (it < 0) rest.length else it }
    val netloc = rest.substring(0, end)
    val hostinfo = netloc.substringAfterLast('@')
    val hasOpen = hostinfo.contains('[')
    val hasClose = hostinfo.contains(']')
    if (hasOpen != hasClose) return null
    val hostname = if (hasOpen) {
        hostinfo.substringAfter('[').substringBefore(']')
    } else {
        hostinfo.substringBefore(':')
    }
    return hostname.lowercase()
}

private fun bestEffortHost(url: String): String {
    var text = url.trim()
    if ("://" in text) text = text.substringAfter("://")
    val end = text.indexOfFirst { it in AUTHORITY_END }
    var authority = if (end < 0) text else text.substring(0, end)
    authority = authority.substringAfterLast('@').replace("[", "").replace("]", "")
    if (authority.count { it == ':' } == 1) authority = authority.substringBefore(':')
    return authority.lowercase().trimEnd('.')
}

/** eTLD+1 of the URL's host ("" when unknown). */
fun registeredDomain(url: String): String = splitHost(hostOf(url)).top

private fun brandInHostNotRegistered(host: String, registered: String): Boolean {
    if (host.isEmpty()) return false
    val labels = host.replace(".", " ").replace("-", " ").split(Regex("\\s+"))
        .filter { it.isNotEmpty() }.toHashSet()
    for (info in RuleData.brands.values) {
        if (registered in info.domains) continue
        for (alias in info.hostAliases) {
            if (alias.length <= 3) {
                if (alias in labels) return true
            } else if (alias in host) {
                return true
            }
        }
    }
    return false
}

/** The url_features values the rules read. */
data class UrlFeatures(
    val tldRisky: Boolean,
    val punycode: Boolean,
    val ipLiteral: Boolean,
    val hasAtSign: Boolean,
    val brandHostMismatch: Boolean,
    val suspiciousTokenCount: Int,
    val nSubdomains: Int,
    val registeredDomain: String,
)

fun urlFeatures(url: String): UrlFeatures {
    val host = hostOf(url)
    val split = splitHost(host)
    val full = url.lowercase()
    val tokens = TOKEN_RE.findAll(full).map { it.value }.toHashSet()
    val suspicious = SUSPICIOUS_TOKENS.count { t -> tokens.any { tok -> t in tok } }
    return UrlFeatures(
        tldRisky = split.suffix in RISKY_TLDS,
        punycode = "xn--" in host,
        ipLiteral = IPV4_RE.matches(host) || host.startsWith("["),
        hasAtSign = "@" in url,
        brandHostMismatch = brandInHostNotRegistered(host, split.top),
        suspiciousTokenCount = suspicious,
        nSubdomains = if (split.subdomain.isEmpty()) 0 else split.subdomain.split(".").size,
        registeredDomain = split.top,
    )
}

private val brandPatterns: Map<String, Regex> by lazy {
    RuleData.brands.mapValues { (_, b) ->
        val alts = b.aliases.distinct().sortedByDescending { it.length }
            .joinToString("|") { Pattern.quote(it) }
        Pattern.compile("(?<![a-z0-9])(?:$alts)(?![a-z0-9])",
            Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE).toRegex()
    }
}

/** Names of curated brands whose alias appears in the text. */
fun brandsInText(text: String): List<String> =
    brandPatterns.filter { (_, pat) -> pat.containsMatchIn(text) }.keys.toList()

/** True when the text names a curated brand but the URL is not on one of its domains. */
fun brandDomainMismatch(text: String, url: String): Boolean {
    val named = brandsInText(text)
    if (named.isEmpty()) return false
    val reg = registeredDomain(url)
    for (name in named) {
        if (reg.isNotEmpty() && reg in RuleData.brands.getValue(name).domains) return false
    }
    return true
}

/** Each brand named in the text mapped to its official domains (for the grounding check). */
fun mentionedBrandDomains(text: String): Map<String, List<String>> =
    brandsInText(text).associateWith { RuleData.brands.getValue(it).domains }
