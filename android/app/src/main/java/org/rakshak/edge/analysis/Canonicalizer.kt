package org.rakshak.edge.analysis

import java.text.Normalizer

/**
 * Port of src/rakshak/canonicalize.py: NFKC, invisible-character strip, homoglyph fold, URL
 * de-obfuscation and extraction, spacing and leetspeak fold. Devanagari is left intact.
 *
 * Not ported: the Flags counters. Nothing in the rules, the encoding or the decision reads them.
 * Offsets are UTF-16 indices (Kotlin) where Python uses code point indices; they differ only for
 * text with supplementary-plane characters, and only evidence spans and the 30-character negation
 * look-behind can see that.
 */
data class Canon(val original: String, val normalized: String, val urls: List<String>)

private val ZERO_WIDTH: Set<Int> = setOf(
    0x200B, 0x200C, 0x200D, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF, 0x180E, 0x00AD, 0x034F,
)
private val BIDI: Set<Int> = setOf(
    0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069, 0x200E, 0x200F, 0x061C,
)

/** Cyrillic, Greek and fullwidth/mathematical lookalikes, exactly as in _HOMOGLYPH. */
private fun buildHomoglyph(): Map<Int, Char> {
    val m = HashMap<Int, Char>()
    val pairs = listOf(
        // Cyrillic
        'А' to 'A', 'а' to 'a', 'В' to 'B', 'Е' to 'E', 'е' to 'e',
        'К' to 'K', 'М' to 'M', 'Н' to 'H', 'О' to 'O', 'о' to 'o',
        'Р' to 'P', 'р' to 'p', 'С' to 'C', 'с' to 'c', 'Т' to 'T',
        'Х' to 'X', 'х' to 'x', 'У' to 'Y', 'у' to 'y', 'Ѕ' to 'S',
        'ѕ' to 's', 'І' to 'I', 'і' to 'i', 'Ј' to 'J', 'ј' to 'j',
        'Ԁ' to 'D', 'ԁ' to 'd', 'Ԛ' to 'Q', 'ԛ' to 'q', 'Ԝ' to 'W',
        'ԝ' to 'w', 'Ү' to 'Y', 'ү' to 'y', 'Ӏ' to 'l',
        // Greek
        'Α' to 'A', 'Β' to 'B', 'Ε' to 'E', 'Ζ' to 'Z', 'Η' to 'H',
        'Ι' to 'I', 'Κ' to 'K', 'Μ' to 'M', 'Ν' to 'N', 'Ο' to 'O',
        'Ρ' to 'P', 'Τ' to 'T', 'Υ' to 'Y', 'Χ' to 'X', 'ο' to 'o',
        'ν' to 'v', 'α' to 'a', 'ρ' to 'p', 'ι' to 'i', 'κ' to 'k',
        'τ' to 't', 'ϲ' to 'c',
    )
    for ((k, v) in pairs) m[k.code] = v
    // Fullwidth ASCII block U+FF01..U+FF5E maps to U+0021..U+007E.
    for (cp in 0xFF01 until 0xFF5F) m[cp] = (cp - 0xFEE0).toChar()
    // Mathematical alphanumerics whose NFKC form is one ASCII letter.
    for (cp in 0x1D400 until 0x1D7C4) {
        val folded = Normalizer.normalize(String(Character.toChars(cp)), Normalizer.Form.NFKC)
        if (folded.length == 1 && (folded[0] in 'A'..'Z' || folded[0] in 'a'..'z')) {
            m[cp] = folded[0]
        }
    }
    return m
}

private val HOMOGLYPH: Map<Int, Char> = buildHomoglyph()

private val LEET: Map<Char, Char> = mapOf(
    '0' to 'o', '1' to 'i', '3' to 'e', '4' to 'a', '5' to 's', '7' to 't', '@' to 'a', '$' to 's',
)

private val URL_SCHEME = rxOf("""\b(?:https?|ftp)://[^\s<>"']+""", ci = true)
private val BARE_HOST = rxOf(
    """(?<![\w@/.\-])((?:[a-z0-9\-]+(?:\[\.\]|\(dot\)|\.))+[a-z]{2,24})((?::\d{2,5})?(?:/[^\s<>"']*)?)""",
    ci = true,
)
private val HXXP_S_SLASHES = rxOf("""\bhxxp(s?)://""", ci = true)
private val HXXP_S_COLON = rxOf("""\bhxxp(s?):""", ci = true)
private val BRACKET_DOT = rxOf("""\s*[\[(]\s*\.\s*[\])]\s*""")
private val BRACKET_DOT_WORD = rxOf("""\s*[\[(]\s*dot\s*[\])]\s*""", ci = true)
private val SPACED_RUN = rxOf("""(?<![\p{L}\p{N}.\-_*|])\p{L}(?: \p{L}){2,}(?![\p{L}\p{N}])""")
private val DOTTED_RUN = rxOf("""(?<![\p{L}\p{N}])\p{L}(?:[.\-_*|]\p{L}){2,}(?![\p{L}\p{N}])""")
private val TOKEN = rxOf("""[\p{L}\p{N}@$]+""")
private val HAS_LATIN = rxOf("""[A-Za-z]""")
private val DIGIT_RUN = rxOf("""\d{2,}""")
private val SPACE_RUN = rxOf("""[ \t]+""")

private val KNOWN_TLDS: Set<String> = setOf(
    "com", "in", "net", "org", "co", "io", "me", "ly", "gl", "is", "to", "cc", "tk",
    "xyz", "top", "site", "online", "info", "biz", "app", "link", "click", "live",
    "gov", "edu", "ac", "bank", "shop", "store", "cfd", "icu", "vip", "work", "ru", "cn", "uk",
    "us", "de", "fr", "br", "xin", "buzz", "cyou", "rest", "fun",
    "pw", "ml", "ga", "cf", "gq", "su", "ws", "one", "sbs", "bond", "club", "digital",
    "sbi", "pro", "mobi", "tech", "ai", "ph", "jp", "kr", "au", "ca", "eu", "ng",
)
private val FILE_LIKE_TLDS: Set<String> =
    setOf("txt", "pdf", "jpg", "png", "exe", "apk", "doc", "docx", "xls", "csv", "zip")

private const val URL_TRAIL = ".,;:!?)\"'"

private fun stripInvisible(text: String): String {
    val sb = StringBuilder()
    text.codePoints().forEach { cp ->
        val control = Character.getType(cp) == Character.CONTROL.toInt() &&
            cp != '\n'.code && cp != '\t'.code && cp != '\r'.code
        if (!(cp in ZERO_WIDTH || cp in BIDI || control)) sb.appendCodePoint(cp)
    }
    return sb.toString()
}

private fun homoglyphFold(text: String): String {
    val sb = StringBuilder()
    text.codePoints().forEach { cp ->
        val mapped = HOMOGLYPH[cp]
        if (mapped != null) sb.append(mapped) else sb.appendCodePoint(cp)
    }
    return sb.toString()
}

/** Python's _unobfuscate_url_text: hxxp(s)://, hxxp(s):, [.] and (.) and [dot] and (dot). */
internal fun unobfuscateUrlText(text: String): String {
    var t = HXXP_S_SLASHES.replace(text) { m -> "http" + m.groupValues[1].lowercase() + "://" }
    t = HXXP_S_COLON.replace(t) { m -> "http" + m.groupValues[1].lowercase() + ":" }
    t = BRACKET_DOT.replace(t, ".")
    t = BRACKET_DOT_WORD.replace(t, ".")
    return t
}

/** Python's extract_urls: explicit-scheme URLs first, then bare hosts with a known TLD. */
fun extractUrls(text: String): List<String> {
    val work = unobfuscateUrlText(text)
    val found = ArrayList<Pair<Int, String>>()
    val seen = HashSet<String>()
    for (m in URL_SCHEME.findAll(work)) {
        val url = m.value.trimEnd { it in URL_TRAIL }
        if (seen.add(url.lowercase())) found.add(m.range.first to url)
    }
    for (m in BARE_HOST.findAll(work)) {
        val host = m.groupValues[1]
        val rest = m.groupValues[2]
        val last = host.trimEnd('.').substringAfterLast('.').lowercase()
        if (last in FILE_LIKE_TLDS || last !in KNOWN_TLDS) continue
        val candidate = (host + rest).trimEnd { it in URL_TRAIL }
        if (found.any { candidate.lowercase() in it.second.lowercase() }) continue
        if (candidate.lowercase() in seen) continue
        seen.add(candidate.lowercase())
        found.add(m.range.first to candidate)
    }
    return found.sortedBy { it.first }.map { it.second }
}

private fun collapseSpacing(text: String): String {
    var t = SPACED_RUN.replace(text) { m -> lettersOnly(m.value) }
    t = DOTTED_RUN.replace(t) { m -> lettersOnly(m.value) }
    return t
}

private fun leetFold(text: String): String = TOKEN.replace(text) { m ->
    val tok = m.value
    when {
        !HAS_LATIN.containsMatchIn(tok) -> tok
        DIGIT_RUN.containsMatchIn(tok) -> tok
        else -> tok.map { LEET[it] ?: it }.joinToString("")
    }
}

/** Python's canonicalize(text). The result keeps the original text and the extracted URLs. */
fun canonicalize(text: String): Canon {
    var work = Normalizer.normalize(text, Normalizer.Form.NFKC)
    work = stripInvisible(work)
    work = homoglyphFold(work)
    work = unobfuscateUrlText(work)
    val urls = extractUrls(work)
    work = collapseSpacing(work)
    work = leetFold(work)
    work = SPACE_RUN.replace(work, " ").trim()
    return Canon(original = text, normalized = work, urls = urls)
}
