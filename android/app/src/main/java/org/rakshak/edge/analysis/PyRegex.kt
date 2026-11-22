package org.rakshak.edge.analysis

import java.util.regex.Pattern
import org.rakshak.edge.perf.activeRuleRegexTiming
import org.rakshak.edge.perf.measure

/**
 * Helpers that make java.util.regex behave like the Python `regex`/`re` patterns the rakshak
 * rules were written for.
 *
 * Python's Unicode `\w` is letters, decimal and other numbers, and underscore. Combining marks
 * (Mn/Mc) are not `\w`, which matters for Devanagari: a word ending in a matra has no `\b` after
 * it in Python. Java's UNICODE_CHARACTER_CLASS counts marks as word characters, so instead of
 * that flag, [pyRe] rewrites `\w`, `\d`, `\s` and `\b` into explicit classes. The engine then
 * needs no special flags, which also keeps the code on Android's regex implementation.
 */
internal const val WORD_BODY = "\\p{L}\\p{N}_"
internal const val WORD = "[$WORD_BODY]"
internal const val SPACE_BODY = "\\t\\n\\x0B\\f\\r\\x1C-\\x20\\x85\\xA0\\p{Z}"
internal const val SPACE = "[$SPACE_BODY]"
private const val WORD_B =
    "(?:(?<=[$WORD_BODY])(?![$WORD_BODY])|(?<![$WORD_BODY])(?=[$WORD_BODY]))"
private const val NOT_WORD_B =
    "(?:(?<=[$WORD_BODY])(?=[$WORD_BODY])|(?<![$WORD_BODY])(?![$WORD_BODY]))"

/** Rewrites one Python pattern into a java.util.regex pattern. Leading "(?i)" becomes a flag. */
internal fun pyRe(pattern: String): String {
    val out = StringBuilder()
    var inClass = false
    var i = 0
    while (i < pattern.length) {
        val c = pattern[i]
        if (c == '\\' && i + 1 < pattern.length) {
            val n = pattern[i + 1]
            val rep: String? = when (n) {
                'w' -> if (inClass) WORD_BODY else WORD
                'd' -> "\\p{Nd}"
                's' -> if (inClass) SPACE_BODY else SPACE
                'b' -> {
                    require(!inClass) { "\\b inside a class is not supported: $pattern" }
                    WORD_B
                }
                'W' -> {
                    require(!inClass) { "\\W inside a class is not supported: $pattern" }
                    "[^$WORD_BODY]"
                }
                'S' -> {
                    require(!inClass) { "\\S inside a class is not supported: $pattern" }
                    "[^$SPACE_BODY]"
                }
                'D' -> "\\P{Nd}"
                'B' -> {
                    require(!inClass) { "\\B inside a class is not supported: $pattern" }
                    NOT_WORD_B
                }
                else -> null
            }
            out.append(rep ?: "\\$n")
            i += 2
            continue
        }
        if (!inClass && c == '[') {
            inClass = true
            out.append(c)
            i++
            if (i < pattern.length && pattern[i] == '^') {
                out.append('^')
                i++
            }
            if (i < pattern.length && pattern[i] == ']') {
                out.append("\\]")
                i++
            }
            continue
        }
        if (inClass && c == ']') inClass = false
        out.append(c)
        i++
    }
    return out.toString()
}

/** Compiles a Python-style pattern to a Kotlin Regex. `ci` is Python's IGNORECASE. */
internal fun rxOf(pattern: String, ci: Boolean = false): Regex {
    var p = pattern
    var caseInsensitive = ci
    if (p.startsWith("(?i)")) {
        caseInsensitive = true
        p = p.substring(4)
    }
    var flags = Pattern.UNIX_LINES // Python "." matches everything except "\n"
    if (caseInsensitive) flags = flags or Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE
    val timing = activeRuleRegexTiming()
    val translated = timing.measure("rules_regex_translate") { pyRe(p) }
    return timing.measure("rules_regex_compile") { Pattern.compile(translated, flags).toRegex() }
}

/** Letters only, as Python's `"".join(regex.findall(r"\p{L}", s))` would give. */
internal fun lettersOnly(s: String): String {
    val sb = StringBuilder()
    s.codePoints().forEach { cp -> if (Character.isLetter(cp)) sb.appendCodePoint(cp) }
    return sb.toString()
}
