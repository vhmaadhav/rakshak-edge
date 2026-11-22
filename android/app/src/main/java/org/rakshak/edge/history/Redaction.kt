package org.rakshak.edge.history

import org.rakshak.edge.analysis.registeredDomain

/**
 * Best-effort redaction for the optional redacted field. Applied only when the store's policy
 * allows it. Email addresses become "[email]", URLs and hosts become their registrable domain
 * (or "[link]" when none can be read), every digit becomes "#", and the result is capped at
 * [MAX_LEN] characters. It is a reduction, not a guarantee: names and free words can remain.
 * That is why the policy default is off.
 */
object Redaction {
    const val MAX_LEN = 120

    private val EMAIL = Regex("""[\w.+-]+@[\w-]+(?:\.[\w-]+)+""")
    private val URL = Regex("""(?i)(?:https?://|www\.)\S+|[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/\S*)?""")
    private val DIGIT = Regex("""\d""")
    private val SPACES = Regex("""\s+""")

    /** Redacts a free-text SMS body. */
    fun redactText(text: String): String {
        var s = EMAIL.replace(text, "[email]")
        s = URL.replace(s) { m -> redactHost(m.value) }
        s = DIGIT.replace(s, "#")
        s = SPACES.replace(s, " ").trim()
        return if (s.length > MAX_LEN) s.take(MAX_LEN - 3) + "..." else s
    }

    /** Redacts a link. Only the registrable domain survives: no scheme, path, query or user info. */
    fun redactUrl(url: String): String = redactHost(url)

    private fun redactHost(raw: String): String = registeredDomain(raw).ifEmpty { "[link]" }
}
