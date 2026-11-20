package org.rakshak.edge.llm

import org.rakshak.edge.analysis.Evidence
import org.rakshak.edge.analysis.EvidencePackage
import org.rakshak.edge.analysis.Risk
import org.rakshak.edge.analysis.TYPE_PHRASE_EN
import org.rakshak.edge.analysis.collapseByType
import org.rakshak.edge.analysis.pointProblems
import org.rakshak.edge.analysis.registeredDomain
import org.rakshak.edge.analysis.templatePoint
import org.rakshak.edge.analysis.templateSummary
import org.rakshak.edge.analysis.rxOf

/**
 * Port of the slot mode in deploy/pi/bench/rbench/slot.py as the Pi runs it:
 * variant describe_kw, Fix C cleanup, Fix D one call per type, and guards G1 and G2.
 *
 * The model writes one sentence per evidence type. Each sentence is cut at its first sentence
 * end, cleaned, and checked with the same rules as the verifier. A failing point is replaced by
 * its template. The summary is always the template.
 */

/** Anything that can continue a prompt. [LlmExplainer] is the ExecuTorch implementation. */
interface TokenGenerator {
    /** Returns the generated text. [stopWhen] is called with the text so far after each token. */
    suspend fun generate(prompt: String, maxNewTokens: Int, stopWhen: (String) -> Boolean): String
}

/** One point of the advisory. [source] is "llm" or "template". */
data class PointView(
    val evidenceId: String,
    val type: String,
    val text: String,
    val source: String,
    val problems: List<String>,
)

data class Advisory(
    val path: String,
    val mode: String,
    val summary: String,
    val points: List<PointView>,
    val g2Rejected: List<String>,
)

private const val DESCRIBE_SYSTEM =
    "You describe what a text message says or asks the reader to do. You are given one fact " +
        "about the message. Write one short plain-English sentence that restates the fact. Do not " +
        "judge the message or its sender. Use only the fact. Output only the sentence."
private const val BOS = "<|begin_of_text|>"
private const val SOT = "<|start_header_id|>"
private const val EOT_HDR = "<|end_header_id|>"
private const val EOT = "<|eot_id|>"
private const val SLOT_LEAD = "This message"
private const val SLOT_MAX_CHARS = 60
private const val DUP_RATIO = 0.9
private const val STOP_CHARS = "\n\"{}"

private val WHITESPACE = Regex("\\s+")
private val URL_TYPES = setOf("brand_domain_mismatch", "shortened_url", "suspicious_url")
private val VERIFIER_URL_RE = rxOf("""\b(?:https?|hxxps?)://\S+""", ci = true)
private val PAREN_RE = rxOf("""\s*\([^)]*\)""")
private val TERMINATOR = rxOf("""[.!?](?=\s|$)""")
private val ECHO_RE = rxOf(
    """\s*,?\s*(?:and\s+|which\s+)?""" +
        """(?:restates?\s+the\s+fact|(?:has|have|is|was)\s+been\s+matched|(?:is|was)\s+matched)""" +
        """\s*([.!?])\s*$""",
    ci = true,
)
private val DANGLING = ("and or but because to the a an of with that which is are was has have been " +
    "by for in on at it this be from as").split(" ").toHashSet()

/** Prompt construction, Llama 3 chat format, as rbench.slot.variant_prompt("describe_kw"). */
object SlotPrompt {
    const val MAX_NEW_TOKENS = 20

    fun factSlot(e: Evidence): String {
        if (e.type in URL_TYPES) return registeredDomain(e.text)
        var text = VERIFIER_URL_RE.replace(e.text) { m -> registeredDomain(m.value).ifEmpty { "a link" } }
        text = WHITESPACE.split(text.replace("\"", " ").trim()).filter { it.isNotEmpty() }
            .joinToString(" ")
        if (text.length > SLOT_MAX_CHARS) text = text.substring(0, SLOT_MAX_CHARS - 3).trimEnd() + "..."
        return text
    }

    fun factText(e: Evidence): String {
        val phrase = TYPE_PHRASE_EN.getValue(e.type).replaceFirstChar { it.uppercaseChar() }
        val slot = factSlot(e)
        return if (slot.isNotEmpty()) "$phrase. Matched text: \"$slot\"" else "$phrase."
    }

    fun keywordClause(type: String): String =
        WHITESPACE.split(PAREN_RE.replace(TYPE_PHRASE_EN.getValue(type), "").trim())
            .filter { it.isNotEmpty() }.joinToString(" ")

    /** (assistant prefix, full prompt ending inside the assistant turn). */
    fun describeKw(e: Evidence): Pair<String, String> {
        val prefix = "$SLOT_LEAD ${keywordClause(e.type)} and"
        val user = "FACT: ${factText(e)}\n\nWrite the sentence for this fact."
        val prompt = BOS +
            "${SOT}system$EOT_HDR\n\n$DESCRIBE_SYSTEM$EOT" +
            "${SOT}user$EOT_HDR\n\n$user$EOT" +
            "${SOT}assistant$EOT_HDR\n\n$prefix"
        return prefix to prompt
    }
}

/** Stop once the continuation holds a sentence end or a character that cannot be in a point. */
fun slotStop(generated: String): Boolean =
    TERMINATOR.containsMatchIn(generated) || generated.any { it in STOP_CHARS }

private fun pySplitJoin(s: String): String =
    WHITESPACE.split(s.trim()).filter { it.isNotEmpty() }.joinToString(" ")

/** clean_sentence: (point, status). Status is "ok", "empty" or "unterminated". */
fun cleanSentence(generated: String, prefix: String): Pair<String, String> {
    if (generated.isBlank()) return "" to "empty"
    val term = TERMINATOR.find(generated) ?: return "" to "unterminated"
    val termEnd = term.range.last + 1
    val cut = generated.indexOfFirst { it in STOP_CHARS }
    if (cut >= 0 && cut < termEnd) return "" to "unterminated"
    return pySplitJoin(prefix + generated.substring(0, termEnd)) to "ok"
}

/** strip_echo: (sentence, changed). An echo that leaves a short or dangling remainder gives "". */
fun stripEcho(sentence: String, lead: String = "This message "): Pair<String, Boolean> {
    val m = ECHO_RE.find(sentence) ?: return sentence to false
    val remainder = sentence.substring(0, m.range.first).trimEnd()
    val words = WHITESPACE.split(remainder.trim()).filter { it.isNotEmpty() }
    val last = words.lastOrNull()?.lowercase()?.trim { it in ".,;:!?\"'" } ?: ""
    if (!remainder.startsWith(lead) || words.size < 4 || last in DANGLING) return "" to true
    return (remainder + m.groupValues[1]) to true
}

private fun normPoint(text: String): String =
    pySplitJoin(text.lowercase().replace(Regex("[^a-z0-9 ]+"), " "))

/** difflib.SequenceMatcher(None, a, b).ratio(), without the autojunk heuristic (points are short). */
internal fun sequenceRatio(a: String, b: String): Double {
    val total = a.length + b.length
    if (total == 0) return 1.0
    val b2j = HashMap<Char, MutableList<Int>>()
    for (j in b.indices) b2j.getOrPut(b[j]) { ArrayList() }.add(j)
    return 2.0 * matchedChars(a, 0, a.length, b, 0, b.length, b2j) / total
}

private fun matchedChars(
    a: String, alo: Int, ahi: Int, b: String, blo: Int, bhi: Int, b2j: Map<Char, List<Int>>,
): Int {
    if (alo >= ahi || blo >= bhi) return 0
    var besti = alo
    var bestj = blo
    var bestsize = 0
    var j2len = HashMap<Int, Int>()
    for (i in alo until ahi) {
        val newj2len = HashMap<Int, Int>()
        for (j in b2j[a[i]].orEmpty()) {
            if (j < blo) continue
            if (j >= bhi) break
            val k = (j2len[j - 1] ?: 0) + 1
            newj2len[j] = k
            if (k > bestsize) {
                besti = i - k + 1
                bestj = j - k + 1
                bestsize = k
            }
        }
        j2len = newj2len
    }
    if (bestsize == 0) return 0
    return bestsize +
        matchedChars(a, alo, besti, b, blo, bestj, b2j) +
        matchedChars(a, besti + bestsize, ahi, b, bestj + bestsize, bhi, b2j)
}

/** duplicate_points: for each point, the index of the earlier point it repeats, or null. */
fun duplicatePoints(points: List<String>): List<Int?> {
    val kept = ArrayList<Pair<Int, String>>()
    val out = ArrayList<Int?>()
    for ((i, text) in points.withIndex()) {
        val norm = normPoint(text)
        var dupOf: Int? = null
        if (norm.isNotEmpty()) {
            for ((j, prev) in kept) {
                if (norm == prev || sequenceRatio(norm, prev) >= DUP_RATIO) {
                    dupOf = j
                    break
                }
            }
            if (dupOf == null) kept.add(i to norm)
        }
        out.add(dupOf)
    }
    return out
}

/** Mirrors SlotExplainer.advise in the Python edge pipeline for English packages. */
object SlotExplainer {
    /** The template advisory for [llmPkg] (already collapsed to one item per type). */
    fun templateOnly(llmPkg: EvidencePackage, path: String, summary: String): Advisory =
        Advisory(
            path = path,
            mode = "template",
            summary = summary,
            points = llmPkg.evidence.map {
                PointView(it.id, it.type, templatePoint(it.type, "en"), "template", emptyList())
            },
            g2Rejected = emptyList(),
        )

    /**
     * Advisory for [pkg]. With a null [generator] (no model installed) or a LOW-risk package
     * (guard G1) no model call is made and the output is the template advisory.
     */
    suspend fun advise(pkg: EvidencePackage, generator: TokenGenerator?): Advisory {
        val llmPkg = collapseByType(pkg)
        val summary = templateSummary(pkg, "en")
        if (pkg.risk == Risk.LOW) return templateOnly(llmPkg, "low_template", summary)
        if (generator == null) return templateOnly(llmPkg, "template_no_model", summary)

        val raw = ArrayList<String>()
        val prefixes = ArrayList<String>()
        for (e in llmPkg.evidence) {
            val (prefix, prompt) = SlotPrompt.describeKw(e)
            val generated = generator.generate(prompt, SlotPrompt.MAX_NEW_TOKENS) { slotStop(it) }
            raw.add(cleanSentence(generated, prefix).first)
            prefixes.add(prefix)
        }
        // Fix C: strip trailing echoes, then blank later duplicates.
        for (i in raw.indices) {
            if (raw[i].isEmpty()) continue
            raw[i] = stripEcho(raw[i], "This message ").first
        }
        for ((i, dup) in duplicatePoints(raw).withIndex()) {
            if (dup != null) raw[i] = ""
        }

        val g2 = ArrayList<String>()
        val points = llmPkg.evidence.mapIndexed { i, e ->
            val problems = pointProblems(raw[i], e.type, llmPkg, "en", pkg.risk)
            problems.filter { it.startsWith("attribution of scam intent") }
                .forEach { g2.add("${e.id}: ${raw[i]} ($it)") }
            if (problems.isEmpty()) {
                PointView(e.id, e.type, raw[i].trim(), "llm", emptyList())
            } else {
                PointView(e.id, e.type, templatePoint(e.type, "en"), "template", problems)
            }
        }
        val llmCount = points.count { it.source == "llm" }
        val mode = when {
            llmCount == 0 -> "template"
            llmCount == points.size -> "llm"
            else -> "partial"
        }
        return Advisory(path = "explainer", mode = mode, summary = summary, points = points, g2Rejected = g2)
    }
}

