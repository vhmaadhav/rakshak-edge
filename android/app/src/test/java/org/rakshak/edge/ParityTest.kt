package org.rakshak.edge

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.rakshak.edge.analysis.Canon
import org.rakshak.edge.analysis.MAX_LEN
import org.rakshak.edge.analysis.buildPackage
import org.rakshak.edge.analysis.canonicalize
import org.rakshak.edge.analysis.chooseAction
import org.rakshak.edge.analysis.detectLang
import org.rakshak.edge.analysis.encodeBytes
import org.rakshak.edge.analysis.extractUrls
import org.rakshak.edge.analysis.brandDomainMismatch
import org.rakshak.edge.analysis.registeredDomain
import org.rakshak.edge.analysis.ruleFlags
import org.rakshak.edge.analysis.runRules
import org.rakshak.edge.analysis.templateSummary
import org.rakshak.edge.analysis.urlFeatures
import org.rakshak.edge.analysis.EvidencePackage
import org.rakshak.edge.analysis.ScamState
import org.rakshak.edge.llm.SlotPrompt

/**
 * Checks the Kotlin port against src/rakshak (the Python ground truth) on every row of
 * parity_fixture.json. Each check collects its mismatches; the assertion prints the first 25.
 */
class ParityTest {

    companion object {
        private lateinit var rows: JSONArray

        @JvmStatic
        @BeforeClass
        fun setUp() {
            TestData.install()
            rows = JSONObject(TestData.fixtureText()).getJSONArray("rows")
        }
    }

    private fun report(name: String, problems: List<String>) {
        assertTrue(
            "$name: ${problems.size} mismatches, first: " + problems.take(25).joinToString(" | "),
            problems.isEmpty(),
        )
    }

    private fun JSONObject.strings(key: String): List<String> =
        getJSONArray(key).let { a -> List(a.length()) { a.getString(it) } }

    private fun JSONObject.ints(key: String): IntArray =
        getJSONArray(key).let { a -> IntArray(a.length()) { a.getInt(it) } }

    private fun rowsSeq(): Sequence<JSONObject> = sequence {
        for (i in 0 until rows.length()) yield(rows.getJSONObject(i))
    }

    private fun rowText(row: JSONObject): String = row.getString("text")

    @Test
    fun canonicalForm_and_urls_match() {
        val problems = ArrayList<String>()
        for (row in rowsSeq()) {
            val canon = canonicalize(rowText(row))
            val id = row.getString("id")
            if (canon.normalized != row.getString("canonical")) problems.add("$id canonical")
            if (canon.urls != row.strings("urls")) problems.add("$id urls ${canon.urls}")
        }
        report("canonicalize", problems)
    }

    @Test
    fun rule_evidence_matches() {
        val problems = ArrayList<String>()
        for (row in rowsSeq()) {
            val id = row.getString("id")
            val evidence = runRules(canonicalize(rowText(row)))
            val expected = row.getJSONArray("evidence")
            if (evidence.size != expected.length()) {
                problems.add("$id evidence count ${evidence.size} != ${expected.length()}")
                continue
            }
            for (i in 0 until expected.length()) {
                val e = expected.getJSONObject(i)
                val got = evidence[i]
                if (got.id != e.getString("id") || got.type != e.getString("type") ||
                    got.weight != e.getDouble("weight") || got.text != e.getString("text")
                ) {
                    problems.add("$id evidence[$i] got ${got.id}/${got.type}/${got.weight}/${got.text}")
                }
            }
        }
        report("rules", problems)
    }

    @Test
    fun side_flags_match_for_both_views() {
        val problems = ArrayList<String>()
        for (row in rowsSeq()) {
            val text = rowText(row)
            val id = row.getString("id")
            val canonFlags = ruleFlags(canonicalize(text)).map { it.toInt() }
            val rawCanon = Canon(text, text, extractUrls(text))
            val rawFlags = ruleFlags(rawCanon).map { it.toInt() }
            if (canonFlags != row.ints("flags_canon").toList()) problems.add("$id flags_canon")
            if (rawFlags != row.ints("flags_raw").toList()) problems.add("$id flags_raw")
        }
        report("side flags", problems)
    }

    @Test
    fun byte_ids_match_the_student_input() {
        val problems = ArrayList<String>()
        for (row in rowsSeq()) {
            val id = row.getString("id")
            val raw = encodeBytes(rowText(row), MAX_LEN)
            val canon = encodeBytes(row.getString("canonical"), MAX_LEN)
            if (raw.toList() != row.getJSONArray("raw_ids").let { a -> List(a.length()) { a.getLong(it) } }) {
                problems.add("$id raw_ids")
            }
            if (canon.toList() != row.getJSONArray("canon_ids").let { a -> List(a.length()) { a.getLong(it) } }) {
                problems.add("$id canon_ids")
            }
        }
        report("byte ids", problems)
    }

    @Test
    fun url_features_match() {
        val problems = ArrayList<String>()
        for (row in rowsSeq()) {
            val canonical = row.getString("canonical")
            val infos = row.getJSONArray("url_info")
            for (i in 0 until infos.length()) {
                val info = infos.getJSONObject(i)
                val url = info.getString("url")
                val f = urlFeatures(url)
                val id = "${row.getString("id")} url=$url"
                if (f.registeredDomain != info.getString("registered_domain")) {
                    problems.add("$id registered ${f.registeredDomain}")
                }
                if (registeredDomain(url) != info.getString("registered_domain")) problems.add("$id reg-fn")
                if (brandDomainMismatch(canonical, url) != info.getBoolean("brand_domain_mismatch")) {
                    problems.add("$id brand_domain_mismatch")
                }
                if (f.tldRisky != info.getBoolean("tld_risky")) problems.add("$id tld_risky")
                if (f.punycode != info.getBoolean("punycode")) problems.add("$id punycode")
                if (f.ipLiteral != info.getBoolean("ip_literal")) problems.add("$id ip_literal")
                if (f.hasAtSign != info.getBoolean("has_at_sign")) problems.add("$id has_at_sign")
                if (f.brandHostMismatch != info.getBoolean("brand_host_mismatch")) {
                    problems.add("$id brand_host_mismatch")
                }
                if (f.suspiciousTokenCount != info.getInt("suspicious_token_count")) {
                    problems.add("$id suspicious_token_count ${f.suspiciousTokenCount}")
                }
                if (f.nSubdomains != info.getInt("n_subdomains")) problems.add("$id n_subdomains ${f.nSubdomains}")
            }
        }
        report("url features", problems)
    }

    @Test
    fun language_detection_matches() {
        val problems = ArrayList<String>()
        for (row in rowsSeq()) {
            if (detectLang(rowText(row)) != row.getString("lang")) {
                problems.add("${row.getString("id")} lang ${detectLang(rowText(row))}")
            }
        }
        report("lang", problems)
    }

    @Test
    fun facts_and_llama_prompts_match() {
        val problems = ArrayList<String>()
        for (row in rowsSeq()) {
            val id = row.getString("id")
            val evidence = runRules(canonicalize(rowText(row)))
            val facts = row.strings("facts")
            val prompts = row.getJSONArray("prompts")
            if (evidence.size != facts.size) continue
            for (i in evidence.indices) {
                if (SlotPrompt.factText(evidence[i]) != facts[i]) {
                    problems.add("$id fact[$i] '${SlotPrompt.factText(evidence[i])}'")
                }
                val (prefix, prompt) = SlotPrompt.describeKw(evidence[i])
                val p = prompts.getJSONObject(i)
                if (prefix != p.getString("prefix") || prompt != p.getString("prompt")) {
                    problems.add("$id prompt[$i]")
                }
            }
        }
        report("facts and prompts", problems)
    }

    @Test
    fun decision_packages_match_for_each_state() {
        val problems = ArrayList<String>()
        val names = mapOf("scam" to ScamState.SCAM, "uncertain" to ScamState.UNCERTAIN, "benign" to ScamState.BENIGN)
        for (row in rowsSeq()) {
            val id = row.getString("id")
            val canon = canonicalize(rowText(row))
            val evidence = runRules(canon)
            val packages = row.getJSONObject("packages")
            for (key in packages.keys()) {
                val expected = packages.getJSONObject(key)
                val pkg: EvidencePackage = buildPackage(canon, evidence, names.getValue(key))
                if (pkg.risk.name != expected.getString("risk")) problems.add("$id $key risk ${pkg.risk}")
                if (pkg.allowedActions.map { it.name } != expected.strings("allowed_actions")) {
                    problems.add("$id $key allowed ${pkg.allowedActions}")
                }
                if (chooseAction(pkg).name != expected.getString("action")) problems.add("$id $key action")
                if (templateSummary(pkg, "en") != expected.getString("summary")) problems.add("$id $key summary")
            }
        }
        report("packages", problems)
    }
}
