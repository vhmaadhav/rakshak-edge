package org.rakshak.edge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.rakshak.edge.analysis.Action
import org.rakshak.edge.analysis.DetectorOnlyMode
import org.rakshak.edge.analysis.Risk
import org.rakshak.edge.analysis.SCAM_THRESHOLD
import org.rakshak.edge.analysis.ScamState
import org.rakshak.edge.analysis.Tier
import org.rakshak.edge.analysis.canonicalize
import org.rakshak.edge.analysis.detectorOnlyPackage
import org.rakshak.edge.analysis.encodeBytes
import org.rakshak.edge.analysis.isActionSolicitation
import org.rakshak.edge.analysis.isDetectorOnly
import org.rakshak.edge.analysis.isPassiveTransactionalNotice
import org.rakshak.edge.analysis.isTransactionalNotice
import org.rakshak.edge.analysis.registeredDomain
import org.rakshak.edge.analysis.rxOf
import org.rakshak.edge.analysis.scamStateFor
import org.rakshak.edge.analysis.shouldEscalateDetectorOnly
import org.rakshak.edge.analysis.tierFor
import org.rakshak.edge.analysis.urlFeatures
import org.rakshak.edge.analysis.runRules
import org.rakshak.edge.analysis.attributionProblems
import org.rakshak.edge.analysis.collapseByType
import org.rakshak.edge.analysis.remediationFor
import org.rakshak.edge.analysis.buildPackage
import org.rakshak.edge.analysis.chooseAction
import org.rakshak.edge.llm.cleanSentence
import org.rakshak.edge.llm.duplicatePoints
import org.rakshak.edge.llm.sequenceRatio
import org.rakshak.edge.llm.slotStop
import org.rakshak.edge.llm.stripEcho

/** Behaviour checks that do not need the fixture: suffixes, cleanup, decision, encodings. */
class UnitTest {
    companion object {
        @JvmStatic
        @BeforeClass
        fun setUp() = TestData.install()
    }

    @Test
    fun public_suffix_lookup_follows_icann_rules() {
        assertEquals("example.co.in", registeredDomain("https://www.example.co.in/x"))
        assertEquals("evil.xyz", registeredDomain("http://sbi.co.in.evil.xyz/verify"))
        assertEquals("", registeredDomain("http://192.168.1.1/login"))
        assertEquals("github.io", registeredDomain("https://a.b.github.io/"))
        assertEquals("", registeredDomain("http://host.unknowntld"))
    }

    @Test
    fun url_host_is_read_without_userinfo_and_port() {
        val f = urlFeatures("http://user@SBI.co.in:8080/login")
        assertEquals("sbi.co.in", f.registeredDomain)
        assertTrue(f.hasAtSign)
    }

    @Test
    fun bracket_mismatch_falls_back_to_best_effort() {
        assertEquals("sbi-kyc.in", registeredDomain("[http://sbi-kyc.in/x"))
    }

    @Test
    fun zero_width_and_homoglyphs_are_folded() {
        val c = canonicalize("Your s​bi accоunt is blocked")
        assertEquals("Your sbi account is blocked", c.normalized)
    }

    @Test
    fun spaced_and_dotted_letters_collapse() {
        assertEquals("KYC update", canonicalize("K Y C update").normalized)
        assertEquals("sbi.kyc", canonicalize("s.b.i.kyc").normalized)
    }

    @Test
    fun obfuscated_url_is_extracted() {
        assertEquals(listOf("http://paytm.verify-kyc.top/login"), canonicalize("hxxp://paytm[.]verify-kyc.top/login").urls)
    }

    @Test
    fun devanagari_word_boundaries_use_the_python_word_class() {
        // Combining marks are not word characters in Python's \w, so a word ending in a matra has
        // no \b after it. The Kotlin port keeps that behaviour (see PyRegex.kt).
        assertTrue(rxOf("""\bfoo\b""").containsMatchIn("a foo b"))
        assertFalse(rxOf("""\bfoo\b""").containsMatchIn("foox"))
    }

    @Test
    fun negated_python_classes_translate() {
        // The verifier's URL pattern uses \S; Python's \S is "not Unicode whitespace".
        val url = rxOf("""\b(?:https?|hxxps?)://\S+""")
        assertEquals("https://x.in/a", url.find("go https://x.in/a now")?.value)
        assertEquals("12", rxOf("""\D(\d+)""").find("a12")?.groupValues?.get(1))
        assertTrue(rxOf("""o\Bo""").containsMatchIn("foo"))
        assertFalse(rxOf("""o\Bo""").containsMatchIn("o o"))
    }

    @Test
    fun scam_state_uses_threshold_then_evidence() {
        val ev = runRules(canonicalize("Your OTP is needed, share it now"))
        assertEquals(ScamState.SCAM, scamStateFor(SCAM_THRESHOLD, SCAM_THRESHOLD, emptyList()))
        assertEquals(ScamState.UNCERTAIN, scamStateFor(0.1, SCAM_THRESHOLD, ev))
        assertEquals(ScamState.BENIGN, scamStateFor(0.1, SCAM_THRESHOLD, emptyList()))
        assertTrue(isDetectorOnly(ScamState.SCAM, emptyList()))
    }

    @Test
    fun detector_only_message_is_low_and_ignored() {
        val canon = canonicalize("Meeting at 5pm tomorrow")
        val pkg = buildPackage(canon, runRules(canon), ScamState.SCAM)
        assertEquals(Risk.LOW, pkg.risk)
        assertEquals(Action.IGNORE, chooseAction(pkg))
        assertEquals(Tier.SAFE, tierFor(pkg.risk))
    }

    @Test
    fun tiers_map_from_risk() {
        assertEquals(Tier.SCAM, tierFor(Risk.HIGH))
        assertEquals(Tier.SCAM, tierFor(Risk.MEDIUM))
        assertEquals(Tier.VERIFY, tierFor(Risk.UNCERTAIN))
    }

    @Test
    fun byte_view_truncates_and_pads() {
        val ids = encodeBytes("a".repeat(300))
        assertEquals(256, ids.size)
        assertEquals(97L, ids[0])
        val short = encodeBytes("hi")
        assertEquals(256L, short[2])
    }

    @Test
    fun slot_sentence_is_cut_at_its_first_sentence_end() {
        val prefix = "This message asks"
        assertEquals("This message asks for an OTP, PIN or password." to "ok",
            cleanSentence(" for an OTP, PIN or password. Extra words", prefix))
        assertEquals("" to "unterminated", cleanSentence(" for an OTP", prefix))
        assertEquals("" to "empty", cleanSentence("   ", prefix))
    }

    @Test
    fun slot_stop_fires_on_sentence_end_or_stop_character() {
        assertTrue(slotStop(" for an OTP."))
        assertTrue(slotStop(" for an OTP\n"))
        assertFalse(slotStop(" for an OTP"))
    }

    @Test
    fun echo_is_removed_only_when_a_real_sentence_remains() {
        assertEquals("This message asks for an OTP." to true,
            stripEcho("This message asks for an OTP and restates the fact."))
        assertEquals("" to true, stripEcho("This message is and restates the fact."))
        assertEquals("This message asks for an OTP." to false, stripEcho("This message asks for an OTP."))
    }

    @Test
    fun duplicate_points_keep_the_first_occurrence() {
        assertEquals(listOf(null, 0), duplicatePoints(listOf("This message asks for an OTP.", "This message asks for an OTP!")))
        assertEquals(1.0, sequenceRatio("abc", "abc"), 1e-9)
    }

    @Test
    fun attribution_guard_rejects_plain_claims_at_low_and_allows_hedges_at_medium() {
        assertFalse(attributionProblems("This is a scam", Risk.LOW).isEmpty())
        assertTrue(attributionProblems("This may be a scam", Risk.MEDIUM).isEmpty())
        assertTrue(attributionProblems("This is a scam", Risk.HIGH).isEmpty())
    }

    @Test
    fun remediation_drops_ignore_when_other_codes_exist() {
        val canon = canonicalize("Your OTP is needed, share it now")
        val ev = runRules(canon)
        val pkg = buildPackage(canon, ev, ScamState.SCAM)
        val items = remediationFor(pkg, chooseAction(pkg), "en")
        assertFalse(items.any { it.code == Action.IGNORE.name } && items.size > 1)
        assertTrue(items.isNotEmpty())
    }

    @Test
    fun collapse_keeps_one_item_per_type_in_order() {
        val canon = canonicalize("Congratulations! You won a lottery prize. Claim prize today")
        val pkg = buildPackage(canon, runRules(canon), ScamState.UNCERTAIN)
        val collapsed = collapseByType(pkg)
        assertEquals(collapsed.evidence.map { it.type }.distinct(), collapsed.evidence.map { it.type })
    }

    @Test
    fun balanced_passive_transactional_and_verify_modes_match_pi_policy() {
        val passiveNotices = listOf(
            "SBI: A/c XX4821 debited by Rs.1,250.00 via UPI Ref 632189451022. Avl Bal Rs.34,820.50.",
            "Myntra: Your return for order 412890 has been processed. Refund of INR 1,299 initiated.",
            "TATA Power: Bill for November is INR 1,420. Due: 2026-12-05. Pay via UPI: 9812345670@oktatapower",
            "849201 is your SBI OTP for online payment. Valid for 5 mins. Do not share with anyone.",
        )
        val fixedHighScorer = StudentScorer { _, _, _ -> 0.97 }
        val extremeScorer = StudentScorer { _, _, _ -> 0.9995 }
        val verifyThresholdScorer = StudentScorer { _, _, _ -> 0.50 }

        for (msg in passiveNotices) {
            val canon = canonicalize(msg)
            assertTrue(msg, isPassiveTransactionalNotice(canon.normalized))

            // Quiet keeps LOW / IGNORE
            val quiet = Analyzer.analyse(msg, fixedHighScorer, mode = DetectorOnlyMode.QUIET)
            assertEquals(Risk.LOW, quiet.pkg.risk)
            assertEquals(Action.IGNORE, quiet.action)
            assertEquals("detector_only_quiet", quiet.advisory.path)

            // Balanced holds passive transactional notice at LOW / IGNORE at p=0.97
            val balMid = Analyzer.analyse(msg, fixedHighScorer, mode = DetectorOnlyMode.BALANCED)
            assertEquals(msg, Risk.LOW, balMid.pkg.risk)
            assertEquals(msg, Action.IGNORE, balMid.action)

            // Balanced escalates passive notice at extreme confidence p=0.9995
            val balHi = Analyzer.analyse(msg, extremeScorer, mode = DetectorOnlyMode.BALANCED)
            assertEquals(Risk.UNCERTAIN, balHi.pkg.risk)
            assertEquals(Action.VERIFY_INDEPENDENTLY, balHi.action)
            assertEquals("detector_only", balHi.advisory.path)
        }

        val actionScams = listOf(
            "Dear SBI customer, your A/c XX4821 is on hold. Call 9876543210 immediately to unblock.",
            "BESCOM Alert: Your electricity bill is overdue and will be disconnected tonight. Contact 8812345678.",
            "Axis Bank: Your account is frozen. Dial 9345678901 within 1 hour to reactivate.",
        )
        for (msg in actionScams) {
            val canon = canonicalize(msg)
            assertTrue(isTransactionalNotice(canon.normalized))
            assertTrue(isActionSolicitation(canon.normalized))
            assertFalse(isPassiveTransactionalNotice(canon.normalized))
            assertTrue(shouldEscalateDetectorOnly(canon.normalized, 0.97, DetectorOnlyMode.BALANCED))
        }
        // Detector-only action-soliciting transactional message (no rule evidence) escalates in BALANCED
        val actionNoRuleMsg = "BESCOM billing notice: Confirm your account payment status to avoid penalty."
        val actionNoRuleCanon = canonicalize(actionNoRuleMsg)
        assertTrue(runRules(actionNoRuleCanon).isEmpty())
        assertTrue(isTransactionalNotice(actionNoRuleCanon.normalized))
        assertTrue(isActionSolicitation(actionNoRuleCanon.normalized))
        assertFalse(isPassiveTransactionalNotice(actionNoRuleCanon.normalized))
        val balAction = Analyzer.analyse(actionNoRuleMsg, fixedHighScorer, mode = DetectorOnlyMode.BALANCED)
        assertEquals(Risk.UNCERTAIN, balAction.pkg.risk)
        assertEquals(Action.VERIFY_INDEPENDENTLY, balAction.action)
        assertEquals("detector_only", balAction.advisory.path)

        // General non-transactional message in verify mode vs balanced at p=0.50 vs p=0.97
        val plain = "Special exclusive bonus awaiting confirmation."
        val verifyRes = Analyzer.analyse(plain, verifyThresholdScorer, mode = DetectorOnlyMode.VERIFY)
        assertEquals(Risk.UNCERTAIN, verifyRes.pkg.risk)
        assertEquals(Action.VERIFY_INDEPENDENTLY, verifyRes.action)
        assertEquals("detector_only", verifyRes.advisory.path)

        val balGeneralMid = Analyzer.analyse(plain, verifyThresholdScorer, mode = DetectorOnlyMode.BALANCED)
        assertEquals(Risk.LOW, balGeneralMid.pkg.risk)
        assertEquals(Action.IGNORE, balGeneralMid.action)

        val balGeneralHi = Analyzer.analyse(plain, fixedHighScorer, mode = DetectorOnlyMode.BALANCED)
        assertEquals(Risk.UNCERTAIN, balGeneralHi.pkg.risk)
        assertEquals(Action.VERIFY_INDEPENDENTLY, balGeneralHi.action)
        assertFalse(shouldEscalateDetectorOnly(plain, 0.50, DetectorOnlyMode.QUIET))
        val rawPkg = buildPackage(canonicalize(plain), emptyList(), ScamState.SCAM)
        assertEquals(Risk.UNCERTAIN, detectorOnlyPackage(rawPkg).risk)
    }
}
