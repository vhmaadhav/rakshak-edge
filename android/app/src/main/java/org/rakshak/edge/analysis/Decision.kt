package org.rakshak.edge.analysis

/** Port of deploy/pi/service/edge_decision.py (quiet, balanced passive-tx, and verify modes). */

/** Threshold of the shipped SMS student (docs/SYSTEM_V1.md section 1a). */
const val SCAM_THRESHOLD: Double = 0.35127517580986023

/** Balanced-mode escalation thresholds matching deploy/pi/service/edge_decision.py. */
const val DEFAULT_VERIFY_THRESHOLD: Double = 0.9566
const val DEFAULT_TX_VERIFY_THRESHOLD: Double = 0.999

enum class DetectorOnlyMode(val label: String, val wireName: String) {
    QUIET("Quiet", "quiet"),
    BALANCED("Balanced", "balanced"),
    VERIFY("Verify", "verify"),
}

private val TX_NOTICE_RE = Regex(
    """\b(?:""" +
        """bank|a/c|acct|account|debit(?:ed)?|credit(?:ed)?|""" +
        """upi|vpa|imps|neft|rtgs|""" +
        """otp|one[ -]time password|verification code|""" +
        """parcel|courier|delivery|delivered|shipment|tracking|""" +
        """recharge|telecom|data pack|electricity|utility|bill|broadband|""" +
        """do not share|don.?t share|never share|share mat|mat share|""" +
        """balance|rs\.?|inr|emi|loan|statement|txn|transaction|order""" +
        """)\b""",
    RegexOption.IGNORE_CASE,
)

private val ACTION_CUE_RE = Regex(
    """\b(?:""" +
        """call|contact|ring|dial|whatsapp|reply|text back|sms back|click|tap|visit|open link|""" +
        """verify|confirm|update|reactivate|re-activate|unblock|unlock|kyc|suspend(?:ed)?|""" +
        """block(?:ed)?|deactivat(?:e|ed|ion)|disconnect(?:ed|ion)?|freeze|frozen|expired?|""" +
        """urgent|immediately|today|within \d+ (?:hr|hour|min)|last chance|final notice|""" +
        """claim|collect|won|winner|prize|reward|lottery|cashback|bonus|approved|""" +
        """free|congratulations|selected|shortlisted|apply now|pay now|overdue|penalty|fine|challan""" +
        """)\b|(?:(?<![\d@])(?:\+91[\s-]?)?[6-9]\d{9}(?!\d|@)|\b1800[\s-]?\d{3,4}[\s-]?\d{3,4}\b)""",
    RegexOption.IGNORE_CASE,
)

fun isTransactionalNotice(text: String): Boolean = TX_NOTICE_RE.containsMatchIn(text)

fun isActionSolicitation(text: String): Boolean = ACTION_CUE_RE.containsMatchIn(text)

fun isPassiveTransactionalNotice(text: String): Boolean =
    isTransactionalNotice(text) && !isActionSolicitation(text)

/** scam when score >= threshold; else uncertain when any rule fired; else benign. */
fun scamStateFor(score: Double, threshold: Double, evidence: List<Evidence>): ScamState = when {
    score >= threshold -> ScamState.SCAM
    evidence.isNotEmpty() -> ScamState.UNCERTAIN
    else -> ScamState.BENIGN
}

/** Detector-only flag: scam state with no rule evidence (the Fix B case). */
fun isDetectorOnly(state: ScamState, evidence: List<Evidence>): Boolean =
    state == ScamState.SCAM && evidence.isEmpty()

/** Escalates a detector-only package to UNCERTAIN / VERIFY_INDEPENDENTLY. */
fun detectorOnlyPackage(pkg: EvidencePackage): EvidencePackage =
    pkg.copy(risk = Risk.UNCERTAIN, allowedActions = listOf(Action.VERIFY_INDEPENDENTLY))

/**
 * Matches deploy/pi/service/edge_decision.py decide_edge:
 * - QUIET: never escalate detector-only messages.
 * - VERIFY: escalate every detector-only message.
 * - BALANCED: passive transactional notices require [txVerifyThreshold] (default 0.999);
 *   action-soliciting transactional messages and general messages escalate at [verifyThreshold] (0.9566).
 */
fun shouldEscalateDetectorOnly(
    canonText: String,
    score: Double,
    mode: DetectorOnlyMode,
    verifyThreshold: Double = DEFAULT_VERIFY_THRESHOLD,
    txVerifyThreshold: Double? = null,
): Boolean = when (mode) {
    DetectorOnlyMode.QUIET -> false
    DetectorOnlyMode.VERIFY -> true
    DetectorOnlyMode.BALANCED -> {
        val txThr = txVerifyThreshold ?: maxOf(DEFAULT_TX_VERIFY_THRESHOLD, verifyThreshold)
        val effThr = if (isPassiveTransactionalNotice(canonText)) txThr else verifyThreshold
        score >= effThr
    }
}

/** The user-facing tier. SAFE means "no known pattern" (LOW), never a guarantee. */
enum class Tier { SCAM, VERIFY, SAFE }

fun tierFor(risk: Risk): Tier = when (risk) {
    Risk.HIGH, Risk.MEDIUM -> Tier.SCAM
    Risk.UNCERTAIN -> Tier.VERIFY
    Risk.LOW -> Tier.SAFE
}
