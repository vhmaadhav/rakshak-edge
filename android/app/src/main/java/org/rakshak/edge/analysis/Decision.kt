package org.rakshak.edge.analysis

/** Port of deploy/pi/service/edge_decision.py in quiet mode, the Pi default. */

/** Threshold of the shipped SMS student (docs/SYSTEM_V1.md section 1a). */
const val SCAM_THRESHOLD: Double = 0.35127517580986023

/** scam when score >= threshold; else uncertain when any rule fired; else benign. */
fun scamStateFor(score: Double, threshold: Double, evidence: List<Evidence>): ScamState = when {
    score >= threshold -> ScamState.SCAM
    evidence.isNotEmpty() -> ScamState.UNCERTAIN
    else -> ScamState.BENIGN
}

/** Detector-only flag: scam state with no rule evidence (the quiet-mode case). */
fun isDetectorOnly(state: ScamState, evidence: List<Evidence>): Boolean =
    state == ScamState.SCAM && evidence.isEmpty()

/** The user-facing tier. SAFE means "no known pattern" (LOW), never a guarantee. */
enum class Tier { SCAM, VERIFY, SAFE }

fun tierFor(risk: Risk): Tier = when (risk) {
    Risk.HIGH, Risk.MEDIUM -> Tier.SCAM
    Risk.UNCERTAIN -> Tier.VERIFY
    Risk.LOW -> Tier.SAFE
}
