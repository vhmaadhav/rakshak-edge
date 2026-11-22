package org.rakshak.edge

import org.rakshak.edge.analysis.Action
import org.rakshak.edge.analysis.Canon
import org.rakshak.edge.analysis.EvidencePackage
import org.rakshak.edge.analysis.Evidence
import org.rakshak.edge.analysis.SCAM_THRESHOLD
import org.rakshak.edge.analysis.ScamState
import org.rakshak.edge.analysis.Tier
import org.rakshak.edge.analysis.buildPackage
import org.rakshak.edge.analysis.canonicalize
import org.rakshak.edge.analysis.chooseAction
import org.rakshak.edge.analysis.collapseByType
import org.rakshak.edge.analysis.detectLang
import org.rakshak.edge.analysis.isDetectorOnly
import org.rakshak.edge.analysis.ruleFlags
import org.rakshak.edge.analysis.RemediationItem
import org.rakshak.edge.analysis.remediationFor
import org.rakshak.edge.analysis.routeLang
import org.rakshak.edge.analysis.runRules
import org.rakshak.edge.analysis.scamStateFor
import org.rakshak.edge.analysis.templatePoint
import org.rakshak.edge.analysis.templateSummary
import org.rakshak.edge.analysis.tierFor
import org.rakshak.edge.llm.Advisory
import org.rakshak.edge.llm.PointView
import org.rakshak.edge.llm.SlotExplainer
import org.rakshak.edge.llm.TokenGenerator
import org.rakshak.edge.perf.InferenceTiming
import org.rakshak.edge.perf.measure
import org.rakshak.edge.perf.measureRules

/** The student score for one message. The ONNX implementation is ml.StudentDetector. */
fun interface StudentScorer {
    fun score(text: String, canon: Canon, side: FloatArray): Double
}

/** Everything the screen shows for one message, before or after the explainer. */
data class Analysis(
    val text: String,
    val lang: String,
    val score: Double,
    val state: ScamState,
    val evidence: List<Evidence>,
    val pkg: EvidencePackage,
    val action: Action,
    val detectorOnly: Boolean,
    val remediation: List<RemediationItem>,
    val advisory: Advisory,
) {
    val tier: Tier get() = tierFor(pkg.risk)
}

/** Quiet-mode texts of deploy/pi/service/edge_pipeline.py (Fix B). */
private const val DETECTOR_ONLY_QUIET_EN = "No specific scam pattern was found in this message."
private const val DETECTOR_ONLY_QUIET_HI = "Is message mein koi specific scam pattern nahi mila."

internal data class PreparedMessage(val canon: Canon, val evidence: List<Evidence>, val flags: FloatArray)

/**
 * The decision and the template advisory for one message. No model is called here; [explain]
 * adds the Llama sentences on top.
 */
object Analyzer {
    fun analyse(
        text: String,
        scorer: StudentScorer,
        threshold: Double = SCAM_THRESHOLD,
        timing: InferenceTiming? = null,
    ): Analysis {
        return analyse(text, prepare(text, timing), scorer, threshold, timing)
    }

    internal fun prepare(text: String, timing: InferenceTiming?): PreparedMessage {
        val canon = timing.measure("canonicalize") { canonicalize(text) }
        val evidence = timing.measureRules { runRules(canon, timing) }
        val flags = timing.measure("rule_flags") { ruleFlags(evidence) }
        return PreparedMessage(canon, evidence, flags)
    }

    internal fun analyse(
        text: String,
        prepared: PreparedMessage,
        scorer: StudentScorer,
        threshold: Double = SCAM_THRESHOLD,
        timing: InferenceTiming? = null,
    ): Analysis {
        val (canon, evidence, flags) = prepared
        val score = scorer.score(text, canon, flags)
        return timing.measure("decision") { decide(text, canon, score, evidence, threshold) }
    }

    private fun decide(text: String, canon: Canon, score: Double, evidence: List<Evidence>, threshold: Double): Analysis {
        val state = scamStateFor(score, threshold, evidence)
        val pkg = buildPackage(canon, evidence, state)
        val action = chooseAction(pkg)
        val lang = routeLang(detectLang(text))
        val detectorOnly = isDetectorOnly(state, evidence)
        val remediation = remediationFor(pkg, action, lang)
        val advisory = when {
            detectorOnly -> Advisory(
                path = "detector_only_quiet",
                mode = "template",
                summary = if (lang == "hinglish") DETECTOR_ONLY_QUIET_HI else DETECTOR_ONLY_QUIET_EN,
                points = emptyList(),
                g2Rejected = emptyList(),
            )
            lang == "hinglish" -> Advisory(
                path = "template_hinglish",
                mode = "template",
                summary = templateSummary(pkg, "hinglish"),
                points = collapseByType(pkg).evidence.map {
                    PointView(it.id, it.type, templatePoint(it.type, "hinglish"), "template", emptyList())
                },
                g2Rejected = emptyList(),
            )
            else -> SlotExplainer.templateOnly(
                collapseByType(pkg),
                "template_no_model",
                templateSummary(pkg, "en"),
            )
        }
        return Analysis(text, lang, score, state, evidence, pkg, action, detectorOnly, remediation, advisory)
    }

    /**
     * Adds the explainer sentences. Quiet-mode detector-only messages and Hinglish messages
     * (template mode, the Pi default) make no model call and keep the template advisory.
     */
    suspend fun explain(
        analysis: Analysis,
        generator: TokenGenerator?,
        onVerifiedPoint: (() -> Unit)? = null,
    ): Analysis {
        if (analysis.detectorOnly || analysis.lang == "hinglish") return analysis
        return analysis.copy(advisory = SlotExplainer.advise(analysis.pkg, generator, onVerifiedPoint))
    }
}
