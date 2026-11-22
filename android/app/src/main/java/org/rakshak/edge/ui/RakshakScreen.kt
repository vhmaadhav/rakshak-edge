package org.rakshak.edge.ui

import android.os.Build
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.rakshak.edge.Analysis
import org.rakshak.edge.Engine
import org.rakshak.edge.SharedMessage
import org.rakshak.edge.analysis.Tier
import org.rakshak.edge.analysis.TYPE_PHRASE_EN
import org.rakshak.edge.data.RuleData
import org.rakshak.edge.perf.InferenceTiming

/** Verdict, evidence, remediation and the optional explanation for one message. */
@Composable
fun RakshakScreen(engine: Engine, shared: SharedMessage?) {
    val scope = rememberCoroutineScope()
    var text by rememberSaveable { mutableStateOf("") }
    var analysis by remember { mutableStateOf<Analysis?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var showAbout by remember { mutableStateOf(false) }
    var activeRequest by remember { mutableStateOf<Job?>(null) }
    var pendingFrame by remember { mutableStateOf<InferenceFrame?>(null) }

    fun submit(timing: InferenceTiming?, explaining: Boolean = false, run: suspend () -> Result<Analysis>) {
        activeRequest?.cancel()
        pendingFrame = null
        busy = true
        error = null
        activeRequest = scope.launch {
            run().onSuccess { result ->
                analysis = result
                pendingFrame = timing?.let {
                    if (explaining) {
                        val points = result.advisory.points
                        InferenceFrame(it, result.advisory.mode, points.count { p -> p.source == "llm" }, points.size)
                    } else {
                        InferenceFrame(it, "verdict")
                    }
                }
            }.onFailure {
                val message = it.message ?: "Analysis failed"
                error = if (explaining) "Explanation failed: $message" else message
                pendingFrame = timing?.let { sample -> InferenceFrame(sample, "error") }
            }
            busy = false
        }
    }

    LaunchedEffect(shared) {
        if (shared != null && shared.text.isNotBlank()) {
            text = shared.text
            submit(shared.timing) { engine.check(shared.text, shared.timing) }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Rakshak Edge", style = androidx.compose.material3.MaterialTheme.typography.headlineSmall)
        Text(
            "Checks a message on this phone. The app asks for no permissions and sends nothing anywhere.",
            style = androidx.compose.material3.MaterialTheme.typography.bodySmall,
        )
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.fillMaxWidth(),
            minLines = 4,
            label = { Text("Paste an SMS or message") },
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                val message = text
                val timing = InferenceTiming.start("check")
                submit(timing) { engine.check(message, timing) }
            }, enabled = text.isNotBlank() && !busy) {
                Text(if (busy) "Checking..." else "Check message")
            }
            TextButton(onClick = { showAbout = true }) { Text("About") }
        }
        error?.let { Text(it, color = androidx.compose.material3.MaterialTheme.colorScheme.error) }

        analysis?.let { a ->
            ResultCard(
                analysis = a,
                explainerInstalled = engine.explainerInstalled(),
                onExplain = {
                    val timing = InferenceTiming.start("explain")
                    submit(timing, explaining = true) { engine.explain(a, timing) }
                },
                busy = busy,
            )
        }
    }

    pendingFrame?.let { RecordInferenceFrame(it) }

    if (showAbout) {
        AboutDialog(onDismiss = { showAbout = false })
    }
}

private data class InferenceFrame(
    val timing: InferenceTiming,
    val outcome: String,
    val modelPoints: Int? = null,
    val totalPoints: Int? = null,
) {
    fun record(submitted: Boolean) = timing.finish(outcome, submitted, modelPoints, totalPoints)
}

@Composable
private fun RecordInferenceFrame(frame: InferenceFrame) {
    val view = LocalView.current
    DisposableEffect(view, frame) {
        val observer = view.viewTreeObserver
        if (Build.VERSION.SDK_INT >= 29 && view.isHardwareAccelerated && observer.isAlive) {
            val callback = Runnable { frame.record(submitted = true) }
            observer.registerFrameCommitCallback(callback)
            view.invalidate()
            onDispose { if (observer.isAlive) observer.unregisterFrameCommitCallback(callback) }
        } else {
            frame.record(submitted = false)
            onDispose { }
        }
    }
}

@Composable
private fun ResultCard(
    analysis: Analysis,
    explainerInstalled: Boolean,
    onExplain: () -> Unit,
    busy: Boolean,
) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(tierLabel(analysis.tier), style = androidx.compose.material3.MaterialTheme.typography.titleLarge)
            if (analysis.tier == Tier.SAFE) {
                Text(
                    "No known scam pattern was found. This does not make the message safe.",
                    style = androidx.compose.material3.MaterialTheme.typography.bodySmall,
                )
            }
            Text(analysis.advisory.summary)

            if (analysis.advisory.points.isNotEmpty()) {
                Text("What the checks found", style = androidx.compose.material3.MaterialTheme.typography.titleSmall)
                analysis.advisory.points.forEach { p ->
                    val origin = if (p.source == "llm") "explained by Llama" else "template"
                    Text("- ${p.text} ($origin)")
                }
            }

            if (analysis.evidence.isNotEmpty()) {
                Text("Evidence", style = androidx.compose.material3.MaterialTheme.typography.titleSmall)
                analysis.evidence.forEach { e ->
                    val phrase = TYPE_PHRASE_EN[e.type] ?: e.type
                    Text("${e.id}: $phrase (matched \"${e.text.take(60)}\")")
                }
            }

            if (analysis.remediation.isNotEmpty()) {
                Text("What to do", style = androidx.compose.material3.MaterialTheme.typography.titleSmall)
                analysis.remediation.forEach { item ->
                    Text(item.title, style = androidx.compose.material3.MaterialTheme.typography.bodyMedium)
                    item.steps.forEach { step -> Text("  - $step") }
                }
            }

            val explainable = !analysis.detectorOnly && analysis.lang != "hinglish" &&
                analysis.advisory.path != "explainer" && analysis.pkg.risk != org.rakshak.edge.analysis.Risk.LOW
            if (explainable) {
                if (explainerInstalled) {
                    OutlinedButton(onClick = onExplain, enabled = !busy) { Text("Explain with Llama") }
                } else {
                    Text(
                        "Explanation model not installed. Showing the built-in wording. " +
                            "See docs/ANDROID_APP.md to add the model.",
                        style = androidx.compose.material3.MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
    }
}

private fun tierLabel(tier: Tier): String = when (tier) {
    Tier.SCAM -> "SCAM"
    Tier.VERIFY -> "VERIFY"
    Tier.SAFE -> "SAFE (no known pattern)"
}

@Composable
private fun AboutDialog(onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        confirmButton = { TextButton(onClick = onDismiss) { Text("Close") } },
        title = { Text("About Rakshak Edge") },
        text = {
            Column(modifier = Modifier.verticalScroll(rememberScrollState())) {
                Text("Built with Llama.")
                Text(
                    "The explanation layer uses Llama 3.2 1B Instruct under the Llama 3.2 Community " +
                        "License. The license text is below.",
                    style = androidx.compose.material3.MaterialTheme.typography.bodySmall,
                )
                Text(
                    runCatching { RuleData.llamaLicense }.getOrDefault("License text unavailable."),
                    style = androidx.compose.material3.MaterialTheme.typography.bodySmall,
                )
            }
        },
    )
}
