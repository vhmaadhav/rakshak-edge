package org.rakshak.edge.ui

import android.os.Build
import android.provider.Settings
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.roundToInt
import org.rakshak.edge.Analysis
import org.rakshak.edge.Engine
import org.rakshak.edge.SharedMessage
import org.rakshak.edge.analysis.Action
import org.rakshak.edge.analysis.Evidence
import org.rakshak.edge.analysis.RemediationItem
import org.rakshak.edge.analysis.Risk
import org.rakshak.edge.analysis.Tier
import org.rakshak.edge.analysis.TYPE_PHRASE_EN
import org.rakshak.edge.data.RuleData
import org.rakshak.edge.perf.InferenceTiming

/** Synthetic example texts for the demo. They are invented, not real messages. */
private data class Sample(val label: String, val text: String)

private val SAMPLES = listOf(
    Sample(
        "Try: KYC scam",
        "Dear customer, your bank KYC will be blocked today. Update it now at " +
            "http://kyc-update.example.test/verify or your account will be suspended.",
    ),
    Sample(
        "Try: OTP bank alert",
        "Your OTP for a Rs 4500 transfer is 000000. Do not share this OTP with anyone, " +
            "including bank staff.",
    ),
    Sample(
        "Try: Hinglish job scam",
        "Ghar baithe part-time kaam. Roz Rs 2000 kamayein. Task complete karne par turant " +
            "paisa milega. Registration fee bhejein aur reply karein.",
    ),
    Sample(
        "Try: safe delivery SMS",
        "Your parcel is out for delivery today. Track it in the official courier app. " +
            "No action is needed.",
    ),
)

private const val STAGGER_MS = 60L

/** Verdict, evidence, remediation and the optional explanation for one message. */
@Composable
fun RakshakScreen(engine: Engine, shared: SharedMessage?) {
    RakshakTheme {
        RakshakContent(engine, shared)
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun RakshakContent(engine: Engine, shared: SharedMessage?) {
    val c = LocalRakshakColors.current
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

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(c.bg)
            .drawBehind {
                drawRect(
                    brush = Brush.radialGradient(
                        colors = listOf(c.accent.copy(alpha = 0.20f), Color.Transparent),
                        center = Offset(size.width * 0.15f, 0f),
                        radius = size.width * 0.9f,
                    ),
                )
            },
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Header()
            StatusChips()

            PanelCard {
                Text(
                    "Checks a message on this phone. The app asks for no permissions and sends nothing anywhere.",
                    style = MaterialTheme.typography.bodySmall,
                    color = c.muted,
                )
                Text("Try a synthetic example", style = MaterialTheme.typography.labelLarge, color = c.muted)
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    SAMPLES.forEach { sample ->
                        SampleChip(label = sample.label) { text = sample.text }
                    }
                }
                OutlinedTextField(
                    value = text,
                    onValueChange = { text = it },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 4,
                    label = { Text("Paste an SMS or message") },
                )
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Button(
                        onClick = {
                            val message = text
                            val timing = InferenceTiming.start("check")
                            submit(timing) { engine.check(message, timing) }
                        },
                        enabled = text.isNotBlank() && !busy,
                        modifier = Modifier.heightIn(min = 48.dp),
                    ) {
                        Text(if (busy) "Checking..." else "Check message")
                    }
                    TextButton(
                        onClick = { showAbout = true },
                        modifier = Modifier.heightIn(min = 48.dp),
                    ) { Text("About") }
                }
                error?.let {
                    Text(it, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.error)
                }
            }

            if (busy && analysis == null) {
                ScanningCard()
            }

            analysis?.let { a ->
                FadeUp(0) { VerdictHero(a) }
                if (a.evidence.isNotEmpty()) {
                    FadeUp(1) { EvidenceCard(a) }
                    FadeUp(2) { PreviewCard(a) }
                }
                if (a.remediation.isNotEmpty()) {
                    FadeUp(3) { RemediationCard(a.remediation, a.tier) }
                }
                FadeUp(4) {
                    AdvisoryCard(
                        analysis = a,
                        explainerInstalled = engine.explainerInstalled(),
                        busy = busy,
                        onExplain = {
                            val timing = InferenceTiming.start("explain")
                            submit(timing, explaining = true) { engine.explain(a, timing) }
                        },
                    )
                }
            }
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

// ---------------------------------------------------------------------------
// Header and input
// ---------------------------------------------------------------------------

@Composable
private fun Header() {
    val c = LocalRakshakColors.current
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        ShieldMark(modifier = Modifier.size(44.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(
                "Rakshak Edge",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = c.text,
            )
            Text(
                "On-device threat detection. Nothing leaves this device.",
                style = MaterialTheme.typography.bodySmall,
                color = c.muted,
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun StatusChips() {
    val c = LocalRakshakColors.current
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Chip("● OFFLINE", c.low)
        Chip("This phone", c.accent2)
        Chip("SMS student · ONNX", c.accent)
        Chip("Llama 3.2 1B · ExecuTorch", c.accent)
    }
}

@Composable
private fun ShieldMark(modifier: Modifier = Modifier) {
    val c = LocalRakshakColors.current
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val shield = Path().apply {
            moveTo(w * 0.5f, 0f)
            lineTo(w, h * 0.14f)
            lineTo(w, h * 0.48f)
            cubicTo(w, h * 0.76f, w * 0.74f, h * 0.92f, w * 0.5f, h)
            cubicTo(w * 0.26f, h * 0.92f, 0f, h * 0.76f, 0f, h * 0.48f)
            lineTo(0f, h * 0.14f)
            close()
        }
        drawPath(
            path = shield,
            brush = Brush.linearGradient(
                colors = listOf(c.accent, c.accent2),
                start = Offset(0f, 0f),
                end = Offset(w, h),
            ),
        )
        val tick = Path().apply {
            moveTo(w * 0.32f, h * 0.5f)
            lineTo(w * 0.46f, h * 0.64f)
            lineTo(w * 0.7f, h * 0.38f)
        }
        drawPath(
            path = tick,
            color = c.bg,
            style = Stroke(width = w * 0.08f, cap = StrokeCap.Round),
        )
    }
}

@Composable
private fun Chip(text: String, color: Color, modifier: Modifier = Modifier) {
    val shape = RoundedCornerShape(percent = 50)
    Box(
        modifier = modifier
            .clip(shape)
            .background(color.copy(alpha = 0.14f))
            .border(1.dp, color.copy(alpha = 0.45f), shape)
            .padding(horizontal = 10.dp, vertical = 6.dp),
    ) {
        Text(text, style = MaterialTheme.typography.labelMedium, color = color)
    }
}

@Composable
private fun SampleChip(label: String, onClick: () -> Unit) {
    val c = LocalRakshakColors.current
    Surface(
        onClick = onClick,
        modifier = Modifier.heightIn(min = 48.dp),
        shape = RoundedCornerShape(percent = 50),
        color = c.surface2,
        border = BorderStroke(1.dp, c.line),
    ) {
        Text(
            label,
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
            style = MaterialTheme.typography.labelLarge,
            color = c.accent,
        )
    }
}

// ---------------------------------------------------------------------------
// Result cards
// ---------------------------------------------------------------------------

/** Rounded surface card. [glow] adds a soft risk-coloured halo; [scanPhase] draws a sweep. */
@Composable
private fun PanelCard(
    glow: Color? = null,
    scanPhase: Float? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    val c = LocalRakshakColors.current
    val shape = RoundedCornerShape(20.dp)
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(shape)
            .background(c.surface)
            .drawBehind {
                if (glow != null) {
                    drawRect(
                        brush = Brush.radialGradient(
                            colors = listOf(glow, Color.Transparent),
                            center = Offset(size.width / 2f, 0f),
                            radius = size.width * 0.8f,
                        ),
                    )
                }
                if (scanPhase != null) {
                    val w = size.width
                    val x = -w + scanPhase * 2f * w
                    drawRect(
                        brush = Brush.linearGradient(
                            colors = listOf(Color.Transparent, c.accent.copy(alpha = 0.16f), Color.Transparent),
                            start = Offset(x, 0f),
                            end = Offset(x + w * 0.6f, size.height),
                        ),
                    )
                }
            }
            .border(1.dp, c.line, shape)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        content = content,
    )
}

@Composable
private fun SectionTitle(title: String) {
    val c = LocalRakshakColors.current
    Text(title, style = MaterialTheme.typography.titleSmall, color = c.muted)
}

@Composable
private fun ScanningCard() {
    val c = LocalRakshakColors.current
    val transition = rememberInfiniteTransition(label = "scan")
    val phase by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 1400, easing = LinearEasing),
            repeatMode = RepeatMode.Restart,
        ),
        label = "scan-phase",
    )
    PanelCard(scanPhase = phase) {
        Text(
            "Checking on this phone",
            style = MaterialTheme.typography.titleMedium,
            color = c.text,
        )
        Text(
            "Scoring the message and matching rules. Nothing leaves this device.",
            style = MaterialTheme.typography.bodySmall,
            color = c.muted,
        )
    }
}

@Composable
private fun VerdictHero(analysis: Analysis) {
    val c = LocalRakshakColors.current
    val reduced = rememberReducedMotion()
    val tier = analysis.tier
    val color = tierColor(tier, c)
    val target = if (analysis.score.isNaN()) 0f else analysis.score.toFloat().coerceIn(0f, 1f)
    val percent = if (analysis.score.isNaN()) {
        "score --"
    } else {
        "score ${(analysis.score * 100.0).coerceIn(0.0, 100.0).roundToInt()}%"
    }
    val sweep = remember { Animatable(0f) }
    LaunchedEffect(analysis) {
        if (reduced) {
            sweep.snapTo(target)
        } else {
            sweep.snapTo(0f)
            sweep.animateTo(target, tween(durationMillis = 600, easing = FastOutSlowInEasing))
        }
    }

    PanelCard(glow = color.copy(alpha = 0.22f)) {
        SectionTitle("Verdict")
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            VerdictGauge(
                progress = sweep.value,
                color = color,
                centerWord = tierWord(tier),
                centerSub = percent,
            )
            Text(
                actionLabel(analysis.action),
                style = MaterialTheme.typography.titleMedium,
                color = c.text,
                textAlign = TextAlign.Center,
            )
            if (tier == Tier.SAFE) {
                Text(
                    "No known scam pattern was found. This does not make the message safe.",
                    style = MaterialTheme.typography.bodySmall,
                    color = c.muted,
                    textAlign = TextAlign.Center,
                )
            }
            Text(
                analysis.advisory.summary,
                style = MaterialTheme.typography.bodyMedium,
                color = c.text,
                textAlign = TextAlign.Center,
            )
        }
    }
}

@Composable
private fun EvidenceCard(analysis: Analysis) {
    PanelCard {
        SectionTitle("Evidence")
        analysis.evidence.forEach { e ->
            EvidenceItem(e)
        }
    }
}

@Composable
private fun EvidenceItem(e: Evidence) {
    val c = LocalRakshakColors.current
    val phrase = TYPE_PHRASE_EN[e.type] ?: e.type
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Chip("${e.id} · $phrase", c.accent)
        Text(
            text = "\"${e.text.take(60)}\"",
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(c.surface2)
                .padding(10.dp),
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            color = c.text,
        )
    }
}

@Composable
private fun PreviewCard(analysis: Analysis) {
    val c = LocalRakshakColors.current
    val highlight = tierColor(analysis.tier, c)
    PanelCard {
        SectionTitle("Message preview")
        Text(
            text = highlightMatches(analysis.text, analysis.evidence, highlight),
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(c.surface2)
                .padding(12.dp),
            style = MaterialTheme.typography.bodyMedium,
            color = c.text,
        )
    }
}

/** Case-insensitive highlight of every evidence text inside the message. Bounds are checked. */
private fun highlightMatches(message: String, evidence: List<Evidence>, color: Color): AnnotatedString =
    buildAnnotatedString {
        append(message)
        val style = SpanStyle(color = color, fontWeight = FontWeight.Bold)
        evidence.forEach { e ->
            val needle = e.text
            if (needle.isEmpty()) return@forEach
            var from = 0
            while (from < message.length) {
                val at = message.indexOf(needle, from, ignoreCase = true)
                if (at < 0) break
                val end = at + needle.length
                if (end > message.length) break
                addStyle(style, at, end)
                from = end
            }
        }
    }

@Composable
private fun RemediationCard(items: List<RemediationItem>, tier: Tier) {
    val c = LocalRakshakColors.current
    val rail = tierColor(tier, c)
    PanelCard {
        SectionTitle("What to do")
        items.forEach { item ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(IntrinsicSize.Min),
            ) {
                Box(
                    modifier = Modifier
                        .width(4.dp)
                        .fillMaxHeight()
                        .clip(RoundedCornerShape(percent = 50))
                        .background(rail),
                ) {}
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .padding(start = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text(item.title, style = MaterialTheme.typography.bodyMedium, color = c.text)
                    item.steps.forEachIndexed { index, step ->
                        Text(
                            "${index + 1}. $step",
                            style = MaterialTheme.typography.bodySmall,
                            color = c.muted,
                        )
                    }
                }
            }
        }
    }
}

private fun isExplainable(analysis: Analysis): Boolean =
    !analysis.detectorOnly && analysis.lang != "hinglish" &&
        analysis.advisory.path != "explainer" && analysis.pkg.risk != Risk.LOW

@Composable
private fun AdvisoryCard(
    analysis: Analysis,
    explainerInstalled: Boolean,
    busy: Boolean,
    onExplain: () -> Unit,
) {
    val c = LocalRakshakColors.current
    val explainable = isExplainable(analysis)
    val points = analysis.advisory.points
    if (points.isEmpty() && !explainable) return

    PanelCard {
        if (points.isNotEmpty()) {
            SectionTitle("What the checks found")
            points.forEach { p ->
                val fromLlm = p.source == "llm"
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.Top,
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Chip(if (fromLlm) "Llama" else "Template", if (fromLlm) c.accent else c.muted)
                    Column(
                        modifier = Modifier.weight(1f),
                        verticalArrangement = Arrangement.spacedBy(2.dp),
                    ) {
                        Text(p.text, style = MaterialTheme.typography.bodyMedium, color = c.text)
                        if (fromLlm) {
                            Text(
                                "Verifier ✓",
                                style = MaterialTheme.typography.labelSmall,
                                color = c.low,
                            )
                        } else if (p.problems.isNotEmpty()) {
                            Text(
                                "Verifier ✗: Llama sentence replaced by template",
                                style = MaterialTheme.typography.labelSmall,
                                color = c.high,
                            )
                        }
                    }
                }
            }
        }

        if (explainable) {
            if (explainerInstalled) {
                OutlinedButton(
                    onClick = onExplain,
                    enabled = !busy,
                    modifier = Modifier.heightIn(min = 48.dp),
                ) { Text("Explain with Llama") }
            } else {
                Text(
                    "Explanation model not installed. Showing the built-in wording. " +
                        "See docs/ANDROID_APP.md to add the model.",
                    style = MaterialTheme.typography.bodySmall,
                    color = c.muted,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Motion and helpers
// ---------------------------------------------------------------------------

/** True when the system animator duration scale is 0 (the Android reduced-motion setting). */
@Composable
private fun rememberReducedMotion(): Boolean {
    val context = LocalContext.current
    return remember(context) {
        runCatching {
            Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
        }.getOrDefault(false)
    }
}

/** Fades a card in and slides it up 16 dp, [index] x 60 ms after it first appears. */
@Composable
private fun FadeUp(index: Int, content: @Composable () -> Unit) {
    val reduced = rememberReducedMotion()
    val progress = remember { Animatable(if (reduced) 1f else 0f) }
    val density = LocalDensity.current.density
    LaunchedEffect(Unit) {
        if (!reduced) {
            delay(index * STAGGER_MS)
            progress.animateTo(1f, tween(durationMillis = 450, easing = FastOutSlowInEasing))
        }
    }
    Box(
        modifier = Modifier.graphicsLayer {
            alpha = progress.value
            translationY = (1f - progress.value) * 16f * density
        },
    ) {
        content()
    }
}

private fun tierWord(tier: Tier): String = when (tier) {
    Tier.SCAM -> "SCAM"
    Tier.VERIFY -> "VERIFY"
    Tier.SAFE -> "SAFE"
}

private fun actionLabel(action: Action): String = when (action) {
    Action.DO_NOT_CLICK -> "Do not click any link in this message"
    Action.DO_NOT_OPEN_LINK -> "Do not open the link in this message"
    Action.DO_NOT_SHARE_OTP -> "Do not share any OTP, PIN or password"
    Action.OPEN_OFFICIAL_APP -> "Use the official app or website instead"
    Action.CONTACT_BANK_OFFICIAL -> "Contact the organisation on its official number"
    Action.REPORT_1930_OR_CHAKSHU -> "Report it on 1930 or Chakshu"
    Action.BLOCK_SENDER -> "Block the sender"
    Action.IGNORE -> "No action needed from this check"
    Action.VERIFY_INDEPENDENTLY -> "Verify through an official channel"
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
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    runCatching { RuleData.llamaLicense }.getOrDefault("License text unavailable."),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        },
    )
}
