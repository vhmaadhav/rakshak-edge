package org.rakshak.edge.ui

import android.os.Build
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
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
import androidx.compose.runtime.mutableStateListOf
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
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import org.rakshak.edge.Analysis
import org.rakshak.edge.Engine
import org.rakshak.edge.SharedMessage
import org.rakshak.edge.analysis.Action
import org.rakshak.edge.analysis.DetectorOnlyMode
import org.rakshak.edge.analysis.Evidence
import org.rakshak.edge.analysis.RemediationItem
import org.rakshak.edge.analysis.Risk
import org.rakshak.edge.analysis.SCAM_THRESHOLD
import org.rakshak.edge.analysis.TYPE_PHRASE_EN
import org.rakshak.edge.analysis.Tier
import org.rakshak.edge.data.RuleData
import org.rakshak.edge.llm.PointView
import org.rakshak.edge.perf.InferenceTiming

/** Synthetic example texts. They are invented for the demo, not real messages. */
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

private const val TAB_HOME = 0
private const val TAB_CHECK = 1
private const val TAB_PROOF = 2
private const val MAX_HISTORY = 20
private const val MAX_SIGNS = 5
private val ITEM_HEIGHT = 58.dp
private val PARENTHETICAL = Regex("""\s*\([^)]*\)""")

/** State of the detector as seen by this session's checks. */
internal enum class DetectorStatus { NOT_TRIED, LOADED, ERROR }

/** One verdict from this session. Kept in memory only, never persisted. */
internal data class SessionEntry(
    val id: Long,
    val atMillis: Long,
    val checkMs: Long,
    val analysis: Analysis,
    val explainMs: Long?,
)

private fun elapsedMs(startNs: Long): Long = (System.nanoTime() - startNs) / 1_000_000L

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
    val context = LocalContext.current
    val haptics = LocalHapticFeedback.current
    val scope = rememberCoroutineScope()
    val reducedMotion = rememberReducedMotion()
    var selectedTab by rememberSaveable { mutableStateOf(TAB_HOME) }
    var detectorMode by rememberSaveable { mutableStateOf(DetectorOnlyMode.QUIET) }
    var text by rememberSaveable { mutableStateOf("") }
    var analysis by remember { mutableStateOf<Analysis?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }
    var showAbout by remember { mutableStateOf(false) }
    var activeRequest by remember { mutableStateOf<Job?>(null) }
    var pendingFrame by remember { mutableStateOf<InferenceFrame?>(null) }
    val history = remember { mutableStateListOf<SessionEntry>() }
    var currentEntryId by remember { mutableStateOf<Long?>(null) }
    var checkMs by remember { mutableStateOf<Long?>(null) }
    var explainMs by remember { mutableStateOf<Long?>(null) }
    var detectorStatus by remember { mutableStateOf(DetectorStatus.NOT_TRIED) }
    var verdictSerial by remember { mutableStateOf(0) }
    val explainerInstalled = remember { engine.explainerInstalled() }
    val permissions = remember(context) { requestedPermissions(context) }
    val offline = permissions != null && "android.permission.INTERNET" !in permissions
    val threat = history.firstOrNull()?.analysis?.tier == Tier.SCAM
    val wash = if (threat) c.high else c.accent
    val scroll = rememberScrollState()
    val transition = rememberInfiniteTransition(label = "aurora")
    val drift = transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 16000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "aurora-drift",
    )

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

    /** engine.check with a wall-clock time around it. A success is recorded in this session. */
    suspend fun runCheck(message: String, timing: InferenceTiming?): Result<Analysis> {
        val startNs = System.nanoTime()
        val result = engine.check(message, timing, mode = detectorMode)
        val ms = elapsedMs(startNs)
        result.onSuccess { fresh ->
            val id = System.nanoTime()
            currentEntryId = id
            checkMs = ms
            explainMs = null
            detectorStatus = DetectorStatus.LOADED
            history.add(0, SessionEntry(id = id, atMillis = System.currentTimeMillis(), checkMs = ms, analysis = fresh, explainMs = null))
            while (history.size > MAX_HISTORY) {
                history.removeAt(history.lastIndex)
            }
            verdictSerial += 1
        }.onFailure {
            detectorStatus = DetectorStatus.ERROR
        }
        return result
    }

    /** engine.explain with a wall-clock time around it. The session entry keeps the explanation. */
    suspend fun runExplain(base: Analysis, timing: InferenceTiming?): Result<Analysis> {
        val startNs = System.nanoTime()
        val result = engine.explain(base, timing)
        val ms = elapsedMs(startNs)
        result.onSuccess { updated ->
            explainMs = ms
            val index = history.indexOfFirst { it.id == currentEntryId }
            if (index >= 0) {
                history[index] = history[index].copy(analysis = updated, explainMs = ms)
            }
        }
        return result
    }

    fun startCheck() {
        val message = text
        val timing = InferenceTiming.start("check")
        submit(timing) { runCheck(message, timing) }
    }

    fun startExplain(a: Analysis) {
        val timing = InferenceTiming.start("explain")
        submit(timing, explaining = true) { runExplain(a, timing) }
    }

    fun openEntry(entry: SessionEntry) {
        activeRequest?.cancel()
        pendingFrame = null
        busy = false
        error = null
        text = entry.analysis.text
        analysis = entry.analysis
        currentEntryId = entry.id
        checkMs = entry.checkMs
        explainMs = entry.explainMs
        selectedTab = TAB_CHECK
    }

    fun selectTab(tab: Int) {
        if (tab != selectedTab) {
            haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove)
            selectedTab = tab
        }
    }

    LaunchedEffect(shared) {
        if (shared != null && shared.text.isNotBlank()) {
            selectedTab = TAB_CHECK
            text = shared.text
            submit(shared.timing) { runCheck(shared.text, shared.timing) }
        }
    }

    LaunchedEffect(selectedTab) {
        scroll.scrollTo(0)
    }

    LaunchedEffect(verdictSerial) {
        val current = analysis
        if (verdictSerial > 0 && current != null) {
            haptics.performHapticFeedback(
                if (current.tier == Tier.SCAM) HapticFeedbackType.LongPress else HapticFeedbackType.TextHandleMove,
            )
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(c.bg)
            .drawBehind {
                val p = if (reducedMotion) 0.5f else drift.value
                drawRect(
                    brush = Brush.radialGradient(
                        colors = listOf(wash.copy(alpha = 0.22f), Color.Transparent),
                        center = Offset(size.width * (0.1f + 0.5f * p), 0f),
                        radius = size.width * 0.95f,
                    ),
                )
            },
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scroll)
                .padding(start = 16.dp, top = 16.dp, end = 16.dp, bottom = 120.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Header()
            StatusChips(offline = offline, permissionsKnown = permissions != null)
            when (selectedTab) {
                TAB_HOME -> HomeTab(
                    history = history,
                    detectorStatus = detectorStatus,
                    explainerInstalled = explainerInstalled,
                    offline = offline,
                    onOpen = { entry -> openEntry(entry) },
                )
                TAB_PROOF -> ProofTab(
                    detectorStatus = detectorStatus,
                    explainerInstalled = explainerInstalled,
                    permissions = permissions,
                    detectorMode = detectorMode,
                    onDetectorMode = { detectorMode = it },
                    onAbout = { showAbout = true },
                )
                else -> CheckTab(
                    text = text,
                    onText = { text = it },
                    busy = busy,
                    error = error,
                    analysis = analysis,
                    verdictKey = currentEntryId,
                    checkMs = checkMs,
                    explainMs = explainMs,
                    explainerInstalled = explainerInstalled,
                    onCheck = { startCheck() },
                    onExplain = { a -> startExplain(a) },
                    onAbout = { showAbout = true },
                )
            }
        }

        BottomNav(
            selected = selectedTab,
            onSelect = { tab -> selectTab(tab) },
            modifier = Modifier.align(Alignment.BottomCenter),
        )
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
// Header, status and navigation
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
private fun StatusChips(offline: Boolean, permissionsKnown: Boolean) {
    val c = LocalRakshakColors.current
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        if (offline) {
            Chip("● OFFLINE", c.low)
        } else if (!permissionsKnown) {
            Chip("Permissions could not be read", c.muted)
        } else {
            Chip("INTERNET permission requested", c.medium)
        }
        Chip("This phone", c.muted)
        Chip("SMS student · ONNX", c.accent)
        Chip("Llama 3.2 1B · ExecuTorch", c.accent)
    }
}

@Composable
private fun BottomNav(selected: Int, onSelect: (Int) -> Unit, modifier: Modifier = Modifier) {
    val c = LocalRakshakColors.current
    val shape = RoundedCornerShape(28.dp)
    val pill = animateFloatAsState(
        targetValue = selected.toFloat(),
        animationSpec = tween<Float>(durationMillis = 320, easing = FastOutSlowInEasing),
        label = "nav-pill",
    )
    Box(
        modifier = modifier
            .fillMaxWidth()
            .padding(start = 16.dp, end = 16.dp, bottom = 16.dp)
            .clip(shape)
            .background(c.glass)
            .border(1.dp, c.line, shape)
            .padding(6.dp),
    ) {
        BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
            val density = LocalDensity.current
            val itemW = maxWidth / 3
            val itemPx = with(density) { itemW.toPx() }
            Box(
                modifier = Modifier
                    .offset { IntOffset((itemPx * pill.value).roundToInt(), 0) }
                    .width(itemW)
                    .height(ITEM_HEIGHT)
                    .clip(RoundedCornerShape(22.dp))
                    .background(c.accent.copy(alpha = 0.18f)),
            )
            Row(modifier = Modifier.fillMaxWidth()) {
                NavItem(
                    label = "Home",
                    glyph = Glyph.HOME,
                    selected = selected == TAB_HOME,
                    onClick = { onSelect(TAB_HOME) },
                    modifier = Modifier.weight(1f),
                )
                NavItem(
                    label = "Check",
                    glyph = Glyph.CHECK,
                    selected = selected == TAB_CHECK,
                    onClick = { onSelect(TAB_CHECK) },
                    modifier = Modifier.weight(1f),
                )
                NavItem(
                    label = "Proof",
                    glyph = Glyph.PROOF,
                    selected = selected == TAB_PROOF,
                    onClick = { onSelect(TAB_PROOF) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }
}

@Composable
private fun NavItem(
    label: String,
    glyph: Glyph,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val c = LocalRakshakColors.current
    val tint = if (selected) c.accent else c.muted
    Column(
        modifier = modifier
            .height(ITEM_HEIGHT)
            .selectable(selected = selected, role = Role.Tab, onClick = onClick),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        GlyphIcon(glyph = glyph, tint = tint, modifier = Modifier.size(22.dp))
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Normal,
            color = tint,
        )
    }
}

// ---------------------------------------------------------------------------
// Check tab
// ---------------------------------------------------------------------------

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun CheckTab(
    text: String,
    onText: (String) -> Unit,
    busy: Boolean,
    error: String?,
    analysis: Analysis?,
    verdictKey: Long?,
    checkMs: Long?,
    explainMs: Long?,
    explainerInstalled: Boolean,
    onCheck: () -> Unit,
    onExplain: (Analysis) -> Unit,
    onAbout: () -> Unit,
) {
    val c = LocalRakshakColors.current
    FadeUp(0) {
        GlassCard {
            Text(
                "Checks a message on this phone. The app asks for no permissions and sends nothing anywhere.",
                style = MaterialTheme.typography.bodySmall,
                color = c.muted,
            )
            SectionLabel("Try a synthetic example")
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                SAMPLES.forEach { sample ->
                    SampleChip(label = sample.label) { onText(sample.text) }
                }
            }
            Text(
                "Samples are invented for the demo. They are not real messages.",
                style = MaterialTheme.typography.labelSmall,
                color = c.muted,
            )
            OutlinedTextField(
                value = text,
                onValueChange = onText,
                modifier = Modifier.fillMaxWidth(),
                minLines = 4,
                label = { Text("Paste an SMS or message") },
            )
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = onCheck,
                    enabled = text.isNotBlank() && !busy,
                    modifier = Modifier.heightIn(min = 48.dp),
                ) {
                    Text(if (busy) "Checking..." else "Check message")
                }
                TextButton(
                    onClick = onAbout,
                    modifier = Modifier.heightIn(min = 48.dp),
                ) { Text("About") }
            }
            error?.let {
                Text(it, style = MaterialTheme.typography.bodyMedium, color = c.high)
            }
        }
    }

    if (busy && analysis == null) {
        FadeUp(1) { ScanningCard() }
    }

    analysis?.let { a ->
        FadeUp(2) { VerdictHero(analysis = a, verdictKey = verdictKey) }
        if (a.evidence.isNotEmpty()) {
            FadeUp(3) { WarningSignsCard(a) }
            FadeUp(4) { PreviewCard(a) }
        }
        FadeUp(5) {
            WhyCard(
                analysis = a,
                explainerInstalled = explainerInstalled,
                busy = busy,
                explainMs = explainMs,
                onExplain = { onExplain(a) },
            )
        }
        if (a.advisory.points.isNotEmpty()) {
            FadeUp(6) { VerifierCard(a.advisory.points) }
        }
        if (a.remediation.isNotEmpty()) {
            FadeUp(7) { RemediationCard(a.remediation, a.tier) }
        }
        checkMs?.let { ms ->
            Text(
                "checked in $ms ms on this phone",
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                color = c.muted,
            )
        }
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
    GlassCard(scanPhase = phase) {
        Text(
            "Checking on this phone",
            style = MaterialTheme.typography.titleMedium,
            color = c.text,
        )
        Text(
            "Scoring the message and matching rules. Nothing leaves this phone.",
            style = MaterialTheme.typography.bodySmall,
            color = c.muted,
        )
    }
}

@Composable
private fun VerdictHero(analysis: Analysis, verdictKey: Long?) {
    val c = LocalRakshakColors.current
    val reduced = rememberReducedMotion()
    val tier = analysis.tier
    val color = tierColor(tier, c)
    val target = if (analysis.score.isNaN()) 0f else analysis.score.toFloat().coerceIn(0f, 1f)
    val sweep = remember { Animatable(0f) }
    LaunchedEffect(verdictKey) {
        if (reduced) {
            sweep.snapTo(target)
        } else {
            sweep.snapTo(0f)
            sweep.animateTo(target, tween(durationMillis = 600, easing = FastOutSlowInEasing))
        }
    }
    val centerSub = if (analysis.score.isNaN()) {
        "score --"
    } else {
        "score ${(sweep.value * 100f).roundToInt()}%"
    }
    val top = analysis.evidence.maxByOrNull { it.weight }

    GlassCard(glow = color.copy(alpha = 0.22f)) {
        SectionLabel("Verdict")
        Column(
            modifier = Modifier.fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            VerdictGauge(
                progress = sweep.value,
                color = color,
                centerWord = tierWord(tier),
                centerSub = centerSub,
                threshold = SCAM_THRESHOLD.toFloat(),
            )
            Text(
                headlineFor(tier, top?.let { phraseOf(it.type) }),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = c.text,
                textAlign = TextAlign.Center,
            )
            Text(
                actionLabel(analysis.action),
                style = MaterialTheme.typography.bodyLarge,
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
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun WarningSignsCard(analysis: Analysis) {
    val c = LocalRakshakColors.current
    val tint = tierColor(analysis.tier, c)
    val phrases = analysis.evidence.map { phraseOf(it.type) }.distinct()
    val shown = phrases.take(MAX_SIGNS)
    val hidden = phrases.size - shown.size
    GlassCard {
        SectionLabel("Warning signs")
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            shown.forEach { phrase -> Chip(phrase, tint) }
            if (hidden > 0) Chip("+$hidden more", c.muted)
        }
    }
}

@Composable
private fun PreviewCard(analysis: Analysis) {
    val c = LocalRakshakColors.current
    val highlight = tierColor(analysis.tier, c)
    GlassCard {
        SectionLabel("Message, matched text highlighted")
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

private fun isExplainable(analysis: Analysis): Boolean =
    !analysis.detectorOnly && analysis.lang != "hinglish" &&
        analysis.advisory.path != "explainer" && analysis.pkg.risk != Risk.LOW

/** The explanation card. Names Llama only when Llama wrote the wording shown. */
@Composable
private fun WhyCard(
    analysis: Analysis,
    explainerInstalled: Boolean,
    busy: Boolean,
    explainMs: Long?,
    onExplain: () -> Unit,
) {
    val c = LocalRakshakColors.current
    val explainable = isExplainable(analysis)
    val points = analysis.advisory.points
    if (points.isEmpty() && !explainable) return
    val fromModel = analysis.advisory.path == "explainer"

    GlassCard(glass = true) {
        SectionLabel("Why")
        Text(
            text = if (fromModel) {
                "Llama 3.2 1B · ExecuTorch · on this phone"
            } else {
                "Built-in wording · Llama did not run for this message"
            },
            style = MaterialTheme.typography.labelMedium,
            color = if (fromModel) c.accent else c.muted,
        )
        Text(
            text = analysis.advisory.summary,
            style = MaterialTheme.typography.bodyMedium,
            color = c.text,
        )
        points.forEach { p ->
            val fromLlm = p.source == "llm"
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.Top,
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Chip(if (fromLlm) "Llama" else "Template", if (fromLlm) c.accent else c.muted)
                Text(
                    text = p.text,
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.bodyMedium,
                    color = c.text,
                )
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
        if (fromModel && explainMs != null) {
            Text(
                "explained in $explainMs ms · measured on this phone",
                style = MaterialTheme.typography.labelSmall,
                fontFamily = FontFamily.Monospace,
                color = c.muted,
            )
        }
    }
}

/** One row per point: verified, or rejected with the first reason and the template shown. */
@Composable
private fun VerifierCard(points: List<PointView>) {
    val c = LocalRakshakColors.current
    GlassCard {
        SectionLabel("Verifier receipt")
        Text(
            "Each point is checked by a deterministic verifier before it is shown.",
            style = MaterialTheme.typography.bodySmall,
            color = c.muted,
        )
        points.forEach { p ->
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    phraseOf(p.type),
                    style = MaterialTheme.typography.labelLarge,
                    color = c.text,
                )
                if (p.source == "llm") {
                    Text("✓ verified", style = MaterialTheme.typography.labelMedium, color = c.accent2)
                } else if (p.problems.isNotEmpty()) {
                    Text(
                        "✗ rejected, template shown",
                        style = MaterialTheme.typography.labelMedium,
                        color = c.high,
                    )
                    Text(
                        p.problems.first(),
                        style = MaterialTheme.typography.bodySmall,
                        color = c.muted,
                    )
                } else {
                    Text(
                        "Template wording, no model sentence to check",
                        style = MaterialTheme.typography.labelSmall,
                        color = c.muted,
                    )
                }
            }
        }
    }
}

@Composable
private fun RemediationCard(items: List<RemediationItem>, tier: Tier) {
    val c = LocalRakshakColors.current
    val rail = tierColor(tier, c)
    GlassCard {
        SectionLabel("What to do now")
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
        Text(
            "Pre-checked steps, not written by the AI",
            style = MaterialTheme.typography.labelSmall,
            color = c.muted,
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** The phrase for an evidence type, without its parenthetical examples. */
internal fun phraseOf(type: String): String =
    (TYPE_PHRASE_EN[type] ?: type).replace(PARENTHETICAL, "").trim()

/** One-line verdict headline built from the top evidence phrase. */
private fun headlineFor(tier: Tier, phrase: String?): String = when (tier) {
    Tier.SCAM -> if (phrase == null) "Flagged by the on-device detector" else "Likely scam: $phrase"
    Tier.VERIFY -> if (phrase == null) "Unclear, check before acting" else "Unclear: $phrase"
    Tier.SAFE -> if (phrase == null) "No known scam pattern" else "Weak signal only: $phrase"
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
