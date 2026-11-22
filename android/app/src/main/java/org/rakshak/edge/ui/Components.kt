package org.rakshak.edge.ui

import android.provider.Settings
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateIntAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

/** Gap between staggered card entrances. */
internal const val STAGGER_MS = 60L

/**
 * Rounded card with a hairline border. [glass] uses the translucent surface. [glow] adds a soft
 * halo at the top. [scanPhase] (0..1) draws a loading sweep.
 */
@Composable
internal fun GlassCard(
    modifier: Modifier = Modifier,
    glass: Boolean = false,
    glow: Color? = null,
    scanPhase: Float? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    val c = LocalRakshakColors.current
    val shape = RoundedCornerShape(24.dp)
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(shape)
            .background(if (glass) c.glass else c.surface)
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
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        content = content,
    )
}

/** Small pill with a tinted fill and outline. */
@Composable
internal fun Chip(text: String, color: Color, modifier: Modifier = Modifier) {
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

/** Small uppercase label above a card's content. */
@Composable
internal fun SectionLabel(text: String, modifier: Modifier = Modifier) {
    val c = LocalRakshakColors.current
    Text(
        text = text.uppercase(),
        modifier = modifier,
        style = MaterialTheme.typography.labelSmall,
        fontWeight = FontWeight.SemiBold,
        letterSpacing = 1.2.sp,
        color = c.muted,
    )
}

/** An integer that counts to its new value when it changes. The first value is shown at once. */
@Composable
internal fun CountUpText(
    target: Int,
    style: TextStyle,
    color: Color,
    modifier: Modifier = Modifier,
) {
    val shown by animateIntAsState(
        targetValue = target,
        animationSpec = tween<Int>(durationMillis = 500),
        label = "count-up",
    )
    Text(text = shown.toString(), modifier = modifier, style = style, color = color)
}

/** One guard in the Home status hero. [on] fills its ring in [color]. */
internal data class GuardState(
    val label: String,
    val on: Boolean,
    val word: String,
    val color: Color,
)

/** Concentric rings, one per guard, outermost first. A ring is filled only when its guard is on. */
@Composable
internal fun GuardRings(guards: List<GuardState>, modifier: Modifier = Modifier) {
    val c = LocalRakshakColors.current
    Box(modifier = modifier.size(176.dp), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.size(176.dp)) {
            val stroke = 11.dp.toPx()
            val gap = 6.dp.toPx()
            guards.forEachIndexed { index, guard ->
                val inset = stroke / 2f + index * (stroke + gap)
                val arc = Size(size.width - 2f * inset, size.height - 2f * inset)
                val topLeft = Offset(inset, inset)
                drawArc(
                    color = c.line,
                    startAngle = -90f,
                    sweepAngle = 360f,
                    useCenter = false,
                    topLeft = topLeft,
                    size = arc,
                    style = Stroke(width = stroke),
                )
                if (guard.on) {
                    drawArc(
                        color = guard.color,
                        startAngle = -90f,
                        sweepAngle = 360f,
                        useCenter = false,
                        topLeft = topLeft,
                        size = arc,
                        style = Stroke(width = stroke),
                    )
                }
            }
        }
        GlyphIcon(glyph = Glyph.SHIELD, tint = c.accent, modifier = Modifier.size(34.dp))
    }
}

/** Filled shield with a tick. Used as the app mark in the header. */
@Composable
internal fun ShieldMark(modifier: Modifier = Modifier) {
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
        drawPath(path = shield, color = c.accent)
        drawPath(
            path = tickPath(w, h),
            color = c.bg,
            style = Stroke(width = w * 0.08f, cap = StrokeCap.Round, join = StrokeJoin.Round),
        )
    }
}

internal enum class Glyph { SHIELD, HOME, CHECK, PROOF }

/** Outline icons drawn with Canvas paths. No icon library is used. */
@Composable
internal fun GlyphIcon(glyph: Glyph, tint: Color, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val w = size.width
        val h = size.height
        val line = Stroke(width = w * 0.09f, cap = StrokeCap.Round, join = StrokeJoin.Round)
        when (glyph) {
            Glyph.SHIELD -> {
                val shield = Path().apply {
                    moveTo(w * 0.5f, h * 0.04f)
                    lineTo(w * 0.92f, h * 0.18f)
                    lineTo(w * 0.92f, h * 0.5f)
                    cubicTo(w * 0.92f, h * 0.76f, w * 0.72f, h * 0.92f, w * 0.5f, h * 0.98f)
                    cubicTo(w * 0.28f, h * 0.92f, w * 0.08f, h * 0.76f, w * 0.08f, h * 0.5f)
                    lineTo(w * 0.08f, h * 0.18f)
                    close()
                }
                drawPath(path = shield, color = tint, style = line)
                drawPath(path = tickPath(w, h), color = tint, style = line)
            }
            Glyph.HOME -> {
                val house = Path().apply {
                    moveTo(w * 0.14f, h * 0.46f)
                    lineTo(w * 0.5f, h * 0.14f)
                    lineTo(w * 0.86f, h * 0.46f)
                    moveTo(w * 0.24f, h * 0.4f)
                    lineTo(w * 0.24f, h * 0.86f)
                    lineTo(w * 0.76f, h * 0.86f)
                    lineTo(w * 0.76f, h * 0.4f)
                }
                drawPath(path = house, color = tint, style = line)
            }
            Glyph.CHECK -> {
                drawCircle(color = tint, radius = w * 0.4f, center = Offset(w / 2f, h / 2f), style = line)
                drawPath(path = tickPath(w, h), color = tint, style = line)
            }
            Glyph.PROOF -> {
                drawRoundRect(
                    color = tint,
                    topLeft = Offset(w * 0.18f, h * 0.08f),
                    size = Size(w * 0.64f, h * 0.84f),
                    cornerRadius = CornerRadius(w * 0.12f),
                    style = line,
                )
                drawLine(
                    color = tint,
                    start = Offset(w * 0.34f, h * 0.36f),
                    end = Offset(w * 0.66f, h * 0.36f),
                    strokeWidth = w * 0.09f,
                    cap = StrokeCap.Round,
                )
                drawLine(
                    color = tint,
                    start = Offset(w * 0.34f, h * 0.52f),
                    end = Offset(w * 0.66f, h * 0.52f),
                    strokeWidth = w * 0.09f,
                    cap = StrokeCap.Round,
                )
                drawLine(
                    color = tint,
                    start = Offset(w * 0.34f, h * 0.68f),
                    end = Offset(w * 0.56f, h * 0.68f),
                    strokeWidth = w * 0.09f,
                    cap = StrokeCap.Round,
                )
            }
        }
    }
}

private fun tickPath(w: Float, h: Float): Path = Path().apply {
    moveTo(w * 0.32f, h * 0.5f)
    lineTo(w * 0.46f, h * 0.64f)
    lineTo(w * 0.7f, h * 0.38f)
}

/** True when the system animator duration scale is 0 (the Android reduced-motion setting). */
@Composable
internal fun rememberReducedMotion(): Boolean {
    val context = LocalContext.current
    return remember(context) {
        runCatching {
            Settings.Global.getFloat(context.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
        }.getOrDefault(false)
    }
}

/** Fades a block in and slides it up 16 dp, [index] x 60 ms after it first appears. */
@Composable
internal fun FadeUp(index: Int, content: @Composable () -> Unit) {
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
