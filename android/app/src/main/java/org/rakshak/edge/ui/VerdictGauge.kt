package org.rakshak.edge.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Ring gauge for the verdict. [progress] is 0..1 and is already animated by the caller, so the
 * caller can count the score up with it. The arc starts at 12 o'clock. [threshold], when given,
 * draws a tick on the ring at that score (the SCAM threshold of the detector).
 */
@Composable
fun VerdictGauge(
    progress: Float,
    color: Color,
    centerWord: String,
    centerSub: String,
    threshold: Float? = null,
    modifier: Modifier = Modifier,
) {
    val c = LocalRakshakColors.current
    Box(modifier = modifier.size(184.dp), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.size(184.dp)) {
            val stroke = size.minDimension * 0.085f
            val inset = stroke / 2f
            val arcSize = Size(size.width - stroke, size.height - stroke)
            val topLeft = Offset(inset, inset)
            drawArc(
                color = c.line,
                startAngle = -90f,
                sweepAngle = 360f,
                useCenter = false,
                topLeft = topLeft,
                size = arcSize,
                style = Stroke(width = stroke),
            )
            val sweep = 360f * progress.coerceIn(0f, 1f)
            if (sweep > 0.5f) {
                drawArc(
                    color = color,
                    startAngle = -90f,
                    sweepAngle = sweep,
                    useCenter = false,
                    topLeft = topLeft,
                    size = arcSize,
                    style = Stroke(width = stroke, cap = StrokeCap.Round),
                )
            }
            if (threshold != null) {
                val rad = (-90.0 + 360.0 * threshold.coerceIn(0f, 1f)) * PI / 180.0
                val radius = size.minDimension / 2f
                val cx = size.width / 2f
                val cy = size.height / 2f
                val inner = radius - stroke - 6f
                val outer = radius + 2f
                drawLine(
                    color = c.text,
                    start = Offset(cx + inner * cos(rad).toFloat(), cy + inner * sin(rad).toFloat()),
                    end = Offset(cx + outer * cos(rad).toFloat(), cy + outer * sin(rad).toFloat()),
                    strokeWidth = 3f,
                    cap = StrokeCap.Round,
                )
            }
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = centerWord,
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.ExtraBold,
                color = color,
            )
            Text(
                text = centerSub,
                style = MaterialTheme.typography.labelLarge,
                color = c.muted,
            )
        }
    }
}
