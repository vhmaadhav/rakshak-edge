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

/**
 * Ring gauge for the verdict. [progress] is 0..1 (already animated by the caller) and is drawn
 * as an arc starting at 12 o'clock. The centre shows the risk word and a caption.
 */
@Composable
fun VerdictGauge(
    progress: Float,
    color: Color,
    centerWord: String,
    centerSub: String,
    modifier: Modifier = Modifier,
) {
    val c = LocalRakshakColors.current
    Box(modifier = modifier.size(176.dp), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.size(176.dp)) {
            val stroke = size.minDimension * 0.09f
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
