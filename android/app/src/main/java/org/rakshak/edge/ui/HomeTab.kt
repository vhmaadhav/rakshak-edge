package org.rakshak.edge.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import org.rakshak.edge.analysis.Tier

private const val FEED_SIZE = 8
private const val FEED_PREVIEW_CHARS = 60

/**
 * Status hero (guard rings, title, nothing-leaves line), this session's counters, and the recent
 * feed. Every number comes from this session's own checks or from the device.
 */
@Composable
internal fun HomeTab(
    history: List<SessionEntry>,
    detectorStatus: DetectorStatus,
    explainerInstalled: Boolean,
    offline: Boolean,
    onOpen: (SessionEntry) -> Unit,
) {
    val c = LocalRakshakColors.current
    val threat = history.firstOrNull()?.analysis?.tier == Tier.SCAM
    val detectorOn = detectorStatus != DetectorStatus.ERROR
    val guards = listOf(
        GuardState(label = "Detector", on = detectorOn, word = if (detectorOn) "ready" else "error", color = c.accent),
        GuardState(
            label = "Explainer",
            on = explainerInstalled,
            word = if (explainerInstalled) "installed" else "missing",
            color = c.accent,
        ),
        GuardState(label = "Offline", on = offline, word = if (offline) "on" else "off", color = c.accent2),
    )
    val guardsOn = guards.count { it.on }
    val flagged = history.count { it.analysis.tier == Tier.SCAM }
    val heroGlow = if (threat) c.high.copy(alpha = 0.22f) else c.accent.copy(alpha = 0.20f)

    FadeUp(0) {
        GlassCard(glass = true, glow = heroGlow) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                GuardRings(guards = guards)
                Text(
                    text = if (threat) "Threat found" else "Protected",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.ExtraBold,
                    color = if (threat) c.high else c.text,
                    textAlign = TextAlign.Center,
                )
                Text(
                    text = "$guardsOn of 3 guards on",
                    style = MaterialTheme.typography.bodyMedium,
                    color = c.muted,
                    textAlign = TextAlign.Center,
                )
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                ) {
                    guards.forEach { guard -> GuardLegend(guard) }
                }
                Chip("Nothing leaves this phone", c.accent2)
            }
        }
    }

    FadeUp(1) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionLabel("This session")
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                StatTile(
                    label = "checked",
                    value = history.size,
                    color = c.text,
                    modifier = Modifier.weight(1f),
                )
                StatTile(
                    label = "flagged",
                    value = flagged,
                    color = if (flagged > 0) c.high else c.text,
                    modifier = Modifier.weight(1f),
                )
                StatTile(
                    label = "median ms",
                    value = medianMs(history),
                    color = c.text,
                    modifier = Modifier.weight(1f),
                )
            }
        }
    }

    FadeUp(2) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionLabel("Recent checks")
            if (history.isEmpty()) {
                GlassCard {
                    Text(
                        text = "No checks yet this session",
                        style = MaterialTheme.typography.titleSmall,
                        color = c.text,
                    )
                    Text(
                        text = "Results are kept in memory only and clear when the app closes.",
                        style = MaterialTheme.typography.bodySmall,
                        color = c.muted,
                    )
                }
            } else {
                GlassCard {
                    history.take(FEED_SIZE).forEach { entry ->
                        FeedRow(entry = entry, onClick = { onOpen(entry) })
                    }
                }
            }
        }
    }
}

@Composable
private fun GuardLegend(guard: GuardState) {
    val c = LocalRakshakColors.current
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        Text(guard.label, style = MaterialTheme.typography.labelMedium, color = c.muted)
        Text(
            text = guard.word,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
            color = if (guard.on) guard.color else c.muted,
        )
    }
}

@Composable
private fun StatTile(label: String, value: Int?, color: Color, modifier: Modifier = Modifier) {
    val c = LocalRakshakColors.current
    val shape = RoundedCornerShape(20.dp)
    val valueStyle = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold)
    Column(
        modifier = modifier
            .clip(shape)
            .background(c.surface)
            .border(1.dp, c.line, shape)
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        if (value == null) {
            Text(text = "—", style = valueStyle, color = c.muted)
        } else {
            CountUpText(target = value, style = valueStyle, color = color)
        }
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = c.muted)
    }
}

@Composable
private fun FeedRow(entry: SessionEntry, onClick: () -> Unit) {
    val c = LocalRakshakColors.current
    val tier = entry.analysis.tier
    val color = tierColor(tier, c)
    val preview = entry.analysis.text.replace('\n', ' ').trim().take(FEED_PREVIEW_CHARS)
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .clickable(onClick = onClick),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Box(modifier = Modifier.size(10.dp).clip(CircleShape).background(color)) { }
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = preview.ifEmpty { "(empty message)" },
                style = MaterialTheme.typography.bodyMedium,
                color = c.text,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                text = "${tierWord(tier)} · ${entry.checkMs} ms · ${clockTime(entry.atMillis)}",
                style = MaterialTheme.typography.labelSmall,
                color = c.muted,
            )
        }
    }
}

/** Median of this session's check times in ms, or null when there are none. */
private fun medianMs(history: List<SessionEntry>): Int? {
    if (history.isEmpty()) return null
    val sorted = history.map { it.checkMs }.sorted()
    val mid = sorted.size / 2
    val median = if (sorted.size % 2 == 1) sorted[mid] else (sorted[mid - 1] + sorted[mid]) / 2
    return median.toInt()
}

private fun clockTime(atMillis: Long): String =
    SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(atMillis))
