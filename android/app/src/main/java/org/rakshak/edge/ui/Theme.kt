package org.rakshak.edge.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.ColorScheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import org.rakshak.edge.analysis.Tier

/**
 * Semantic colours for the v2 graphite theme. One signature accent (indigo). Cyan is reserved
 * for sealed or verified states. [glass] is the translucent surface used by the nav and hero.
 */
class RakshakColors(
    val isDark: Boolean,
    val bg: Color,
    val surface: Color,
    val surface2: Color,
    val line: Color,
    val text: Color,
    val muted: Color,
    val accent: Color,
    val accent2: Color,
    val high: Color,
    val medium: Color,
    val low: Color,
    val glass: Color,
)

val DarkRakshakColors = RakshakColors(
    isDark = true,
    bg = Color(0xFF0B0D12),
    surface = Color(0xFF13161D),
    surface2 = Color(0xFF1A1E27),
    line = Color(0xFF262B36),
    text = Color(0xFFF2F4F8),
    muted = Color(0xFF8A93A6),
    accent = Color(0xFF7C8CFF),
    accent2 = Color(0xFF2EE6D6),
    high = Color(0xFFFF5B5B),
    medium = Color(0xFFFFB547),
    low = Color(0xFF3DDC84),
    glass = Color(0xD113161D),
)

val LightRakshakColors = RakshakColors(
    isDark = false,
    bg = Color(0xFFF5F7FB),
    surface = Color(0xFFFFFFFF),
    surface2 = Color(0xFFEEF1F8),
    line = Color(0xFFD9DFEA),
    text = Color(0xFF111827),
    muted = Color(0xFF5B6475),
    accent = Color(0xFF4A55E0),
    accent2 = Color(0xFF0E7490),
    high = Color(0xFFD92D3F),
    medium = Color(0xFFB45309),
    low = Color(0xFF15803D),
    glass = Color(0xDBFFFFFF),
)

val LocalRakshakColors = staticCompositionLocalOf<RakshakColors> { DarkRakshakColors }

/** Risk colour for a tier: SCAM is high, VERIFY is medium, SAFE is low. */
fun tierColor(tier: Tier, colors: RakshakColors): Color = when (tier) {
    Tier.SCAM -> colors.high
    Tier.VERIFY -> colors.medium
    Tier.SAFE -> colors.low
}

/** The risk word shown with the colour, so risk is never carried by colour alone. */
fun tierWord(tier: Tier): String = when (tier) {
    Tier.SCAM -> "SCAM"
    Tier.VERIFY -> "VERIFY"
    Tier.SAFE -> "SAFE"
}

private fun schemeFor(c: RakshakColors): ColorScheme =
    if (c.isDark) {
        darkColorScheme(
            primary = c.accent,
            onPrimary = c.bg,
            secondary = c.accent2,
            onSecondary = c.bg,
            background = c.bg,
            onBackground = c.text,
            surface = c.surface,
            onSurface = c.text,
            surfaceVariant = c.surface2,
            onSurfaceVariant = c.muted,
            outline = c.line,
            error = c.high,
            onError = c.bg,
        )
    } else {
        lightColorScheme(
            primary = c.accent,
            onPrimary = Color.White,
            secondary = c.accent2,
            onSecondary = Color.White,
            background = c.bg,
            onBackground = c.text,
            surface = c.surface,
            onSurface = c.text,
            surfaceVariant = c.surface2,
            onSurfaceVariant = c.muted,
            outline = c.line,
            error = c.high,
            onError = Color.White,
        )
    }

/** Dark or light Rakshak theme. Follows the system setting unless [darkTheme] is given. */
@Composable
fun RakshakTheme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    val colors = if (darkTheme) DarkRakshakColors else LightRakshakColors
    CompositionLocalProvider(LocalRakshakColors provides colors) {
        MaterialTheme(colorScheme = schemeFor(colors), content = content)
    }
}
