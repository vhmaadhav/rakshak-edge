package org.rakshak.edge.ui

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.rakshak.edge.analysis.DetectorOnlyMode

/**
 * The permissions this app declares, read at runtime from its own package info. Null when the
 * package info cannot be read. An empty set means no permissions are requested.
 */
internal fun requestedPermissions(context: Context): Set<String>? =
    runCatching { readRequestedPermissions(context) }.getOrNull()

@Suppress("DEPRECATION")
private fun readRequestedPermissions(context: Context): Set<String> {
    val pm = context.packageManager
    val info = if (Build.VERSION.SDK_INT >= 33) {
        pm.getPackageInfo(
            context.packageName,
            PackageManager.PackageInfoFlags.of(PackageManager.GET_PERMISSIONS.toLong()),
        )
    } else {
        pm.getPackageInfo(context.packageName, PackageManager.GET_PERMISSIONS)
    }
    return info.requestedPermissions?.toSet() ?: emptySet()
}

private const val PERM_INTERNET = "android.permission.INTERNET"
private const val PERM_NETWORK_STATE = "android.permission.ACCESS_NETWORK_STATE"
private const val PERM_READ_SMS = "android.permission.READ_SMS"
private const val PERM_RECEIVE_SMS = "android.permission.RECEIVE_SMS"

/**
 * Offline proof: the permissions read at runtime, the status of the two models on this phone,
 * the proof-carrying explanation, and the Llama credit with the licence dialog.
 */
@Composable
internal fun ProofTab(
    detectorStatus: DetectorStatus,
    explainerInstalled: Boolean,
    permissions: Set<String>?,
    detectorMode: DetectorOnlyMode,
    onDetectorMode: (DetectorOnlyMode) -> Unit,
    onAbout: () -> Unit,
) {
    val c = LocalRakshakColors.current
    val detectorLine = when (detectorStatus) {
        DetectorStatus.NOT_TRIED -> "Detector: loads on the first check"
        DetectorStatus.LOADED -> "Detector: loaded on this phone"
        DetectorStatus.ERROR -> "Detector: error on the last check"
    }
    val explainerLine = if (explainerInstalled) {
        "Explainer: installed on this phone"
    } else {
        "Explainer: not installed, so the built-in wording is used"
    }

    FadeUp(0) {
        GlassCard(glass = true) {
            SectionLabel("Offline proof")
            Text(
                text = "Read at runtime from this app's own manifest, through PackageManager.",
                style = MaterialTheme.typography.bodySmall,
                color = c.muted,
            )
            PermissionRow("Network (INTERNET)", PERM_INTERNET, permissions)
            PermissionRow("Network state (ACCESS_NETWORK_STATE)", PERM_NETWORK_STATE, permissions)
            PermissionRow("Read SMS (READ_SMS)", PERM_READ_SMS, permissions)
            PermissionRow("Receive SMS (RECEIVE_SMS)", PERM_RECEIVE_SMS, permissions)
        }
    }

    FadeUp(1) {
        GlassCard {
            SectionLabel("Detector-only policy")
            Text(
                text = "Controls messages flagged by the ONNX model when no deterministic rule matched.",
                style = MaterialTheme.typography.bodySmall,
                color = c.muted,
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                DetectorOnlyMode.entries.forEach { m ->
                    FilterChip(
                        selected = detectorMode == m,
                        onClick = { onDetectorMode(m) },
                        label = { Text(m.label) },
                        modifier = Modifier.weight(1f),
                    )
                }
            }
            val policyNote = when (detectorMode) {
                DetectorOnlyMode.QUIET -> "Quiet (default): detector-only flags stay LOW / IGNORE."
                DetectorOnlyMode.BALANCED ->
                    "Balanced: escalates at >= 0.9566, while passive bank/UPI/OTP notices require >= 0.999."
                DetectorOnlyMode.VERIFY ->
                    "Verify: escalates every detector-only flag (>= 0.3513) to Verify independently."
            }
            Text(text = policyNote, style = MaterialTheme.typography.bodySmall, color = c.text)
        }
    }

    FadeUp(2) {
        GlassCard {
            SectionLabel("On this phone")
            Text(text = detectorLine, style = MaterialTheme.typography.bodyMedium, color = c.text)
            Text(text = explainerLine, style = MaterialTheme.typography.bodyMedium, color = c.text)
            Text(
                text = "Models: SMS student (ONNX) and Llama 3.2 1B (ExecuTorch)",
                style = MaterialTheme.typography.bodySmall,
                color = c.muted,
            )
        }
    }

    FadeUp(3) {
        GlassCard {
            Text(
                text = "Proof-carrying explanation",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = c.text,
            )
            Text(
                text = "Every sentence Llama writes is checked by a deterministic verifier before it is " +
                    "shown. A sentence that fails any check is replaced by a fixed template. The Check tab " +
                    "shows a verifier receipt for each point, with the reason for any rejection.",
                style = MaterialTheme.typography.bodyMedium,
                color = c.muted,
            )
        }
    }

    FadeUp(4) {
        GlassCard {
            Text(
                text = "Built with Llama",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = c.text,
            )
            Text(
                text = "The explanation layer uses Llama 3.2 1B Instruct under the Llama 3.2 Community License.",
                style = MaterialTheme.typography.bodySmall,
                color = c.muted,
            )
            OutlinedButton(
                onClick = onAbout,
                modifier = Modifier.heightIn(min = 48.dp),
            ) { Text("About") }
        }
    }
}

@Composable
private fun PermissionRow(label: String, permission: String, permissions: Set<String>?) {
    val c = LocalRakshakColors.current
    val (status, statusColor) = when {
        permissions == null -> "could not read" to c.muted
        permission in permissions -> "requested" to c.medium
        else -> "not requested" to c.accent2
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 40.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(
            text = label,
            modifier = Modifier.weight(1f),
            style = MaterialTheme.typography.bodyMedium,
            color = c.text,
        )
        Text(
            text = status,
            style = MaterialTheme.typography.labelLarge,
            fontWeight = FontWeight.SemiBold,
            color = statusColor,
        )
    }
}
