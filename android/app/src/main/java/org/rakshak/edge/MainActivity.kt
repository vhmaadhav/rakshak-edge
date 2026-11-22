package org.rakshak.edge

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import org.rakshak.edge.data.RuleData
import org.rakshak.edge.perf.InferenceTiming
import org.rakshak.edge.ui.RakshakScreen

class SharedMessage(val text: String, val timing: InferenceTiming?)

/**
 * Entry point. Text arrives by paste inside the app or by the share sheet (ACTION_SEND,
 * text/plain). The app requests no permission at all.
 */
class MainActivity : ComponentActivity() {
    private lateinit var engine: Engine
    private var incoming: SharedMessage? by mutableStateOf(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        RuleData.reader = { name ->
            assets.open("rakshak/$name").bufferedReader(Charsets.UTF_8).use { it.readText() }
        }
        engine = Engine(applicationContext)
        val coldLaunch = !createdInThisProcess
        createdInThisProcess = true
        takeShared(intent, coldLaunch)
        setContent {
            MaterialTheme {
                RakshakScreen(engine, incoming)
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        takeShared(intent, coldLaunch = false)
    }

    private fun takeShared(intent: Intent?, coldLaunch: Boolean) {
        if (intent?.action != Intent.ACTION_SEND || intent.type != "text/plain") return
        val text = intent.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString() ?: return
        val id = intent.getLongExtra(InferenceTiming.EXTRA_SAMPLE_ID, 0L).takeIf { it > 0L }
        val path = if (coldLaunch) "cold_share" else "warm_share"
        incoming = SharedMessage(text, InferenceTiming.start(path, id, fromProcessStart = coldLaunch))
    }

    override fun onDestroy() {
        if (::engine.isInitialized) engine.close()
        super.onDestroy()
    }

    private companion object {
        var createdInThisProcess = false
    }
}
