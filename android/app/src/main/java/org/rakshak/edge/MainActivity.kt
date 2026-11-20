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
import org.rakshak.edge.ui.RakshakScreen

/**
 * Entry point. Text arrives by paste inside the app or by the share sheet (ACTION_SEND,
 * text/plain). The app requests no permission at all.
 */
class MainActivity : ComponentActivity() {
    private lateinit var engine: Engine
    private var incoming: String? by mutableStateOf(null)
    private var incomingNonce by mutableStateOf(0)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        RuleData.reader = { name ->
            assets.open("rakshak/$name").bufferedReader(Charsets.UTF_8).use { it.readText() }
        }
        engine = Engine(applicationContext)
        takeShared(intent)
        setContent {
            MaterialTheme {
                RakshakScreen(engine = engine, sharedText = incoming, sharedNonce = incomingNonce)
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        takeShared(intent)
    }

    private fun takeShared(intent: Intent?) {
        if (intent?.action != Intent.ACTION_SEND || intent.type != "text/plain") return
        val text = intent.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString() ?: return
        incoming = text
        incomingNonce++
    }

    override fun onDestroy() {
        if (::engine.isInitialized) engine.close()
        super.onDestroy()
    }
}
