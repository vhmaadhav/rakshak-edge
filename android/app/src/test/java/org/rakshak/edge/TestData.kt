package org.rakshak.edge

import java.io.File
import org.rakshak.edge.data.RuleData

/** Points the rule data at the source assets, so the JVM tests read what the app ships. */
object TestData {
    fun install() {
        RuleData.reader = { name -> File("src/main/assets/rakshak/$name").readText(Charsets.UTF_8) }
    }

    fun fixtureText(): String =
        checkNotNull(javaClass.classLoader?.getResourceAsStream("parity_fixture.json")) {
            "parity_fixture.json is not on the test classpath"
        }.bufferedReader(Charsets.UTF_8).use { it.readText() }
}
