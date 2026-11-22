package org.rakshak.edge.analysis

import org.rakshak.edge.data.RuleData

/** Port of datasets/lang.py detect_lang: "en", "hinglish", "hi_deva" or "other". */
fun detectLang(text: String): String {
    val lex = RuleData.langLexicon
    var letters = 0
    var deva = 0
    text.codePoints().forEach { cp ->
        if (Character.isLetter(cp)) {
            letters++
            if (cp in 0x0900..0x097F) deva++
        }
    }
    if (letters == 0) return "other"
    if (deva.toDouble() / letters >= 0.3) return "hi_deva"
    val tokens = Regex("[a-z]+").findAll(text.lowercase()).map { it.value }.toList()
    if (tokens.isEmpty()) return "other"
    val strong = tokens.count { it in lex.hinglishStrong }
    val weak = tokens.count { it in lex.hinglishWeak }
    val hin = strong + weak
    val eng = tokens.count { it in lex.english }
    if (strong >= 1 && hin >= lex.minHinglishHits &&
        hin.toDouble() / (hin + eng) >= lex.hinglishRatio
    ) {
        return "hinglish"
    }
    if (eng >= 2 || (eng >= 1 && hin == 0 && eng.toDouble() / tokens.size >= 0.2)) return "en"
    return "other"
}

/** route_lang with the default "auto" request: only a hinglish detection routes to hinglish. */
fun routeLang(detected: String): String = if (detected == "hinglish") "hinglish" else "en"
