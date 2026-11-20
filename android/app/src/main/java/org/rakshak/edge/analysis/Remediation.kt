package org.rakshak.edge.analysis

/**
 * Port of the SMS catalogue in src/rakshak/remediation.py (English and Hinglish). The Hinglish
 * wording is a first draft and needs review by a native speaker before it ships. The network
 * catalogue is not ported: the app handles SMS only.
 */

const val SFC_URL = "https://sancharsaathi.gov.in/sfc/"

data class RemediationItem(val code: String, val title: String, val steps: List<String>)

private data class Entry(val title: String, val steps: List<String>)

private val SMS_EN: Map<Action, Entry> = mapOf(
    Action.DO_NOT_OPEN_LINK to Entry(
        "Do not open the link; check the sender through the official app or number",
        listOf(
            "Do not open the link in this message.",
            "Check the sender through the official app, or the number on your card or official website.",
            "Delete the message if you are not sure.",
        ),
    ),
    Action.DO_NOT_CLICK to Entry(
        "Do not click the links in this message",
        listOf(
            "Do not open any link or button in this message.",
            "If you already clicked, do not enter any details on that page.",
            "Delete the message.",
        ),
    ),
    Action.DO_NOT_SHARE_OTP to Entry(
        "Do not share any OTP, PIN or password",
        listOf(
            "Do not type or reply with any OTP, PIN, CVV or password.",
            "Do not share an OTP with anyone, including someone who says they are from a bank.",
            "If you already shared one, call your bank on the number printed on your card and " +
                "ask them to block the account or payment.",
        ),
    ),
    Action.OPEN_OFFICIAL_APP to Entry(
        "Use the official app or website instead",
        listOf(
            "Do not use the link in this message.",
            "Open the official app, or type the official website address yourself.",
            "Check your account or the request there.",
        ),
    ),
    Action.CONTACT_BANK_OFFICIAL to Entry(
        "Contact the organisation through its official number",
        listOf(
            "Find the official helpline number on your card, statement or official website.",
            "Do not use a number given in this message.",
            "Ask them to confirm whether the message is genuine.",
        ),
    ),
    Action.REPORT_1930_OR_CHAKSHU to Entry(
        "Report it on 1930 or Chakshu",
        listOf(
            "If money was lost, call the 1930 cyber crime helpline as soon as possible.",
            "You can also report the message on Chakshu (Sanchar Saathi): $SFC_URL",
            "Keep the message and any transaction details for the report.",
        ),
    ),
    Action.BLOCK_SENDER to Entry(
        "Block the sender",
        listOf(
            "Block the sender in your messaging app.",
            "Report the message as spam in your messaging app.",
            "Delete the message.",
        ),
    ),
    Action.VERIFY_INDEPENDENTLY to Entry(
        "Check with the sender through an official channel",
        listOf(
            "Do not reply to this message or use its links or numbers.",
            "Check the request through an app, website or number you already know is official.",
            "Do not act on the request until you have checked it.",
        ),
    ),
    Action.IGNORE to Entry(
        "No action needed from this check",
        listOf(
            "No action is needed based on this check.",
            "If you are still unsure, check with the sender through an official channel.",
        ),
    ),
)

private val SMS_HINGLISH: Map<Action, Entry> = mapOf(
    Action.DO_NOT_OPEN_LINK to Entry(
        "Link na kholein; sender ko official app ya official number se check karein",
        listOf(
            "Is message ka link na kholein.",
            "Sender ko official app, ya card ya official website par diye number se check karein.",
            "Shak ho to message delete kar dein.",
        ),
    ),
    Action.DO_NOT_CLICK to Entry(
        "Is message ke links par click na karein",
        listOf(
            "Is message ke kisi link ya button ko na kholein.",
            "Agar pehle hi click ho gaya hai, to us page par koi detail na dalein.",
            "Message delete kar dein.",
        ),
    ),
    Action.DO_NOT_SHARE_OTP to Entry(
        "OTP, PIN ya password kisi ko na batayein",
        listOf(
            "Koi bhi OTP, PIN, CVV ya password reply mein ya kisi form mein na likhein.",
            "Bank ke naam par call ya message karne wale ko bhi OTP na batayein.",
            "Agar pehle hi share ho gaya hai, to card par likhe official helpline number par call " +
                "karke account ya payment block karne ko kahein.",
        ),
    ),
    Action.OPEN_OFFICIAL_APP to Entry(
        "Official app ya website ka istemal karein",
        listOf(
            "Is message ka link use na karein.",
            "Official app kholein, ya official website ka address khud type karein.",
            "Wahan apna account ya request check karein.",
        ),
    ),
    Action.CONTACT_BANK_OFFICIAL to Entry(
        "Official number par sampark karein",
        listOf(
            "Official helpline number apne card, statement ya official website par se dhoondhein.",
            "Is message mein diya gaya number use na karein.",
            "Unse poochhein ki message asli hai ya nahi.",
        ),
    ),
    Action.REPORT_1930_OR_CHAKSHU to Entry(
        "1930 ya Chakshu par report karein",
        listOf(
            "Agar paise kati hain, to jaldi se 1930 cyber crime helpline par call karein.",
            "Chakshu (Sanchar Saathi) par bhi message report kar sakte hain: $SFC_URL",
            "Message aur transaction ki details sambhal kar rakhein.",
        ),
    ),
    Action.BLOCK_SENDER to Entry(
        "Sender ko block karein",
        listOf(
            "Apne messaging app mein sender ko block karein.",
            "Message ko spam ki tarah report karein.",
            "Message delete kar dein.",
        ),
    ),
    Action.VERIFY_INDEPENDENTLY to Entry(
        "Official channel se verify karein",
        listOf(
            "Is message ka reply, link ya number use na karein.",
            "Jo app, website ya number aap pehle se official jaante hain, usse check karein.",
            "Check karne tak request par amal na karein.",
        ),
    ),
    Action.IGNORE to Entry(
        "Is check ke hisaab se koi action zaroori nahi",
        listOf(
            "Is check ke hisaab se koi action zaroori nahi hai.",
            "Agar phir bhi shak ho, to sender se official channel par poochh lein.",
        ),
    ),
)

/** remediation_steps for one code (SMS only). */
fun remediationSteps(action: Action, lang: String): RemediationItem {
    val catalogue = if (lang == "hinglish") SMS_HINGLISH else SMS_EN
    val entry = catalogue.getValue(action)
    return RemediationItem(action.name, entry.title, entry.steps)
}

/**
 * remediation_for: the chosen action first, then the other allowed ones. IGNORE is dropped when
 * any other code is present. Every code comes from the package's allowed actions.
 */
fun remediationFor(pkg: EvidencePackage, chosen: Action, lang: String): List<RemediationItem> {
    var ordered: List<Action> = listOf(chosen) + pkg.allowedActions.filter { it != chosen }
    if (ordered.size > 1) ordered = ordered.filter { it != Action.IGNORE }
    return ordered.distinct().map { remediationSteps(it, lang) }
}
