package org.rakshak.edge.analysis

/** Port of src/rakshak/evidence.py for SMS packages (the URL model is not ported, see docs). */

enum class Risk { HIGH, MEDIUM, LOW, UNCERTAIN }

/** Action codes. The enum name is the wire value used by the Python code. */
enum class Action {
    DO_NOT_CLICK,
    DO_NOT_OPEN_LINK,
    DO_NOT_SHARE_OTP,
    OPEN_OFFICIAL_APP,
    CONTACT_BANK_OFFICIAL,
    REPORT_1930_OR_CHAKSHU,
    BLOCK_SENDER,
    IGNORE,
    VERIFY_INDEPENDENTLY,
}

/** Scam state from the fusion of the student score and the rule evidence. */
enum class ScamState(val wire: String) { SCAM("scam"), UNCERTAIN("uncertain"), BENIGN("benign") }

val ACTION_MAP: Map<String, List<Action>> = mapOf(
    "credential_request" to listOf(Action.DO_NOT_SHARE_OTP, Action.CONTACT_BANK_OFFICIAL),
    "upi_collect_or_pay" to listOf(Action.DO_NOT_SHARE_OTP, Action.REPORT_1930_OR_CHAKSHU),
    "kyc_block_urgency" to listOf(
        Action.OPEN_OFFICIAL_APP, Action.CONTACT_BANK_OFFICIAL, Action.DO_NOT_CLICK,
    ),
    "account_suspension" to listOf(
        Action.CONTACT_BANK_OFFICIAL, Action.OPEN_OFFICIAL_APP, Action.DO_NOT_CLICK,
    ),
    "prize_lottery" to listOf(Action.DO_NOT_CLICK, Action.BLOCK_SENDER),
    "apk_install" to listOf(Action.DO_NOT_CLICK, Action.BLOCK_SENDER, Action.REPORT_1930_OR_CHAKSHU),
    "remote_access_app" to listOf(
        Action.DO_NOT_CLICK, Action.REPORT_1930_OR_CHAKSHU, Action.BLOCK_SENDER,
    ),
    "electricity_disconnect" to listOf(
        Action.VERIFY_INDEPENDENTLY, Action.OPEN_OFFICIAL_APP, Action.DO_NOT_CLICK,
    ),
    "parcel_customs" to listOf(Action.DO_NOT_CLICK, Action.VERIFY_INDEPENDENTLY),
    "job_task_scam" to listOf(Action.BLOCK_SENDER, Action.REPORT_1930_OR_CHAKSHU),
    "investment_returns" to listOf(
        Action.BLOCK_SENDER, Action.REPORT_1930_OR_CHAKSHU, Action.VERIFY_INDEPENDENTLY,
    ),
    "digital_arrest" to listOf(
        Action.REPORT_1930_OR_CHAKSHU, Action.VERIFY_INDEPENDENTLY, Action.BLOCK_SENDER,
        Action.DO_NOT_SHARE_OTP,
    ),
    "brand_domain_mismatch" to listOf(Action.DO_NOT_CLICK, Action.OPEN_OFFICIAL_APP),
    "shortened_url" to listOf(Action.DO_NOT_CLICK),
    "suspicious_url" to listOf(Action.DO_NOT_CLICK),
)

private val PHONE_RE = rxOf("""\+?\d[\d \-]{8,}\d""")

data class EvidencePackage(
    val risk: Risk,
    val confidenceState: ScamState,
    val evidence: List<Evidence>,
    val allowedActions: List<Action>,
    val urls: List<String>,
    val domains: List<String>,
    val phones: List<String>,
    val brandDomains: Map<String, List<String>>,
) {
    val evidenceIds: Set<String> get() = evidence.map { it.id }.toSet()

    /** Every domain the summary may mention: message domains plus official brand domains. */
    fun allowedDomainNames(): Set<String> {
        val names = HashSet<String>(domains)
        brandDomains.values.forEach { names.addAll(it) }
        return names.map { it.lowercase() }.toSet()
    }
}

private fun allowedActionsFor(evidence: List<Evidence>, risk: Risk): List<Action> {
    val acts = HashSet<Action>()
    for (e in evidence) ACTION_MAP[e.type]?.let { acts.addAll(it) }
    if (acts.isEmpty()) acts.add(if (risk == Risk.LOW) Action.IGNORE else Action.VERIFY_INDEPENDENTLY)
    if (risk == Risk.LOW) acts.add(Action.IGNORE)
    if (risk == Risk.UNCERTAIN || risk == Risk.MEDIUM) acts.add(Action.VERIFY_INDEPENDENTLY)
    return acts.sortedBy { it.name }
}

internal fun riskFor(evidence: List<Evidence>, state: ScamState): Risk {
    val top = evidence.maxOfOrNull { it.weight } ?: 0.0
    if (top >= 0.85) return Risk.HIGH
    if (top >= 0.6) return Risk.MEDIUM
    if (evidence.isNotEmpty() && evidence.all { it.type == "urgency" }) {
        return if (state == ScamState.SCAM) Risk.UNCERTAIN else Risk.LOW
    }
    if (state == ScamState.SCAM && evidence.isNotEmpty()) return Risk.MEDIUM
    if (state == ScamState.UNCERTAIN) return Risk.UNCERTAIN
    return Risk.LOW
}

fun buildPackage(canon: Canon, evidence: List<Evidence>, state: ScamState): EvidencePackage {
    val risk = riskFor(evidence, state)
    val domains = canon.urls.map { registeredDomain(it) }.filter { it.isNotEmpty() }
        .toSortedSet().toList()
    val phones = PHONE_RE.findAll(canon.normalized)
        .map { it.value.filter { c -> c.isDigit() } }.toSortedSet().toList()
    return EvidencePackage(
        risk = risk,
        confidenceState = state,
        evidence = evidence,
        allowedActions = allowedActionsFor(evidence, risk),
        urls = canon.urls,
        domains = domains,
        phones = phones,
        brandDomains = mentionedBrandDomains(canon.normalized),
    )
}

/** choose_action: the first evidence (highest weight, earliest on ties) with an ACTION_MAP entry. */
fun chooseAction(pkg: EvidencePackage): Action {
    val ranked = pkg.evidence.withIndex().sortedWith(
        compareBy<IndexedValue<Evidence>>({ -it.value.weight }, { it.index }),
    )
    var chosen: Action? = null
    for (iv in ranked) {
        val entry = ACTION_MAP[iv.value.type]
        if (!entry.isNullOrEmpty()) {
            chosen = entry[0]
            break
        }
    }
    val action = chosen ?: if (pkg.risk == Risk.LOW) Action.IGNORE else Action.VERIFY_INDEPENDENTLY
    check(action in pkg.allowedActions) { "chosen action $action is not allowed" }
    return action
}
