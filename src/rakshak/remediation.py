"""Deterministic remediation steps for SMS and network advisories.

Each action code (``rakshak.evidence.Action`` for SMS, ``rakshak.nids.evidence.NetAction`` for
network) has a fixed title and steps. The steps only tell the user what to do with that code;
they add no new claim about the message or the traffic. The list is built from the chosen action
and the package's allowed actions, so it can never name a code the package does not allow.

SMS and network catalogues are separate. ``remediation_problems`` checks that a response has no
SMS-only code in a network package and no network-only code in an SMS package.

Hinglish SMS wording is a first draft and needs review by a native speaker before use. The network
steps are English only.
"""

from __future__ import annotations

from typing import Any

from rakshak.evidence import Action, EvidencePackage
from rakshak.nids.evidence import NetAction

SFC_URL = "https://sancharsaathi.gov.in/sfc/"
LANGS = ("en", "hinglish")

# (title, steps) per code. Steps are plain sentences.
SMS_EN: dict[str, tuple[str, tuple[str, ...]]] = {
    Action.DO_NOT_OPEN_LINK.value: (
        "Do not open the link; check the sender through the official app or number",
        (
            "Do not open the link in this message.",
            "Check the sender through the official app, or the number on your card or official "
            "website.",
            "Delete the message if you are not sure.",
        ),
    ),
    Action.DO_NOT_CLICK.value: (
        "Do not click the links in this message",
        (
            "Do not open any link or button in this message.",
            "If you already clicked, do not enter any details on that page.",
            "Delete the message.",
        ),
    ),
    Action.DO_NOT_SHARE_OTP.value: (
        "Do not share any OTP, PIN or password",
        (
            "Do not type or reply with any OTP, PIN, CVV or password.",
            "Do not share an OTP with anyone, including someone who says they are from a bank.",
            "If you already shared one, call your bank on the number printed on your card and "
            "ask them to block the account or payment.",
        ),
    ),
    Action.OPEN_OFFICIAL_APP.value: (
        "Use the official app or website instead",
        (
            "Do not use the link in this message.",
            "Open the official app, or type the official website address yourself.",
            "Check your account or the request there.",
        ),
    ),
    Action.CONTACT_BANK_OFFICIAL.value: (
        "Contact the organisation through its official number",
        (
            "Find the official helpline number on your card, statement or official website.",
            "Do not use a number given in this message.",
            "Ask them to confirm whether the message is genuine.",
        ),
    ),
    Action.REPORT_1930_OR_CHAKSHU.value: (
        "Report it on 1930 or Chakshu",
        (
            "If money was lost, call the 1930 cyber crime helpline as soon as possible.",
            f"You can also report the message on Chakshu (Sanchar Saathi): {SFC_URL}",
            "Keep the message and any transaction details for the report.",
        ),
    ),
    Action.BLOCK_SENDER.value: (
        "Block the sender",
        (
            "Block the sender in your messaging app.",
            "Report the message as spam in your messaging app.",
            "Delete the message.",
        ),
    ),
    Action.VERIFY_INDEPENDENTLY.value: (
        "Check with the sender through an official channel",
        (
            "Do not reply to this message or use its links or numbers.",
            "Check the request through an app, website or number you already know is official.",
            "Do not act on the request until you have checked it.",
        ),
    ),
    Action.IGNORE.value: (
        "No action needed from this check",
        (
            "No action is needed based on this check.",
            "If you are still unsure, check with the sender through an official channel.",
        ),
    ),
}

SMS_HINGLISH: dict[str, tuple[str, tuple[str, ...]]] = {
    Action.DO_NOT_OPEN_LINK.value: (
        "Link na kholein; sender ko official app ya official number se check karein",
        (
            "Is message ka link na kholein.",
            "Sender ko official app, ya card ya official website par diye number se check karein.",
            "Shak ho to message delete kar dein.",
        ),
    ),
    Action.DO_NOT_CLICK.value: (
        "Is message ke links par click na karein",
        (
            "Is message ke kisi link ya button ko na kholein.",
            "Agar pehle hi click ho gaya hai, to us page par koi detail na dalein.",
            "Message delete kar dein.",
        ),
    ),
    Action.DO_NOT_SHARE_OTP.value: (
        "OTP, PIN ya password kisi ko na batayein",
        (
            "Koi bhi OTP, PIN, CVV ya password reply mein ya kisi form mein na likhein.",
            "Bank ke naam par call ya message karne wale ko bhi OTP na batayein.",
            "Agar pehle hi share ho gaya hai, to card par likhe official helpline number par call "
            "karke account ya payment block karne ko kahein.",
        ),
    ),
    Action.OPEN_OFFICIAL_APP.value: (
        "Official app ya website ka istemal karein",
        (
            "Is message ka link use na karein.",
            "Official app kholein, ya official website ka address khud type karein.",
            "Wahan apna account ya request check karein.",
        ),
    ),
    Action.CONTACT_BANK_OFFICIAL.value: (
        "Official number par sampark karein",
        (
            "Official helpline number apne card, statement ya official website par se dhoondhein.",
            "Is message mein diya gaya number use na karein.",
            "Unse poochhein ki message asli hai ya nahi.",
        ),
    ),
    Action.REPORT_1930_OR_CHAKSHU.value: (
        "1930 ya Chakshu par report karein",
        (
            "Agar paise kati hain, to jaldi se 1930 cyber crime helpline par call karein.",
            f"Chakshu (Sanchar Saathi) par bhi message report kar sakte hain: {SFC_URL}",
            "Message aur transaction ki details sambhal kar rakhein.",
        ),
    ),
    Action.BLOCK_SENDER.value: (
        "Sender ko block karein",
        (
            "Apne messaging app mein sender ko block karein.",
            "Message ko spam ki tarah report karein.",
            "Message delete kar dein.",
        ),
    ),
    Action.VERIFY_INDEPENDENTLY.value: (
        "Official channel se verify karein",
        (
            "Is message ka reply, link ya number use na karein.",
            "Jo app, website ya number aap pehle se official jaante hain, usse check karein.",
            "Check karne tak request par amal na karein.",
        ),
    ),
    Action.IGNORE.value: (
        "Is check ke hisaab se koi action zaroori nahi",
        (
            "Is check ke hisaab se koi action zaroori nahi hai.",
            "Agar phir bhi shak ho, to sender se official channel par poochh lein.",
        ),
    ),
}

NET_EN: dict[str, tuple[str, tuple[str, ...]]] = {
    NetAction.BLOCK_SOURCE_IP.value: (
        "Block the source address",
        (
            "Note the source address shown in the alert.",
            "Add a block for that address in your router or firewall settings.",
            "Check the router logs for more traffic from that address.",
        ),
    ),
    NetAction.CHECK_ROUTER_FIREWALL.value: (
        "Review your router firewall",
        (
            "Open the router admin page and review the firewall rules.",
            "Look for port-forward rules you do not recognise.",
            "Close any open port that no device needs.",
        ),
    ),
    NetAction.CHANGE_PASSWORD.value: (
        "Change the password of the targeted account or device",
        (
            "Change the password of the account or device that was targeted.",
            "Use a long, unique password.",
            "Turn on two-step verification where the service supports it.",
        ),
    ),
    NetAction.DISABLE_PASSWORD_LOGIN.value: (
        "Turn off password login for SSH",
        (
            "On the SSH server, switch to key-based login.",
            "Set PasswordAuthentication to no in the SSH server settings and reload the service.",
            "Check the authorised keys file for keys you do not recognise.",
        ),
    ),
    NetAction.RATE_LIMIT.value: (
        "Limit the traffic rate",
        (
            "Apply a rate limit to the affected service or connection in the router or device.",
            "Watch whether the traffic drops.",
            "If it does not drop, contact your internet provider.",
        ),
    ),
    NetAction.CONTACT_ISP.value: (
        "Contact your internet provider",
        (
            "Tell your internet provider what you saw, with the time and addresses from the alert.",
            "Ask whether they can help with this traffic.",
        ),
    ),
    NetAction.ISOLATE_DEVICE.value: (
        "Move the device to a guest network or unplug it",
        (
            "Move the device to a guest network, or unplug it from the network.",
            "Do not use the device until you have checked it.",
            "Keep its logs or settings for later review.",
        ),
    ),
    NetAction.FACTORY_RESET_IOT.value: (
        "Factory-reset the IoT device",
        (
            "Factory-reset the device with its own reset steps.",
            "Install the vendor's latest firmware update.",
            "Change its default password before you connect it again.",
        ),
    ),
    NetAction.USE_TRUSTED_DNS.value: (
        "Use a DNS resolver you trust",
        (
            "Set the router's DNS to a resolver you trust.",
            "Watch for DNS queries from the device afterwards.",
        ),
    ),
    NetAction.VERIFY_INDEPENDENTLY.value: (
        "Check with the device owner or vendor",
        (
            "Check the device list and confirm the device is yours.",
            "Ask the owner or the vendor whether this traffic is expected.",
            "Do not change settings until you have checked.",
        ),
    ),
    NetAction.IGNORE.value: (
        "No action needed from this check",
        ("No action is needed based on this check.",),
    ),
}

SMS_CODES = frozenset(a.value for a in Action)
NET_CODES = frozenset(a.value for a in NetAction)
# Codes that exist in both catalogues (the shared SMS codes). Their steps differ by domain.
SHARED_CODES = SMS_CODES & NET_CODES
# SMS-only codes: must never appear in a network package or response.
SMS_ONLY = SMS_CODES - NET_CODES
NET_ONLY = NET_CODES - SMS_CODES


def _catalogue(domain: str, lang: str) -> dict[str, tuple[str, tuple[str, ...]]]:
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {LANGS}, got {lang!r}")
    if domain == "sms":
        return SMS_HINGLISH if lang == "hinglish" else SMS_EN
    if domain == "network":
        return NET_EN  # no Hinglish network catalogue: English for any request
    raise ValueError(f"unknown domain {domain!r}")


def remediation_steps(code: str, domain: str, lang: str = "en") -> dict[str, Any]:
    """{code, title, steps} for one action code in one domain."""
    catalogue = _catalogue(domain, lang)
    if code not in catalogue:
        raise ValueError(f"no remediation for {code!r} in domain {domain!r}")
    title, steps = catalogue[code]
    return {"code": code, "title": title, "steps": list(steps)}


def remediation_for(
    package: EvidencePackage, action: str, lang: str = "en"
) -> list[dict[str, Any]]:
    """Ordered remediation for a package: the chosen action first, then the other allowed ones.

    IGNORE is dropped when any other code is present. Every code comes from
    ``package.allowed_actions``, and the domain is the package's own domain.
    """
    chosen = action.value if hasattr(action, "value") else str(action)
    allowed = [a.value if hasattr(a, "value") else str(a) for a in package.allowed_actions]
    ordered = [chosen] + [a for a in allowed if a != chosen]
    if len(ordered) > 1:
        ordered = [a for a in ordered if a != Action.IGNORE.value]
    seen: list[str] = []
    for code in ordered:
        if code not in seen:
            seen.append(code)
    items = [remediation_steps(code, package.domain, lang) for code in seen]
    problems = remediation_problems(items, package)
    if problems:
        raise ValueError("; ".join(problems))
    return items


def remediation_problems(items: list[dict[str, Any]], package: EvidencePackage) -> list[str]:
    """Reasons a remediation list is not valid for the package (empty when valid)."""
    problems: list[str] = []
    allowed = {a.value if hasattr(a, "value") else str(a) for a in package.allowed_actions}
    own = SMS_CODES if package.domain == "sms" else NET_CODES
    other = NET_ONLY if package.domain == "sms" else SMS_ONLY
    for item in items:
        code = item.get("code")
        if code in other:
            problems.append(f"{code} is not valid in the {package.domain} domain")
        elif code not in own:
            problems.append(f"{code!r} is not a known {package.domain} action")
        elif code not in allowed:
            problems.append(f"{code} is not in the package's allowed actions")
        if not item.get("steps"):
            problems.append(f"{code} has no steps")
    return problems
