"""Conservative deterministic redaction; human review is required for free-form names."""

import re
from urllib.parse import urlsplit

VERSION = '1'
BRANDS = {'SBI', 'HDFC', 'HDFCBK', 'ICICI', 'AXIS', 'AIRTEL', 'JIO', 'BSNL', 'VI'}
URL = re.compile(r'(?i)\b(?:https?://|www\.)[^\s<>]+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>]*)?')
TOKEN = re.compile(r'<[A-Z_]+>')


def _url(match):
    value = match.group().rstrip('.,;!')
    suffix = match.group()[len(value):]
    parts = urlsplit(value if '://' in value else 'https://' + value)
    host = parts.hostname or '<DOMAIN>'
    # Personal subdomains can still require manual review; never retain URL credentials.
    prefix = parts.scheme + '://' if '://' in value else ''
    path = '/<PATH>' if parts.path and parts.path != '/' else parts.path
    return (prefix + host + path + ('?<QUERY>' if parts.query else '')
            + ('#<FRAGMENT>' if parts.fragment else '') + suffix)


def scrub(text, *, names=()):
    text = str(text)
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]', '', text)
    # URL components are replaced before generic numeric/name processing.
    text = URL.sub(_url, text)
    text = re.sub(r'(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', '<EMAIL>', text)
    text = re.sub(r'(?i)(?<![\w@])[\w.+-]+@[a-z][a-z0-9.-]*(?![\w@])', '<UPI_ID>', text)
    for name in sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True):
        text = re.sub(r'(?<!\w)' + re.escape(name) + r'(?!\w)', '<NAME>', text,
                      flags=re.IGNORECASE)
    def greeting(match):
        value = match.group(2).strip()
        if value.lower() in {'customer', 'user', 'sir', 'madam', 'sir/madam', 'subscriber'}:
            return match.group()
        return match.group(1) + '<NAME>' + match.group(3)
    text = re.sub(r'(?i)(\b(?:dear|hi|hello|namaste)\s+)([^,\n<>]{1,60})([,])', greeting, text)
    text = re.sub(r'(?i)(?:₹|\b(?:rs\.?|inr|usd|amount)\s*)[\s:]*[\d,]+(?:\.\d+)?',
                  '<AMOUNT>', text)
    text = re.sub(r'(?i)\b((?:otp|one[ -]time password|verification code|pin)\s*'
                  r'(?:(?:is|:|=|code)\s*)?)[\d -]{4,12}\b', r'\1<OTP>', text)
    text = re.sub(r'(?i)\b((?:a/c|ac|acct|account|card)\s*(?:no\.?|number|ending|xx)?\s*[:*-]?\s*)'
                  r'[xX*]*\d[\d xX*.-]{1,30}\b', r'\1<ACCOUNT_CARD>', text)
    text = re.sub(r'(?<!\w)(?:\+?91[ -]?)?[6-9](?:[ -]?\d){9}(?!\w)', '<PHONE>', text)
    # Masked accounts, residual OTPs/references/dates are never retained as digits.
    text = re.sub(r'(?i)[x*]{2,}[\d*x -]*\d[\d*x -]*', '<ACCOUNT_CARD>', text)
    text = re.sub(r'\d+(?:[.,/-]\d+)*', '<NUMBER>', text)
    return text


DLT_SENDER = r'[A-Z]{2}-[A-Z0-9]{3,9}(?:-[PSTG])?'


def sender_style(address):
    address = str(address).strip()
    if re.fullmatch(DLT_SENDER, address):
        return 'alphanumeric_id'
    if re.search(r'\d', address):
        return 'numeric'
    return 'other'


def scrub_sender(address, *, names=()):
    address = str(address).strip()
    if any(n.strip() and n.lower() in address.lower() for n in names):
        return '<NAME>'
    if sender_style(address) == 'alphanumeric_id' or address in BRANDS:
        return re.sub(r'\d+', '<NUMBER>', address)
    return '<PHONE>' if re.search(r'\d', address) else '<SENDER>'
