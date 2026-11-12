"""Surrogate values for placeholder tags such as <AMOUNT> or <NAME>.

What it is for: the scrubbed SMS text (inbox export, IMC25) carries <TAG> placeholders, and the
tag itself became a scam shortcut (docs/SURROGATE_RETRAIN_PLAN.md, sections 1 and 3). This module
replaces each tag with a realistic fake value of the same type, so text with and without tags looks
alike to a model.

Rules: (1) the generator takes no label and no source, so a surrogate cannot carry label
information; (2) each row's RNG is seeded from sha256 of (SURROGATE_SEED, row id), so a row always
gets the same realisation, and nothing depends on Python's salted hash(); (3) it runs in memory
only, and no dataset is rewritten; (4) one fixed realisation per row, with no resampling per epoch.
Plan: docs/SURROGATE_RETRAIN_PLAN.md, section 3.

Standard library only, plus the brand table that rakshak.url_features already loads.
"""

from __future__ import annotations

import functools
import hashlib
import math
import random
import re
import string
from collections.abc import Callable, Sequence

from rakshak.url_features import load_brands

SURROGATE_SEED = 20261107
TAG_RE = re.compile(r"<[A-Z_]+>")

GIVEN_NAMES = (
    "Rahul",
    "Priya",
    "Amit",
    "Sneha",
    "Vikram",
    "Anjali",
    "Rohan",
    "Neha",
    "Arjun",
    "Kavya",
    "Suresh",
    "Meera",
    "Karthik",
    "Divya",
    "Sanjay",
    "Pooja",
    "Ravi",
    "Lakshmi",
    "Aditya",
    "Ananya",
    "Manoj",
    "Ritu",
    "Vijay",
    "Sunita",
    "Imran",
    "Fatima",
    "John",
    "Michael",
    "Sarah",
    "Emily",
    "David",
    "Olivia",
    "Liam",
    "Chen",
    "Maria",
    "Ahmed",
    "Yuki",
    "Carlos",
)
FAMILY_NAMES = (
    "Sharma",
    "Verma",
    "Patel",
    "Iyer",
    "Nair",
    "Reddy",
    "Gupta",
    "Singh",
    "Kumar",
    "Mehta",
    "Joshi",
    "Rao",
    "Menon",
    "Das",
    "Chatterjee",
    "Khan",
    "Shah",
    "Pillai",
    "Bose",
    "Desai",
    "Smith",
    "Johnson",
    "Brown",
    "Taylor",
    "Wilson",
    "Garcia",
    "Martin",
    "Lee",
    "Kim",
    "Nguyen",
    "Miller",
    "Davis",
    "Chopra",
    "Bhat",
    "Kaur",
)
NATIONALITIES = (
    "Indian",
    "American",
    "British",
    "Chinese",
    "Japanese",
    "Canadian",
    "Australian",
    "German",
    "French",
    "Nepali",
    "Bangladeshi",
    "Sri Lankan",
    "Mexican",
    "Brazilian",
)
CITIES = (
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Hyderabad",
    "Kolkata",
    "Pune",
    "Ahmedabad",
    "Jaipur",
    "Lucknow",
    "Kochi",
    "Indore",
    "Bhopal",
    "Nagpur",
    "Surat",
    "London",
    "New York",
    "Dubai",
    "Singapore",
    "Toronto",
    "Sydney",
    "Berlin",
    "Paris",
    "Chicago",
    "Houston",
)
WEBMAIL_DOMAINS = (
    "gmail.com",
    "yahoo.co.in",
    "outlook.com",
    "rediffmail.com",
    "hotmail.com",
    "icloud.com",
    "protonmail.com",
)
PSP_HANDLES = ("okhdfcbank", "oksbi", "okicici", "okaxis", "ybl", "paytm", "ibl", "axl", "upi")
SHORTENER_DOMAINS = ("bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy", "ow.ly")
RANDOM_WORDS = (
    "sunny",
    "market",
    "quick",
    "hello",
    "pixel",
    "rivet",
    "mango",
    "cloud",
    "garden",
    "harbor",
    "tiger",
    "silver",
    "orbit",
    "bright",
    "coral",
    "maple",
    "river",
    "basket",
    "anchor",
    "meadow",
)
RANDOM_WORD_TLDS = ("com", "net", "info", "online", "site")
LOOKALIKE_TLDS = ("top", "xyz", "in", "co")
LOOKALIKE_TEMPLATES = (
    "{b}-kyc.{t}",
    "{b}-verify.{t}",
    "secure-{b}.{t}",
    "{b}-update.{t}",
    "{b}-login.{t}",
)
QUERY_KEYS = ("id", "ref", "token", "session", "code", "q")
NEUTRAL_WORDS = ("item", "details", "reference", "status", "update", "info", "service", "notice")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
AMOUNT_PREFIXES = ("Rs.", "Rs ", "INR ", "₹")
AMOUNT_MIN = 1.0
AMOUNT_MAX = 200_000.0
NUMBER_LENGTHS = tuple(range(1, 9))
NUMBER_WEIGHTS = (2, 6, 12, 24, 14, 10, 6, 4)  # mode 4 digits
ALNUM = string.ascii_lowercase + string.digits
DIGITS = string.digits


# ---- brand-lookalike and shortener pools (built once from the brand table) ----


@functools.lru_cache(maxsize=1)
def _brand_pool() -> tuple[tuple[str, ...], frozenset[str]]:
    """Synthetic lookalike domains built from the brand aliases, and every registered brand domain.
    A lookalike never equals a registered domain of any brand."""
    brands = load_brands()
    registered: set[str] = set()
    for info in brands.values():
        registered.update(d.lower() for d in info.get("domains", []))
    lookalikes: set[str] = set()
    for name in sorted(brands):
        info = brands[name]
        aliases = info.get("host_aliases") or info.get("aliases") or [name]
        base = re.sub(r"[^a-z0-9]", "", str(aliases[0]).lower())
        if not base:
            continue
        for template in LOOKALIKE_TEMPLATES:
            for tld in LOOKALIKE_TLDS:
                domain = template.format(b=base, t=tld)
                if domain not in registered:
                    lookalikes.add(domain)
    return tuple(sorted(lookalikes)), frozenset(registered)


def lookalike_domains() -> tuple[str, ...]:
    return _brand_pool()[0]


def registered_brand_domains() -> frozenset[str]:
    return _brand_pool()[1]


# ---- small helpers ----


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(DIGITS) for _ in range(n))


def _alnum(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(ALNUM) for _ in range(n))


def _group(int_str: str, indian: bool) -> str:
    """1,23,456 (Indian) or 123,456 (Western) grouping of a digit string."""
    if len(int_str) <= 3:
        return int_str
    head, tail = int_str[:-3], int_str[-3:]
    step = 2 if indian else 3
    parts: list[str] = []
    while len(head) > step:
        parts.insert(0, head[-step:])
        head = head[:-step]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


def _domain(rng: random.Random) -> str:
    """A bare domain from the mixed pool: about 40 % lookalike, 30 % shortener, 30 % random word."""
    r = rng.random()
    if r < 0.4:
        return rng.choice(lookalike_domains())
    if r < 0.7:
        return rng.choice(SHORTENER_DOMAINS)
    word = rng.choice(RANDOM_WORDS)
    if rng.random() < 0.5:
        word = word + rng.choice(RANDOM_WORDS)
    else:
        word = word + _digits(rng, 2)
    return f"{word}.{rng.choice(RANDOM_WORD_TLDS)}"


def _given(rng: random.Random) -> str:
    return rng.choice(GIVEN_NAMES)


# ---- one branch per tag ----


def _amount(rng: random.Random) -> str:
    value = math.exp(rng.uniform(math.log(AMOUNT_MIN), math.log(AMOUNT_MAX)))
    indian = rng.random() < 0.5
    prefix = rng.choice(AMOUNT_PREFIXES)
    if rng.random() < 0.5:
        cents = int(round(value * 100))
        whole, frac = divmod(cents, 100)
        return f"{prefix}{_group(str(whole), indian)}.{frac:02d}"
    return f"{prefix}{_group(str(int(round(value))), indian)}"


def _number(rng: random.Random) -> str:
    n = rng.choices(NUMBER_LENGTHS, weights=NUMBER_WEIGHTS)[0]
    if n == 1:
        return str(rng.randint(0, 9))
    return str(rng.randint(1, 9)) + _digits(rng, n - 1)


def _decimal(rng: random.Random) -> str:
    whole = str(rng.randint(1, 9999))
    return f"{whole}.{_digits(rng, rng.choice((1, 2)))}"


def _otp(rng: random.Random) -> str:
    return _digits(rng, rng.choice((4, 6)))


def _masked(rng: random.Random) -> str:
    last4 = _digits(rng, 4)
    return rng.choice((f"XX{last4}", f"XXXXXXXX{last4}", f"**** {last4}"))


def _indian_mobile(rng: random.Random) -> str:
    national = str(rng.randint(6, 9)) + _digits(rng, 9)
    form = rng.randrange(4)
    if form == 0:
        return national
    if form == 1:
        return "+91" + national
    if form == 2:
        return f"+91 {national[:5]} {national[5:]}"
    return f"{national[:5]} {national[5:]}"


def _name(rng: random.Random) -> str:
    if rng.random() < 0.7:
        return f"{_given(rng)} {rng.choice(FAMILY_NAMES)}"
    return _given(rng)


def _upi(rng: random.Random) -> str:
    return f"{_given(rng).lower()}{_digits(rng, rng.randint(2, 4))}@{rng.choice(PSP_HANDLES)}"


def _email(rng: random.Random) -> str:
    return f"{_given(rng).lower()}{_digits(rng, rng.randint(2, 4))}@{rng.choice(WEBMAIL_DOMAINS)}"


def _path(rng: random.Random) -> str:
    return "/".join(_alnum(rng, rng.randint(3, 8)) for _ in range(rng.randint(1, 2)))


def _query(rng: random.Random) -> str:
    return f"{rng.choice(QUERY_KEYS)}={_alnum(rng, rng.randint(4, 8))}"


def _fragment(rng: random.Random) -> str:
    return _alnum(rng, rng.randint(4, 8))


def _url(rng: random.Random) -> str:
    domain = _domain(rng)
    scheme = "https://" if rng.random() < 0.8 else "http://"
    if domain in SHORTENER_DOMAINS:
        path = "/" + _alnum(rng, rng.randint(5, 7))
    else:
        path = "/" + _path(rng) if rng.random() < 0.8 else "/"
    return scheme + domain + path


def _time_of_day(rng: random.Random) -> str:
    hour, minute = rng.randint(0, 23), rng.randint(0, 59)
    if rng.random() < 0.5:
        return f"{hour:02d}:{minute:02d}"
    h12 = hour % 12 or 12
    return f"{h12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def _date(rng: random.Random) -> str:
    day, month = rng.randint(1, 28), rng.randint(1, 12)
    year = rng.choice((2024, 2025, 2026))
    mon = MONTHS[month - 1]
    forms = (
        f"{day:02d}/{month:02d}/{year}",
        f"{day:02d}-{month:02d}-{year}",
        f"{day:02d}/{month:02d}/{year % 100:02d}",
        f"{day} {mon} {year}",
        f"{mon} {day}, {year}",
        f"{month:02d}/{day:02d}/{year}",
        f"{year}-{month:02d}-{day:02d}",
        f"{day:02d}-{mon}-{year}",
    )
    return rng.choice(forms)


def _date_time(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return _date(rng)
    return f"{_date(rng)} {_time_of_day(rng)}"


def _location(rng: random.Random) -> str:
    return rng.choice(CITIES)


def _us_driver_license(rng: random.Random) -> str:
    return "X000" + _digits(rng, 5)


def _us_passport(rng: random.Random) -> str:
    return "000" + _digits(rng, 6)


def _us_ssn(rng: random.Random) -> str:
    return f"9{_digits(rng, 2)}-00-{_digits(rng, 4)}"


def _us_itin(rng: random.Random) -> str:
    return f"9{_digits(rng, 2)}-9{_digits(rng, 1)}-{_digits(rng, 4)}"


def _uk_nhs(rng: random.Random) -> str:
    return f"999 {_digits(rng, 3)} {_digits(rng, 4)}"


def _medical_license(rng: random.Random) -> str:
    return "ML" + _digits(rng, 6)


def _ip_address(rng: random.Random) -> str:
    # RFC 5737 documentation ranges, so no real host is named.
    prefix = rng.choice(("192.0.2", "198.51.100", "203.0.113"))
    return f"{prefix}.{rng.randint(1, 254)}"


def _crypto(rng: random.Random) -> str:
    return "bc1q" + "".join(rng.choice("023456789acdefghjklmnpqrstuvwxyz") for _ in range(38))


def _sender(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_uppercase) for _ in range(6))


def _nationality(rng: random.Random) -> str:
    return rng.choice(NATIONALITIES)


def _neutral(rng: random.Random) -> str:
    return rng.choice(NEUTRAL_WORDS)


TAG_BRANCHES: dict[str, Callable[[random.Random], str]] = {
    "AMOUNT": _amount,
    "NUMBER": _number,
    "DECIMAL": _decimal,
    "OTP": _otp,
    "ACCOUNT_CARD": _masked,
    "US_BANK_NUMBER": _masked,
    "CREDIT_CARD": _masked,
    "IBAN_CODE": _masked,
    "PHONE": _indian_mobile,
    "PHONE_NUMBER": _indian_mobile,
    "NAME": _name,
    "NAMED_ENTITY": _name,
    "UPI_ID": _upi,
    "EMAIL": _email,
    "EMAIL_ADDRESS": _email,
    "PATH": _path,
    "QUERY": _query,
    "FRAGMENT": _fragment,
    "URL": _url,
    "DOMAIN": _domain,
    "DATE_TIME": _date_time,
    "DATE": _date,
    "TIME": _time_of_day,
    "LOCATION": _location,
    "US_DRIVER_LICENSE": _us_driver_license,
    "US_PASSPORT": _us_passport,
    "US_SSN": _us_ssn,
    "US_ITIN": _us_itin,
    "UK_NHS": _uk_nhs,
    "MEDICAL_LICENSE": _medical_license,
    "IP_ADDRESS": _ip_address,
    "CRYPTO": _crypto,
    "NRP": _nationality,
    "SENDER": _sender,
}


def _surrogate(tag: str, rng: random.Random) -> str:
    branch = TAG_BRANCHES.get(tag, _neutral)
    return branch(rng)


# ---- public API ----


def fill(text: str, rng: random.Random) -> str:
    """Replace every <TAG> with a type-specific surrogate drawn from ``rng``. Text without a tag is
    returned unchanged. Takes no label and no source."""
    return TAG_RE.sub(lambda m: _surrogate(m.group()[1:-1], rng), text)


def row_rng(row_id: str, seed: int = SURROGATE_SEED) -> random.Random:
    """A RNG seeded from sha256 of "<seed>:<row id>" (never Python's salted hash())."""
    digest = hashlib.sha256(f"{seed}:{row_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def fill_row(row_id: str, text: str, seed: int = SURROGATE_SEED) -> str:
    return fill(text, row_rng(row_id, seed))


def fill_many(ids: Sequence[str], texts: Sequence[str], seed: int = SURROGATE_SEED) -> list[str]:
    if len(ids) != len(texts):
        raise ValueError(f"fill_many: {len(ids)} ids for {len(texts)} texts")
    return [fill_row(i, t, seed) for i, t in zip(ids, texts, strict=True)]


def assert_no_tags(texts: Sequence[str]) -> None:
    """Raise ValueError with the number of rows that still hold a tag. The message never includes
    any text."""
    count = sum(1 for t in texts if TAG_RE.search(t))
    if count:
        raise ValueError(f"{count} row(s) still contain a <TAG> placeholder after filling")
