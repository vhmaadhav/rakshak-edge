"""Deterministic language bucketing for SMS text: en | hinglish | hi_deva | other.

Rules, in order (Latin tokens are lower-cased ``[a-z]+`` runs):
1. hi_deva: at least 30% of the letters are Devanagari.
2. hinglish: at least one strong romanised-Hindi token, at least ``MIN_HINGLISH_HITS`` Hindi
   tokens H (strong and weak), and a Hindi share ``H / (H + E) >= HINGLISH_RATIO``, where E counts
   English lexicon tokens. Weak tokens are words shared with other Latin-script languages.
   Code-mixed text counts here: an English-dominant message with a real Hindi share is still
   hinglish.
3. en: at least two English lexicon tokens, or one English token that is at least 20% of the
   tokens and no Hindi token. A single common word ("in", "is", "no") is also Dutch, German or
   Spanish, so it is not enough alone.
4. other: everything else (other Latin-script languages, digits only, unknown).

Ambiguous tokens (``to``, ``me``, ``do``, ``hi``) and acronyms (``otp``, ``kyc``, ``upi``,
``rs``) sit in neither lexicon, so they never push a message towards either language.

The lexicons are hand-written so the decision is auditable. ``HINGLISH_RATIO`` was swept on
``tests/fixtures/lang_cases.json`` (hand-labelled, see tests/test_lang.py): agreement is flat
for 0.10 to 0.25 and falls above 0.3, so 0.2 sits in the middle of the flat region.
Agreement with the source's own language column is reported by the build (see docs/DATACARD.md).
That column is heuristic (much of the ScamShield "hinglish" label is English bank and telecom
templates), so it is a weak check on this detector.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z]+")
MIN_HINGLISH_HITS = 2
HINGLISH_RATIO = 0.2

# Romanised Hindi and Hinglish (Hindi/Urdu function words, verbs, time words, and the common
# scam and banking vocabulary: turant, jaldi, sampark, khata, paisa, bijli, kat, inaam ...).
# Words that are common English words ("tab", "the", "no", "hi", "to", "do") are left out.
# Words that are also common in Spanish, French, Italian, Polish, Romanian or German ("se", "le",
# "na", "par", "din" ...) are in HINGLISH_WEAK instead, and never count alone.
HINGLISH_LEXICON = frozenset(
    """
    aap aapka aapke aapki aapko aapse aapne apna apni apne apki apko tum tumhara tumhari tumhare
    tumhe tera teri tere tujhe tujhko mera meri mere mujhe mujhko humara hamara hamari humare
    hume hum unka unki unke unhe uska uski uske inka inki inhe hai hain hoon hun hona hota hoti
    hote hoga hogi hoge hua hui hue gaya gayi gaye raha rahe rahi rahega rahegi thi karo karein
    karen kare karna karenge karke karwa karwao karwaye karwayein karwai karvai kiya kijiye
    kijie dijiye dijie diya diye liye lena lijiye bhi sab sabhi koi kisi kuch kuchh kya kyu kyun
    kyunki kaise kaisa kaun kahan kab jab phir fir lekin warna agar toh nahi nahin nhi nai mat
    yeh woh wo wala wali wale yaar bhai bhaiya behen didi ji dost sahi galat theek thik accha
    achha bura badhiya bahut bohot zyada thoda thodi sirf bas bina saath sath jaldi jaldee
    turant abhi aaj kal raat subah shaam dopahar baje ghante ghanta dino dinon hafte mahine saal
    ghar kaam paisa paise paiso rupaye rupya rupiya khata khate khaate bijli bijlee kat kaat
    katega katenge kata band bandh jayega jayegi jayenge jaayega jao jaao jana aana aao aaye
    aayega aaya bhejo bhejein bhejen bhej bheja bhejiye batao bataye bata bataiye bataunga bolo
    bol bolna sunao dekho dekhna dekhiye chahiye sakte sakta sakti zaroor zaroori jeeta jeete
    jeeto jeetne jeet inaam inam sampark sampark karein kripya krupya kripaya shikayat giraftar
    giraftaar adalat jurmana bhugtaan bhugtan munafa munafaa nivesh kamao kamaiye kamayi kamai
    kamaye naukri shulk shulka jama bharo bhariye mubarak shadi vasool pakka pata mauka mehenga
    sasta kharid kharido mat ghabrao chalo milte milo bakwas bekar khush dukh zaroorat madad
    hisaab hisab kahani
    """.split()
)

# Hindi/Urdu words that are also ordinary words in other Latin-script languages. They count
# towards H but never alone: a message needs at least one strong token (HINGLISH_LEXICON).
HINGLISH_WEAK = frozenset(
    """ka ke ki ko se pe par tak na ye vo tha din ho le lo main mai mein aur kar""".split()
)

# English function words and common English vocabulary in SMS. "to", "do", "me" and "hi"
# are ambiguous (Hindi words too) and are left out of both lexicons.
ENGLISH_LEXICON = frozenset(
    """
    an the and or of in is are was were be been being am you your yours yourself we our ours
    they their it its this that these those with for from on at by have has had will would can could
    should shall may might must not no yes please click call now get got account bank verify
    link here today if update updated kyc otp dear sir madam thank thanks thank hello hey
    customer user what when where which who why how all any some each every more most other
    such only also just so than too very up down out off over under again once about into
    through during before after above below between because while until against both few nor
    same own don dont won sent send share shared offer free win
    prize congratulations congrats congratulation download install app details
    valid visit check contact support message regards team order delivered shipped
    booking confirmed payment paid due bill credited debited balance available
    """.split()
)

_DEVA_LO, _DEVA_HI = "ऀ", "ॿ"


def detect_lang(text: str, *, hinglish_ratio: float = HINGLISH_RATIO) -> str:
    """Return one of ``en``, ``hinglish``, ``hi_deva`` or ``other``."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "other"
    deva = sum(_DEVA_LO <= ch <= _DEVA_HI for ch in letters)
    if deva / len(letters) >= 0.3:
        return "hi_deva"
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return "other"
    strong = sum(tok in HINGLISH_LEXICON for tok in tokens)
    weak = sum(tok in HINGLISH_WEAK for tok in tokens)
    hin = strong + weak
    eng = sum(tok in ENGLISH_LEXICON for tok in tokens)
    if strong >= 1 and hin >= MIN_HINGLISH_HITS and hin / (hin + eng) >= hinglish_ratio:
        return "hinglish"
    if eng >= 2 or (eng >= 1 and hin == 0 and eng / len(tokens) >= 0.2):
        return "en"
    return "other"
