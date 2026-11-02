"""Corpus-independent, detector-only synthetic benign notifications. Training only."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass

import numpy as np

from rakshak.canonicalize import canonicalize
from rakshak.datasets.common import dedup_key
from rakshak.models.features import Featurized, featurize

VERSION = 'transactional-v1'
FAMILIES = ('bank_debit', 'bank_credit', 'upi_completed', 'otp', 'parcel', 'telecom_utility')
# Original code templates from the task brief/public semantics; no corpus/model-written targets.
TEMPLATES = {
    'bank_debit': {
        'en': ('{bank}: A/c XX{account} debited Rs.{amount} on {date}. Ref {ref}. '
               'Available balance Rs.{balance}.',
               'Debit alert from {bank}. INR {amount} spent using card XX{account}. '
               'Reference {ref}. Check your statement in the bank app.'),
        'hinglish': ('{bank}: A/c XX{account} se Rs.{amount} debit hue on {date}. Ref {ref}. '
                     'Bacha balance Rs.{balance}.',
                     '{bank} card XX{account} se INR {amount} ka payment ho gaya. Ref {ref}. '
                     'Statement apne bank app mein dekhein.')},
    'bank_credit': {
        'en': ('{bank}: Rs.{amount} credited to A/c XX{account} on {date}. Ref {ref}. '
               'Available balance Rs.{balance}.',
               'Credit notification: INR {amount} received in {bank} account XX{account}. '
               'Transaction reference {ref}.'),
        'hinglish': ('{bank}: Rs.{amount} A/c XX{account} mein credit hue on {date}. Ref {ref}. '
                     'Balance Rs.{balance}.',
                     '{bank} account XX{account} mein INR {amount} aa gaye. '
                     'Transaction reference {ref}.')},
    'upi_completed': {
        'en': ('UPI transfer completed: Rs.{amount} debited from {bank} A/c XX{account}. '
               'Reference {ref}. This is a transaction notification.',
               '{bank}: UPI payment of INR {amount} received in A/c XX{account}. Ref {ref}. '
               'No PIN is needed to receive money.'),
        'hinglish': ('UPI transfer complete: {bank} A/c XX{account} se Rs.{amount} gaye. '
                     'Reference {ref}. Yeh transaction ki jaankari hai.',
                     '{bank}: UPI se INR {amount} A/c XX{account} mein mile. Ref {ref}. '
                     'Paise receive karne ke liye PIN nahi chahiye.')},
    'otp': {
        'en': ('{otp} is your {bank} OTP for login. Valid for 5 minutes. Do not share this '
               'OTP with anyone, including bank staff.',
               '{bank} verification code {otp} for your requested transaction. Expires in '
               '10 minutes. Never share this code with anyone.'),
        'hinglish': ('{bank} login ka OTP {otp} hai. 5 minute tak valid. OTP kisi ko mat share '
                     'karein, bank staff ko bhi nahi.',
                     'Aapke requested transaction ka {bank} verification code {otp}. '
                     '10 minute mein expire hoga. Kisi ke saath share mat karein.')},
    'parcel': {
        'en': ('Parcel status: shipment {ref} has reached the local delivery centre. '
               'Delivery expected on {date}. View status in your courier app.',
               'Your parcel with tracking ID {ref} was delivered on {date}. '
               'This is a delivery confirmation; no payment is requested.'),
        'hinglish': ('Parcel {ref} local delivery centre pahunch gaya hai. '
                     'Expected delivery {date}. Status courier app mein dekhein.',
                     'Tracking ID {ref} wala parcel {date} ko deliver ho gaya. '
                     'Yeh confirmation hai; koi payment nahi maanga gaya.')},
    'telecom_utility': {
        'en': ('Your mobile recharge of Rs.{amount} is successful. Plan validity 28 days. '
               'Receipt {ref}. Check usage in your operator app.',
               'Utility bill for account XX{account}: Rs.{amount}, due on {date}. '
               'Reference {ref}. View bill in the official utility app.'),
        'hinglish': ('Rs.{amount} ka mobile recharge successful hai. Plan validity 28 din. '
                     'Receipt {ref}. Usage operator app mein dekhein.',
                     'Utility account XX{account} ka bill Rs.{amount}, due date {date}. '
                     'Reference {ref}. Bill official utility app mein dekhein.')},
}
BANKS = (('SBI', 'SBISYN'), ('HDFC Bank', 'HDFSYN'), ('ICICI Bank', 'ICISYN'))


@dataclass(frozen=True)
class SyntheticSMS:
    id: str
    text: str
    sender: str
    family: str
    lang: str
    template_id: str
    generator_seed: int
    generator_version: str = VERSION
    synthetic: bool = True
    partition: str = 'train'
    label_binary: int = 0
    label_scam: int = 0
    label_policy: str = 'benign_transactional'
    source: str = 'synthetic_transactional_v1'

    def record(self):
        return asdict(self)


def fingerprint():
    return hashlib.sha256(json.dumps(TEMPLATES, sort_keys=True).encode()).hexdigest()


def key(text):
    return dedup_key(canonicalize(str(text)).normalized)


def generate(n, *, seed=20261101):
    if n < 0:
        raise ValueError('nonnegative synthetic count required')
    rng, rows, seen = random.Random(seed), [], set()
    index = 0
    while len(rows) < n:
        if index >= max(1, n * 4):
            raise ValueError('could not produce the fixed unique synthetic prefix')
        family = FAMILIES[index % 6]
        lang = ('en', 'hinglish')[(index // 6) % 2]
        variant = (index // 12) % 2
        bank, header = BANKS[rng.randrange(len(BANKS))]
        values = {'bank': bank, 'account': f'{rng.randrange(10000):04d}',
                  'amount': f'{rng.randrange(10, 25000)}.{rng.randrange(100):02d}',
                  'balance': f'{rng.randrange(100, 90000)}.{rng.randrange(100):02d}',
                  'date': f'{rng.randrange(1, 29):02d}-11-2026',
                  'ref': f'{rng.randrange(10**11, 10**12):012d}',
                  'otp': f'{rng.randrange(10**6):06d}'}
        text = TEMPLATES[family][lang][variant].format(**values)
        sender = ('VM', 'AD', 'JD')[rng.randrange(3)] + '-' + (
            'PRCSYN' if family == 'parcel' else 'UTLSYN' if family == 'telecom_utility' else header)
        template_id = f'{VERSION}/{family}/{lang}/{variant}'
        index += 1
        if len(text.encode('utf-8')) > 256:
            raise ValueError('synthetic notification exceeds the frozen byte budget')
        exact = key(text)
        if exact in seen:
            continue
        seen.add(exact)
        digest = hashlib.sha256(f'{seed}|{template_id}|{sender}|{text}'.encode()).hexdigest()
        rows.append(SyntheticSMS('txn-syn-v1-' + digest[:24], text, sender, family, lang,
                                 template_id, seed))
    return rows


def append_training(train, labels, teacher, rows, *, forbidden_keys):
    if train.buckets is not None:
        raise ValueError('this fixed experiment supports byte-CNN features only')
    if any(not r.synthetic or r.partition != 'train' or r.label_binary != 0
           or r.label_scam != 0 or r.label_policy != 'benign_transactional'
           or not r.id.startswith('txn-syn-v1-') for r in rows):
        raise ValueError('synthetic negative rows must be explicitly training-only')
    keys = [key(r.text) for r in rows]
    if len(set(keys)) != len(rows) or set(keys) & set(forbidden_keys):
        raise ValueError('synthetic duplicates or original/test/external text collision; abort arm')
    labels, teacher = np.asarray(labels), np.asarray(teacher)
    if labels.shape != (len(train),) or teacher.shape != (len(train), 2):
        raise ValueError('original training labels/teacher rows must align')
    extra = featurize([r.text for r in rows], max_len=train.raw_bytes.shape[1])
    combined = Featurized(train.texts + extra.texts, *(
        np.concatenate((getattr(train, name), getattr(extra, name)), axis=0)
        for name in ('raw_bytes', 'canon_bytes', 'side_raw', 'side_canon')))
    return (combined, np.r_[labels, np.zeros(len(rows), dtype=int)],
            np.concatenate((teacher, np.full((len(rows), 2), np.nan)), axis=0))
