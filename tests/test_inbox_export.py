"""Entirely invented SMS fixtures; no real inbox data."""

import json

import pandas as pd
import pytest

from tools.inbox_export.export import load_messages, prepare, review, write_parquet
from tools.inbox_export.scrub import scrub


def test_scrub_identifiers_preserves_brand_domain_structure():
    text = ('Dear Rohan Sharma, HDFC a/c XX1234 debited INR 2,345.67. OTP: 482915. '
            'Do not share. Call +91 98765 43210. UPI rohan@okaxis. '
            'https://user:secret@bank.example.com/customer/rohan?otp=482915#secret')
    clean = scrub(text, names=['Rohan Sharma', 'rohan'])
    for secret in ('Rohan', 'rohan', '1234', '2,345', '482915', '98765', 'okaxis', 'secret'):
        assert secret not in clean
    assert 'HDFC' in clean and 'bank.example.com/<PATH>?<QUERY>#<FRAGMENT>' in clean
    assert all(token in clean for token in ('<NAME>', '<AMOUNT>', '<OTP>', '<UPI_ID>', '<PHONE>'))
    assert 'Do not share' in clean


def test_xml_json_only_received_and_no_raw_metadata(tmp_path):
    xml = tmp_path / 'backup.xml'
    xml.write_text('<smses><sms type="1" address="VM-HDFCBK" contact_name="(Unknown)" '
                   'body="Dear Priya, OTP is 123456" date="1234567890"/>'
                   '<sms type="2" address="9999999999" body="outgoing"/>'
                   '<mms><parts><part text="private attachment"/></parts></mms></smses>')
    xml.write_bytes(b'\xef\xbb\xbf' + xml.read_bytes())
    messages = list(load_messages(xml))
    assert len(messages) == 1 and set(messages[0]) == {'body', 'address', 'contact_name'}
    row = prepare(messages[0])
    assert 'Priya' not in row['text'] and row['sender'] == 'VM-HDFCBK'
    assert row['synthetic'] is False and row['human_reviewed'] is False
    js = tmp_path / 'backup.json'
    js.write_text(json.dumps({'messages': [dict(messages[0], type=1)]}))
    assert list(load_messages(js)) == messages


def test_contact_and_hinglish_name_scrub():
    row = prepare({'body': 'नमस्ते अमित, amit ko Rs. 250 bheje. OTP 928133 share mat karo.',
                   'address': '9876543210', 'contact_name': 'अमित'}, names=['amit'])
    assert 'अमित' not in row['text'] and 'amit' not in row['text']
    assert not any(c.isdigit() for c in row['text']) and row['sender'] == '<PHONE>'


def test_review_requires_confirmation_and_exports_only_approved(tmp_path):
    row = prepare({'body': 'Synthetic fixture personal details', 'address': 'test'})
    answers = iter(['1', 'no', 'e', 'Dear customer, OTP 654321. Do not share.', '1', 'yes'])
    reviewed, stop = review(row, ask=lambda _: next(answers), show=lambda _: None)
    assert not stop and reviewed['human_reviewed'] and '654321' not in reviewed['text']
    target = tmp_path / 'reviewed.parquet'
    with pytest.raises(ValueError, match='approved'):
        write_parquet([row], target)
    write_parquet([reviewed], target)
    saved = pd.read_parquet(target)
    assert saved.label_policy.tolist() == ['benign_transactional']
    assert target.stat().st_mode & 0o777 == 0o600
    assert 'contact_name' not in saved and 'body' not in saved
    with pytest.raises(FileExistsError):
        write_parquet([reviewed], target)


def test_rejects_entities_and_git_output(tmp_path):
    path = tmp_path / 'bad.xml'
    path.write_text('<?xml version="1.0"?><!DOCTYPE smses [<!ENTITY p "secret">]>'
                    '<smses><sms type="1" body="&p;"/></smses>')
    with pytest.raises(ValueError, match='DTD'):
        list(load_messages(path))
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git/HEAD').write_text('ref: refs/heads/main\n')
    with pytest.raises(ValueError, match='git checkout'):
        write_parquet([], tmp_path / 'result.parquet')
