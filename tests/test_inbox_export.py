"""Entirely invented SMS fixtures; no real inbox data."""

import json

import pandas as pd
import pytest

from tools.inbox_export import export
from tools.inbox_export.export import (
    COLUMNS,
    load_messages,
    main,
    prepare,
    review,
    select_messages,
    write_parquet,
)
from tools.inbox_export.scrub import scrub, scrub_sender, sender_style


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
    assert len(messages) == 1 and set(messages[0]) == {'body', 'address', 'contact_name', 'date'}
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


def _msg(body, address, date=None):
    date = '' if date is None else str(date)
    return {'body': body, 'address': address, 'contact_name': '', 'date': date}


def test_max_per_sender_keeps_newest_in_file_order_undated_last():
    messages = [
        _msg('a1', 'AX-HDFCBK-S', 1000000000001),
        _msg('a5', 'AX-HDFCBK-S', 1000000000005),
        _msg('a3', 'AX-HDFCBK-S', 1000000000003),
        _msg('a0', 'AX-HDFCBK-S'),
        _msg('a4', 'AX-HDFCBK-S', 1000000000004),
        _msg('b1', 'VM-AIRTEL-T', 100),
        _msg('b2', 'VM-AIRTEL-T', 200),
        _msg('c1', 'JD-ICICIT-G'), _msg('c2', 'JD-ICICIT-G'), _msg('c3', 'JD-ICICIT-G'),
    ]
    kept, counts = select_messages(messages, max_per_sender=2)
    assert [m['body'] for m in kept] == ['a5', 'a4', 'b1', 'b2', 'c1', 'c2']
    assert counts == {'senders': 3, 'kept': 6, 'dropped_by_cap': 4,
                      'skipped_person_to_person': 0}


def test_business_only_skips_person_to_person_and_flag_disables_it():
    messages = [_msg('bank', 'VM-AIRTEL-T', 1), _msg('friend', '+919876543210', 2),
                _msg('other', '9876543210', 3)]
    kept, counts = select_messages(messages)
    assert [m['body'] for m in kept] == ['bank']
    assert counts['skipped_person_to_person'] == 2
    kept, counts = select_messages(messages, business_only=False)
    assert len(kept) == 3 and counts['skipped_person_to_person'] == 0


def test_main_applies_cap_and_skip_exports_no_timestamp_or_counts(tmp_path, monkeypatch, capsys):
    xml = tmp_path / 'backup.xml'
    xml.write_text(
        '<smses>'
        '<sms type="1" address="AX-HDFCBK-S" body="Alpha offer" date="1700000000005"/>'
        '<sms type="1" address="AX-HDFCBK-S" body="Bravo offer" date="1700000000009"/>'
        '<sms type="1" address="AX-HDFCBK-S" body="Charlie offer" date="1700000000001"/>'
        '<sms type="1" address="AX-HDFCBK-S" body="Delta offer"/>'
        '<sms type="1" address="VM-AIRTEL-T" body="Echo offer" date="1700000000007"/>'
        '<sms type="1" address="+919876543210" body="Foxtrot private" date="1700000000008"/>'
        '</smses>')
    seen = []

    def approve(row, **_):
        seen.append(row['text'])
        return dict(row, label_policy='benign_transactional', human_reviewed=True), False

    monkeypatch.setattr(export, 'review', approve)
    output = tmp_path / 'reviewed.parquet'
    main([str(xml), '--output', str(output), '--max-per-sender', '2'])
    assert seen == ['Alpha offer', 'Bravo offer', 'Echo offer']
    out = capsys.readouterr().out
    assert 'dropped by cap 2' in out and 'skipped person-to-person 1' in out
    saved = pd.read_parquet(output)
    assert list(saved.columns) == list(COLUMNS)
    raw = output.read_bytes()
    for leak in (b'1700000000', b'dropped', b'skipped', b'Foxtrot', b'private'):
        assert leak not in raw


def test_dlt_suffix_survives_scrub_and_export(tmp_path):
    for address in ('AX-HDFCBK-S', 'VM-AIRTEL-T', 'JD-ICICIT-G', 'AX-HDFCBK-P'):
        assert sender_style(address) == 'alphanumeric_id'
        assert scrub_sender(address) == address
    assert scrub_sender('VM-AIRTEL5-T') == 'VM-AIRTEL<NUMBER>-T'
    row = prepare({'body': 'Synthetic offer text', 'address': 'AX-HDFCBK-S'})
    row.update(label_policy='benign_transactional', human_reviewed=True)
    target = tmp_path / 'reviewed.parquet'
    write_parquet([row], target)
    assert pd.read_parquet(target).sender.tolist() == ['AX-HDFCBK-S']


def test_merge_codex_keeps_only_private_nonpersonal_labels(tmp_path):
    import pandas as pd

    from tools.inbox_export import merge_codex as mc
    rows = [{'id': f'r{i}', 'sender': 'AX-TESTBK-S', 'sender_style': 'dlt',
             'text': f'synthetic {i}'}
            for i in range(5)]
    labels = [{'id': 'r0', 'label': 'scam', 'privacy_ok': True},
              {'id': 'r1', 'label': 'benign_transactional', 'privacy_ok': True},
              {'id': 'r2', 'label': 'promotion', 'privacy_ok': False},
              {'id': 'r3', 'label': 'benign_personal', 'privacy_ok': True},
              {'id': 'r4', 'label': 'not-a-label', 'privacy_ok': True}]
    out, counts = mc.merge(rows, labels)
    assert [o['id'] for o in out] == ['r0', 'r1']
    assert counts == {'rows': 5, 'unlabelled': 1, 'privacy_dropped': 1, 'personal_dropped': 1,
                      'kept': 2, 'by_label': {**{k: 0 for k in mc.LABELS},
                                              'scam': 1, 'benign_transactional': 1}}
    assert all(o['human_reviewed'] is False and o['label_source'] == 'codex' for o in out)
    dest = tmp_path / 'out.parquet'
    mc.write(out, dest)
    assert oct(dest.stat().st_mode & 0o777) == '0o600'
    assert list(pd.read_parquet(dest).columns) == list(mc.OUT_COLUMNS)
