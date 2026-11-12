"""Run with python -m tools.inbox_export.export; no network or model labelling."""

import argparse
import json
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.inbox_export.scrub import VERSION, scrub, scrub_sender, sender_style

DEFAULT_MAX_PER_SENDER = 40
LABELS = ('benign_transactional', 'benign_personal', 'promotion', 'scam', 'unclear')
COLUMNS = ('id', 'text', 'sender', 'sender_style', 'label_policy', 'synthetic',
           'human_reviewed', 'scrubber_version')


def load_messages(path):
    """SMS only. Drop MMS, calls and metadata; received type1 only."""
    path = Path(path)
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError('backup exceeds 256 MiB; split locally first')
    raw = path.read_bytes()
    if path.suffix.lower() == '.xml':
        # Reject declarations even if differently encoded (e.g. UTF16).
        prefix = raw.removeprefix(b'\xef\xbb\xbf').lstrip()
        if not prefix.startswith(b'<?xml') and not prefix.startswith(b'<smses'):
            raise ValueError('only UTF8 SMS XML is supported')
        text = raw.decode('utf-8-sig')
        if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
            raise ValueError('DTD/entity declarations are not supported')
        root = ET.fromstring(text)
        if root.tag != 'smses':
            raise ValueError('expected SMS Backup & Restore smses root')
        rows = [node.attrib for node in root if node.tag == 'sms']
    elif path.suffix.lower() == '.json':
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = rows.get('sms', rows.get('messages'))
        if not isinstance(rows, list):
            raise ValueError('JSON requires an array, or sms/messages array')
    else:
        raise ValueError('only uncompressed XML or JSON is supported')
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('body'), str):
            raise ValueError('each SMS requires a string body')
        if str(row.get('type')) != '1':
            continue
        yield {key: row.get(key, '') for key in ('body', 'address', 'contact_name', 'date')}


def _sort_date(value):
    # XML/JSON dates are epoch milliseconds; anything else counts as undated.
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def select_messages(messages, *, max_per_sender=DEFAULT_MAX_PER_SENDER, business_only=True):
    """Group by raw sender address, keep the newest N per sender, return in file order.

    Undated messages rank after dated ones, in file order. Dates are used only for ranking
    and are never exported. Returns (messages, counts); counts are for the screen only.
    """
    groups, personal_skipped = {}, 0
    for seq, message in enumerate(messages):
        if business_only and sender_style(message['address']) == 'numeric':
            personal_skipped += 1
            continue
        groups.setdefault(str(message['address']), []).append((seq, message))

    def rank(item):
        seq, message = item
        date = _sort_date(message.get('date'))
        return (0, -date, seq) if date is not None else (1, 0, seq)

    kept, dropped = [], 0
    for members in groups.values():
        chosen = sorted(members, key=rank)[:max_per_sender]
        dropped += len(members) - len(chosen)
        kept.extend(chosen)
    kept.sort(key=lambda item: item[0])
    counts = {'senders': len(groups), 'kept': len(kept), 'dropped_by_cap': dropped,
              'skipped_person_to_person': personal_skipped}
    return [message for _, message in kept], counts


def prepare(message, names=()):
    # Caller-provided personal names and contact metadata stay in memory only.
    contact = str(message.get('contact_name') or '').strip()
    names = list(names)
    if contact and contact not in {'(Unknown)', 'null'} and contact.upper() not in {
        'SBI', 'HDFC', 'ICICI', 'AIRTEL', 'JIO', 'BSNL'
    }:
        names.append(contact)
    return {'id': uuid.uuid4().hex, 'text': scrub(message['body'], names=names),
            'sender': scrub_sender(message['address'], names=names),
            'sender_style': sender_style(message['address']), 'label_policy': None,
            'synthetic': False, 'human_reviewed': False, 'scrubber_version': VERSION}


def review(row, *, names=(), ask=input, show=print):
    row = dict(row)
    while True:
        show(f"\nSender: {row['sender']} ({row['sender_style']})\n{row['text']}")
        show('1 transactional | 2 personal | 3 promotion | 4 scam | 5 unclear')
        action = ask('Choose label, e=edit text, s=skip, q=finish: ').strip()
        if action == 'q':
            return None, True
        if action == 's':
            return None, False
        if action == 'e':
            row['text'] = scrub(ask('Replace with fully scrubbed text: '), names=names)
            continue
        if action not in {'1', '2', '3', '4', '5'}:
            continue
        if ask('Confirm no personal names/details remain (yes/no): ').strip().lower() != 'yes':
            continue
        row.update(label_policy=LABELS[int(action)-1], human_reviewed=True)
        return row, False


def write_parquet(rows, output):
    import pandas as pd

    path = Path(output).expanduser().resolve()
    def git_marker(parent):
        marker = parent / '.git'
        return marker.is_file() or (marker / 'HEAD').is_file()
    if any(git_marker(parent) for parent in (path.parent, *path.parents)):
        raise ValueError('write outside every git checkout; never store SMS in git')
    if not rows or any(set(r) != set(COLUMNS) or not r['human_reviewed']
                       or r['label_policy'] not in LABELS for r in rows):
        raise ValueError('only manually approved labelled rows can be exported')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pd.DataFrame(rows, columns=COLUMNS).to_parquet(stream, index=False)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('backup', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--names-file', type=Path,
                   help='Local personal names, one per line; never exported')
    p.add_argument('--max-per-sender', type=int, default=DEFAULT_MAX_PER_SENDER,
                   help='Keep only the newest N messages per raw sender address (default 40)')
    p.add_argument('--business-only', action=argparse.BooleanOptionalAction, default=True,
                   help='Skip person-to-person messages (phone-number sender); on by default. '
                        'Use --no-business-only to include them.')
    args = p.parse_args(argv)
    if args.max_per_sender < 1:
        p.error('--max-per-sender must be at least 1')
    names = args.names_file.read_text().splitlines() if args.names_file else []
    approved = []
    try:
        messages, counts = select_messages(load_messages(args.backup),
                                           max_per_sender=args.max_per_sender,
                                           business_only=args.business_only)
        # Counts are printed for the reviewer only; they are never written to the output.
        print(f"Senders {counts['senders']}; to review {counts['kept']}; dropped by cap "
              f"{counts['dropped_by_cap']}; skipped person-to-person "
              f"{counts['skipped_person_to_person']}.")
        for message in messages:
            row, stop = review(prepare(message, names), names=names)
            if row:
                approved.append(row)
            if stop:
                break
        if approved:
            write_parquet(approved, args.output)
        print(f'Approved {len(approved)} rows. No uploads performed.')
    except (ValueError, OSError, ET.ParseError, UnicodeError):
        # Do not echo parser errors containing raw SMS, paths or credentials.
        p.exit(2, 'Could not process backup/output. Check format and a new local output path.\n')


if __name__ == '__main__':
    main()
