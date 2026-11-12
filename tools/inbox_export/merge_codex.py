"""Merge Codex labels into an inbox parquet for sms_inbox_retrain (plan amendment 2026-11-07).

    python -m tools.inbox_export.merge_codex ~/private/codex_review/to_label.jsonl \
        ~/private/codex_review/labels.jsonl --output ~/private/reviewed.parquet

Rows keep human_reviewed=False and get label_source="codex". Rows Codex marks privacy_ok=false,
labels benign_personal, or leaves unlabelled are dropped. Counts only are printed.
"""
import argparse
import json
import os
from pathlib import Path

from tools.inbox_export.export import COLUMNS, LABELS
from tools.inbox_export.scrub import VERSION

OUT_COLUMNS = (*COLUMNS, 'label_source')


def merge(rows, labels):
    by_id = {}
    for lab in labels:
        if lab.get('label') in LABELS and lab.get('id'):
            by_id[lab['id']] = lab
    out, counts = [], {'rows': 0, 'unlabelled': 0, 'privacy_dropped': 0, 'personal_dropped': 0}
    for r in rows:
        counts['rows'] += 1
        lab = by_id.get(r['id'])
        if lab is None:
            counts['unlabelled'] += 1
            continue
        if lab.get('privacy_ok') is not True:
            counts['privacy_dropped'] += 1
            continue
        if lab['label'] == 'benign_personal':
            counts['personal_dropped'] += 1
            continue
        out.append({'id': r['id'], 'text': r['text'], 'sender': r['sender'],
                    'sender_style': r['sender_style'], 'label_policy': lab['label'],
                    'synthetic': False, 'human_reviewed': False,
                    'scrubber_version': VERSION, 'label_source': 'codex'})
    counts['kept'] = len(out)
    counts['by_label'] = {k: sum(o['label_policy'] == k for o in out) for k in LABELS}
    return out, counts


def write(rows, output):
    import pandas as pd

    path = Path(output).expanduser().resolve()
    def git_marker(parent):  # same rule as export.write_parquet
        marker = parent / '.git'
        return marker.is_file() or (marker / 'HEAD').is_file()
    if any(git_marker(p) for p in (path.parent, *path.parents)):
        raise ValueError('write outside every git checkout; never store SMS in git')
    if not rows:
        raise ValueError('no rows to write')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pd.DataFrame(rows, columns=OUT_COLUMNS).to_parquet(stream, index=False)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _jsonl(path):
    with open(Path(path).expanduser(), encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rows', type=Path)
    p.add_argument('labels', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        out, counts = merge(_jsonl(args.rows), _jsonl(args.labels))
        write(out, args.output)
    except (ValueError, OSError, json.JSONDecodeError, KeyError):
        p.exit(2, 'Could not merge. Check both files and a new output path outside git.\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
