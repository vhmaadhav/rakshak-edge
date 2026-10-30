#!/usr/bin/env python3
"""One ntfy push subscription for a rakshak Kaggle job, replaying events since the launch.

Usage: kaggle/wait_event.py [JOB] [--timeout SECONDS] [--platform kaggle|colab]
(kaggle/wait_event.sh runs this). --platform colab reads colab/out/JOB and waits for
rakshak-colab-<job>; it needs the JOB name, since there is no latest Colab launch.

Opens one stream on the job's notify topic, replays cached messages since
kaggle/out/JOB/launch_epoch, and returns on the first completion message for this job and this
build's git label. There is no polling, sleep or retry. --timeout ends the wait with exit 124 and
writes nothing. The topic URL is never printed. On a match, writes kaggle/out/JOB/completion.json
(the event time and message, the job and the git label).

The constants are read from the generated kaggle/out/JOB/kernel/run.py with ast, so nothing is
executed.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import signal
import sys
import urllib.parse
import urllib.request
from pathlib import Path

KAGGLE = Path(__file__).resolve().parent
OUT = KAGGLE / 'out'
COLAB_OUT = KAGGLE.parent / 'colab' / 'out'
NAMES = ('JOB', 'GIT_LABEL', 'NOTIFY_URL')


def runner_constants(path: Path) -> dict:
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in NAMES:
                    found[target.id] = ast.literal_eval(node.value)
    missing = [name for name in NAMES if name not in found]
    if missing:
        raise SystemExit(f'wait_event: {path} lacks {missing}; rebuild the kernel')
    return found


def matches(event: dict, slug: str, label: str, since: int, platform: str = 'kaggle') -> bool:
    """A completion for this job and this build, not a stale run or another job. Colab runners
    send rakshak-colab-<slug>, Kaggle runners rakshak-<slug>."""
    if event.get('event') != 'message' or int(event.get('time', 0)) < since:
        return False
    prefix = 'rakshak-colab-' if platform == 'colab' else 'rakshak-'
    parts = str(event.get('message', '')).split()
    return (len(parts) >= 3 and parts[0] == f'{prefix}{slug}'
            and parts[1] in ('ok', 'partial', 'fail') and f'git={label}' in parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('job', nargs='?')
    parser.add_argument('--timeout', type=int, default=0,
                        help='seconds before giving up (0 = no limit)')
    parser.add_argument('--platform', choices=('kaggle', 'colab'), default='kaggle',
                        help='which launcher ran the job (default: kaggle)')
    args = parser.parse_args()

    job = args.job
    platform = args.platform
    if job is None and platform == 'kaggle' and (OUT / 'latest_launch.json').exists():
        job = json.loads((OUT / 'latest_launch.json').read_text())['job']
    if not job or not job.replace('_', '').replace('-', '').isalnum() or '/' in job:
        parser.error('a valid job name is required' + (
            '' if platform == 'colab' else ' (or run kaggle/launch.sh first)'))

    colab = platform == 'colab'
    folder = (COLAB_OUT if colab else OUT) / job
    constants = runner_constants(folder / 'kernel' / 'run.py')
    if constants['JOB'] != job:
        raise SystemExit(f"wait_event: the kernel in {folder} is for job {constants['JOB']!r}")
    url = constants['NOTIFY_URL']
    if not url:
        parser.error('this kernel was built without --internet, so it sends no notification; '
                     'rebuild with --internet to wait on it')
    label = constants['GIT_LABEL']
    if not (folder / 'launch_epoch').exists():
        launcher = 'colab/launch.sh' if colab else 'kaggle/launch.sh'
        parser.error(f'no launch recorded for {job}; run {launcher} {job} first')
    since = int((folder / 'launch_epoch').read_text())
    slug = job.replace('_', '-')
    request_url = url.rstrip('/') + '/json?' + urllib.parse.urlencode({'since': since})

    if args.timeout > 0:
        def expire(_signum, _frame):
            sys.stderr.write(f'wait_event: no completion within {args.timeout} s; '
                             'no retry performed\n')
            os._exit(124)
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(args.timeout)

    try:
        with urllib.request.urlopen(request_url, timeout=None) as stream:
            for raw in stream:
                event = json.loads(raw)
                if not matches(event, slug, label, since, platform):
                    continue
                receipt = {key: event.get(key) for key in ('event', 'time', 'message')}
                receipt.update(job=job, git_label=label)
                (folder / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
                print(json.dumps(receipt), flush=True)
                return 0
    except (OSError, ValueError) as exc:
        # urllib errors can contain the topic URL; only the exception type is printed.
        print(f'wait_event: stream failed ({type(exc).__name__}); no retry performed',
          file=sys.stderr)
        return 1
    print('wait_event: the stream ended without a completion event; no retry performed',
          file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
