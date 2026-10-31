"""Runs on the Colab VM: start /content/rakshak_run.py detached, record its PID and return at once.

colab/launch.sh uploads the runner to /content/rakshak_run.py and sends this file with a short
colab exec -f. The runner gets its own session (start_new_session), so it keeps running after this
cell returns and after the laptop's connection drops. Its stdout and stderr go to
/content/out/runner.stdout.log and its PID to /content/out/runner.pid. Nothing here waits for the
job: its end is reported by the ntfy completion line, which colab/wait.sh subscribes to.

A runner that exits within the first second counts as a failed start. Standard library only.
RAKSHAK_REMOTE_ROOT replaces /content (the tests use it).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('RAKSHAK_REMOTE_ROOT', '/content'))
RUNNER = ROOT / 'rakshak_run.py'
OUT = ROOT / 'out'
LOG = OUT / 'runner.stdout.log'
PID = OUT / 'runner.pid'
GRACE_S = 1.0


def main():
    if not RUNNER.is_file():
        raise SystemExit(f'start: {RUNNER} is missing; the runner upload did not land')
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    with open(LOG, 'a', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, str(RUNNER)], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
    PID.write_text(f'{proc.pid}\n')
    time.sleep(GRACE_S)
    code = proc.poll()
    if code is not None:
        raise SystemExit(f'start: runner exited {code} within {GRACE_S} s; see {LOG}')
    print(f'start: runner pid {proc.pid} detached; log {LOG}', flush=True)


if __name__ == '__main__':
    main()
