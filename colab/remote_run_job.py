"""Runs on the Colab VM: run /content/rakshak_run.py in a fresh Python process.

colab/launch.sh uploads the generated runner to /content/rakshak_run.py and sends this file with
colab exec -f. The kernel that runs this file already has numpy imported. If the runner ran inside
that kernel, its pinned pip install would upgrade numpy under a live process and the imports would
fail (ImportError: cannot import name '_slice' from 'numpy._core.umath'). A child interpreter
starts clean: its pip install runs before its own imports.

Every line the runner writes to stdout or stderr is printed to the cell output and appended to
/content/out/runner.stdout.log, which the launcher packs with /content/out. The runner's exit code
becomes this file's: a non-zero code raises SystemExit, so launch.sh sees the failure. The
launcher's trap still stops the VM. Standard library only. RAKSHAK_REMOTE_ROOT replaces /content
(the tests use it).
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get('RAKSHAK_REMOTE_ROOT', '/content'))
RUNNER = ROOT / 'rakshak_run.py'
OUT = ROOT / 'out'
LOG = OUT / 'runner.stdout.log'


def main():
    if not RUNNER.is_file():
        raise SystemExit(f'run: {RUNNER} is missing; the runner upload did not land')
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    with open(LOG, 'w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, str(RUNNER)], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', env=env)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        code = proc.wait()
    print(f'run: runner exited {code}; log {LOG}', flush=True)
    if code != 0:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
