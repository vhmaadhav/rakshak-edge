"""Runs on the Colab VM: pack /content/out into /content/rakshak-out.tar.gz.

colab/launch.sh sends this file with colab exec -f after the runner finishes, then downloads the
archive. Standard library only; the paths are fixed. The archive holds out/<files>.
"""

import tarfile
from pathlib import Path

OUT = Path('/content/out')
ARCHIVE = Path('/content/rakshak-out.tar.gz')

if not OUT.is_dir():
    raise SystemExit(f'{OUT} does not exist: the runner did not start')
with tarfile.open(ARCHIVE, mode='w:gz') as tar:
    tar.add(OUT, arcname='out')
print(f'packed {OUT} into {ARCHIVE} ({ARCHIVE.stat().st_size} bytes)')
