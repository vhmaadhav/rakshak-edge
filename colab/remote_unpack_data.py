"""Runs on the Colab VM: rebuild /content/rakshak-data from the parts that launch.sh uploaded.

colab/launch.sh uploads /content/rakshak-data.manifest.json (written by colab/data_bundle.py) and
the parts /content/rakshak-data.part-NNN, then runs this file with colab exec -f. Each part is
checked against its size and sha256 as it is joined into the archive; the joined archive is
checked against the manifest too. Only then is it extracted (filter='data'), and every extracted
file is checked against its size. The parts, the archive and the manifest are removed at the end.
Any mismatch stops with a non-zero exit before anything is extracted.

Standard library only. RAKSHAK_REMOTE_ROOT replaces /content (the tests use it).
"""

import hashlib
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(os.environ.get('RAKSHAK_REMOTE_ROOT', '/content'))
MANIFEST = ROOT / 'rakshak-data.manifest.json'
ARCHIVE = ROOT / 'rakshak-data.tar.gz'
DEST = ROOT / 'rakshak-data'
BLOCK = 1024 * 1024
FORMAT = 1


def fail(message):
    raise SystemExit(f'unpack: {message}')


def join_parts(manifest):
    """Write the archive from the parts, checking each part and the whole archive."""
    archive_info = manifest['archive']
    archive_sha = hashlib.sha256()
    archive_bytes = 0
    with open(ARCHIVE, 'wb') as out:
        for part in manifest['parts']:
            path = ROOT / part['name']
            if not path.is_file():
                fail(f"missing part {path}; the upload is incomplete")
            size = path.stat().st_size
            if size != part['bytes']:
                fail(f"part {path} has {size} bytes, the manifest says {part['bytes']}")
            part_sha = hashlib.sha256()
            with open(path, 'rb') as source:
                for block in iter(lambda: source.read(BLOCK), b''):
                    part_sha.update(block)
                    archive_sha.update(block)
                    out.write(block)
                    archive_bytes += len(block)
            if part_sha.hexdigest() != part['sha256']:
                fail(f'part {path} fails its sha256; the upload is corrupt')
    if archive_bytes != archive_info['bytes']:
        fail(f"archive has {archive_bytes} bytes, the manifest says {archive_info['bytes']}")
    if archive_sha.hexdigest() != archive_info['sha256']:
        fail('the joined archive fails its sha256')


def main():
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('format') != FORMAT:
        fail(f"unknown manifest format {manifest.get('format')!r}")
    if not manifest.get('parts'):
        fail('the manifest lists no parts')
    join_parts(manifest)
    DEST.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, mode='r:gz') as tar:
        tar.extractall(DEST, filter='data')
    for entry in manifest['files']:
        path = DEST / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            fail(f"{path} is missing or has the wrong size after extraction")
    for part in manifest['parts']:
        os.remove(ROOT / part['name'])
    os.remove(ARCHIVE)
    os.remove(MANIFEST)
    print(f"unpacked {len(manifest['files'])} files into {DEST}: "
          f"{sorted(p.name for p in DEST.iterdir())}")


if __name__ == '__main__':
    main()
