#!/usr/bin/env bash
# Usage: kaggle/push_data.sh
#
# Creates or versions the PRIVATE Kaggle dataset maadhavvh/rakshak-data from data/processed/.
# Creates data/processed/ with a README.txt placeholder if missing. Refuses to upload when the folder holds no
# file other than the placeholder. Stages a copy under kaggle/out/rakshak-data/ (the repo data/ folder is never
# modified, and data/ is gitignored). Nothing here runs compute.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/data/processed"
DATASET="maadhavvh/rakshak-data"
STAGE="$HERE/out/rakshak-data"

mkdir -p "$SRC"
if [ ! -f "$SRC/README.txt" ]; then
  cat > "$SRC/README.txt" <<'PLACEHOLDER'
rakshak-data placeholder. Replace this folder's contents with the processed datasets before pushing.
kaggle/push_data.sh refuses to upload while this is the only file here.
PLACEHOLDER
fi
data_files=$(find "$SRC" -type f ! -name README.txt | wc -l)
if [ "$data_files" -eq 0 ]; then
  echo "push_data: $SRC has no data files besides README.txt; nothing to upload" >&2
  exit 2
fi

rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$SRC"/. "$STAGE"/
python3 - "$STAGE" "$DATASET" <<'META'
import json
import sys
from pathlib import Path
folder, dataset = Path(sys.argv[1]), sys.argv[2]
meta = {'id': dataset, 'title': 'rakshak-data', 'licenses': [{'name': 'other'}]}
(folder / 'dataset-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
META

set +e
kaggle datasets status "$DATASET" >/dev/null 2>&1
exists=$?
set -e
set +e
if [ "$exists" -eq 0 ]; then
  kaggle datasets version -p "$STAGE" -m "rakshak data $(date -u +%Y-%m-%dT%H:%M:%SZ)" -r zip >"$HERE/out/push-data.log" 2>&1
else
  kaggle datasets create -p "$STAGE" -r zip >"$HERE/out/push-data.log" 2>&1
fi
rc=$?
set -e
cat "$HERE/out/push-data.log"
if [ "$rc" -ne 0 ] || grep -qiE 'error|failed|exception|forbidden|unauthori[sz]ed|invalid' "$HERE/out/push-data.log"; then
  echo "push_data: the Kaggle CLI reported a failure (see $HERE/out/push-data.log)" >&2
  exit 1
fi
echo "push_data: $DATASET updated from $data_files file(s); attach it with --datasets $DATASET"
