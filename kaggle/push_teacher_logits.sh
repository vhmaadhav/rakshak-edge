#!/usr/bin/env bash
# Usage: kaggle/push_teacher_logits.sh
#
# Creates or versions the PRIVATE Kaggle dataset maadhavvh/rakshak-teacher-logits: the six sms_teacher transfer
# files (teacher jhu-clsp/mmBERT-base, seed 0) plus manifest.json. Sources are the fetched outputs of the two
# teacher runs: kaggle/out/sms_teacher/output (campaign and three source_holdout folds) and
# kaggle/out/sms_teacher-b/output (lang_shift and temporal). manifest.json maps each run name to its file and
# sha256. Stages a copy under kaggle/out/rakshak-teacher-logits/ (the sources are never modified). Nothing here
# runs compute. Attach with: --datasets maadhavvh/rakshak-teacher-logits
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="maadhavvh/rakshak-teacher-logits"
STAGE="$HERE/out/rakshak-teacher-logits"
LOG="$HERE/out/push-teacher-logits.log"
SRC_A="$HERE/out/sms_teacher/output"
SRC_B="$HERE/out/sms_teacher-b/output"
EXPECTED_FILES=6

rm -rf "$STAGE"
mkdir -p "$STAGE"
python3 - "$SRC_A" "$SRC_B" "$STAGE" "$DATASET" "$EXPECTED_FILES" <<'PY'
import hashlib
import json
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sources = [Path(sys.argv[1]), Path(sys.argv[2])]
stage, dataset, expected = Path(sys.argv[3]), sys.argv[4], int(sys.argv[5])
pattern = re.compile(r"^transfer_(?P<teacher>.+?)_s(?P<seed>\d+)_(?P<run>.+)\.parquet$")

splits, teacher, seed = {}, None, None
for src in sources:
    metrics = src / "metrics.json"
    if not metrics.is_file():
        sys.exit(f"push_teacher_logits: {metrics} is missing; fetch the teacher outputs first")
    meta = json.loads(metrics.read_text(encoding="utf-8"))
    # Run keys as the teacher wrote them (source_holdout:holdout_x), keyed by the file-name form.
    run_keys = {name.replace(":", "_"): name for name in meta["splits"]}
    for path in sorted(src.glob("transfer_*.parquet")):
        match = pattern.match(path.name)
        if not match:
            sys.exit(f"push_teacher_logits: {path.name} does not match the transfer file name")
        run = run_keys.get(match["run"])
        if run is None:
            sys.exit(f"push_teacher_logits: {path.name}: run {match['run']} is not in {metrics}")
        if run in splits:
            sys.exit(f"push_teacher_logits: run {run} appears twice")
        teacher = meta["meta"]["models"][0]
        seed = int(match["seed"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        splits[run] = {"file": path.name, "sha256": digest, "source": src.parent.name}
        shutil.copyfile(path, stage / path.name)

if len(splits) != expected:
    sys.exit(f"push_teacher_logits: found {len(splits)} transfer files, expected {expected}")
manifest = {
    "dataset": dataset,
    "teacher": teacher,
    "seed": seed,
    "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "columns": ["id", "split", "logit_0", "logit_1"],
    "splits": dict(sorted(splits.items())),
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
meta = {"id": dataset, "title": "rakshak-teacher-logits", "licenses": [{"name": "other"}]}
(stage / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(f"staged {len(splits)} transfer files for {teacher} seed {seed}")
PY

set +e
kaggle datasets status "$DATASET" >/dev/null 2>&1
exists=$?
set -e
set +e
if [ "$exists" -eq 0 ]; then
  kaggle datasets version -p "$STAGE" -m "rakshak teacher logits $(date -u +%Y-%m-%dT%H:%M:%SZ)" -r zip >"$LOG" 2>&1
else
  kaggle datasets create -p "$STAGE" -r zip >"$LOG" 2>&1
fi
rc=$?
set -e
cat "$LOG"
if [ "$rc" -ne 0 ] || grep -qiE 'error|failed|exception|forbidden|unauthori[sz]ed|invalid' "$LOG"; then
  echo "push_teacher_logits: the Kaggle CLI reported a failure (see $LOG)" >&2
  exit 1
fi
echo "push_teacher_logits: $DATASET updated; attach it with --datasets $DATASET"
