#!/usr/bin/env bash
# Copy the rbench harness and its registered model and dataset files from the laptop to the Pi.
#
#   deploy/pi/bench/sync_bench.sh [--dry-run] [--code-only | --files-only]
#
# Code: src/rakshak, deploy/pi/bench, deploy/pi/service and deploy/pi/pi_common.py, thermal_run.py.
#
# Env: PI_HOST (default hermoine@pi5.local), PI_KEY (default ~/.ssh/pi5),
#      RBENCH_REMOTE_DIR (default rakshak-bench, relative to the Pi home).
#
# Safety rules (do not weaken):
#   - Never deletes anything on the Pi. No rsync --delete, no rm.
#   - Never writes under models/llama32_1b_spinquant/ (the Llama files are placed by hand).
#   - A file whose sha256 on the Pi already matches the local source is skipped.
#   - Every local file is checked against its registry pin before anything is sent.
#   - Each transfer goes to <name>.part, is checked by sha256, then renamed into place.
#
# Exit: 0 ok, 1 usage or environment (ssh, python), 2 a local file is missing or does not match
# its pin (nothing transferred), 3 a remote transfer failed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PI_HOST="${PI_HOST:-hermoine@pi5.local}"
PI_KEY="${PI_KEY:-$HOME/.ssh/pi5}"
REMOTE_DIR="${RBENCH_REMOTE_DIR:-rakshak-bench}"
PROTECTED="models/llama32_1b_spinquant"
SSH_OPTS=(-i "$PI_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10)

DRY=0
DO_CODE=1
DO_FILES=1
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --code-only) DO_FILES=0 ;;
    --files-only) DO_CODE=0 ;;
    -h | --help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "sync_bench: unknown argument $arg" >&2; exit 1 ;;
  esac
done

case "$REMOTE_DIR" in
  "" | /* | *..*) echo "sync_bench: RBENCH_REMOTE_DIR must be a relative path" >&2; exit 1 ;;
esac

ssh_pi() { ssh "${SSH_OPTS[@]}" "$PI_HOST" "$@"; }

if ! ssh_pi true; then
  echo "sync_bench: cannot reach $PI_HOST with $PI_KEY" >&2
  exit 1
fi
echo "target: $PI_HOST:~/$REMOTE_DIR  (dry run: $DRY)"

# -- 1. code: src/rakshak, deploy/pi/bench, deploy/pi/service and the deploy/pi helpers ----
if [[ $DO_CODE == 1 ]]; then
  code_paths=(src/rakshak deploy/pi/bench deploy/pi/service deploy/pi/pi_common.py deploy/pi/thermal_run.py)
  n=$(tar -C "$REPO" --exclude=__pycache__ --exclude='*.pyc' -cf - "${code_paths[@]}" | tar -tf - | wc -l)
  echo "code: $n files from ${code_paths[*]}"
  if [[ $DRY == 0 ]]; then
    tar -C "$REPO" --exclude=__pycache__ --exclude='*.pyc' -cf - "${code_paths[@]}" |
      ssh_pi "mkdir -p $REMOTE_DIR && tar -C $REMOTE_DIR -xf -" ||
      { echo "sync_bench: code transfer failed" >&2; exit 3; }
  fi
fi

if [[ $DO_FILES == 0 ]]; then
  echo "done (code only)"
  exit 0
fi

# -- 2. registry manifest, validated locally before any transfer ------------------------------
manifest="$(python3 "$HERE/bin/rbench" manifest 2>&1)" || {
  echo "sync_bench: rbench manifest failed:" >&2
  echo "$manifest" >&2
  exit 1
}

declare -a owners roles sources paths pins
errors=0
while IFS=$'\t' read -r owner role source path pin; do
  [[ -z "$path" ]] && continue
  case "$path" in
    models/* | data/*) ;;
    *) echo "refuse: $path is outside models/ and data/" >&2; errors=$((errors + 1)); continue ;;
  esac
  if [[ "$path" == "$PROTECTED"/* ]]; then
    echo "refuse: $path is in the protected Llama directory" >&2
    errors=$((errors + 1))
    continue
  fi
  if [[ ! -f "$REPO/$source" ]]; then
    echo "missing local file: $source ($owner)" >&2
    errors=$((errors + 1))
    continue
  fi
  local_sha="$(sha256sum "$REPO/$source" | cut -c1-64)"
  if [[ -n "$pin" && "$local_sha" != "$pin" ]]; then
    echo "pin mismatch: $source has $local_sha, registry pins $pin ($owner)" >&2
    errors=$((errors + 1))
    continue
  fi
  owners+=("$owner"); roles+=("$role"); sources+=("$source"); paths+=("$path"); pins+=("$local_sha")
done <<<"$manifest"

if (( errors > 0 )); then
  echo "sync_bench: $errors problem(s) with local files; nothing transferred" >&2
  exit 2
fi

# -- 3. what is already on the Pi (read only, one ssh call) ------------------------------------
declare -A remote_sha
remote_list="$(printf '%s\n' "${paths[@]}" | ssh_pi "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && while IFS= read -r p; do if [ -f \"\$p\" ]; then printf '%s  %s\n' \"\$(sha256sum < \"\$p\" | cut -c1-64)\" \"\$p\"; else printf 'missing  %s\n' \"\$p\"; fi; done")" || {
  echo "sync_bench: could not read remote hashes" >&2
  exit 3
}
while read -r sha p; do
  [[ -z "$p" ]] && continue
  if [[ "$sha" == missing ]]; then remote_sha["$p"]=""; else remote_sha["$p"]="$sha"; fi
done <<<"$remote_list"

# -- 4. copy what differs --------------------------------------------------------------------
copied=0
skipped=0
bytes=0
for i in "${!paths[@]}"; do
  path="${paths[$i]}"
  source="${sources[$i]}"
  sha="${pins[$i]}"
  size=$(stat -c %s "$REPO/$source")
  if [[ "${remote_sha[$path]:-}" == "$sha" ]]; then
    skipped=$((skipped + 1))
    echo "same    $path"
    continue
  fi
  state="new    "
  [[ -n "${remote_sha[$path]:-}" ]] && state="update "
  echo "$state $path ($size bytes)"
  if [[ $DRY == 1 ]]; then
    copied=$((copied + 1))
    bytes=$((bytes + size))
    continue
  fi
  ssh_pi "mkdir -p '$REMOTE_DIR/$(dirname "$path")'" || { echo "sync_bench: mkdir failed" >&2; exit 3; }
  scp -q -i "$PI_KEY" -o BatchMode=yes -o IdentitiesOnly=yes "$REPO/$source" "$PI_HOST:$REMOTE_DIR/$path.part" ||
    { echo "sync_bench: scp failed for $path" >&2; exit 3; }
  ssh_pi "cd '$REMOTE_DIR' && s=\$(sha256sum '$path.part' | cut -c1-64) && [ \"\$s\" = '$sha' ] && mv -f '$path.part' '$path'" ||
    { echo "sync_bench: checksum after transfer failed for $path (left as $path.part)" >&2; exit 3; }
  copied=$((copied + 1))
  bytes=$((bytes + size))
done

echo "done: $copied to copy ($bytes bytes), $skipped already current$( [[ $DRY == 1 ]] && echo ' (dry run, nothing sent)')"
