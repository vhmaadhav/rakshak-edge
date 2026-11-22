# r3 thermal guard + arm runner (sourced by arm_old.sh / arm_new.sh). No external polling:
# the guard is a background sampler inside the same ssh command, killed when the arm ends.
# Override any of these with env vars; defaults are the Pi 5 paths used for the published run.
PERF=${PERF_DIR:-/home/hermoine/rakshak-perf}
TOOLS=${TOOLS_DIR:-/home/hermoine/pf-tools}
RES=$PERF/run/bench_r3
OLD_HOME=${OLD_HOME:-/home/hermoine/bl-home}
NEW_HOME=${NEW_HOME:-/home/hermoine/pf-home}
TABLE=$PERF/run/explanation_table_v2.json

read_dev() {  # prints: temp_c throttled_hex clock_mhz
  local t th c
  t=$(vcgencmd measure_temp | sed -E "s/^temp=([0-9.]+).*/\1/")
  th=$(vcgencmd get_throttled | sed -E "s/^throttled=//")
  c=$(vcgencmd measure_clock arm | sed -E "s/.*=//")
  echo "$t $th $(( c / 1000000 ))"
}

precheck() {  # $1 dir: read temp; if >= 65 C, one 120 s cooldown, then read again
  local t th c
  mkdir -p "$1"
  read -r t th c < <(read_dev)
  echo "[pre $(date +%T)] temp=$t throttled=$th clock_mhz=$c" | tee -a "$1/precheck.txt"
  if awk "BEGIN{exit !($t>=65)}"; then
    echo "[pre] temp $t >= 65: one sleep 120 before this arm" | tee -a "$1/precheck.txt"
    sleep 120
    read -r t th c < <(read_dev)
    echo "[pre after cooldown $(date +%T)] temp=$t throttled=$th clock_mhz=$c" | tee -a "$1/precheck.txt"
  fi
}

guard() {  # $1 device log, $2 target process group, $3 invalid marker file
  local ep t th c reason start
  start=$(date +%s)
  while :; do
    read -r t th c < <(read_dev)
    ep=$(date +%s)
    echo "$ep $t $th $c" >> "$1"
    reason=""
    awk "BEGIN{exit !($t>=78)}" && reason="temp=$t>=78"
    (( (th & 0xF) != 0 )) && reason="$reason throttled=$th"
    (( ep - start >= 3 && c < 2350 )) && reason="$reason clock=${c}MHz<2350"
    if [[ -n $reason ]]; then
      echo "INVALID at $(date +%T):$reason" > "$3"
      kill -TERM -- "-$2" 2>/dev/null
      exit 0
    fi
    sleep 1
  done
}

summarize() {  # $1 dir, $2 arm name
  local f="$1/device_$2.log" n=0 first="" tmax="" cmin=999999 cmax=0 low=0 sticky=0 ep t th c
  while read -r ep t th c; do
    [[ -z ${c:-} ]] && continue
    n=$((n+1)); [[ -z $first ]] && first=$t
    if [[ -z $tmax ]] || awk "BEGIN{exit !($t>$tmax)}"; then tmax=$t; fi
    (( c < cmin )) && cmin=$c
    (( c > cmax )) && cmax=$c
    low=$(( low | (th & 0xF) ))
    (( th & 0x80000 )) && sticky=1
  done < "$f"
  local st=VALID; [[ -f "$1/invalid_$2.txt" ]] && st="INVALID ($(cat "$1/invalid_$2.txt"))"
  printf "arm=%s status=%s samples=%s temp_start=%s temp_peak=%s clock_min_mhz=%s clock_max_mhz=%s throttled_low_nibble=0x%X sticky_0x80000=%s\n" \
    "$2" "$st" "$n" "$first" "$tmax" "$cmin" "$cmax" "$low" "$sticky" | tee -a "$1/summary.txt"
}

run_arm() {  # $1 dir, $2 arm name, $3 command (run under setsid with the guard attached)
  local d=$1 name=$2 cmd=$3 t0 t1 pg gp rc
  precheck "$d"
  rm -f "$d/invalid_$name.txt"; : > "$d/device_$name.log"
  t0=$(date +%s)
  setsid bash -c "$cmd" > "$d/$name.out" 2> "$d/$name.err" < /dev/null &
  pg=$!
  guard "$d/device_$name.log" "$pg" "$d/invalid_$name.txt" &
  gp=$!
  wait "$pg"; rc=$?
  kill "$gp" 2>/dev/null; wait "$gp" 2>/dev/null
  t1=$(date +%s)
  echo "$name rc=$rc secs=$((t1-t0))" | tee -a "$d/arms.txt"
  summarize "$d" "$name"
}
