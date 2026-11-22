#!/usr/bin/env bash
# Arm A: OLD baseline 2b3c1bd at 2.4 GHz. English eval (40 rows) + 100-request soak (50 if the guard trips).
. "$(dirname "$0")/lib.sh"
export HOME=$OLD_HOME RAKSHAK_BENCH=$OLD_HOME/rakshak-bench
S=$RAKSHAK_BENCH/deploy/pi/service
PY=$RAKSHAK_BENCH/.venv/bin/python
D=$RES/armA_old
mkdir -p "$D"
svc_stop() { "$S/stop_service.sh" >> "$D/stop.txt" 2>&1; }
trap svc_stop EXIT
"$S/start_service.sh" --threads 3 --explainer-cache 0 > "$D/start.txt" 2>&1 || { echo "start failed" | tee -a "$D/arms.txt"; exit 1; }
curl -s http://127.0.0.1:8770/health > "$D/health_before.json"; echo
run_arm "$D" eval_en "$PY $S/eval_service.py run --rows $RAKSHAK_BENCH/results/edge_eval/rows.json --out $D/eval_en"
"$PY" "$S/eval_service.py" report --out "$D/eval_en" > "$D/eval_en_report.txt" 2>&1
run_arm "$D" soak100 "$PY $S/soak.py run --threads 3 --requests 100 --out $D/soak100"
if [[ -f "$D/invalid_soak100.txt" ]]; then
  echo "soak100 tripped the guard: cutting soak to 50" | tee -a "$D/arms.txt"
  run_arm "$D" soak50 "$PY $S/soak.py run --threads 3 --requests 50 --out $D/soak50"
fi
curl -s http://127.0.0.1:8770/health > "$D/health_after.json"; echo
echo DONE armA
