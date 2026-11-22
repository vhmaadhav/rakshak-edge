#!/usr/bin/env bash
# NEW (round-2 final, verdict_first + explanation table v2) at 2.4 GHz. Usage: arm_new.sh WORKERS CLIENT_PER
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/lib.sh"
N=$1; PER=$2
export HOME=$NEW_HOME RAKSHAK_BENCH=$NEW_HOME/rakshak-bench
S=$RAKSHAK_BENCH/deploy/pi/service
B=python3
A=$HERE/armbc2.py
D=$RES/armB_new_w$N
mkdir -p "$D"
svc_stop() { "$S/stop_service.sh" >> "$D/stop.txt" 2>&1; }
trap svc_stop EXIT
rm -f $PERF/run/edge.sock.*   # stale per-worker sockets make the bridge relay to dead sockets
"$S/start_service.sh" --threads 3 --explainer-cache 0 --response-mode verdict_first \
  --explanation-table $TABLE --workers $N > "$D/start.txt" 2>&1 || { echo "start failed" | tee -a "$D/arms.txt"; exit 1; }
curl -s http://127.0.0.1:8770/health > "$D/health_before.json"; echo
run_arm "$D" ttv "$B $A ttv 5"
run_arm "$D" drain "$B $A drain"
run_arm "$D" seq2000 "$B $A seq 2000 $TOOLS/hits.json"
run_arm "$D" batch256 "$B $A batch 256 4096 $TOOLS/hits.json"
run_arm "$D" batch1024 "$B $A batch 1024 4096 $TOOLS/hits.json"
run_arm "$D" clients16 "$B $A clients 16 $PER $TOOLS/hits.json"
run_arm "$D" batchpar256 "$B $A batchpar 256 2 $TOOLS/hits.json"
run_arm "$D" batchpar1024 "$B $A batchpar 1024 1 $TOOLS/hits.json"
run_arm "$D" uds2000 "$B $A uds 2000 $TOOLS/hits.json"
if [[ $N == 1 ]]; then
  $B -c "import json;r=json.load(open(\"$TOOLS/ttv_records.json\"));t=[x[\"text\"] for x in r if x[\"status\"]!=\"ready\"][:1];json.dump(t,open(\"$D/miss_text.json\",\"w\"))"
  run_arm "$D" llm_one "$B $HERE/llm_one.py $D/miss_text.json $D/llm_one.json"
fi
curl -s http://127.0.0.1:8770/health > "$D/health_after.json"; echo
cp $TOOLS/ttv_records.json $TOOLS/hits.json "$D/" 2>/dev/null
echo DONE
