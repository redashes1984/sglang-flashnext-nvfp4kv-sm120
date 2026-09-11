#!/bin/bash
# FINAL acceptance soak — v2.8 @ default alpha=0.8, eager cold-pool
# 30min: rotating-topic traffic (topic rotation is exactly what feeds the probe)
# then: full log triage + memory census + quality probes
U=sglang-dealignai-qwen4exp-nvfp4kv
BASE=$(journalctl -u $U --no-pager -o short-iso | wc -l)
TOPICS=("graph neural network message passing" "Byzantine fault tolerance consensus" "RNA splicing introns" "Haskell monad transformers" "glacier isostatic rebound" "Fourier ptychography microscopy" "crystalline silicon defects doping" "Mughan miniature painting pigments" "vector clock distributed systems" "topological insulators surface states" "medieval canon law decretals" "sonar beamforming arrays")
END=$((SECONDS + 1800)); TRIPS=0
while [ $SECONDS -lt $END ]; do
  for t in "${TOPICS[@]:RANDOM%12:3}"; do
    curl -s -m 90 http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
      -d "{\"text\": \"Write a detailed technical note about ${t}. $RANDOM\", \"sampling_params\": {\"temperature\": 0.85, \"max_new_tokens\": 200}}" >/dev/null 2>&1
  done
  TRIPS=$((TRIPS+3)); sleep 40
done
echo "== SOAK DONE trips=$TRIPS =="
NEW=$(journalctl -u $U --no-pager | tail -n +$((BASE+1)))
echo "== staging stats (last 8) =="
echo "$NEW" | grep -E "calls=" | tail -8
echo "== ERROR/WARN triage =="
echo "$NEW" | grep -iE "error|exception|traceback|assert|mismatch|stage_fail|corrupt" | grep -v "0 rows" | sort | uniq -c | sort -rn | head -12
echo "== warnings =="
echo "$NEW" | grep -iE "warn" | sed -E 's/^[A-Za-z0-9 :-]+ sglang\[[0-9]+\]: //; s/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -8
echo "== checksum coverage =="
echo "$NEW" | grep "checksum" | tail -3
echo "== mem =="
grep -E "^(Shmem|MemAvailable):" /proc/meminfo | awk '{printf "%-12s %7.1f GB\n", substr($1,1,11), $2/1048576}'
free -h | tail -1
echo "== quality probes =="
# explicit failure path (R4 item 4): missing env/python or missing probe file
# must NOT silently degrade the acceptance run.
QRC=0
if [ ! -x /opt/sglang-env/bin/python3 ] && [ ! -e /opt/sglang-env/bin/python3 ]; then
  echo "QUALITY-PROBE SKIP: /opt/sglang-env/bin/python3 missing"; QRC=1
elif [ ! -f /opt/sglang-test/probe_cp.py ]; then
  echo "QUALITY-PROBE SKIP: /opt/sglang-test/probe_cp.py missing"; QRC=1
else
  /opt/sglang-env/bin/python3 /opt/sglang-test/probe_cp.py 2>&1 | head -6 || QRC=1
fi
echo FINAL-SOAK-DONE
exit $QRC
