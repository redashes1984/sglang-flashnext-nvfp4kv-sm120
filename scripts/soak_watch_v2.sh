#!/bin/bash
# soak watcher v2: cold-pool eager, 30min, traffic + staging + memory watch
U=sglang-dealignai-qwen4exp-nvfp4kv
BASE=$(journalctl -u $U --no-pager -o short-iso | wc -l)
END=$((SECONDS + 1800))
declare -i TRIPS=0
while [ $SECONDS -lt $END ]; do
  # drive traffic: varied prompts incl. code + long tail to force cold routing
  for i in 1 2 3; do
    curl -s -m 90 http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
      -d "{\"text\": \"Explain how $i-layer KV cache eviction interacts with sparse attention routing in long-context transformers. marker $RANDOM\", \"sampling_params\": {\"temperature\": 0.6, \"max_new_tokens\": 220}}" >/dev/null 2>&1
  done
  TRIPS+=3
  sleep 50
done
echo "== TRAFFIC DONE trips=$TRIPS =="
# report: staging stats + errors + memory
NEW=$(journalctl -u $U --no-pager | tail -n +$((BASE+1)))
echo "$NEW" | grep -E "COLD-POOL. calls=" | tail -6
echo "-- errors --"
echo "$NEW" | grep -iE "MISMATCH|checksum|CUDA error|device-side assert|OutOfMemory|Traceback|stage_fail" | head -10
echo "-- stats tail --"
echo "$NEW" | grep -E "COLD-POOL" | tail -4
echo "== host mem =="
grep -E "^(Shmem|MemAvailable):" /proc/meminfo | awk '{printf "%-12s %7.1f GB\n", substr($1,1,11), $2/1048576}'
free -h | tail -1
echo "== post-soak greedy probe =="
/opt/sglang-env/bin/python3 /opt/sglang-test/probe_cp.py 2>&1 | head -6
echo SOAK-WATCH-DONE
