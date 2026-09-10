#!/bin/bash
# conc12 short-burst regression: graphs bs12 + accept + zero errors
P='{"text": "用中文写一段关于分布式系统共识算法的说明，包含Paxos与Raft对比。", "sampling_params": {"temperature": 0.7, "max_new_tokens": 200}}'
S=$(systemctl show sglang-dealignai-qwen4exp-nvfp4kv -p ExecMainStartTimestamp --value)
t0=$(date +%s.%N)
for i in $(seq 1 12); do
  curl -s -m 180 http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d "$P" -o /dev/null &
done
wait
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "C12 wall=%.1fs => aggregate %.0f tok/s\n", b-a, 2400/(b-a)}'
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "$S" | grep "Decode batch" | grep "#running-req: 12," | tail -2 \
  | grep -oE "accept len: [0-9.]+, accept rate: [0-9.]+, cuda graph: True, gen throughput \(token/s\): [0-9.]+"
echo "--- errors ---"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "$S" | grep -E "MISMATCH|CUDA error: |retract request|CUDA out of memory|stage_fail" | grep -vc "may lead"
echo C12-REGRESSION-DONE
