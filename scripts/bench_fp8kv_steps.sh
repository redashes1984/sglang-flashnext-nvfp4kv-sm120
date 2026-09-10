#!/bin/bash
# fp8kv steps A/B acceptance bench: C1 + C4 + accept stats + error sweep
# unit for journal = sglang-dealignai-qwen4exp-fp8kv (conc4)
U=sglang-dealignai-qwen4exp-fp8kv
S=$(systemctl show $U -p ExecMainStartTimestamp --value)

# warm the route (radix + draft path) twice, measure the third pass
for w in 1 2; do
  curl -s -m 180 http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
    -d '{"text": "Describe in depth how LSM-trees work in storage engines: compaction strategies, bloom filters, read path, write path.", "sampling_params": {"temperature": 0, "max_new_tokens": 400}}' -o /dev/null
done
t0=$(date +%s.%N)
curl -s -m 180 http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"text": "Describe in depth how LSM-trees work in storage engines: compaction strategies, bloom filters, read path, write path.", "sampling_params": {"temperature": 0, "max_new_tokens": 400}}' -o /dev/null
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "C1 third-pass: %.1fs => %.1f tok/s\n", b-a, 400/(b-a)}'

CTX=$(python3 -c "print('介绍Transformer的注意力机制、KV Cache管理、MoE专家路由、投机解码与量化推理的工程实践。' * 12)")
P='{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"%s 请用中文详细回答。"}],"max_tokens":300,"temperature":0.7}'
t0=$(date +%s.%N)
for i in $(seq 1 4); do
  curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$(printf "$P" "$CTX")" -o /dev/null &
done
wait
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "C4 wall=%.1fs => aggregate %.1f tok/s\n", b-a, 1200/(b-a)}'

echo "--- C1 steady-window (last decode bs1) ---"
journalctl -u $U --no-pager --since "$S" | grep "Decode batch" | grep "#running-req: 1," | tail -2 \
  | grep -oE "accept len: [0-9.]+, accept rate: [0-9.]+, cuda graph: True, gen throughput \(token/s\): [0-9.]+"
echo "--- C4 window (bs4) ---"
journalctl -u $U --no-pager --since "$S" | grep "Decode batch" | grep "#running-req: 4," | tail -2 \
  | grep -oE "accept len: [0-9.]+, accept rate: [0-9.]+, cuda graph: True, gen throughput \(token/s\): [0-9.]+"
echo "--- error lines (expect 0):"
journalctl -u $U --no-pager --since "$S" | grep -E "out of memory|MISMATCH|CUDA error: |stage_fail|retract request" | grep -vc "may lead"
echo BENCH-FP8KV-DONE
