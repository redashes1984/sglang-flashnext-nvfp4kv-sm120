#!/bin/bash
# steps3 replay bench: C1 third pass + C12 aggregate + error sweep
t0=$(date +%s.%N)
curl -s -m 180 http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"text": "Describe in depth how LSM-trees work in storage engines: compaction strategies, bloom filters, read path, write path.", "sampling_params": {"temperature": 0, "max_new_tokens": 400}}' -o /dev/null
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "C1 third: %.1fs => %.1f tok/s\n", b-a, 400/(b-a)}'

CTX=$(python3 -c "print('介绍Transformer的注意力机制、KV Cache管理、MoE专家路由、投机解码与量化推理的工程实践。' * 12)")
P='{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"%s 请用中文详细回答。"}],"max_tokens":300,"temperature":0.7}'
t0=$(date +%s.%N)
for i in $(seq 1 12); do
  curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$(printf "$P" "$CTX")" -o /dev/null &
done
wait
t1=$(date +%s.%N)
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "C12 wall=%.1fs => aggregate %.1f tok/s\n", b-a, 3600/(b-a)}'

journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-2min" | grep "Decode batch" \
  | grep "#running-req: 12" | tail -3 \
  | grep -oE "accept len: [0-9.]+, accept rate: [0-9.]+, cuda graph: True, gen throughput \(token/s\): [0-9.]+"
echo "--- error lines (expect 0):"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-12min" | grep -icE "out of memory|MISMATCH|CUDA error: "
echo BENCH3-DONE
