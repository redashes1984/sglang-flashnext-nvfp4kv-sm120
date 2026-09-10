#!/bin/bash
# conc12 stress: 12 concurrent chat reqs, ~1K-token prompts, 300 tokens each.
# sample GPU peak every 1s for 90s.
PEAKF=/tmp/gpu_peak_conc12.txt
: > $PEAKF
( for i in $(seq 1 90); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $PEAKF; sleep 1; done ) &
MPID=$!

CTX=$(python3 -c "print('介绍Transformer的注意力机制、KV Cache管理、MoE专家路由、投机解码与量化推理的工程实践。' * 12)")
P='{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"%s 请用中文详细回答，覆盖机制、权衡与实践。"}],"max_tokens":300,"temperature":0.7}'
for i in $(seq 1 12); do
  curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "$(printf "$P" "$CTX")" -o /dev/null &
done
wait
sleep 2
kill $MPID 2>/dev/null

echo "=== peak GPU used (MiB / 97887):"
sort -n $PEAKF | tail -1
echo "=== token usage / run level seen:"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-3min" | grep "Decode batch" \
  | grep -oE "#running-req: [0-9]+|full token usage: [0-9.]+" | sort | uniq -c | tail -10
echo "=== error lines (expect 0):"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-3min" | grep -icE "out of memory|MISMATCH|CUDA error"
echo "=== last decode lines:"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-3min" | grep "Decode batch" | tail -2
echo STRESS-DONE
