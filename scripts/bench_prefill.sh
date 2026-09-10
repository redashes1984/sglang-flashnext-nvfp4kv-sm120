#!/bin/bash
# prefill bench v3: build request file entirely inside python (no argv passing)
python3 - <<'PY'
import json, uuid
ctx = '介绍Transformer的注意力机制、KV Cache管理、MoE专家路由、投机解码与量化推理的工程实践细节。' * 2600
text = uuid.uuid4().hex.upper() + ' ' + ctx
with open('/tmp/pf_req.json', 'w') as f:
    json.dump({"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 1}}, f)
print("prompt chars:", len(text))
PY

meas () {
  curl -s -m 600 http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d @/tmp/pf_req.json \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
m = d['meta_info']
pt = m['prompt_tokens']; e2e = m['e2e_latency']
cached = m.get('cached_tokens', 0)
print(f'$1: prompt_tokens={pt} cached={cached} e2e={e2e:.2f}s => {pt/e2e:,.0f} tok/s')"
}

echo "=== COLD (unique uuid prefix)"
meas COLD
echo "=== WARM (same request, radix reuse)"
meas WARM
echo "=== journal chunked prefill (this test window):"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-4min" | grep "Prefill batch" \
  | grep -oE "#new-token: [0-9]+, #cached-token: [0-9]+.*input throughput \(token/s\): [0-9.]+" | tail -22
rm -f /tmp/pf_req.json
echo PREFILL-V3-DONE
