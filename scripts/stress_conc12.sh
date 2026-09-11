#!/bin/bash
# conc12 stress: 12 concurrent chat reqs, ~1K-token prompts, 300 tokens each.
# sample GPU peak every 1s for 90s.
# A2/R4 remediation: marker-correlated journal rows replace fixed "--since -3min"
# windows (fallback = window from run start); curl failures counted explicitly;
# big bodies via --data @file. Hermetic knobs: HERMES_JOURNAL_FILE, A2_MARK.
set -u
U=sglang-dealignai-qwen4exp-nvfp4kv
MARK="${A2_MARK:-a2mk-$(date +%s)-$$}"
START=$(date +%s)
TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
FF="$TMPD/cfail"; : > "$FF"

jdump() {
  if [ -n "${HERMES_JOURNAL_FILE:-}" ]; then cat "$HERMES_JOURNAL_FILE"
  else journalctl -u "$U" --no-pager --since "@$START"
  fi
}

PEAKF=/tmp/gpu_peak_conc12.txt
: > "$PEAKF"
if command -v nvidia-smi >/dev/null 2>&1; then
  ( for i in $(seq 1 90); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$PEAKF"; sleep 1; done ) &
  MPID=$!
else
  echo "(nvidia-smi unavailable; GPU peak sampling skipped)" >&2
fi

CTX=$(python3 -c "print('介绍Transformer的注意力机制、KV Cache管理、MoE专家路由、投机解码与量化推理的工程实践。' * 12)")
BODY="$TMPD/c12.json"
printf '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"%s 请用中文详细回答，覆盖机制、权衡与实践。 [%s]"}],"max_tokens":300,"temperature":0.7}' "$CTX" "$MARK" > "$BODY"
for i in $(seq 1 12); do
  { curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      --data @"$BODY" -o /dev/null || echo x >> "$FF"; } &
done
wait
sleep 2
[ -n "${MPID:-}" ] && kill "$MPID" 2>/dev/null || true

echo "=== peak GPU used (MiB / 97887):"
sort -n "$PEAKF" | tail -1

J="$(jdump)"
if printf '%s\n' "$J" | grep -q "$MARK"; then
  J="$(printf '%s\n' "$J" | awk -v mk="$MARK" 'index($0,mk){f=1} f')"
fi

echo "=== token usage / run level seen:"
printf '%s\n' "$J" | grep "Decode batch" \
  | grep -oE "#running-req: [0-9]+|full token usage: [0-9.]+" | sort | uniq -c | tail -10
echo "=== error lines (expect 0):"
printf '%s\n' "$J" | grep -icE "out of memory|MISMATCH|CUDA error" || true
echo "=== last decode lines:"
printf '%s\n' "$J" | grep "Decode batch" | tail -2

NC="$(wc -l < "$FF" | tr -d ' ')"
echo "curl failures this run: $NC"
echo STRESS-DONE
[ "$NC" != "0" ] && exit 1
exit 0
