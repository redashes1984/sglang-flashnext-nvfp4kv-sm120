#!/bin/bash
# steps3 replay bench: C1 third pass + C12 aggregate + error sweep
# A2/R4 remediation:
#   - unique marker embedded in every prompt tail; journal rows attributed by
#     grep-marker slice instead of fixed --since windows (fallback: window from
#     this run's start, never a stale fixed window)
#   - headline throughput = journal "gen throughput (token/s)" aggregate; awk
#     wall-clock figure demoted to reference only
#   - curl failures counted explicitly and reflected in exit code
#   - big bodies via --data @file
# Hermetic knobs for self-test: HERMES_JOURNAL_FILE, A2_MARK. No systemctl use.
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

# marker-correlated slice: rows from the line carrying MARK onward.
# If the marker never appears (journal doesn't echo prompts), fall back to the
# full since-run-start dump — never a stale fixed window.
jslice() {
  local all="$(jdump)"
  printf '%s\n' "$all" | grep -q "$MARK" && printf '%s\n' "$all" | awk -v mk="$MARK" 'index($0,mk){f=1} f' || printf '%s\n' "$all"
}

FAILS=0

# ---- phase C1: single third-pass long request -----------------------
BODY1="$TMPD/c1.json"
printf '{"text": "Describe in depth how LSM-trees work in storage engines: compaction strategies, bloom filters, read path, write path. [%s]", "sampling_params": {"temperature": 0, "max_new_tokens": 400}}' "$MARK" > "$BODY1"
t0=$(date +%s.%N)
curl -s -m 180 http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  --data @"$BODY1" -o /dev/null || echo x >> "$FF"
t1=$(date +%s.%N)

J1="$(jslice)"
GT="$(printf '%s\n' "$J1" | grep "Decode batch" | grep "#running-req: 1," \
  | grep -oE "gen throughput \(token/s\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -g | tail -1)"
echo "C1 headline (journal gen throughput): ${GT:-n/a} tok/s"
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "C1 wall=%.1fs => nominal-400tok %.1f tok/s (reference only)\n", b-a, 400/(b-a)}'

# ---- phase C12: 12 concurrent chat requests -------------------------
CTX=$(python3 -c "print('介绍Transformer的注意力机制、KV Cache管理、MoE专家路由、投机解码与量化推理的工程实践。' * 12)")
BODY12="$TMPD/c12.json"
printf '{"model":"Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"%s 请用中文详细回答。 [%s]"}],"max_tokens":300,"temperature":0.7}' "$CTX" "$MARK" > "$BODY12"
t0=$(date +%s.%N)
for i in $(seq 1 12); do
  { curl -s -m 300 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
      --data @"$BODY12" -o /dev/null || echo x >> "$FF"; } &
done
wait
t1=$(date +%s.%N)

J2="$(jslice)"
GT12="$(printf '%s\n' "$J2" | grep "Decode batch" | grep "#running-req: 12," \
  | grep -oE "gen throughput \(token/s\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -g | tail -1)"
echo "C12 headline (journal gen throughput): ${GT12:-n/a} tok/s"
awk -v a="$t0" -v b="$t1" 'BEGIN{printf "C12 wall=%.1fs => nominal-3600tok %.1f tok/s (reference only)\n", b-a, 3600/(b-a)}'

# ---- accept-len readout + error sweep within marker slice -----------
printf '%s\n' "$J2" | grep "Decode batch" \
  | grep "#running-req: 12" | tail -3 \
  | grep -oE "accept len: [0-9.]+, accept rate: [0-9.]+, cuda graph: True, gen throughput \(token/s\): [0-9.]+"
echo "--- error lines (expect 0):"
printf '%s\n' "$J2" | grep -icE "out of memory|MISMATCH|CUDA error: " || true

NC="$(wc -l < "$FF" | tr -d ' ')"
echo "curl failures this run: $NC"
[ "$NC" != "0" ] && FAILS=$((FAILS+1))
echo BENCH3-DONE
exit $FAILS
