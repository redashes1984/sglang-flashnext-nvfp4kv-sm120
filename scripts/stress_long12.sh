#!/bin/bash
# long-context stress: 12 concurrent x ~66K-token prompts, 64 tok decode each.
# reproduces the historical 6x74K OOM guard at 2x the concurrency.
cd /tmp
python3 - <<'PY'
import random
words = ['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel','india','juliet','kilo','lima','mike','november','oscar','papa','quebec','romeo','sierra','tango']
for i in range(12):
    random.seed(1000 + i)
    txt = ' '.join(random.choice(words) for _ in range(66000))
    open(f'/tmp/p{i}.txt', 'w').write(txt)
print("FILES-WRITTEN")
PY
# tokenize count via server
N=$(curl -s -m 60 http://127.0.0.1:8000/tokenize -H 'Content-Type: application/json' -d @<(python3 -c "import json;print(json.dumps({'prompt': open('/tmp/p0.txt').read()}))") | python3 -c "import json,sys;d=json.load(sys.stdin);print(len(d.get('tokens',[])))")
echo "tokens-per-prompt: $N"

PEAKF=/tmp/gpu_peak_long.txt
: > $PEAKF
setsid bash -c "for i in \$(seq 1 480); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $PEAKF; sleep 1; done" >/dev/null 2>&1 &
MPID=$!

for i in $(seq 0 11); do
  python3 -c "import json;t=open('/tmp/p$i.txt').read();open('/tmp/body$i.json','w').write(json.dumps({'text':t,'sampling_params':{'temperature':0,'max_new_tokens':64}}))" &
done
wait
for i in $(seq 0 11); do
  curl -s -m 900 http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d @/tmp/body$i.json -o /dev/null &
done
wait
sleep 2
kill -- -$MPID 2>/dev/null; kill $MPID 2>/dev/null

echo "=== peak GPU used (MiB / 97887):"
sort -n $PEAKF | tail -1
echo "=== samples:"
wc -l < $PEAKF
echo "=== OOM / retract / errors in last 9min:"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-9min" | grep -icE "out of memory|retract|MISMATCH|CUDA error"
echo "=== longest-context decode seen:"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-9min" | grep "Decode batch" | grep -oE "#full token: [0-9]+, full token usage: [0-9.]+" | sort -t: -k2 -n | tail -3
echo "=== prefill throughput lines:"
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager --since "-9min" | grep -E "Prefill batch" | tail -2
echo LONGSTRESS-DONE
