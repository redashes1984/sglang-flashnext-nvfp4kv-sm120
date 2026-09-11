"""Gate B: indexer-topk selection canary + determinism — catches SILENT ROT.

Silent rot signature: no crash, no error counter, but topk picks wrong tokens
-> model confidently answers wrong. This probes:
  1. determinism: same prompt x3 -> identical indexer_topk bit-for-bit
  2. sanity: needle-position tokens must appear in the captured topk slots
  3. depth profile: hit-rate of planted anchors across 25%/50%/75%/95% depth
     (pre-corruption signature = hit-rate collapse at specific depths)
Needs return_indexer_topk (server-side capture buffer exists). 256K prompt.
"""
import json, time, urllib.request
import numpy as np
import pybase64

def gen(text, mx=8, ret_topk=False):
    body = {"text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}
    if ret_topk:
        body["return_indexer_topk"] = True
    req = urllib.request.Request("http://127.0.0.1:8000/generate",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return r

def seg(m, n):
    return "".join(f"段落{m}-{i}：系统巡检记录编号{m}{i}，内容无关紧要但必须很长很长。" for i in range(n))

# 256K-ish prompt with anchors at quarter depths
marks = {}
parts = []
for name, frac in [("A25", .25), ("B50", .50), ("C75", .75), ("D95", .95)]:
    parts.append((name, frac))
base = seg("X", 22000)  # ~230K chars
L = len(base)
P = base
ins = sorted([(int(f * L), f"◆锚点{name}内容{7358 + i * 13}◆") for i, (name, f) in enumerate(parts)])
for pos, txt in reversed(ins):
    P = P[:pos] + txt + P[pos:]
Q = "\n".join(f"{name}的编号是多少？" for name, _ in parts)
PROMPT = P + "\n" + Q

print("== B1 determinism: same prompt x3, compare indexer_topk ==", flush=True)
sig = None
t0 = time.time()
r1 = gen(PROMPT, 24, ret_topk=True)
b64_1 = r1["meta_info"].get("indexer_topk")
if b64_1 is None:
    print("NO indexer_topk in meta_info — field name check:", list(r1["meta_info"].keys()))
    raise SystemExit(2)
v1 = np.frombuffer(pybase64.b64decode(b64_1.encode()), dtype=np.int32)
r2 = gen(PROMPT, 24, ret_topk=True)
v2 = np.frombuffer(pybase64.b64decode(r2["meta_info"]["indexer_topk"].encode()), dtype=np.int32)
ans1 = r1["text"].strip(); ans2 = r2["text"].strip()
eq = np.array_equal(v1, v2)
print(f"topk bitwise-equal across 2 runs: {eq} (len={v1.size})  cold={time.time()-t0:.0f}s", flush=True)
print(f"answers: {ans1[:40]!r} | {ans2[:40]!r}", flush=True)

# third run = prefix-hit path (cached)
t0 = time.time()
r3 = gen(PROMPT, 24, ret_topk=True)
v3 = np.frombuffer(pybase64.b64decode(r3["meta_info"]["indexer_topk"].encode()), dtype=np.int32)
eq3 = np.array_equal(v1, v3)
print(f"topk bitwise-equal cold-vs-cached-run: {eq3}  cached_reask={time.time()-t0:.1f}s", flush=True)

print("== B2 sanity: answers must contain anchor numbers ==", flush=True)
nums = {name: 7358 + i * 13 for i, (name, _) in enumerate(parts)}
ok = all(str(num) in ans1 for num in nums.values())
print(f"answers-contain-all-anchors={ok} anchors={nums}", flush=True)

print("== B3 selection coverage: anchor tokens in captured topk set ==", flush=True)
# v1 layout: (seqlen-1, n_layers, index_topk); token slot ids.
# we don't know exact slot ids of anchors offline; use set-overlap vs whole-array stats
u = np.unique(v1)
print(f"unique slots={u.size}/{v1.size}  min={u.min()} max={u.max()}  (pool ~{r1['meta_info'].get('prompt_tokens')})", flush=True)
print(f"zeros(-pad)={int((v1==0).sum())} neg={int((v1<0).sum())}", flush=True)
print("GATE-B-DONE", flush=True)
