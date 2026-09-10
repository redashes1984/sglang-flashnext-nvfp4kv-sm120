"""nvfp4kv mamba anchor-slack gate (mamba 48 -> 64).

fp8kv lesson: long-context radix anchors need mamba slots OUTSIDE the
conc*4 running budget; 48 active at conc12 leaves ZERO slack. This probe
reproduces the cached=0 failure signature at nvfp4kv's scale:

G1: single 500K ctx cold -> re-ask must hit (~99%+ cached, <2s)
G2: two 500K ctxs coexist, re-ask both hit
G3: conc12 short burst still healthy (no regression from mamba64)
G4: mamba usage readout during G2 (expect slack visible in journal)

Run AFTER switch to mamba 64, with cold pool + graphs ON.
"""
import json, subprocess, sys, time, urllib.request

URL = "http://127.0.0.1:8000/generate"

def gen(text, mx=8, timeout=900):
    req = urllib.request.Request(URL, data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return time.time() - t0, r.get("meta_info", {})

def ctx(m, n):
    line = "锚定段落{m}-{i}：nvfp4kv长上下文槽位余量判别实验，文本稳定用于radix复用验证。"
    return "".join(line.format(m=m, i=i) for i in range(n))

fails = 0
N = 20000  # ~480K tokens per ctx (nvfp4kv ctx 1M, pool 2.36M)

print("G1 single ~480K cold/re-ask")
cA = ctx("A", N)
t1, m1 = gen(cA + "答：好"); print(f"  cold={t1:.1f}s pt={m1.get('prompt_tokens')}")
t2, m2 = gen(cA + "答：好"); c = m2.get("cached_tokens") or 0
print(f"  reask={t2:.1f}s cached={c}")
if t2 > 5 or c < 0.95 * (m2.get("prompt_tokens") or 1): fails += 1; print("  G1 FAIL")

print("G2 two ~480K coexist")
cB = ctx("B", N)
t, _ = gen(cB + "答：好"); print(f"  B cold={t:.1f}s")
t, m = gen(cA + "答：好"); print(f"  A reask={t:.1f}s cached={m.get('cached_tokens')}")
if t > 5: fails += 1; print("  G2-A FAIL (anchor evicted by B)")
t, m = gen(cB + "答：好"); print(f"  B reask={t:.1f}s cached={m.get('cached_tokens')}")
if t > 5: fails += 1; print("  G2-B FAIL")

print("G3 conc12 short burst (no regression)")
P = '{"text": "用中文写一段关于分布式系统共识的说明。", "sampling_params": {"temperature": 0.7, "max_new_tokens": 200}}'
t0 = time.time()
procs = [subprocess.Popen(["curl", "-s", "-m", "180", URL, "-H", "Content-Type: application/json", "-d", P],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(12)]
for p in procs: p.wait()
dt = time.time() - t0
print(f"  12x200tok wall={dt:.1f}s => aggregate {2400/dt:.0f} tok/s")
if dt > 60: fails += 1; print("  G3 FAIL (suspiciously slow)")

print("ANCHOR-GATE:", "PASS" if fails == 0 else f"FAIL x{fails}")
sys.exit(1 if fails else 0)
