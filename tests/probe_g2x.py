"""G2 only, scaled: two ~210K ctxs coexist, re-ask both. flush to stdout."""
import json, time, urllib.request, sys

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return time.time() - t0, r.get("meta_info", {})

def log(*a):
    print(*a); sys.stdout.flush()

def ctx(m, n):
    line = "独立分支{m}第{i}段：本段专属内容互不重叠，用于radix多分支共存判别。"
    return "".join(line.format(m=m, i=i) for i in range(n))

N = 9000  # ~210K tokens
A = ctx("甲", N); B = ctx("乙", N); C = ctx("丙", N)
t, m = gen(A + "答：好"); log(f"A cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(B + "答：好"); log(f"B cold={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(A + "答：好"); log(f"A reask={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(B + "答：好"); log(f"B reask={t:.1f}s cached={m.get('cached_tokens')}")

# third branch pressure, then re-ask all (LRU depth at mamba 64: 3 branches)
t, m = gen(C + "答：好"); log(f"C cold={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(A + "答：好"); log(f"A reask2={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(B + "答：好"); log(f"B reask2={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(C + "答：好"); log(f"C reask={t:.1f}s cached={m.get('cached_tokens')}")
log("G2X-DONE")
