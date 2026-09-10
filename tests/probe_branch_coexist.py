import json, time, urllib.request

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {})

def ctx(m, n):
    line_tpl = "分支{m}-{i}：判别实验文本，稳定内容用于radix分支共存测试。"
    return "".join(line_tpl.format(m=m, i=i) for i in range(n))

N = 12000  # ~349K tokens each
A = ctx("AA", N)
B = ctx("BB", N)
t, m = gen(A + "问A。", 8); print(f"A1 cold={t:.1f}s")
t, m = gen(B + "问B。", 8); print(f"B1 cold={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(A + "问A。", 8); print(f"A2 reask={t:.1f}s cached={m.get('cached_tokens')}  <-- 2-branch coexist")
t, m = gen(B + "问B。", 8); print(f"B2 reask={t:.1f}s cached={m.get('cached_tokens')}")
C = ctx("CC", N)
t, m = gen(C + "问C。", 8); print(f"C1 cold={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(A + "问A。", 8); print(f"A3 reask={t:.1f}s cached={m.get('cached_tokens')}  <-- 3-branch, A oldest")
