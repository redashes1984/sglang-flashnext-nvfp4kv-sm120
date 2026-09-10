import json, time, urllib.request

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {})

def brn(m, n):
    return "".join(f"独立分支{m}第{i}段：本段专属内容互不重叠，用于radix多分支共存判别。" for i in range(n))

B = brn("甲2", 20000); C = brn("乙2", 20000)
t, m = gen(B + "答：好", 16); print(f"B cold={t:.1f}s pt={m.get('prompt_tokens')}", flush=True)
t, m = gen(C + "答：好", 16); print(f"C cold={t:.1f}s", flush=True)
t, m = gen(B + "答：好", 16); print(f"B reask={t:.1f}s cached={m.get('cached_tokens')}", flush=True)
t, m = gen(C + "答：好", 16); print(f"C reask={t:.1f}s cached={m.get('cached_tokens')}", flush=True)
print("D2-I8OFF-DONE", flush=True)
