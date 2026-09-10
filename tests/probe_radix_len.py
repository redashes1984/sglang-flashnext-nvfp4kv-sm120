import json, time, urllib.request

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        return time.time() - t0, r.get("meta_info", {})
    except urllib.error.HTTPError as e:
        return time.time() - t0, {"error": e.code, "detail": e.read().decode()[:120]}

line = "控制实验段落Z-{i}：本段用于验证radix树在超长序列下的插入与命中行为，保持稳定文本。"
for n in (9000, 12000):   # ~260K, ~345K tokens
    text = "".join(line.format(i=i) for i in range(n))
    t1, m1 = gen(text + "答：好", 8)
    t2, m2 = gen(text + "答：好", 8)
    pt = m1.get("prompt_tokens")
    print(f"{pt} tok: cold={t1:.1f}s cached1={m1.get('cached_tokens')} err1={m1.get('error')}")
    print(f"    re-ask={t2:.1f}s cached2={m2.get('cached_tokens')} err2={m2.get('error')} {m2.get('detail','')}")
