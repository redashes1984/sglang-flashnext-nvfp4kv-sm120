"""Discriminate the cached=0 anomaly: template vs branch-count.

P1 文档段落-template alone; P2 pair; P3 triple (bench replica);
P4 分支-template alone (control). Same boot, ~100K scale.
"""
import json, time, urllib.request

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {})

def docctx(m, n):   # bench template ("文档段落")
    t = "文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。"
    return "".join(t.format(m=m, i=i) for i in range(n))

def brctx(m, n):    # branch template
    t = "分支{m}-{i}：判别实验文本，稳定内容用于radix分支共存测试。"
    return "".join(t.format(m=m, i=i) for i in range(n))

N = 3500  # ~100K tokens per ctx

def one(label, maker):
    c = maker(f"{label}", N)
    t1, m1 = gen(c + "答：好")
    t2, m2 = gen(c + "答：好")
    print(f"{label}: pt={m1.get('prompt_tokens')} cold={t1:.1f}s | reask={t2:.1f}s cached={m2.get('cached_tokens')}")

print("P1/P4 single-context, template A/B")
one("DOC", docctx)
one("BRN", brctx)

print("P2 doc-template pair")
cA = docctx("D2A", N); cB = docctx("D2B", N)
t, m = gen(cA + "答：好"); print(f"D2A cold={t:.1f}")
t, m = gen(cB + "答：好"); print(f"D2B cold={t:.1f}")
t, m = gen(cA + "答：好"); print(f"D2A reask={t:.1f}s cached={m.get('cached_tokens')}")

print("P3 doc-template triple (bench replica)")
cC = docctx("D3C", N)
t, m = gen(cC + "答：好"); print(f"D3C cold={t:.1f}")
t, m = gen(cA + "答：好"); print(f"D3A reask={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(cB + "答：好"); print(f"D3B reask={t:.1f}s cached={m.get('cached_tokens')}")
