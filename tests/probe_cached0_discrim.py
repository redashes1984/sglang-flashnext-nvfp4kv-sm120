"""cached0 discriminator v2 (lean): size-only vs branch-combination.

  D1  ONE 495K bench-A ctx, reask  -> miss = per-ctx-size effect proven,
                                      template & branches fully bypassed
  D2  TWO 495K branch-template ctxs -> if D1 hits, is 2x495K still fine?
"""
import json, sys, time, urllib.request

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {})

def log(*a): print(*a, flush=True)

def doc(m, n):
    return "".join(f"文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。" for i in range(n))
def brn(m, n):
    return "".join(f"独立分支{m}第{i}段：本段专属内容互不重叠，用于radix多分支共存判别。" for i in range(n))

log("== D1 single 495K (bench-A template) ==")
A = doc("Z1", 22000)
t, m = gen(A + "\n锚点A的编号是什么？只答编号。", 16); log(f"D1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(A + "\n锚点A的编号是什么？只答编号。", 16); log(f"D1 reask={t:.1f}s cached={m.get('cached_tokens')}")

log("== D2 two 495K branch template ==")
B = brn("甲2", 20000); C = brn("乙2", 20000)
t, m = gen(B + "答：好", 16); log(f"D2 B cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(C + "答：好", 16); log(f"D2 C cold={t:.1f}s")
t, m = gen(B + "答：好", 16); log(f"D2 B reask={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(C + "答：好", 16); log(f"D2 C reask={t:.1f}s cached={m.get('cached_tokens')}")
log("DISCRIM-DONE")
