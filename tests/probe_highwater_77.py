"""High-water replica of bench A on nvfp4kv (pool 2.36M, 77% fill).

bench A used 文档段落 template, ~336K ctxs, 77% pool fill -> cached=0 twice.
All hit experiments were at <=60% fill. This replicates EXACT template,
re-ask text and fill ratio (3 x ~605K on 2.36M = 77%) on the final-state
service (pinned, mamba64). Hit -> water level is not the culprit.
"""
import json, sys, time, urllib.request

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {})

def log(*a):
    print(*a, flush=True)

def make_ctx(m, n):   # EXACT bench template
    line_tpl = "文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。"
    return "".join(line_tpl.format(m=m, i=i) for i in range(n))

N = 22000  # ~605K tokens per ctx (bench A's line ~27.6 tok/line)
cA, cB, cC = make_ctx("AA", N), make_ctx("BB", N), make_ctx("CC", N)
t, m = gen(cA + "\n锚点A的编号是什么？只答编号。", 16); log(f"A1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(cB + "\n锚点B的编号是什么？只答编号。", 16); log(f"B1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(cC + "\n锚点C的编号是什么？只答编号。", 16); log(f"C1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(cA + "\n锚点A的编号是什么？只答编号。", 16); log(f"A2 reask={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(cB + "\n锚点B的编号是什么？只答编号。", 16); log(f"B2 reask={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(cC + "\n锚点C的编号是什么？只答编号。", 16); log(f"C2 reask={t:.1f}s cached={m.get('cached_tokens')}")
log("HIGHWATER-DONE")
