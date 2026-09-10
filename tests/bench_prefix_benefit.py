"""Prefix-cache benefit measurement at the 1.31M pool.

Scenario A (the headline claim): TWO distinct 500K-token contexts co-resident
+ a third cold 500K arrives. Under old 552960 pool -> guaranteed eviction of
both anchors. Under 819200 -> exactly two barely fit, third evicts one.
Under 1310720 -> all three fit, zero retract.

Scenario B: hit-rate economics at realistic agent sizes (30K shared prefix).
"""
import json, time, urllib.request

URL = "http://127.0.0.1:8000/generate"

def gen(text, mx=8, temp=0.0):
    req = urllib.request.Request(URL, data=json.dumps({
        "text": text, "sampling_params": {"temperature": temp, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {})

def make_ctx(seed_marker, target_tokens):
    # ~1.6 tokens per CJK char-pair line; build deterministic filler w/ anchor
    line_tpl = "文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。"
    lines, approx = [], 0
    i = 0
    while approx < target_tokens:
        ln = line_tpl.format(m=seed_marker, i=i)
        lines.append(ln)
        approx += len(ln) * 0.9   # rough CJK token ratio
        i += 1
    return "".join(lines)

print("=== A. three-context coexistence (old-pool killer) ===")
cA = make_ctx("AA", 480_000)
cB = make_ctx("BB", 480_000)
cC = make_ctx("CC", 480_000)
t, m = gen(cA + "\n锚点A的编号是什么？只答编号。", 16); print(f"A1 cold: {t:.1f}s in={m.get('prompt_tokens')}")
t, m = gen(cB + "\n锚点B的编号是什么？只答编号。", 16); print(f"B1 cold: {t:.1f}s in={m.get('prompt_tokens')}")
t, m = gen(cC + "\n锚点C的编号是什么？只答编号。", 16); print(f"C1 cold: {t:.1f}s in={m.get('prompt_tokens')}")
# now re-ask A and B: both should still be cached under 1.31M
t, m = gen(cA + "\n锚点A的编号是什么？只答编号。", 16)
print(f"A2 re-ask: {t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(cB + "\n锚点B的编号是什么？只答编号。", 16)
print(f"B2 re-ask: {t:.1f}s cached={m.get('cached_tokens')}")

print("=== B. 30K shared-prefix agent-turn economics ===")
prefix = make_ctx("XX", 30_000)
t, m = gen(prefix + " 第一问：系统状态？", 24); print(f"turn1 cold: {t:.2f}s in={m.get('prompt_tokens')}")
for k in range(2, 6):
    t, m = gen(prefix + f" 第{k}问：继续分析。", 24)
    print(f"turn{k} warm: {t:.2f}s cached={m.get('cached_tokens')}")
