"""fp8kv 1M target-shape validation (corrected: chat endpoint + enable_thinking=false).
Target: ONE ~1M main session + TWO 256K side chains co-resident, NIAH@1M needles."""
import json, time, threading, urllib.request
def chat(msgs, mx=64):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps({
        "model": "Qwen3.8-Flash-Next-NVFP4", "messages": msgs, "max_tokens": mx,
        "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    ch = r.get("choices", [{}])[0]
    return time.time()-t0, r.get("usage", {}), (ch.get("message", {}).get("content") or ""), ch.get("finish_reason")
def log(*a): print(*a, flush=True)

def doc_with_needles(m, n):
    pos = {int(n*0.10): f"【机密编号X{m}-A：本次审计的密钥是 紫罗兰-{m}1】",
           int(n*0.50): f"【机密编号X{m}-B：本次审计的密钥是 琥珀-{m}2】",
           int(n*0.90): f"【机密编号X{m}-C：本次审计的密钥是 青金-{m}3】"}
    out = []
    for i in range(n):
        if i in pos: out.append(pos[i])
        out.append(f"文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。")
    return "".join(out)
def plain(m, n):
    return "".join(f"独立分支{m}第{i}段：本段专属内容互不重叠，用于radix多分支共存判别。" for i in range(n))

MAIN = doc_with_needles("巨", 44000)          # ~22.7 tok/seg -> ~1.0M pt
Q = "问：文档中的三把审计密钥分别是什么？只答三个密钥词。"
log(f"== M main ~1M cold (start {time.strftime('%H:%M:%S')}) ==")
t, u, c, f = chat([{"role": "user", "content": MAIN + "\n" + Q}], 48)
log(f"M cold={t:.1f}s pt={u.get('prompt_tokens')}")
t, u, c, f = chat([{"role": "user", "content": MAIN + "\n" + Q}], 48)
log(f"M reask={t:.1f}s cached_pt={u.get('prompt_tokens')} finish={f}")
log(f"M ANSWER: {c[:100]!r}")
log(f"MAIN-NIAH-1M: {'PASS' if all(k in c for k in ('紫罗兰-巨1','琥珀-巨2','青金-巨3')) else 'FAIL'}")

log("== S side 256K x2 ==")
S1 = plain("侧甲", 9400); S2 = plain("侧乙", 9400)
for nm, s in (("S1", S1), ("S2", S2)):
    t, u, c, f = chat([{"role": "user", "content": s + "\n答：好"}], 16)
    log(f"{nm} cold={t:.1f}s pt={u.get('prompt_tokens')}")
t, u, c, f = chat([{"role": "user", "content": S1 + "\n答：好"}], 16)
log(f"S1 reask={t:.1f}s")
t, u, c, f = chat([{"role": "user", "content": S2 + "\n答：好"}], 16)
log(f"S2 reask={t:.1f}s")

log("== small-lobe x4 while 1.56M resident ==")
lat = []
def small(i):
    t0 = time.time()
    t, u, c, f = chat([{"role": "user", "content": f"用一句话说明备份策略要点{i}。"}], 48)
    lat.append(time.time()-t0)
ts = [threading.Thread(target=small, args=(i,)) for i in range(4)]
[x.start() for x in ts]; [x.join() for x in ts]
log("small lats: " + " ".join(f"{x:.2f}s" for x in sorted(lat)))
log("SHAPE-DONE")
