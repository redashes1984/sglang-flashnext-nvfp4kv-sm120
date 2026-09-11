import json, time, urllib.request
def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return time.time()-t0, r.get("meta_info", {}), r
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

BIG = doc_with_needles("主", 4500)     # ~102K pt, cheap re-check of the same question shape
t, m, r = gen(BIG + "\n问：三把审计密钥分别是什么？只答三个密钥词。", 64)
log(f"small-NIAH cold={t:.1f}s pt={m.get('prompt_tokens')}")
log("RAW TEXT:", repr(r.get("text"))[:300])
log("COMPLETION TOKENS:", m.get("completion_tokens"), "finish:", m.get("finish_reason"))
t, m, r = gen(BIG + "\n问：三把审计密钥分别是什么？只答三个密钥词。", 64)
log(f"small-NIAH reask={t:.1f}s cached={m.get('cached_tokens')}")
log("RAW TEXT2:", repr(r.get("text"))[:300])
