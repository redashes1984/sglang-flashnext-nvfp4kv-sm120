"""fp8kv 1M bring-up gate: target shape 1x1M + 2x256K co-resident + NIAH@1M.

  T1  main session ~1M doc, 3 needles at 10%/50%/90% depth -> cold prefill,
      needle QA (YaRN x4 correctness under fp8 KV = the unverified item)
  T2  two 256K side chains, interleaved while main decodes? no — sequential
      cold fill, then re-ask all three (radix co-existence under cap16)
  T3  side-lobe small request latency while pools are full (Hindsight proxy:
      ~3K json-out + ~30K prompt), full token usage readout
Gate: needles exact; re-ask cached>90% pt; small-req p50 < ~2s on idle card;
      zero OOM/retract in journal.
"""
import json, time, urllib.request

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text, "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return time.time() - t0, r.get("meta_info", {}), r.get("text", "")

def log(*a): print(*a, flush=True)

def doc_with_needles(m, n):
    """~27.4 tok/seg; needles at 10%/50%/90%."""
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

log(f"== T1 main 1M cold fill (start {time.strftime('%H:%M:%S')}) ==")
BIG = doc_with_needles("主", 37500)          # ~1.03M pt incl needles
t, m, _ = gen(BIG + "\n问：三把审计密钥分别是什么？只答三个密钥词。", 64)
log(f"T1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m, txt = gen(BIG + "\n问：三把审计密钥分别是什么？只答三个密钥词。", 64)
log(f"T1 reask={t:.1f}s cached={m.get('cached_tokens')} answer={txt.strip()[:80]}")
ok = all(k in txt for k in ("紫罗兰-主1", "琥珀-主2", "青金-主3"))
log(f"T1 NIAH@1M: {'PASS' if ok else 'FAIL'}")

log("== T2 two 256K side chains ==")
S1 = plain("侧一", 9400); S2 = plain("侧二", 9400)   # ~257K pt each
t, m = gen(S1 + "\n答：好", 16)[:2]; log(f"T2 S1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(S2 + "\n答：好", 16)[:2]; log(f"T2 S2 cold={t:.1f}s")
t, m, _ = gen(BIG + "\n问：三把审计密钥分别是什么？只答三个密钥词。", 64)
log(f"T2 main reask-3={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(S1 + "\n答：好", 16)[:2]; log(f"T2 S1 reask={t:.1f}s cached={m.get('cached_tokens')}")
t, m = gen(S2 + "\n答：好", 16)[:2]; log(f"T2 S2 reask={t:.1f}s cached={m.get('cached_tokens')}")

log("== T3 side-lobe small requests on full pools ==")
import threading
lat = []
def small(i):
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps({"model": "Qwen3.8-Flash-Next-NVFP4",
                         "messages": [{"role": "user", "content": f"用一句话说明备份策略要点{i}。"}],
                         "max_tokens": 48}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    lat.append((time.time() - t0, r.get("usage", {})))
ts = [threading.Thread(target=small, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
for i, (tt, u) in enumerate(sorted(lat)):
    log(f"T3 small{i} {tt:.2f}s pt={u.get('prompt_tokens')}")
log("BRINGUP-DONE")
