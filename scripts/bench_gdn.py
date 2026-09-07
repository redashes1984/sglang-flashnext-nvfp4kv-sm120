#!/usr/bin/env python3
"""GDN/nvfp4kv serve bench: C1 latency + C6 aggregate throughput + accept-len hint.

Hits the OpenAI-compatible endpoint of the nvfp4kv scheme (port 8000).
Run twice — first pass is warmup, second pass is the hot number.

Usage: python3 bench_gdn.py [base_url]  (default http://127.0.0.1:8000)
"""
import json, sys, time, threading

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "Qwen3.8-Flash-Next-NVFP4"
PROMPT = ("用中文简要总结：混合线性注意力(GDN)与稀疏全注意力(QSA)交替的模型为什么"
          "在单卡上能用NVFP4 KV缓存放大上下文窗口？三点即可。")

import urllib.request

def one():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256, "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    el = time.time() - t0
    n = d["choices"][0]["message"]["content"]
    toks = d.get("usage", {}).get("completion_tokens") or len(n)
    return toks, el

def main():
    # C1: sequential single stream
    tot_t, tot_e = 0, 0.0
    for _ in range(3):
        t, e = one(); tot_t += t; tot_e += e
    print(f"C1: {tot_t/tot_e:.1f} tok/s ({tot_t} tok in {tot_e:.2f}s)")
    # C6: six parallel streams
    results = []
    lock = threading.Lock()
    def worker():
        t, e = one()
        with lock: results.append((t, e))
    threads = [threading.Thread(target=worker) for _ in range(6)]
    t0 = time.time()
    [th.start() for th in threads]; [th.join() for th in threads]
    wall = time.time() - t0
    agg = sum(t for t, _ in results) / wall
    print(f"C6: aggregate {agg:.0f} tok/s over {wall:.2f}s wall "
          f"({sum(t for t,_ in results)} tok total, per-stream "
          f"{sum(e for _,e in results)/6:.2f}s avg)")

if __name__ == "__main__":
    main()
