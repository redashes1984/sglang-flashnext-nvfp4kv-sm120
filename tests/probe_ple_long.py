import json, urllib.request, time

def gen(text, mx=64, temp=0.7):
    req = urllib.request.Request("http://127.0.0.1:8000/generate",
        data=json.dumps({"text": text,
            "sampling_params": {"temperature": temp, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return time.time() - t0, r["text"]

# ~36K chars long prefill: heavy PLE row traffic, forces page-cache fault-in
big = "人工智能的发展历程非常悠久。" * 3000
dt, out = gen(big + "\n请续写一段关于深度学习的总结：", mx=64)
print(f"LONG-36K: {dt:.1f}s out={out[:80]!r}")

# second pass same prefix: radix hit path still fine with PLE on new disk
dt, out = gen(big + "\n请续写一段关于深度学习的总结：", mx=64)
print(f"LONG-36K-2nd: {dt:.1f}s out={out[:80]!r}")

# fresh random long prompt (no radix reuse) — cold PLE rows across the whole table
import random
random.seed(42)
words = ["量子计算", "蛋白质折叠", "城市规划", "爵士乐", "热带雨林", "操作系统", "印象派", "板块构造"]
prompt = "、".join(random.choice(words) for _ in range(6000))
dt, out = gen("这些概念如何交织？" + prompt, mx=48)
print(f"LONG-RANDOM: {dt:.1f}s out={out[:80]!r}")
