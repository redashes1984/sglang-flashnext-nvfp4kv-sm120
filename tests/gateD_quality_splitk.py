"""Gate D: end-to-end quality under split-K indexer.
  D-a  json_object grammar (structured output path)
  D-b  mid-doc branch: 500K doc, then ask anchored Q from 30% depth prefix
       (re-prefill from branch point; exercises indexer at interior windows)
  D-c  tool-call parse smoke
"""
import json, time, urllib.request

def chat(msgs, mx=64, extra=None):
    body = {"model": "Qwen3.8-Flash-Next-NVFP4", "messages": msgs,
            "max_tokens": mx, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    if extra: body.update(extra)
    req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return time.time() - t0, r

def doc(m, n):
    return "".join(f"文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。" for i in range(n))

print("== D-a grammar json_object ==", flush=True)
t, r = chat([{"role": "user", "content": "输出一个JSON对象，包含字段name(字符串)、age(整数)"}],
            80, {"response_format": {"type": "json_object"}})
c = r["choices"][0]["message"]["content"]
try:
    obj = json.loads(c); ok = "name" in obj and "age" in obj
except Exception:
    ok = False
print(f"grammar {t:.2f}s parse_ok={ok} content={c[:60]!r}", flush=True)

print("== D-b mid-doc branch 500K == ", flush=True)
D = doc("BD1", 22000)  # ~500K
# anchor at ~30% depth
cut = len(D) // 3
seg_anchor = f"分支锚点BR-{cut % 997}插入于此。"
Dm = D[:cut] + seg_anchor + D[cut:]
t, r = chat([{"role": "user", "content": Dm + "\n问：分支锚点的编号是多少？只答编号。"}], 24)
pt = r["usage"]["prompt_tokens"]
print(f"cold {t:.1f}s pt={pt} ans={r['choices'][0]['message']['content']!r}", flush=True)
# reask same full doc -> should hit
t, r = chat([{"role": "user", "content": Dm + "\n问：分支锚点的编号是多少？只答编号。"}], 24)
cached = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
print(f"reask-full {t:.1f}s cached={cached} ans={r['choices'][0]['message']['content']!r}", flush=True)
# branch: different suffix asking 10%-depth anchor -> interior-window indexer
t, r = chat([{"role": "user", "content": Dm[:cut // 3] + "\n这段文本里有没有出现'分支锚点'四个字？只答有或没有。"}], 16)
print(f"branch-10% {t:.1f}s ans={r['choices'][0]['message']['content']!r}", flush=True)
# same branch again -> cache behavior on interior prefix
t, r = chat([{"role": "user", "content": Dm[:cut // 3] + "\n这段文本里有没有出现'分支锚点'四个字？只答有或没有。"}], 16)
cached = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
print(f"branch-10% reask {t:.1f}s cached={cached} ans={r['choices'][0]['message']['content']!r}", flush=True)

print("== D-c tool-call smoke ==", flush=True)
t, r = chat([{"role": "user", "content": "北京天气怎么样"}], 80,
    {"tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "查询城市天气",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名"}},
            "required": ["city"]}}}]})
m = r["choices"][0]["message"]
tc = m.get("tool_calls") or []
ok = bool(tc) and tc[0]["function"]["name"] == "get_weather"
print(f"toolcall {t:.2f}s ok={ok} arg={tc[0]['function']['arguments'] if tc else m['content'][:40]!r}", flush=True)
print("GATE-D-DONE", flush=True)
