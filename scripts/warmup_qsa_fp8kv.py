#!/opt/sglang-env/bin/python
"""Pre-load QSA/grammar Triton specializations right after sglang is ready.

Closes the late-device-load OOM window (kernels first loaded by real traffic
when free VRAM has dropped <1GiB). Shapes covered (from 2026-09-07 journal):
  - ~3K-token prompt sent twice -> prefix-cache hit -> _sparse_gqa_chunk_prefill,
    _qsa_graph_layout_kernel, _fused_slot_copy_kernel, get_last_loc_kernel,
    alloc_extend_kernel, assign_req_to_token_pool, _fused_commit_track_indices_kernel
  - response_format json_object -> apply_token_bitmask_inplace_kernel (xgrammar)
  - 8 concurrent requests -> bs 4/6/8 decode specializations (conc8 stack, 09-11)
"""
import json, sys, time, threading
import requests

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-Flash-Next-NVFP4"
PROMPT = "这是一段用于服务预热的人工生成文本，目的是触发稀疏注意力与分块预填充内核。" * 90  # ~3.4K chars

def log(m): print(f"[warmup] {time.strftime('%H:%M:%S')} {m}", flush=True)

def wait_ready(max_s=1200):
    t0 = time.time()
    while time.time() - t0 < max_s:
        try:
            if requests.get(BASE + "/health", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    return False

def chat(payload):
    r = requests.post(BASE + "/v1/chat/completions",
                      headers={"Content-Type": "application/json"},
                      json=payload, timeout=300)
    return r.status_code

def main():
    if not wait_ready():
        log("ERROR: server never became ready"); sys.exit(1)
    log("server ready, starting kernel warmup")
    base_msgs = [{"role": "user", "content": PROMPT}]
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    # 1) cold prefill, then 2) identical -> cached-token chunk-prefill (QSA kernels)
    log(f"pass1 cold prefill: {chat({'model':MODEL,'messages':base_msgs,'max_tokens':8,**kw})}")
    log(f"pass2 prefix-cache hit: {chat({'model':MODEL,'messages':base_msgs,'max_tokens':8,**kw})}")
    # 3) grammar bitmask kernel
    log(f"json_object grammar: {chat({'model':MODEL,'messages':[{'role':'user','content':'输出一个JSON对象，包含字段ok，值为true'}],'max_tokens':32,'response_format':{'type':'json_object'},**kw})}")
    # 4) bs=8 concurrent decode (conc8 stack: covers bs 4/6/8 eager specializations
    #    that boot-time graph capture does NOT warm on the Triton/eager fallback path)
    codes = []
    lock = threading.Lock()
    def one():
        c = chat({'model':MODEL,'messages':[{'role':'user','content':'说一句话，'+PROMPT[:200]}],'max_tokens':64,**kw})
        with lock: codes.append(c)
    ts = [threading.Thread(target=one) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    log(f"concurrent bs8: {codes}")
    log("warmup complete - late-load window closed")

if __name__ == "__main__":
    main()
