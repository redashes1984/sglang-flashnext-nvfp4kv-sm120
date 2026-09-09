#!/opt/sglang-env/bin/python
"""Kernel warmup for the fp4kv experiment service (port 8000, radix OFF).

Closes the late-device-load OOM window. Late-load set observed 2026-09-07 r2
(first 200K-token request, free VRAM ~0.8GiB):
  chunk_gated_delta_rule_fwd_kernel_h_blockdim64  (GDN long prefill)
  _fused_gate_sigmoid_mul_add_kernel              (GDN prefill epilogue)
  _fused_sigmoid_mul_kernel                       (GDN prefill epilogue)
  _sparse_gqa_prefill                             (QSA cold prefill)
  _sparse_gqa_chunk_prefill                       (QSA chunked prefill w/ history)
=> needs LONG prefill passes (chunked across 4096-token chunks), which the
prod warmup (3K prompt + radix hit) does not provide, and radix is disabled
here anyway. Also covers grammar bitmask + bs=8 concurrent decode.
"""
import random, sys, time, threading
import requests

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-Flash-Next-NVFP4"
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu").split()

def log(m): print(f"[warmup-fp4kv] {time.strftime('%H:%M:%S')} {m}", flush=True)

def wait_ready(max_s=1500):
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
                      json=payload, timeout=900)
    return r.status_code

def filler(nwords, salt):
    rnd = random.Random(salt)
    return " ".join(rnd.choice(WORDS) for _ in range(nwords))

def main():
    if not wait_ready():
        log("ERROR: server never became ready"); sys.exit(1)
    log("server ready, starting kernel warmup")
    kw = {"chat_template_kwargs": {"enable_thinking": False}}
    # 1) short cold prefill + decode
    log(f"pass1 short: {chat({'model':MODEL,'messages':[{'role':'user','content':'说一句话，带句号。'}],'max_tokens':16,**kw})}")
    # 2) long prefill ~50K words -> GDN chunk kernels + sparse_gqa prefill/chunk_prefill
    p = filler(50000, 11) + "\n\n回答：OK"
    log(f"pass2 long-50K: {chat({'model':MODEL,'messages':[{'role':'user','content':p}],'max_tokens':8,**kw})}")
    # 3) different length -> other divisibility specializations
    p = filler(33000, 22) + "\n\n回答：OK"
    log(f"pass3 long-33K: {chat({'model':MODEL,'messages':[{'role':'user','content':p}],'max_tokens':8,**kw})}")
    # 4) grammar bitmask kernel (xgrammar)
    log(f"pass4 grammar: {chat({'model':MODEL,'messages':[{'role':'user','content':'输出一个JSON对象，包含字段ok，值为true'}],'max_tokens':32,'response_format':{'type':'json_object'},**kw})}")
    # 5) bs=8 concurrent decode (new graph bucket)
    codes = []
    lock = threading.Lock()
    def one():
        c = chat({'model':MODEL,'messages':[{'role':'user','content':'说一个成语并解释，一句话。'}],'max_tokens':64,**kw})
        with lock: codes.append(c)
    ts = [threading.Thread(target=one) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    log(f"pass5 concurrent bs6: {codes}")
    log("warmup complete - late-load window closed")

if __name__ == "__main__":
    main()
