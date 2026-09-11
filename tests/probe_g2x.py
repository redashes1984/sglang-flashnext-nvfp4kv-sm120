"""G2 only, scaled: two ~210K ctxs coexist, re-ask both. flush to stdout.

Evidence chain (R4 item 1/2): unique marker in each prompt; cached readings
cross-check meta_info vs journal "#cached-token:" after the marker line.
Single missing source => INCONCLUSIVE, not FAIL. Marker inside prompt tail.
"""
import json, os, re, subprocess, sys, time, urllib.request

MARK = os.environ.get("A2_MARK") or f"a2mk-{int(time.time())}-{os.urandom(3).hex()}"
SVC = "sglang-dealignai-qwen4exp-nvfp4kv"

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text + f" [{MARK}]", "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return time.time() - t0, r.get("meta_info", {})

def log(*a): print(*a); sys.stdout.flush()

def journal_stats(key, pat):
    txt = None
    try:
        if os.environ.get("HERMES_JOURNAL_FILE"):
            txt = open(os.environ["HERMES_JOURNAL_FILE"]).read()
        else:
            txt = subprocess.run(["journalctl", "-u", SVC, "--no-pager"],
                                 capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:
        print(f"[journal unavailable: {exc}]")
        txt = ""
    blocks = re.split(r"(?m)^.*" + re.escape(MARK) + r".*$", txt or "")
    nums = re.findall(pat, blocks[-1]) if len(blocks) > 1 else []
    return [float(n) for n in nums]

def cached_pair(mi):
    meta = mi.get("cached_tokens")
    jv_list = journal_stats("cached", r"#cached-token:\s*([0-9]+)")
    jv = jv_list[0] if jv_list else None
    if meta is None or jv is None:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "consistent" if abs(meta - jv) <= 8 else "MISMATCH"
    return meta, jv, verdict

def ctx(m, n):
    line = "独立分支{m}第{i}段：本段专属内容互不重叠，用于radix多分支共存判别。"
    return "".join(line.format(m=m, i=i) for i in range(n))

N = 9000  # ~210K tokens
A = ctx("甲", N); B = ctx("乙", N); C = ctx("丙", N)
t, m = gen(A + "答：好"); log(f"A cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(B + "答：好"); log(f"B cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(A + "答：好")
meta, jrn, verdict = cached_pair(m)
log(f"A reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")
t, m = gen(B + "答：好")
meta, jrn, verdict = cached_pair(m)
log(f"B reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")

# third branch pressure, then re-ask all (LRU depth at mamba 64: 3 branches)
t, m = gen(C + "答：好"); log(f"C cold={t:.1f}s pt={m.get('prompt_tokens')}")
for tag, txt in (("A reask2", A), ("B reask2", B), ("C reask", C)):
    t, m = gen(txt + "答：好")
    meta, jrn, verdict = cached_pair(m)
    log(f"{tag}={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")
log("G2X-DONE")
