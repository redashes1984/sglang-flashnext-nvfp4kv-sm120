"""cached0 discriminator v3 (lean): size-only vs branch-combination + dual-source cached.

  D1  ONE 495K bench-A ctx, reask  -> miss = per-ctx-size effect proven,
                                      template & branches fully bypassed
  D2  TWO 495K branch-template ctxs -> if D1 hits, is 2x495K still fine?

Evidence chain (R4 item 1/2): every reading cross-checks meta_info.cached_tokens
against the journal "#cached-token:" line located by this run's unique marker.
Single missing source => INCONCLUSIVE, never a fake FAIL. Marker appended inside
the prompt tail line; semantics unchanged, no syncs (CUDA-graph capture-safe).
"""
import json, os, re, subprocess, sys, time, urllib.request

MARK = os.environ.get("A2_MARK") or f"a2mk-{int(time.time())}-{os.urandom(3).hex()}"
SVC = "sglang-dealignai-qwen4exp-nvfp4kv"

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text + f" [{MARK}]", "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {})

def journal_stats(key, pat):
    """Latest numbers for `pat` in journal blocks after the most recent MARK line."""
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
    """Return (meta_value, journal_value, verdict) from both evidence sources."""
    meta = mi.get("cached_tokens")
    jrn = journal_stats("cached", r"#cached-token:\s*([0-9]+)")
    jv = jrn[0] if jrn else None
    if meta is None or jv is None:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "consistent" if abs(meta - jv) <= 8 else "MISMATCH"
    return meta, jv, verdict

def log(*a): print(*a, flush=True)

def doc(m, n):
    return "".join(f"文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。" for i in range(n))
def brn(m, n):
    return "".join(f"独立分支{m}第{i}段：本段专属内容互不重叠，用于radix多分支共存判别。" for i in range(n))

log("== D1 single 495K (bench-A template) ==")
A = doc("Z1", 22000)
t, m = gen(A + "\n锚点A的编号是什么？只答编号。", 16); log(f"D1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(A + "\n锚点A的编号是什么？只答编号。", 16)
meta, jrn, verdict = cached_pair(m)
log(f"D1 reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")

log("== D2 two 495K branch template ==")
B = brn("甲2", 20000); C = brn("乙2", 20000)
t, m = gen(B + "答：好", 16); log(f"D2 B cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m = gen(C + "答：好", 16); log(f"D2 C cold={t:.1f}s")
t, m = gen(B + "答：好", 16)
meta, jrn, verdict = cached_pair(m)
log(f"D2 B reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")
t, m = gen(C + "答：好", 16)
meta, jrn, verdict = cached_pair(m)
log(f"D2 C reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")
log("DISCRIM-DONE")
