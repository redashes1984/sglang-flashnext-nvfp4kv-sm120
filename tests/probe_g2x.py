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
    return time.time() - t0, r.get("meta_info", {}), t0

def log(*a): print(*a); sys.stdout.flush()

def journal_stats(key, pat, t0=None, t1=None):
    """Numbers matching pat after the newest MARK line; no MARK in journal
    (log_requests=False on live) => syslog timestamp-window fallback."""
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
    if len(blocks) > 1:
        return [float(n) for n in re.findall(pat, blocks[-1])]
    if t0 is None:
        return []
    return _window_nums(txt, pat, t0, t1)

def _window_nums(txt, pat, t0, t1):
    from datetime import datetime
    now = datetime.now()
    out = []
    for line in txt.splitlines():
        m = re.match(r"^([A-Z][a-z]{2}) +(\d{1,2}) (\d{2}):(\d{2}):(\d{2}) ", line)
        if not m:
            continue
        mon, d, hh, mm, ss = m.groups()
        try:
            ts = datetime(now.year, datetime.strptime(mon, "%b").month, int(d), int(hh), int(mm), int(ss))
        except ValueError:
            continue
        if ts > now:
            ts = ts.replace(year=now.year - 1)
        e = ts.timestamp()
        if t0 - 1 <= e <= t1 + 3:
            out += [float(n) for n in re.findall(pat, line)]
    return out

def cached_pair(mi, t0=None, t1=None):
    meta = mi.get("cached_tokens")
    jrn = journal_stats("cached", r"#cached-token:\s*([0-9]+)", t0, t1)
    jv = max(jrn) if jrn else None
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
t, m, tg = gen(A + "答：好"); log(f"A cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m, tg = gen(B + "答：好"); log(f"B cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m, tg = gen(A + "答：好")
meta, jrn, verdict = cached_pair(m, tg, time.time())
log(f"A reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")
t, m, tg = gen(B + "答：好")
meta, jrn, verdict = cached_pair(m, tg, time.time())
log(f"B reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")

# third branch pressure, then re-ask all (LRU depth at mamba 64: 3 branches)
t, m, tg = gen(C + "答：好"); log(f"C cold={t:.1f}s pt={m.get('prompt_tokens')}")
for tag, txt in (("A reask2", A), ("B reask2", B), ("C reask", C)):
    t, m, tg = gen(txt + "答：好")
    meta, jrn, verdict = cached_pair(m, tg, time.time())
    log(f"{tag}={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")
log("G2X-DONE")
