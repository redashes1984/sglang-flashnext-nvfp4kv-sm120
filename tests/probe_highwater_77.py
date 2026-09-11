"""High-water replica of bench A on nvfp4kv (pool 2.36M, 77% fill).

bench A used 文档段落 template, ~336K ctxs, 77% pool fill -> cached=0 twice.
All hit experiments were at <=60% fill. This replicates EXACT template,
re-ask text and fill ratio (3 x ~605K on 2.36M = 77%) on the final-state
service (pinned, mamba64). Hit -> water level is not the culprit.

Evidence chain (R4 item 1/2): reask readings cross-check meta_info.cached_tokens
against journal "#cached-token:" located via this run's unique marker; single
missing source => INCONCLUSIVE instead of FAIL. Marker inside prompt tail line.
"""
import json, os, re, subprocess, sys, time, urllib.request

MARK = os.environ.get("A2_MARK") or f"a2mk-{int(time.time())}-{os.urandom(3).hex()}"
SVC = os.environ.get("SGLANG_SVC", "sglang-dealignai-qwen4exp-nvfp4kv")

def gen(text, mx=8):
    req = urllib.request.Request("http://127.0.0.1:8000/generate", data=json.dumps({
        "text": text + f" [{MARK}]", "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=900).read())
    return time.time() - t0, r.get("meta_info", {}), t0

def log(*a): print(*a, flush=True)

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

def make_ctx(m, n):   # EXACT bench template
    line_tpl = "文档段落{m}-{i}：本报告记录系统状态与测试结论，涉及网络与存储。"
    return "".join(line_tpl.format(m=m, i=i) for i in range(n))

N = 22000  # ~605K tokens per ctx (bench A's line ~27.6 tok/line)
cA, cB, cC = make_ctx("AA", N), make_ctx("BB", N), make_ctx("CC", N)
t, m, tg = gen(cA + "\n锚点A的编号是什么？只答编号。", 16); log(f"A1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m, tg = gen(cB + "\n锚点B的编号是什么？只答编号。", 16); log(f"B1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
t, m, tg = gen(cC + "\n锚点C的编号是什么？只答编号。", 16); log(f"C1 cold={t:.1f}s pt={m.get('prompt_tokens')}")
for tag, txt in (("A2", cA), ("B2", cB), ("C2", cC)):
    t, m, tg = gen(txt + "\n锚点" + tag[0] + "的编号是什么？只答编号。", 16)
    meta, jrn, verdict = cached_pair(m, tg, time.time())
    log(f"{tag} reask={t:.1f}s cached_meta={meta} cached_journal={jrn} verdict={verdict}")
log("HIGHWATER-DONE")
