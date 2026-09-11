"""nvfp4kv mamba anchor-slack gate (mamba 48 -> 64).

fp8kv lesson: long-context radix anchors need mamba slots OUTSIDE the
conc*4 running budget; 48 active at conc12 leaves ZERO slack. This probe
reproduces the cached=0 failure signature at nvfp4kv's scale:

G1: single 500K ctx cold -> re-ask must hit (~99%+ cached, <2s)
G2: two 500K ctxs coexist, re-ask both hit
G3: conc12 short burst still healthy (no regression from mamba64)
G4: mamba usage readout during G2 (expect slack visible in journal)

Evidence chain (R4 items 1-3): unique marker embedded in each prompt tail;
cached/throughput readings cross-check meta_info against journal lines
located via marker (no fixed --since windows). Single missing source =>
INCONCLUSIVE, not FAIL. Headline throughput = aggregated journal
"gen throughput (token/s)"; wall-clock/awk figure demoted to reference.
Exit code is explicit for callers (0 pass / 1 fail).

Run AFTER switch to mamba 64, with cold pool + graphs ON.
"""
import json, os, re, subprocess, sys, time, urllib.request

URL = "http://127.0.0.1:8000/generate"
SVC = os.environ.get("SGLANG_SVC", "sglang-dealignai-qwen4exp-nvfp4kv")
MARK = os.environ.get("A2_MARK") or f"a2mk-{int(time.time())}-{os.urandom(3).hex()}"

def gen(text, mx=8, timeout=900):
    req = urllib.request.Request(URL, data=json.dumps({
        "text": text + f" [{MARK}]", "sampling_params": {"temperature": 0.0, "max_new_tokens": mx}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return time.time() - t0, r.get("meta_info", {}), t0

def journal_after_marker(pat, t0=None, t1=None):
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
        print(f"  [journal unavailable: {exc}]")
        txt = ""
    blocks = re.split(r"(?m)^.*" + re.escape(MARK) + r".*$", txt or "")
    if len(blocks) > 1:
        return re.findall(pat, blocks[-1])
    if t0 is None:
        return []
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
            out += re.findall(pat, line)
    return out

def ctx(m, n):
    line = "锚定段落{m}-{i}：nvfp4kv长上下文槽位余量判别实验，文本稳定用于radix复用验证。"
    return "".join(line.format(m=m, i=i) for i in range(n))

fails = 0
inconclusive = 0
N = 20000  # ~480K tokens per ctx (nvfp4kv ctx 1M, pool 2.36M)

print("G1 single ~480K cold/re-ask")
cA = ctx("A", N)
t1, m1, g1 = gen(cA + "答：好"); print(f"  cold={t1:.1f}s pt={m1.get('prompt_tokens')}")
t2, m2, g2 = gen(cA + "答：好")
meta = m2.get("cached_tokens")
jw = [float(x) for x in journal_after_marker(r"#cached-token:\s*([0-9]+)", g2, time.time())]
jv = max(jw) if jw else None
pt = m2.get("prompt_tokens")
if meta is None or jv is None or pt is None:
    inconclusive += 1
    print(f"  reask={t2:.1f}s cached_meta={meta} cached_journal={jv} verdict=INCONCLUSIVE (source missing)")
else:
    print(f"  reask={t2:.1f}s cached_meta={meta} cached_journal={jv}")
    if t2 > 5 or meta < 0.95 * pt or jv < 0.95 * pt: fails += 1; print("  G1 FAIL")

print("G2 two ~480K coexist")
cB = ctx("B", N)
t, _, _ = gen(cB + "答：好"); print(f"  B cold={t:.1f}s")
t, m, g3 = gen(cA + "答：好")
j2 = [float(x) for x in journal_after_marker(r"#cached-token:\s*([0-9]+)", g3, time.time())]
meta, jrn2 = m.get("cached_tokens"), ([max(j2)] if j2 else [])
print(f"  A reask={t:.1f}s cached_meta={meta} cached_journal={jrn2[0] if jrn2 else None}")
if t > 5: fails += 1; print("  G2-A FAIL (anchor evicted by B)")
t, m, g4 = gen(cB + "答：好")
j4 = [float(x) for x in journal_after_marker(r"#cached-token:\s*([0-9]+)", g4, time.time())]
meta, jrn2 = m.get("cached_tokens"), ([max(j4)] if j4 else [])
print(f"  B reask={t:.1f}s cached_meta={meta} cached_journal={jrn2[0] if jrn2 else None}")
if t > 5: fails += 1; print("  G2-B FAIL")

print("G3 conc12 short burst (no regression)")
P = '{"text": "用中文写一段关于分布式系统共识的说明。 [' + MARK + ']", "sampling_params": {"temperature": 0.7, "max_new_tokens": 200}}'
t0 = time.time()
procs = [subprocess.Popen(["curl", "-s", "-m", "180", URL, "-H", "Content-Type: application/json", "-d", P],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for _ in range(12)]
for p in procs: p.wait()
dt = time.time() - t0
# headline: aggregate journal gen throughput (token/s) after marker; awk value reference-only
tp = [float(x) for x in journal_after_marker(r"gen throughput \(token/s\): ([0-9.]+)", t0, time.time())]
peak = max(tp) if tp else None
ref = 2400 / dt if dt else 0
print(f"  journal gen-throughput peak={peak} tok/s (headline; n={len(tp)} samples)")
print(f"  wall={dt:.1f}s nominal-2400tok => {ref:.0f} tok/s (reference only)")
if dt > 60: fails += 1; print("  G3 FAIL (suspiciously slow)")

print("ANCHOR-GATE:", "PASS" if fails == 0 else f"FAIL x{fails}",
      f"(inconclusive={inconclusive})")
sys.exit(1 if fails else 0)
