"""Discriminating probe: is graph-mode non-repeatability a staging TRANSIENT
(expected: output converges once slots settle) or real CORRUPTION (output keeps
drifting while staged counter is frozen)?

Method: hammer the same greedy prompt N times, hash each reply, and read the
cold-pool staged counter before/after. staged grows in lockstep with the changes
=> transient, benign. staged flat while replies change => R1/R2 violation.
"""
import hashlib, json, subprocess, re, sys

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "List three fruits:"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
UNIT = "sglang-dealignai-qwen4exp-nvfp4kv"


def gen(text, mx=32):
    payload = json.dumps({"text": text,
                          "sampling_params": {"temperature": 0, "max_new_tokens": mx}})
    out = subprocess.run(["curl", "-s", "-m", "120", "http://127.0.0.1:8000/generate",
                          "-H", "Content-Type: application/json", "-d", payload],
                         capture_output=True, text=True).stdout
    try:
        return json.loads(out)["text"]
    except Exception:
        return "ERR"


def stats():
    j = subprocess.run(["journalctl", "-u", UNIT, "--no-pager", "--since", "-40min"],
                       capture_output=True, text=True).stdout
    m = re.findall(r"calls=(\d+) strong=(\d+) staged=(\d+)", j)
    return tuple(int(x) for x in m[-1]) if m else (0, 0, 0)


def errcount():
    j = subprocess.run(["journalctl", "-u", UNIT, "--no-pager", "--since", "-40min"],
                       capture_output=True, text=True).stdout
    return len(re.findall(r"MISMATCH|stage_fail|CUDA error|Traceback", j))


c0 = stats()
e0 = errcount()
hashes = []
for i in range(N):
    t = gen(PROMPT)
    h = hashlib.md5(t.encode()).hexdigest()[:8]
    hashes.append(h)
    print(f"rep{i:2d} {h} staged={stats()[2]:5d}  {t[:64]!r}")

c1 = stats()
tail = 4
stable = len(set(hashes[-tail:])) == 1
print(f"\nhashes: {hashes}")
print(f"distinct={len(set(hashes))}/{N}  last{tail}-stable={stable}")
print(f"staged delta={c1[2]-c0[2]}  calls delta={c1[0]-c0[0]}  errors delta={errcount()-e0}")
print("VERDICT:", "CONVERGED-transient (benign)" if stable and c1[2] > c0[2]
      else ("STABLE-and-identical (benign)" if stable and c1[2] == c0[2]
            else "DRIFTING with staged frozen => REAL BUG" if c1[2] == c0[2]
            else "still changing after N reps => inspect"))
