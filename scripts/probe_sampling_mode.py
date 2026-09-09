import json, subprocess
def gen(text, mx=24, **extra):
    sp = {"max_new_tokens": mx}; sp.update(extra)
    payload = json.dumps({"text": text, "sampling_params": sp})
    out = subprocess.run(["curl", "-s", "-m", "120", "http://127.0.0.1:8000/generate",
                          "-H", "Content-Type: application/json", "-d", payload],
                         capture_output=True, text=True).stdout
    try: return json.loads(out)["text"]
    except Exception: return "ERR"

P = "List three fruits:"
# 1) pure defaults (no temperature at all) x4
r = [gen(P) for _ in range(4)]
uniq = len(set(r)); print(f"default-sampling x4: {uniq} distinct")
for x in r[:2]: print("  ", repr(x)[:100])
# 2) explicit temp=0 x4
r2 = [gen(P, temperature=0.0) for _ in range(4)]
print(f"temp=0 x4: {len(set(r2))} distinct")
for x in r2[:2]: print("  ", repr(x)[:100])
# 3) temp=0 different prompt x3
r3 = [gen("Name three primary colors:", temperature=0.0) for _ in range(3)]
print(f"colors temp=0 x3: {len(set(r3))} distinct")
