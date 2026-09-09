#!/usr/bin/env python3
"""Step-by-step trace of after_forward_hook internals for the T3 scenario."""
import os, sys, json, tempfile
os.environ["SGLANG_EXPERT_KEEP_MASK"] = ""
import torch
sys.path.insert(0, "/opt/sglang-test")
import expert_cold_pool as ecp

E, keep_len, slots, k = 32, 20, 4, 10
keep = list(range(keep_len))
tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
json.dump({"0": keep}, tmp); tmp.close()
os.environ["SGLANG_EXPERT_KEEP_MASK"] = tmp.name
os.environ["SGLANG_EXPERT_KEEP_OFFLOAD"] = "1"
os.environ["SGLANG_EXPERT_COLD_POOL_SLOTS"] = str(slots)
ecp._STATE.update({"shrunk": True, "dynamic": True, "layers": {}, "stats": {
    "calls": 0, "strong": 0, "staged": 0, "evicted": 0, "h2d_bytes": 0,
    "demoted": 0, "promoted": 0, "stage_fail": 0}})
ecp._KEEP, ecp._KEEP_LOADED = None, False
ecp._BIAS_CACHE.clear(); ecp._REMAP_CACHE.clear(); ecp._STASH.clear()

m = torch.nn.Module()
m.w13_weight = torch.nn.Parameter(torch.zeros(E, 4, 2), requires_grad=False)
pool = ecp.LayerPool(0, keep, list(range(keep_len, E)), slots, m, E)
# fake host rows for all cold gids
for gid in range(keep_len, E):
    pool.host[gid] = {"w13_weight": torch.ones(4, 2) * gid}
ecp._STATE["layers"][0] = pool
ecp.build_tables(0, keep, E - keep_len, slots, E, "cpu")

logits2 = torch.full((4, E), -10.0)
logits2[:, 31] = 5.0
logits2[:, :10] = 4.0
topk2 = logits2.topk(k, dim=-1).indices
ecp.stash_demand(0, logits2, topk2, k)

# ---- manual replay of the hook body ----
buf = ecp._STASH[0]
ids = buf["ids"]
flat = ids[ids >= 0]
print("valid ids:", flat.numel())
n = (int(flat.numel()) // k) * k
flat = flat[:n].reshape(-1, k)
sc = buf["scores"][:n].reshape(-1, k)
print("flat[0]:", flat[0].tolist())
print("sc[0]:", [round(x,3) for x in sc[0].tolist()])
is_cold = pool.cold_mask_gpu[flat]
print("is_cold[0]:", is_cold[0].tolist())
hot_min = torch.where(~is_cold, sc, torch.full_like(sc, float("inf"))).amin(dim=-1, keepdim=True)
print("hot_min:", hot_min.flatten().tolist())
alpha = ecp.strong_alpha()
print("alpha:", alpha)
strong = is_cold & (sc >= alpha * hot_min)
print("strong[0]:", strong[0].tolist())
sel = flat[strong]
print("sel:", sel.tolist())

# now the actual hook
for it in range(3):
    ecp.stash_demand(0, logits2, topk2, k)
    ecp._STATE["stats"]["calls"] += 7
    ecp.after_forward_hook()
    s = ecp._STATE["stats"]
    print(f"it{it} calls={s['calls']} strong={s['strong']} staged={s['staged']} "
          f"need31={float(pool.need_counts[31]) if pool.need_counts is not None else None} "
          f"row31={pool.gid_row.get(31)} dirty={ecp._DIRTY}")
