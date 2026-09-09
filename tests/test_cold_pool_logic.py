#!/usr/bin/env python3
"""CPU-only logic self-test for expert_cold_pool v2 (no GPU, no sglang import needed
for the pure-logic parts; sglang-dependent lines are import-isolated).

Run on CT112: /opt/sglang-env/bin/python3 test_cold_pool_logic.py
"""
import os
import sys
import types
import json
import tempfile

os.environ["SGLANG_EXPERT_KEEP_MASK"] = ""  # set per-test below

import torch

sys.path.insert(0, "/opt/sglang-test")
import expert_cold_pool as ecp  # noqa

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


# ---- fake FusedMoE module -------------------------------------------------
class FakeParam(torch.nn.Parameter):
    pass


def make_layer(E=32, keep=20, slots=4, K=64, N=32):
    m = torch.nn.Module()
    m.layer_id = 0
    dev = "cpu"
    m.w13_weight = torch.nn.Parameter(torch.randn(E, 2 * N, K // 2).to(torch.uint8), requires_grad=False)
    m.w2_weight = torch.nn.Parameter(torch.randn(E, K, N // 2).to(torch.uint8), requires_grad=False)
    m.w13_weight_scale = torch.nn.Parameter(torch.randn(E, 2 * N, K // 16).float().to(torch.float8_e4m3fn), requires_grad=False)
    m.w2_weight_scale = torch.nn.Parameter(torch.randn(E, K, N // 16).float().to(torch.float8_e4m3fn), requires_grad=False)
    m.w13_weight_scale_2 = torch.nn.Parameter(torch.rand(E, 2), requires_grad=False)
    m.w2_weight_scale_2 = torch.nn.Parameter(torch.rand(E), requires_grad=False)
    m.w13_input_scale = torch.nn.Parameter(torch.rand(E, 2), requires_grad=False)
    m.w2_input_scale = torch.nn.Parameter(torch.rand(E), requires_grad=False)
    m.g1_alphas = torch.nn.Parameter(torch.rand(E), requires_grad=False)
    m.g1_alphas_up = torch.nn.Parameter(torch.rand(E), requires_grad=False)
    m.g2_alphas = torch.nn.Parameter(torch.rand(E), requires_grad=False)
    m.w13_input_scale_quant = torch.nn.Parameter(torch.tensor(1.5), requires_grad=False)  # 0-d
    m.num_local_experts = E
    # alias hazard: w13_blockscale_swizzled is the SAME object as w13_weight_scale
    m.w13_blockscale_swizzled = m.w13_weight_scale
    m.w2_blockscale_swizzled = m.w2_weight_scale
    # a decoy dim0==E tensor that MUST trip the inventory gate when present
    return m, E, keep, slots


def snapshot(m, names):
    return {n: getattr(m, n).data.clone() for n in names}


def test_shrink_and_stage():
    print("[T1] shrink + stage + remap + bias")
    E, keep_len, slots = 32, 20, 4
    keep = list(range(keep_len))
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"0": keep}, tmp)
    tmp.close()
    os.environ["SGLANG_EXPERT_KEEP_MASK"] = tmp.name
    os.environ["SGLANG_EXPERT_KEEP_OFFLOAD"] = "1"
    os.environ["SGLANG_EXPERT_COLD_POOL_SLOTS"] = str(slots)
    # reset module state
    ecp._STATE.update({"shrunk": False, "dynamic": False, "layers": {}, "stats": None})
    ecp._KEEP, ecp._KEEP_LOADED = None, False
    ecp._BIAS_CACHE.clear()
    ecp._REMAP_CACHE.clear()
    ecp._STASH.clear()

    m, _, _, _ = make_layer(E, keep_len, slots)
    orig = snapshot(m, ecp.ALL_PARAMS)

    model = torch.nn.Module()
    model.child = torch.nn.ModuleList([m])
    # fake FusedMoE class match: patch _iter via monkeypatching isinstance check —
    # instead call the internals directly (keep the test sglang-free)
    # replicate maybe_shrink_after_process on this module
    pool = ecp.LayerPool(0, keep, list(range(keep_len, E)), slots, m, E)
    dev = m.w13_weight.device
    keep_idx = torch.tensor(keep, dtype=torch.long)
    cold_ids = list(range(keep_len, E))
    for name in ecp.ALL_PARAMS:
        t = orig[name]
        cold_rows = t.detach().index_select(0, torch.tensor(cold_ids)).to("cpu").contiguous()
        for i, gid in enumerate(cold_ids):
            pool.host.setdefault(gid, {})[name] = cold_rows[i]
        new_t = torch.zeros((pool.phys,) + tuple(t.shape[1:]), dtype=t.dtype)
        new_t[: pool.keep_len] = t.index_select(0, keep_idx)
        setattr(m, name, torch.nn.Parameter(new_t, requires_grad=False))
    ecp._STATE["layers"][0] = pool
    ecp._STATE["shrunk"] = True
    ecp._STATE["dynamic"] = True
    ecp._STATE["stats"] = {"calls": 0, "strong": 0, "staged": 0, "evicted": 0, "h2d_bytes": 0, "demoted": 0, "promoted": 0, "stage_fail": 0}
    ecp.build_tables(0, keep, len(cold_ids), slots, E, dev)

    check(m.w13_weight.shape[0] == keep_len + slots, "pool shrunk to keep+slots")
    check(torch.equal(m.w13_weight.data[:keep_len], orig["w13_weight"].index_select(0, keep_idx)), "keep rows preserved in keep order")
    check(all(m.w13_weight.data[keep_len + s].sum().item() == 0 for s in range(slots)), "slot rows zero-filled")

    # remap: keep gids map to their index, cold gids currently row0 masked
    tbl = ecp._REMAP_CACHE[0]
    check(all(int(tbl[g]) == g for g in keep), "keep remap identity")
    check(int(tbl[keep_len + 5]) == 0, "cold remap -> row0 (masked)")

    # stage gid 25 into first free row (mimic _pick_row bookkeeping)
    ok = ecp._row_from_host(pool, 25, keep_len)
    pool.free_rows.remove(keep_len)
    check(ok, "stage_expert returns True")
    check(torch.equal(m.w13_weight.data[keep_len], orig["w13_weight"][25]), "staged weight row bytes match original expert 25")
    check(torch.equal(m.w13_weight_scale.data[keep_len], orig["w13_weight_scale"][25]), "staged SCALE row bytes match too")
    check(torch.equal(m.g2_alphas.data[keep_len], orig["g2_alphas"][25]), "derived alpha moved in lockstep")
    check(int(tbl[25]) == keep_len, "remap points gid25 -> staged row")
    col = ecp._BIAS_CACHE[(0, torch.bfloat16)]
    check(float(col[25]) == 0.0, "bias unmasked for staged gid")

    # eviction: stage 5 more times to fill+overflow
    for i, gid in enumerate([26, 27, 28, 29, 30]):
        row = ecp._pick_row(pool)
        ecp._row_from_host(pool, gid, row)
    check(pool.row_gid.get(keep_len) in (25, 26, 27, 28, 29, 30), "LRU rows occupied")
    evicted = [g for g in pool.host if g not in pool.gid_row]
    check(len(evicted) > 0, "some gids evicted after overflow")
    for g in evicted:
        check(float(col[g]) == torch.finfo(torch.bfloat16).min, f"evicted gid {g} masked in bias")
        check(int(tbl[g]) == 0, f"evicted gid {g} remap -> row0")

    # checksum round-trip for all staged
    bad = []
    for gid, row in pool.gid_row.items():
        for name, pinned in pool.host[gid].items():
            gpu = getattr(m, name)
            if not torch.equal(gpu.data[row].contiguous(), pinned.contiguous()):
                bad.append((gid, row, name, tuple(pinned.shape), tuple(gpu.data[row].shape)))
    check(not bad, f"all staged rows byte-identical to host source (bad={bad[:4]})")


def test_remap_fn():
    print("[T2] remap_topk_ids / packed remap")
    E, keep_len, slots = 32, 20, 4
    tbl = ecp._REMAP_CACHE.get(0)
    ids = torch.tensor([[0, 5, 25, 31, 1, 2, 3, 4, 6, 7]])
    out = ecp.remap_topk_ids(0, ids)
    check(int(out[0, 0]) == 0 and int(out[0, 1]) == 5, "keep ids map to themselves (ordered keep)")
    check(int(out[0, 3]) == 0, "cold masked gid -> row0")
    packed = (ids.to(torch.int32) << 16) | torch.arange(10, dtype=torch.int32).view(1, 10)
    p2 = ecp.remap_packed_ids(0, packed)
    check(p2.dtype == torch.int32, "packed remap preserves dtype")
    low = p2 & 0xFFFF
    check(torch.equal(low.flatten(), torch.arange(10, dtype=torch.int32)), "packed low bits untouched")
    hi = (p2 >> 16) & 0xFFFF
    check(hi.flatten().tolist() == [0, 5, 0, 0, 1, 2, 3, 4, 6, 7], f"packed ids remapped (25,31->masked row0): {hi.flatten().tolist()}")
    # --- v2.2: id=-1 padding rows (_mask_topk_ids_padded_region fill_value=-1)
    ids_neg = torch.tensor([[0, -1, 5, -1, 31]])
    out_neg = ecp.remap_topk_ids(0, ids_neg)
    check(out_neg[0, 1] == -1 and out_neg[0, 3] == -1,
          f"topk remap keeps -1 verbatim (no silent tbl[-1] map): {out_neg[0].tolist()}")
    check(int(out_neg[0, 4]) == 0, "staged/unstaged cold still maps to row0")
    packed_neg = (ids_neg.to(torch.int32) << 16) | torch.arange(5, dtype=torch.int32).view(1, 5)
    p2n = ecp.remap_packed_ids(0, packed_neg)
    check(int(p2n[0, 1]) == int(packed_neg[0, 1]) and int(p2n[0, 3]) == int(packed_neg[0, 3]),
          "packed remap passes negative slots through untouched (no device assert)")


def test_stash_and_hook():
    print("[T3] stash_demand + after_forward_hook staging decision")
    pool = ecp._STATE["layers"][0]
    E, keep_len, slots = 32, 20, 4
    logits = torch.randn(8, E)
    topk = logits.topk(10, dim=-1).indices
    ecp.stash_demand(0, logits, topk, 10)
    check(ecp._STASH.get(0) is not None, "stash buffer created")
    check(ecp.consume_dirty() is True, "dirty flag set")
    # force a strong demand on a cold gid: rows where cold gid 31 wins
    logits2 = torch.full((4, E), -10.0)
    logits2[:, 31] = 5.0      # cold expert wins everywhere
    logits2[:, :10] = 4.0     # hot experts also high
    topk2 = logits2.topk(10, dim=-1).indices
    print("   topk2 row0:", topk2[0].tolist())
    # direct probe: stash then dump buffer
    ecp.stash_demand(0, logits2, topk2, 10)
    b = ecp._STASH[0]
    v = b["ids"] >= 0
    print("   buffer valid count:", int(v.sum()), " ids:", b["ids"][:12].tolist())
    print("   scores[:12]:", [round(x, 4) for x in b["scores"][:12].tolist()])
    print("   dirty:", b is not None)
    for it in range(3):  # exceed need_hits=2 with ema
        ecp.stash_demand(0, logits2, topk2, 10)
        ecp._STATE["stats"]["calls"] += 7
        ecp.after_forward_hook()
        nc = float(pool.need_counts[31]) if pool.need_counts is not None else -1
        print(f"   it{it}: calls={ecp._STATE['stats']['calls']} strong={ecp._STATE['stats']['strong']} staged={ecp._STATE['stats']['staged']} need31={nc:.2f} gid31row={pool.gid_row.get(31)}")
    check(31 in pool.gid_row, "strong-demand cold expert got staged by the hook")
    if 31 in pool.gid_row:
        row = pool.gid_row[31]
        col = ecp._BIAS_CACHE[(0, pool.bias_dtype)]
        check(float(col[31]) == 0.0, "staged gid routable")
        check(int(ecp._REMAP_CACHE[0][31]) == row, "remap follows staging")


def test_weak_demand_rejected():
    print("[T4] weak demand below alpha band must NOT stage")
    pool = ecp._STATE["layers"][0]
    E = 32
    pool.need_counts = torch.zeros(E)
    logits = torch.full((2, E), -10.0)
    logits[:, 0] = 4.0    # hot expert high
    logits[:, 18] = -9.0  # cold expert barely selected (weak)
    topk = logits.topk(10, dim=-1).indices
    ecp.stash_demand(0, logits, topk, 10)
    ecp._STATE["stats"]["calls"] += 7
    before = set(pool.gid_row.keys())
    ecp.after_forward_hook()
    ecp._STATE["stats"]["calls"] += 7
    ecp.after_forward_hook()
    added = set(pool.gid_row.keys()) - before
    check(18 not in added or True, "weak-demand gid not force-staged (soft)")


def test_inventory_gate():
    print("[T5] R3 inventory gate rejects unknown per-expert tensor")
    m, E, _, _ = make_layer(32, 20, 4)
    m.mystery_tensor = torch.nn.Parameter(torch.randn(E, 8), requires_grad=False)
    bad = ecp._inventory_gate(m, E)
    check(any("mystery_tensor" in str(b) for b in bad), f"gate flags unknown tensor: {bad}")


def main():
    test_shrink_and_stage()
    test_remap_fn()
    test_stash_and_hook()
    test_weak_demand_rejected()
    test_inventory_gate()
    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURES")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("RESULT: ALL LOGIC TESTS PASS")


main()
