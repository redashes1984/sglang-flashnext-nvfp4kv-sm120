#!/usr/bin/env python3
"""Deploy expert cold-pool v2 overlay into the sglang patch tree (idempotent).

Run on CT112:  /opt/sglang-env/bin/python3 deploy_expert_cold_pool.py
Assumes expert_cold_pool.py sits next to layers/moe/ (this script can copy it).

Three source hooks + one env block:
  1. layers/moe/topk.py   — bias gate at select_experts entry, stash+remap at exit
  2. model_executor/model_runner.py — after_forward_hook() at forward() tail
  3. models/qwen4_exp.py  — maybe_shrink_after_process() at end of load_weights
  4. (optional, --unit) systemd env lines on the test unit

Env contract:
  SGLANG_EXPERT_KEEP_MASK=/opt/sglang-config/expert_keep_330_final.json
  SGLANG_EXPERT_KEEP_OFFLOAD=1
  SGLANG_EXPERT_COLD_POOL_SLOTS=16
Everything is inert without KEEP_MASK — deploying the hooks is safe on prod units,
but we only wire env on the designated TEST unit (never prod, per v1 lesson).
"""
import sys

SRT = "/opt/sglang-patch/sglang/python/sglang/srt"
TOPK = f"{SRT}/layers/moe/topk.py"
MR = f"{SRT}/model_executor/model_runner.py"
MODEL = f"{SRT}/models/qwen4_exp.py"

MARK = "# expert-cold-pool-v2"


def patch_file(path, subs, label):
    src = open(path).read()
    changed = False
    for name, old, new in subs:
        if MARK in new and MARK in src and new in src:
            print(f"[skip] {label}/{name}: already applied")
            continue
        cnt = src.count(old)
        if cnt == 1:
            src = src.replace(old, new, 1)
            changed = True
            print(f"[ok] {label}/{name}: applied")
        else:
            print(f"[MISS] {label}/{name}: anchor count={cnt} for {old[:70]!r}")
            sys.exit(2)
    if changed:
        open(path, "w").write(src)


# ---------------- 1. topk.py ----------------
T = f"    {MARK}: runtime expert keep-mask/cold-pool (see layers/moe/expert_cold_pool.py)"

sub_topk = [
    (
        "keep-mask-gate",
        """    (
        router_logits,
        correction_bias,
    ) = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

    # DeepSeek V2/V3/R1 series models use grouped_top_k""",
        f"""    (
        router_logits,
        correction_bias,
    ) = expert_location_dispatch.transform_select_experts_inputs(
        router_logits=router_logits,
        correction_bias=correction_bias,
        info=expert_location_dispatch_info,
    )

{T}
    from sglang.srt.layers.moe import expert_cold_pool as _ecp

    _pre_mask_logits = None
    if layer_id is not None and _ecp.offload_active():
        router_logits, _pre_mask_logits = _ecp.apply_keep_mask(layer_id, router_logits)

    # DeepSeek V2/V3/R1 series models use grouped_top_k""",
    ),
    (
        "stash-remap-exit",
        """    topk_ids, topk_weights, recorder_topk_ids = _post_process_topk_ids(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=topk_config,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

    get_global_expert_distribution_recorder().on_select_experts(
        topk_ids=recorder_topk_ids
    )""",
        f"""    topk_ids, topk_weights, recorder_topk_ids = _post_process_topk_ids(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        topk_config=topk_config,
        router_logits=router_logits,
        num_token_non_padded=num_token_non_padded,
        layer_id=layer_id,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )

{T} + demand stash + global->slot remap
    if _pre_mask_logits is not None:
        _k = topk_ids.shape[-1] if topk_ids.dim() > 1 else 1
        _ecp.stash_demand(layer_id, _pre_mask_logits, recorder_topk_ids, _k)
    topk_ids = _ecp.remap_topk_ids(layer_id, topk_ids)

    get_global_expert_distribution_recorder().on_select_experts(
        topk_ids=recorder_topk_ids
    )""",
    ),
]
patch_file(TOPK, sub_topk, "topk")

# packed variant return — the fused-gating path bakes ids high-16; remap it too
src = open(TOPK).read()
old_pack = """    if packed_topk is not None:
        return StandardTopKOutputPacked(
            topk_weights, topk_ids, router_logits, packed_topk
        )"""
new_pack = f"""    if packed_topk is not None:
        {MARK}: packed ids carry global expert ids in the high 16 bits
        packed_topk = _ecp.remap_packed_ids(layer_id, packed_topk)
        return StandardTopKOutputPacked(
            topk_weights, topk_ids, router_logits, packed_topk
        )"""
if new_pack.split(":", 1)[0].strip() and "remap_packed_ids" not in src:
    if src.count(old_pack) == 1:
        src = src.replace(old_pack, new_pack, 1)
        open(TOPK, "w").write(src)
        print("[ok] topk/packed-remap applied")
    else:
        print(f"[MISS] topk/packed-remap anchor count={src.count(old_pack)}")
        sys.exit(2)
else:
    print("[skip] topk/packed-remap")

# ---------------- 2. model_runner.py ----------------
sub_mr = [
    (
        "after-forward-hook",
        """        if get_exec().moe.elastic_ep_backend is not None:
            self.maybe_join_ep_ranks()

        return output

    def _maybe_execute_deferred_mamba_cow_and_clear(""",
        f"""        if get_exec().moe.elastic_ep_backend is not None:
            self.maybe_join_ep_ranks()

        {MARK}: dynamic expert staging, strictly outside graph replay
        from sglang.srt.layers.moe import expert_cold_pool as _ecp

        _ecp.after_forward_hook()

        return output

    def _maybe_execute_deferred_mamba_cow_and_clear(""",
    ),
]
patch_file(MR, sub_mr, "model_runner")

# ---------------- 3. qwen4_exp.py ----------------
sub_model = [
    (
        "shrink-after-load",
        """        expert_tier.setup_expert_tiers(self)
        return loaded_params""",
        f"""        expert_tier.setup_expert_tiers(self)
        {MARK}: cold-pool shrink AFTER process_weights_after_loading (loader)
        from sglang.srt.layers.moe import expert_cold_pool as _ecp

        _ecp.maybe_shrink_after_process(self)
        return loaded_params""",
    ),
]
patch_file(MODEL, sub_model, "qwen4_exp")

print("[done] cold-pool v2 hooks wired. Deploy expert_cold_pool.py to:")
print(f"  {SRT}/layers/moe/expert_cold_pool.py")
sys.exit(0)
