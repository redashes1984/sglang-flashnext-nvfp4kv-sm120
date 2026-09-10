#!/usr/bin/env python3
"""Deploy expert cold-pool v2 overlay into the sglang patch tree (idempotent).

Run on CT112:  /opt/sglang-env/bin/python3 deploy_expert_cold_pool.py
Assumes expert_cold_pool.py sits next to layers/moe/ (this script can copy it).

Three source hooks + one env block:
  1. layers/moe/topk.py   — bias gate at select_experts entry, stash+remap at exit
  2. model_executor/model_runner.py — after_forward_hook() at forward() tail
  3. model_executor/model_runner.py — maybe_shrink_after_process(model) in
     load_model() right after the loader returns (strictly post-pwal; v1's
     qwen4_exp end-of-load_weights point ran BEFORE pwal and was the
     silent-corruption root-cause candidate)
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

# --- v2.9 UPGRADE pass: rewrite the v2.8 layer_id-only gate into the identity
# gate BEFORE the pristine-anchor subs run (they then idempotent-skip). MTP
# draft layers reuse decoder layer_id 0..47 in the SAME process; layer_id-only
# gating masked+remapped the drafts (accept rate 0.21 on CT112 2026-09-10).
src_t = open(TOPK).read()
UPG29 = [
    ("""    _pre_mask_logits = None
    if layer_id is not None and _ecp.offload_active():
        router_logits, _pre_mask_logits = _ecp.apply_keep_mask(layer_id, router_logits)""",
     """    _pre_mask_logits = None
    # v2.9 IDENTITY gate: keep the layer_id, but only act when this caller's
    # TopKConfig was registered by the shrink — MTP draft layers reuse
    # decoder layer_id 0..47 inside the same process and must stay untouched.
    _cp_lid = _ecp.is_managed_config(topk_config) if layer_id is not None else None
    if _cp_lid is not None:
        router_logits, _pre_mask_logits = _ecp.apply_keep_mask(_cp_lid, router_logits)"""),
    ("""    if _pre_mask_logits is not None:
        _k = topk_ids.shape[-1] if topk_ids.dim() > 1 else 1
        _ecp.stash_demand(layer_id, _pre_mask_logits, recorder_topk_ids, _k)
    topk_ids = _ecp.remap_topk_ids(layer_id, topk_ids)""",
     """    if _cp_lid is not None:
        _k = topk_ids.shape[-1] if topk_ids.dim() > 1 else 1
        _ecp.stash_demand(_cp_lid, _pre_mask_logits, recorder_topk_ids, _k)
        topk_ids = _ecp.remap_topk_ids(_cp_lid, topk_ids)"""),
    ("""        packed_topk = _ecp.remap_packed_ids(layer_id, packed_topk)""",
     """        if _cp_lid is not None:
            packed_topk = _ecp.remap_packed_ids(_cp_lid, packed_topk)"""),
]
upgraded = 0
for old28, new29 in UPG29:
    if old28 in src_t:
        src_t = src_t.replace(old28, new29, 1)
        upgraded += 1
if upgraded:
    open(TOPK, "w").write(src_t)
    print(f"[ok] topk v2.8->v2.9 identity-gate upgrade: {upgraded} blocks")

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
    # v2.9 IDENTITY gate: keep the layer_id, but only act when this caller's
    # TopKConfig was registered by the shrink — MTP draft layers reuse
    # decoder layer_id 0..47 inside the same process and must stay untouched.
    _cp_lid = _ecp.is_managed_config(topk_config) if layer_id is not None else None
    if _cp_lid is not None:
        router_logits, _pre_mask_logits = _ecp.apply_keep_mask(_cp_lid, router_logits)

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
    if _cp_lid is not None:
        _k = topk_ids.shape[-1] if topk_ids.dim() > 1 else 1
        _ecp.stash_demand(_cp_lid, _pre_mask_logits, recorder_topk_ids, _k)
        topk_ids = _ecp.remap_topk_ids(_cp_lid, topk_ids)

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
        if _cp_lid is not None:
            packed_topk = _ecp.remap_packed_ids(_cp_lid, packed_topk)
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

# ---------------- 3. model_runner.py (shrink AFTER loader returns => post-pwal) ----------------
# v2.1 fix: shrink was originally hooked at end of qwen4_exp.load_weights, which
# runs BEFORE DefaultModelLoader calls process_weights_after_loading -> host pool
# would snapshot pre-pwal bytes (no swizzle/deinterleave/alphas) against post-pwal
# GPU rows. Moved to model_runner.load_model right after `self.model = loaded.model`,
# which is loader-agnostic (covers Default/Layered/QuantizedRL/ModelOpt paths).
sub_mr2 = [
    (
        "shrink-after-model-load",
        """        self.loader = loaded.loader
        self.model = loaded.model
""",
        f"""        self.loader = loaded.loader
        self.model = loaded.model
        {MARK}: cold-pool shrink strictly AFTER process_weights_after_loading;
        # draft workers never shrink (their unquant FusedMoE layer_id can
        # collide with keep-mask keys)
        from sglang.srt.layers.moe import expert_cold_pool as _ecp

        if not self.is_draft_worker:
            _ecp.maybe_shrink_after_process(self.model)
""",
    ),
]
patch_file(MR, sub_mr2, "model_runner-shrink")

print("[done] cold-pool v2 hooks wired. Deploy expert_cold_pool.py to:")
print(f"  {SRT}/layers/moe/expert_cold_pool.py")
sys.exit(0)
