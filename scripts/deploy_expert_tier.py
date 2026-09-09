#!/usr/bin/env python3
"""Deploy expert-tier overlay into the sglang patch tree (idempotent).

Run on CT112: /opt/sglang-env/bin/python3 deploy_expert_tier.py
Assumes expert_tier.py already copied next to layers/moe/.
"""
import re
import sys

SRT = "/opt/sglang-patch/sglang/python/sglang/srt"
LAYER = f"{SRT}/layers/moe/fused_moe_triton/layer.py"
MODEL = f"{SRT}/models/qwen4_exp.py"
UNIT = "/etc/systemd/system/sglang-dealignai-qwen4exp-nvfp4kv.service"


def patch(path, subs):
    src = open(path).read()
    for old, new in subs:
        if new.split("\n")[0].strip() and old in src and new not in src:
            src = src.replace(old, new, 1)
            print(f"[ok] {path}: applied ({old[:40]!r})")
        elif new in src:
            print(f"[skip] {path}: already applied")
        else:
            print(f"[MISS] {path}: anchor not found: {old[:60]!r}")
    open(path, "w").write(src)


# 1. layer.py: import + remap at top of forward_impl
patch(LAYER, [
    (
        "    def forward_impl(\n        self,\n        hidden_states: torch.Tensor,\n        topk_output: TopKOutput,",
        "    def forward_impl(\n        self,\n        hidden_states: torch.Tensor,\n        topk_output: TopKOutput,",
    ),
])
src = open(LAYER).read()
if "expert_tier" not in src:
    src = src.replace(
        "from sglang.srt.layers.moe.fused_moe_triton.fused_moe import ",
        "from sglang.srt.layers.moe import expert_tier\nfrom sglang.srt.layers.moe.fused_moe_triton.fused_moe import ",
        1,
    )
    src = src.replace(
        "        origin_hidden_states_dim = hidden_states.shape[-1]\n        assert self.quant_method is not None",
        "        origin_hidden_states_dim = hidden_states.shape[-1]\n        assert self.quant_method is not None\n        topk_output = expert_tier.apply_to_topk_output(self, topk_output)",
        1,
    )
    open(LAYER, "w").write(src)
    print("[ok] layer.py hooked")
else:
    print("[skip] layer.py already hooked")

# 2. model: call setup at end of load_weights
src = open(MODEL).read()
if "setup_expert_tiers" not in src:
    src = src.replace(
        "from sglang.srt.runtime_context import get_parallel",
        "from sglang.srt.layers.moe import expert_tier\nfrom sglang.srt.runtime_context import get_parallel",
        1,
    )
    src = src.replace(
        "            if isinstance(module, Qwen3_5GatedDeltaNet):\n                module.finalize_fused_in_proj()\n\n        return loaded_params",
        "            if isinstance(module, Qwen3_5GatedDeltaNet):\n                module.finalize_fused_in_proj()\n\n        expert_tier.setup_expert_tiers(self)\n        return loaded_params",
        1,
    )
    open(MODEL, "w").write(src)
    print("[ok] qwen4_exp.py hooked")
else:
    print("[skip] qwen4_exp.py already hooked")

# 3. systemd env lines
src = open(UNIT).read()
if "SGLANG_EXPERT_KEEP_MASK" not in src:
    src = src.replace(
        "Environment=PYTHONPATH=/opt/sglang-patch/sglang/python",
        "Environment=PYTHONPATH=/opt/sglang-patch/sglang/python\nEnvironment=SGLANG_EXPERT_KEEP_MASK=/opt/sglang-config/expert_keep_330_final.json\nEnvironment=SGLANG_EXPERT_KEEP_OFFLOAD=1",
        1,
    )
    open(UNIT, "w").write(src)
    print("[ok] unit env added")
else:
    print("[skip] unit env present")

sys.exit(0)
