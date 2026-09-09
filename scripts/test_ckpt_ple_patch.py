"""Integration test for the int8 ckpt x PLE side-states patch (CPU-only)."""
import torch
from sglang.srt.mem_cache.mamba_checkpoint_pool import MambaCheckpointPool
from sglang.srt.mem_cache.ple_state_pool import ShortConvPool, NGramPool

DEV = "cpu"
fails = []

def check(name, cond):
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        fails.append(name)

class Cache: pass

def make_ap(L, S, H, DV, DK, cw, seed=42):
    ap = Cache(); ap.mamba_cache = Cache()
    torch.manual_seed(seed)
    ap.mamba_cache.temporal = torch.randn(L, S, H, DV, DK, dtype=torch.bfloat16)
    ap.mamba_cache.conv = [torch.randn(L, S, *cw, dtype=torch.bfloat16)]
    return ap

# ---- T1: dual-layout roundtrip with realistic Qwen4 shapes ----
L, NS, H, DV, DK = 48, 96, 16, 64, 128
sc = ShortConvPool(size=6, state_shape=(1, 1620), layer_ids=list(range(4)),
                   dtype=torch.bfloat16, device=DEV)
ng = NGramPool(size=6, context_len=8, eos_token_id=151645, device=DEV)
pool = MambaCheckpointPool(num_layers=L, num_slots=NS, num_heads=H, head_v_dim=DV,
                           head_k_dim=DK, conv_shapes=[(1, 1620)],
                           conv_dtype=torch.bfloat16, device=DEV,
                           temporal_dtype=torch.bfloat16,
                           ple_side_states=[sc, ng])
ap = make_ap(L, 7, H, DV, DK, (1, 1620))

a = torch.tensor([1, 2]); c = torch.tensor([3, 4])
before_t = ap.mamba_cache.temporal[:, a].clone()
before_conv = ap.mamba_cache.conv[0][:, a].clone()
sc.conv_state[:, 1] = 0.75; sc.conv_state[:, 2] = -0.25
ng.context[1] = torch.arange(8) + 100; ng.context[2] = torch.arange(8) + 200
ref_sc = sc.conv_state[:, a].clone(); ref_ng = ng.context[a].clone()

pool.store_from_active(ap, a, c)
ap.mamba_cache.temporal[:, a] = 0          # simulate slot donation
ap.mamba_cache.conv[0][:, a] = 0
sc.conv_state[:, a] = 0
ng.context[a] = 151645                     # simulate reset_slots

pool.load_to_active(ap, c, a)
check("T1 temporal roundtrip",
      torch.allclose(ap.mamba_cache.temporal[:, a].float(), before_t.float(), atol=0.06))
check("T1 conv-window roundtrip",
      torch.allclose(ap.mamba_cache.conv[0][:, a].float(), before_conv.float()))
check("T1 short-conv rows restored", torch.equal(sc.conv_state[:, a], ref_sc))
check("T1 ngram rows restored", torch.equal(ng.context[a], ref_ng))
check("T1 ngram mirror dtype exact (int64)", ng.context[a].dtype == torch.int64)

# ---- T2: two checkpoints from one active slot; independent ckpt slots ----
pool2 = MambaCheckpointPool(num_layers=2, num_slots=4, num_heads=2, head_v_dim=4,
                            head_k_dim=8, conv_shapes=[(4,)], conv_dtype=torch.bfloat16,
                            device=DEV, temporal_dtype=torch.bfloat16,
                            ple_side_states=[sc, ng])
ap2 = make_ap(2, 5, 2, 4, 8, (4,))
a1 = torch.tensor([1]); c1 = torch.tensor([1]); c2 = torch.tensor([2])
sc.conv_state[:, 1] = 1.0
pool2.store_from_active(ap2, a1, c1)
sc.conv_state[:, 1] = 2.0
pool2.store_from_active(ap2, a1, c2)
sc.conv_state[:, 1] = 0.0                  # wipe active row
pool2.load_to_active(ap2, c2, a1)
check("T2 latest checkpoint wins", sc.conv_state[:, 1].float().mean().item() == 2.0)
pool2.load_to_active(ap2, c1, a1)
check("T2 earlier checkpoint intact", sc.conv_state[:, 1].float().mean().item() == 1.0)

# ---- T3: estimate arithmetic covers mirrors exactly ----
real_ple = sum(m.numel() * m.element_size() for _, m, _ in pool.ple_mirrors)
est = MambaCheckpointPool.estimate_mem_usage_bytes(
    num_layers=L, num_slots=NS, num_heads=H, head_v_dim=DV, head_k_dim=DK,
    conv_shapes=[(1, 1620)], conv_dtype=torch.bfloat16, temporal_dtype=torch.bfloat16,
    ple_extra_bytes=(NS + 1) * 4 * 1 * 1620 * 2 + (NS + 1) * 8 * 8)
check("T3 ple estimate matches mirror bytes", est["ple"] == real_ple)

# ---- T4: allocator lifecycle ----
before_avail = pool.available_size()
s1 = pool.alloc(1)
check("T4 alloc decrements availability", pool.available_size() == before_avail - 1)
pool.clear()
check("T4 clear restores availability", pool.available_size() == before_avail)

# ---- T5: mirror slots auto-size to ckpt pool, not active pool ----
check("T5 short-conv mirror sized ckpt+1", pool.ple_mirrors[0][1].shape[1] == NS + 1)
check("T5 ngram mirror sized ckpt+1", pool.ple_mirrors[1][1].shape[0] == NS + 1)
check("T5 source pools untouched size",
      sc.conv_state.shape[1] == 7 and ng.context.shape[0] == 7)

# ---- T6: maybe_init factory path must not leak ple_side_states into estimate kwargs ----
import sglang.srt.mem_cache.mamba_checkpoint_pool as mcp
from types import SimpleNamespace

class _Shape:
    temporal = (H, DV, DK)
    conv = [(1, 8)]
class _DType:
    temporal = torch.bfloat16
    conv = torch.bfloat16

fake_params = SimpleNamespace(shape=_Shape(), dtype=_DType())
fake_exec = SimpleNamespace(mamba=SimpleNamespace(
    enable_int8_mamba_checkpoint=True, int8_mamba_ckpt_size=8))
import sglang.srt.runtime_context as rc
_orig_get_exec = rc.get_exec
rc.get_exec = lambda: fake_exec
try:
    p6 = mcp.maybe_init_int8_mamba_checkpoint_pool(
        mamba_size=6, cache_params=fake_params,
        mamba_layer_ids=[0, 1, 2, 3], device="cpu",
        ple_side_states=[sc, ng])
    check("T6 factory returns pool", p6 is not None)
    check("T6 factory built mirrors", len(p6.ple_mirrors) == 2)
    check("T6 factory mirror ckpt-sized", p6.ple_mirrors[0][1].shape[1] == 9)
except TypeError as e:
    check(f"T6 factory path TypeError: {e}", False)
finally:
    rc.get_exec = _orig_get_exec

print("\nFAILURES:", fails if fails else "none")
